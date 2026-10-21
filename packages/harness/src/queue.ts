import { putCheckpoint, removeCheckpoint, toRecord, type Checkpoint } from "./checkpoint"
import { CHECKPOINT_STORES, CONTRIBUTION, OUTBOX, done, openDb, result } from "./db"
import type { LogFile } from "./log"

/** A change to the file's learner's checkpoint, made in the same transaction as the put. */
export type CheckpointChange = { save: Checkpoint<unknown> } | { remove: string }

/**
 * How a learner's logs fare at the gate. `lastPr`: the highest of its PRs already looked at (0: none). `closedInARow`:
 * of those, how many in a row (newest last) the gate closed without merging. `localOnly`: after
 * MAX_CLOSED_IN_A_ROW of them the learner keeps learning in its tab but its logs are no longer sent.
 */
export interface Contribution { learnerId: string; lastPr: number; closedInARow: number; localOnly: boolean }

export interface Queue {
  /**
   * Adds a file. Rejects if (learnerId, seq) already exists: a queued file is never overwritten.
   * With `checkpoint`, also saves or removes that learner's checkpoint, atomically: both happen or neither.
   */
  put(f: LogFile, checkpoint?: CheckpointChange): Promise<void>
  /** Unsent files only, ordered by learnerId, then seq; never a local-only learner's (they are not sent). */
  pending(): Promise<LogFile[]>
  /** Marks a file sent; `pr`: the pull request it went in (so dropPr can find it if the gate closes that PR). */
  markSent(learnerId: string, seq: number, pr?: number): Promise<void>
  /**
   * The gate closed PR `pr` without merging it: that PR's files of the learner are marked dropped (they stay sent, so
   * they are never proposed again; prune removes them). Resolves to the number marked.
   */
  dropPr(learnerId: string, pr: number): Promise<number>
  /** Highest-seq file for the learner, sent or not (used to resume numbering). */
  latest(learnerId: string): Promise<LogFile | undefined>
  /** Every file of the learner, sent or not, by seq (the page redraws a resumed learner's curve from them). */
  files(learnerId: string): Promise<LogFile[]>
  /**
   * Deletes every learner's sent (and dropped) files except its newest file and a final one: they are on GitHub (or
   * rejected), and a life lives forever. The newest file stays so latest() still resumes numbering, and says a finished
   * life is over (its learnerId is never lived again). Resolves to the number of files deleted.
   */
  prune(): Promise<number>
  /**
   * Bounds the unsent files a browser keeps per learner (a visitor with no token never sends any; a local-only learner
   * never will): of a learner with more than `max` unsent, its oldest unsent files are deleted, this tab's learner's
   * too (the gate allows seq gaps). Never its newest file, nor a final one. Resolves to the number deleted.
   */
  capUnsent(max: number): Promise<number>
  /** The learner's contribution state (a fresh one if it has none). */
  contribution(learnerId: string): Promise<Contribution>
  setContribution(c: Contribution): Promise<void>
  /** The learners that no longer contribute. */
  localOnly(): Promise<string[]>
}

/** The most unsent files a browser keeps per learner (Queue.capUnsent): at 8 logs an hour, a day of a tab's life. */
export const MAX_UNSENT = 200

/** `pr`: the PR the file was sent in, if known. `dropped`: that PR was closed unmerged (the file stays sent, never resent). */
interface Row { learnerId: string; seq: number; sent: boolean; file: LogFile; pr?: number; dropped?: boolean }

/** Each learner's highest seq. */
function newestSeq(all: Row[]): Map<string, number> {
  const m = new Map<string, number>()
  for (const r of all) if ((m.get(r.learnerId) ?? -1) < r.seq) m.set(r.learnerId, r.seq)
  return m
}

/** The database is (re)opened on every call, so a version change that closed the connection does not break the queue. */
export async function openQueue(idb: IDBFactory): Promise<Queue> {
  await openDb(idb) // fail early (e.g. an upgrade blocked by an old tab)
  const rows = async (): Promise<Row[]> => result((await openDb(idb)).transaction(OUTBOX).objectStore(OUTBOX).getAll() as IDBRequest<Row[]>)
  return {
    async put(f, checkpoint) {
      const target = checkpoint === undefined ? undefined : "save" in checkpoint ? checkpoint.save.learnerId : checkpoint.remove
      if (target !== undefined && target !== f.learnerId) throw new Error(`a checkpoint of learner ${target} cannot travel with a log of ${f.learnerId}`)
      const record = checkpoint && "save" in checkpoint ? await toRecord(checkpoint.save) : undefined // before the transaction
      const tx = (await openDb(idb)).transaction(checkpoint ? [OUTBOX, ...CHECKPOINT_STORES] : [OUTBOX], "readwrite")
      const row: Row = { learnerId: f.learnerId, seq: f.seq, sent: false, file: f }
      const finished = done(tx)
      try {
        tx.objectStore(OUTBOX).add(row).onerror = () => { /* surfaces via tx.onabort */ }
        if (record) putCheckpoint(tx, record)
        else if (checkpoint && "remove" in checkpoint) removeCheckpoint(tx, checkpoint.remove)
      } catch (e) {
        // A request that throws at once (a value that cannot be stored) must not leave the others to commit alone.
        finished.catch(() => {})
        tx.abort()
        throw e
      }
      await finished
    },
    async pending() {
      const tx = (await openDb(idb)).transaction([OUTBOX, CONTRIBUTION])
      const [all, states] = await Promise.all([
        result(tx.objectStore(OUTBOX).getAll() as IDBRequest<Row[]>),
        result(tx.objectStore(CONTRIBUTION).getAll() as IDBRequest<Contribution[]>),
      ])
      const local = new Set(states.filter(c => c.localOnly).map(c => c.learnerId))
      return all.filter(r => !r.sent && !local.has(r.learnerId)).map(r => r.file)
    },
    async markSent(learnerId, seq, pr) {
      const tx = (await openDb(idb)).transaction(OUTBOX, "readwrite")
      const store = tx.objectStore(OUTBOX)
      const finished = done(tx)
      const row = await result(store.get([learnerId, seq]) as IDBRequest<Row | undefined>)
      if (row) store.put({ ...row, sent: true, ...(pr !== undefined ? { pr } : {}) })
      await finished
    },
    async dropPr(learnerId, pr) {
      const tx = (await openDb(idb)).transaction(OUTBOX, "readwrite")
      const store = tx.objectStore(OUTBOX)
      const finished = done(tx)
      const all = await result(store.getAll() as IDBRequest<Row[]>)
      const hit = all.filter(r => r.learnerId === learnerId && r.pr === pr && !r.dropped)
      for (const r of hit) store.put({ ...r, sent: true, dropped: true })
      await finished
      return hit.length
    },
    async latest(learnerId) {
      const mine = (await rows()).filter(r => r.learnerId === learnerId)
      return mine.reduce<Row | undefined>((m, r) => (m === undefined || r.seq > m.seq ? r : m), undefined)?.file
    },
    async files(learnerId) {
      return (await rows()).filter(r => r.learnerId === learnerId).sort((a, b) => a.seq - b.seq).map(r => r.file)
    },
    async capUnsent(max) {
      const tx = (await openDb(idb)).transaction(OUTBOX, "readwrite")
      const store = tx.objectStore(OUTBOX)
      const finished = done(tx)
      const all = await result(store.getAll() as IDBRequest<Row[]>)
      const newest = newestSeq(all)
      let dropped = 0
      for (const [id, top] of newest) {
        const unsent = all.filter(r => r.learnerId === id && !r.sent).sort((a, b) => a.seq - b.seq)
        for (const r of unsent.slice(0, Math.max(0, unsent.length - max))) {
          if (r.seq === top || r.file.final) continue
          store.delete([r.learnerId, r.seq])
          dropped++
        }
      }
      await finished
      return dropped
    },
    async prune() {
      const tx = (await openDb(idb)).transaction(OUTBOX, "readwrite")
      const store = tx.objectStore(OUTBOX)
      const finished = done(tx)
      const all = await result(store.getAll() as IDBRequest<Row[]>)
      const newest = newestSeq(all)
      const gone = all.filter(r => r.sent && !r.file.final && r.seq !== newest.get(r.learnerId))
      for (const r of gone) store.delete([r.learnerId, r.seq])
      await finished
      return gone.length
    },
    async contribution(learnerId) {
      const c = await result((await openDb(idb)).transaction(CONTRIBUTION).objectStore(CONTRIBUTION).get(learnerId) as IDBRequest<Contribution | undefined>)
      return c ?? { learnerId, lastPr: 0, closedInARow: 0, localOnly: false }
    },
    async setContribution(c) {
      const tx = (await openDb(idb)).transaction(CONTRIBUTION, "readwrite")
      const finished = done(tx)
      tx.objectStore(CONTRIBUTION).put({ learnerId: c.learnerId, lastPr: c.lastPr, closedInARow: c.closedInARow, localOnly: c.localOnly })
      await finished
    },
    async localOnly() {
      const all = await result((await openDb(idb)).transaction(CONTRIBUTION).objectStore(CONTRIBUTION).getAll() as IDBRequest<Contribution[]>)
      return all.filter(c => c.localOnly).map(c => c.learnerId).sort()
    },
  }
}
