import { toRecord, type Checkpoint } from "./checkpoint"
import { CHECKPOINTS, OUTBOX, done, openDb, result } from "./db"
import type { LogFile } from "./log"

/** A change to the file's learner's checkpoint, made in the same transaction as the put. */
export type CheckpointChange = { save: Checkpoint<unknown> } | { remove: string }

export interface Queue {
  /**
   * Adds a file. Rejects if (learnerId, seq) already exists: a queued file is never overwritten.
   * With `checkpoint`, also saves or removes that learner's checkpoint, atomically: both happen or neither.
   */
  put(f: LogFile, checkpoint?: CheckpointChange): Promise<void>
  /** Unsent files only, ordered by learnerId, then seq. */
  pending(): Promise<LogFile[]>
  markSent(learnerId: string, seq: number): Promise<void>
  /** Highest-seq file for the learner, sent or not (used to resume numbering). */
  latest(learnerId: string): Promise<LogFile | undefined>
  /** Every file of the learner, sent or not, by seq (the page redraws a resumed learner's curve from them). */
  files(learnerId: string): Promise<LogFile[]>
  /**
   * Deletes the sent files of every learner whose final file has been sent: their lives are over and on GitHub.
   * The final file itself stays, so latest() still says the life is over (its learnerId is never lived again).
   * Resolves to the number of files deleted.
   */
  prune(): Promise<number>
}

interface Row { learnerId: string; seq: number; sent: boolean; file: LogFile }

/** The database is (re)opened on every call, so a version change that closed the connection does not break the queue. */
export async function openQueue(idb: IDBFactory): Promise<Queue> {
  await openDb(idb) // fail early (e.g. an upgrade blocked by an old tab)
  const rows = async (): Promise<Row[]> => result((await openDb(idb)).transaction(OUTBOX).objectStore(OUTBOX).getAll() as IDBRequest<Row[]>)
  return {
    async put(f, checkpoint) {
      const target = checkpoint === undefined ? undefined : "save" in checkpoint ? checkpoint.save.learnerId : checkpoint.remove
      if (target !== undefined && target !== f.learnerId) throw new Error(`a checkpoint of learner ${target} cannot travel with a log of ${f.learnerId}`)
      const record = checkpoint && "save" in checkpoint ? await toRecord(checkpoint.save) : undefined // before the transaction
      const tx = (await openDb(idb)).transaction(checkpoint ? [OUTBOX, CHECKPOINTS] : [OUTBOX], "readwrite")
      const row: Row = { learnerId: f.learnerId, seq: f.seq, sent: false, file: f }
      const finished = done(tx)
      try {
        tx.objectStore(OUTBOX).add(row).onerror = () => { /* surfaces via tx.onabort */ }
        if (record) tx.objectStore(CHECKPOINTS).put(record)
        else if (checkpoint && "remove" in checkpoint) tx.objectStore(CHECKPOINTS).delete(checkpoint.remove)
      } catch (e) {
        // A request that throws at once (a value that cannot be stored) must not leave the others to commit alone.
        finished.catch(() => {})
        tx.abort()
        throw e
      }
      await finished
    },
    async pending() {
      return (await rows()).filter(r => !r.sent).map(r => r.file)
    },
    async markSent(learnerId, seq) {
      const tx = (await openDb(idb)).transaction(OUTBOX, "readwrite")
      const store = tx.objectStore(OUTBOX)
      const finished = done(tx)
      const row = await result(store.get([learnerId, seq]) as IDBRequest<Row | undefined>)
      if (row) store.put({ ...row, sent: true })
      await finished
    },
    async latest(learnerId) {
      const mine = (await rows()).filter(r => r.learnerId === learnerId)
      return mine.reduce<Row | undefined>((m, r) => (m === undefined || r.seq > m.seq ? r : m), undefined)?.file
    },
    async files(learnerId) {
      return (await rows()).filter(r => r.learnerId === learnerId).sort((a, b) => a.seq - b.seq).map(r => r.file)
    },
    async prune() {
      const tx = (await openDb(idb)).transaction(OUTBOX, "readwrite")
      const store = tx.objectStore(OUTBOX)
      const finished = done(tx)
      const all = await result(store.getAll() as IDBRequest<Row[]>)
      const over = new Set(all.filter(r => r.sent && r.file.final).map(r => r.learnerId))
      const gone = all.filter(r => over.has(r.learnerId) && r.sent && !r.file.final)
      for (const r of gone) store.delete([r.learnerId, r.seq])
      await finished
      return gone.length
    },
  }
}
