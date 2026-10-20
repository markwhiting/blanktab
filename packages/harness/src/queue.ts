import type { Checkpoint } from "./checkpoint"
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
}

interface Row { learnerId: string; seq: number; sent: boolean; file: LogFile }

export async function openQueue(idb: IDBFactory): Promise<Queue> {
  const db = await openDb(idb)
  const rows = async (): Promise<Row[]> => result(db.transaction(OUTBOX).objectStore(OUTBOX).getAll() as IDBRequest<Row[]>)
  return {
    async put(f, checkpoint) {
      const target = checkpoint === undefined ? undefined : "save" in checkpoint ? checkpoint.save.learnerId : checkpoint.remove
      if (target !== undefined && target !== f.learnerId) throw new Error(`a checkpoint of learner ${target} cannot travel with a log of ${f.learnerId}`)
      const tx = db.transaction(checkpoint ? [OUTBOX, CHECKPOINTS] : [OUTBOX], "readwrite")
      const row: Row = { learnerId: f.learnerId, seq: f.seq, sent: false, file: f }
      const finished = done(tx)
      tx.objectStore(OUTBOX).add(row).onerror = () => { /* surfaces via tx.onabort */ }
      if (checkpoint && "save" in checkpoint) tx.objectStore(CHECKPOINTS).put(checkpoint.save)
      else if (checkpoint) tx.objectStore(CHECKPOINTS).delete(checkpoint.remove)
      await finished
    },
    async pending() {
      return (await rows()).filter(r => !r.sent).map(r => r.file)
    },
    async markSent(learnerId, seq) {
      const tx = db.transaction(OUTBOX, "readwrite")
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
  }
}
