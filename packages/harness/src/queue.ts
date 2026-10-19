import type { LogFile } from "./log"

export interface Queue {
  /** Adds a file. Rejects if (learnerId, seq) already exists: a queued file is never overwritten. */
  put(f: LogFile): Promise<void>
  /** Unsent files only, ordered by learnerId, then seq. */
  pending(): Promise<LogFile[]>
  markSent(learnerId: string, seq: number): Promise<void>
  /** Highest-seq file for the learner, sent or not (used to resume numbering). */
  latest(learnerId: string): Promise<LogFile | undefined>
}

interface Row { learnerId: string; seq: number; sent: boolean; file: LogFile }

const DB = "blanktab"
const STORE = "outbox"

const done = (tx: IDBTransaction): Promise<void> =>
  new Promise((res, rej) => {
    tx.oncomplete = () => res()
    tx.onerror = () => rej(tx.error)
    tx.onabort = () => rej(tx.error ?? new Error("transaction aborted"))
  })
const result = <T>(r: IDBRequest<T>): Promise<T> =>
  new Promise((res, rej) => {
    r.onsuccess = () => res(r.result)
    r.onerror = () => rej(r.error)
  })

export function openQueue(idb: IDBFactory): Promise<Queue> {
  return new Promise((resolve, reject) => {
    const req = idb.open(DB, 1)
    req.onupgradeneeded = () => req.result.createObjectStore(STORE, { keyPath: ["learnerId", "seq"] })
    req.onerror = () => reject(req.error)
    req.onsuccess = () => {
      const db = req.result
      const rows = async (): Promise<Row[]> => result(db.transaction(STORE).objectStore(STORE).getAll() as IDBRequest<Row[]>)
      resolve({
        async put(f) {
          const tx = db.transaction(STORE, "readwrite")
          const row: Row = { learnerId: f.learnerId, seq: f.seq, sent: false, file: f }
          const finished = done(tx)
          tx.objectStore(STORE).add(row).onerror = () => { /* surfaces via tx.onabort */ }
          await finished
        },
        async pending() {
          return (await rows()).filter(r => !r.sent).map(r => r.file)
        },
        async markSent(learnerId, seq) {
          const tx = db.transaction(STORE, "readwrite")
          const store = tx.objectStore(STORE)
          const finished = done(tx)
          const row = await result(store.get([learnerId, seq]) as IDBRequest<Row | undefined>)
          if (row) store.put({ ...row, sent: true })
          await finished
        },
        async latest(learnerId) {
          const mine = (await rows()).filter(r => r.learnerId === learnerId)
          return mine.reduce<Row | undefined>((m, r) => (m === undefined || r.seq > m.seq ? r : m), undefined)?.file
        },
      })
    }
  })
}
