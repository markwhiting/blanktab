// The tab's one IndexedDB database: the outbox of log files and the learners' checkpoints, so a log file and the
// checkpoint taken with it can be written in a single transaction.

export const DB_NAME = "blanktab"
/** 1: outbox. 2: + checkpoints. */
export const DB_VERSION = 2
export const OUTBOX = "outbox"
export const CHECKPOINTS = "checkpoints"

const open = new WeakMap<IDBFactory, Promise<IDBDatabase>>()

/** One shared connection per IDBFactory. Upgrades a v1 database in place (its outbox rows are kept). */
export function openDb(idb: IDBFactory): Promise<IDBDatabase> {
  const cached = open.get(idb)
  if (cached) return cached
  const p = new Promise<IDBDatabase>((resolve, reject) => {
    const req = idb.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(OUTBOX)) db.createObjectStore(OUTBOX, { keyPath: ["learnerId", "seq"] })
      if (!db.objectStoreNames.contains(CHECKPOINTS)) db.createObjectStore(CHECKPOINTS, { keyPath: "learnerId" })
    }
    req.onerror = () => reject(req.error)
    req.onsuccess = () => {
      const db = req.result
      // A newer page wants to upgrade: step aside rather than block it (the next call reopens).
      db.onversionchange = () => {
        db.close()
        open.delete(idb)
      }
      resolve(db)
    }
  })
  p.catch(() => open.delete(idb))
  open.set(idb, p)
  return p
}

export const done = (tx: IDBTransaction): Promise<void> =>
  new Promise((res, rej) => {
    tx.oncomplete = () => res()
    tx.onerror = () => rej(tx.error)
    tx.onabort = () => rej(tx.error ?? new Error("transaction aborted"))
  })

export const result = <T>(r: IDBRequest<T>): Promise<T> =>
  new Promise((res, rej) => {
    r.onsuccess = () => res(r.result)
    r.onerror = () => rej(r.error)
  })
