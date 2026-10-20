// The tab's one IndexedDB database: the outbox of log files and the learners' checkpoints, so a log file and the
// checkpoint taken with it can be written in a single transaction.

export const DB_NAME = "blanktab"
/** 1: outbox. 2: + checkpoints. */
export const DB_VERSION = 2
export const OUTBOX = "outbox"
export const CHECKPOINTS = "checkpoints"

/** How long an upgrade may wait on another tab's connection before giving up. */
export const DB_BLOCKED_TIMEOUT_MS = 10_000
/** What the page shows when an older tab holds the database open and the upgrade cannot proceed. */
export const DB_BLOCKED_MESSAGE = "Close other learner tabs to update"

const open = new WeakMap<IDBFactory, Promise<IDBDatabase>>()

/**
 * One shared connection per IDBFactory; callers ask for it on every use, so after a version change closed it the
 * next call reopens. Upgrades a v1 database in place (its outbox rows are kept). If another tab (older code without
 * a versionchange handler) blocks the upgrade for `blockedTimeoutMs`, rejects with DB_BLOCKED_MESSAGE.
 */
export function openDb(idb: IDBFactory, blockedTimeoutMs = DB_BLOCKED_TIMEOUT_MS): Promise<IDBDatabase> {
  const cached = open.get(idb)
  if (cached) return cached
  const p = new Promise<IDBDatabase>((resolve, reject) => {
    let gaveUp = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const req = idb.open(DB_NAME, DB_VERSION)
    req.onblocked = () => {
      timer ??= setTimeout(() => {
        gaveUp = true
        reject(new Error(DB_BLOCKED_MESSAGE))
      }, blockedTimeoutMs)
    }
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(OUTBOX)) db.createObjectStore(OUTBOX, { keyPath: ["learnerId", "seq"] })
      if (!db.objectStoreNames.contains(CHECKPOINTS)) db.createObjectStore(CHECKPOINTS, { keyPath: "learnerId" })
    }
    req.onerror = () => {
      clearTimeout(timer)
      reject(req.error)
    }
    req.onsuccess = () => {
      clearTimeout(timer)
      const db = req.result
      if (gaveUp) {
        db.close() // the caller has moved on; the next openDb opens afresh
        return
      }
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
