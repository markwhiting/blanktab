// The tab's one IndexedDB database: the outbox of log files and the learners' checkpoints, so a log file and the
// checkpoint taken with it can be written in a single transaction.

export const DB_NAME = "blanktab"
/** 1: outbox. 2: + checkpoints. 3: + checkpointMeta. 4: + contribution. */
export const DB_VERSION = 4
export const OUTBOX = "outbox"
export const CHECKPOINTS = "checkpoints"
/**
 * A small record per checkpoint (CheckpointMeta), written and removed in the same transaction as the checkpoint, so
 * the startup sweep can judge every checkpoint without reading tens of MB each.
 */
export const CHECKPOINT_META = "checkpointMeta"
/**
 * Per learner, how its logs fare at the gate (queue.ts Contribution): the last PR looked at, closed-unmerged PRs in a
 * row, and whether the learner is local-only (it keeps learning but no longer contributes).
 */
export const CONTRIBUTION = "contribution"
/** The stores a checkpoint write or removal touches. */
export const CHECKPOINT_STORES = [CHECKPOINTS, CHECKPOINT_META]
/** Every store this code reads or writes. */
export const REQUIRED_STORES = [OUTBOX, CHECKPOINTS, CHECKPOINT_META, CONTRIBUTION]

/** What the sweep needs to know of a checkpoint. */
export interface CheckpointMeta {
  learnerId: string
  /** ISO time ("" if the record had none). */
  savedAt: string
  /** Its learner state's version (null if it had none). */
  version: unknown
  steps: unknown
  seq: unknown
}

/** The meta record of a checkpoint record (whatever its shape). */
export function metaOf(record: unknown): CheckpointMeta {
  const r = (typeof record === "object" && record !== null ? record : {}) as Record<string, unknown>
  const st = r.learnerState
  return {
    learnerId: String(r.learnerId), savedAt: typeof r.savedAt === "string" ? r.savedAt : "",
    version: typeof st === "object" && st !== null ? ((st as Record<string, unknown>).version ?? null) : null,
    steps: r.steps ?? null, seq: r.seq ?? null,
  }
}

/** How long an upgrade may wait on another tab's connection before giving up. */
export const DB_BLOCKED_TIMEOUT_MS = 10_000
/** What the page shows when an older tab holds the database open and the upgrade cannot proceed. */
export const DB_BLOCKED_MESSAGE = "Close other learner tabs to update"

const open = new WeakMap<IDBFactory, Promise<IDBDatabase>>()

/**
 * One shared connection per IDBFactory; callers ask for it on every use, so after a version change closed it the
 * next call reopens. Upgrades a v1 database in place (its outbox rows are kept). If another tab (older code without
 * a versionchange handler) blocks the upgrade for `blockedTimeoutMs`, rejects with DB_BLOCKED_MESSAGE. A database a
 * newer page has already upgraded past DB_VERSION (a VersionError) is opened at its own version and used as it is if it
 * still has every store this code needs (REQUIRED_STORES), so a tab of older code keeps working after a version bump.
 * The rule that makes this safe (decision 0014): a future DB_VERSION may add stores or indexes but must never change
 * the format of an existing store (its key path or its records' shape), nor remove one.
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
      if (!db.objectStoreNames.contains(CONTRIBUTION)) db.createObjectStore(CONTRIBUTION, { keyPath: "learnerId" })
      if (!db.objectStoreNames.contains(CHECKPOINT_META)) {
        const meta = db.createObjectStore(CHECKPOINT_META, { keyPath: "learnerId" })
        // Checkpoints from before v3 get their meta records now (once: each record is read a single time).
        const cursor = req.transaction!.objectStore(CHECKPOINTS).openCursor()
        cursor.onsuccess = () => {
          const c = cursor.result
          if (!c) return
          meta.put({ ...metaOf(c.value), learnerId: String(c.key) })
          c.continue()
        }
      }
    }
    const use = (db: IDBDatabase) => {
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
    req.onerror = () => {
      clearTimeout(timer)
      const error = req.error
      if (error?.name !== "VersionError") return reject(error)
      // Newer code upgraded it: open it at whatever version it is now (no upgrade), and use it if nothing we need is gone.
      const now = idb.open(DB_NAME)
      now.onerror = () => reject(now.error)
      now.onsuccess = () => {
        const db = now.result
        const missing = REQUIRED_STORES.filter(name => !db.objectStoreNames.contains(name))
        if (missing.length === 0) return use(db)
        db.close()
        reject(new Error(`the database is at version ${db.version}, newer than this page's ${DB_VERSION}, and lacks the stores ${missing.join(", ")} (VersionError)`))
      }
    }
    req.onsuccess = () => {
      clearTimeout(timer)
      use(req.result)
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
