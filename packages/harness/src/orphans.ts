// Orphaned checkpoints: learners whose tab was closed mid-life (spec §10, "Liveness and adoption"). At startup a tab
// adopts the newest one it may, and deletes the ones nobody will resume, so they do not fill the visitor's disk.
import { LEARNER_STATE_VERSION } from "@blanktab/learner"
import { checkCheckpoint, checkpointEntry, checkpointIndex, deleteCheckpoint, readCheckpointRecord } from "./checkpoint"

/** An orphan checkpointed this long ago or more is not adopted, but deleted (its logs still go out). */
export const ORPHAN_MAX_AGE_MS = 24 * 60 * 60 * 1000
/** At most this many orphans are kept for later tabs to adopt; older ones are deleted. */
export const MAX_ORPHANS = 3
/** How long a tab waits for its own learner's lock (a sweep in another tab may hold it for a moment). */
export const OWN_LOCK_WAIT_MS = 10_000

/** The Web Lock a live learner holds for its whole life. */
export const learnerLockName = (learnerId: string) => `blanktab:learner:${learnerId}`

/** A lock this tab holds. */
export interface Held {
  release(): void
  /** Whether another tab is waiting for this lock (its own learner, reloading): then it is not an orphan. */
  wanted(): Promise<boolean>
}

/** Takes the learner's lock if no tab holds it, without waiting: the held lock, or null if another tab holds it. */
export type TryLock = (learnerId: string) => Promise<Held | null>

/** The part of the Web Locks API used here (navigator.locks). */
export interface Locks {
  request(name: string, options: LockOptions, callback: (lock: Lock | null) => unknown): Promise<unknown>
  query(): Promise<LockManagerSnapshot>
}

function request(locks: Locks, learnerId: string, options: LockOptions): Promise<Held | null> {
  const name = learnerLockName(learnerId)
  return new Promise(resolve => {
    let granted = false
    const wanted = async () => {
      try {
        return ((await locks.query()).pending ?? []).some(l => l.name === name)
      } catch {
        return false
      }
    }
    locks.request(name, options, lock => {
      if (lock === null) { resolve(null); return }
      granted = true
      return new Promise<void>(release => resolve({ release, wanted }))
    }).catch(() => { if (!granted) resolve(null) }) // a rejected request (aborted, unavailable): not acquired
  })
}

/** tryLock on the Web Locks API: ifAvailable; the lock is held until released. A rejected request is "not acquired". */
export function webLock(locks: Locks): TryLock {
  return id => request(locks, id, { ifAvailable: true })
}

/**
 * This tab's own learner's lock: waits for it up to `ms` (another tab's sweep may be looking at it), then gives up
 * (null: another tab lives this learner, e.g. a duplicated tab).
 */
export function waitLock(locks: Locks, learnerId: string, ms = OWN_LOCK_WAIT_MS): Promise<Held | null> {
  return request(locks, learnerId, { signal: AbortSignal.timeout(ms) })
}

export type OrphanVerdict = "adopted" | "kept" | "locked" | "wanted" | "stale" | "invalid" | "version" | "final" | "surplus"

export interface SweepResult {
  /**
   * The orphan this tab adopted, its lock held (hand `release` to its life), with its checkpoint record as stored
   * (already read and checked: pass it on so it is not read twice), or null.
   */
  adopted: { learnerId: string; release: () => void; record: unknown } | null
  /** What became of each checkpoint looked at. */
  verdicts: Record<string, OrphanVerdict>
}

/**
 * Looks at every checkpoint but those in `exclude` (this tab's own learner, from its sessionStorage). The ids, their
 * small meta records and the final learners are listed in one transaction. Then each checkpoint is judged on its own,
 * under its lock taken with ifAvailable and let go at once (no lock is held across another checkpoint's work):
 *  - its lock held by another tab: alive there, left alone ("locked");
 *  - another tab waiting for its lock (its own learner, reloading): left alone ("wanted");
 *  - deleted if it has no meta record, is of another LEARNER_STATE_VERSION, was checkpointed ORPHAN_MAX_AGE_MS or more
 *    ago, or its learner's latest log file is final;
 *  - otherwise eligible.
 * With `adopt`, the newest eligible one is locked again and read in full (checkCheckpoint): adopted if well formed and
 * not wanted meanwhile, its lock left held; a malformed one is deleted and the next tried. This is the only lock held
 * across a full read (a GPU checkpoint is tens of MB). Of the remaining eligible ones the newest MAX_ORPHANS (counting
 * the adopted one) are kept, the rest deleted, each under its own brief lock.
 *
 * What this cannot close: between a tab's unload and its reload's first line (main(), which asks for its own lock at
 * once) nobody holds or waits for its learner's lock, so a sweep running in exactly that gap still sees an orphan and
 * may adopt it (the reloaded tab then starts a newborn) or, if it is surplus or stale, delete it. The gap is narrowed
 * to the page's load time, not closed: Web Locks cannot outlive a page.
 */
export async function sweepCheckpoints(o: {
  idb: IDBFactory
  tryLock: TryLock
  now: () => Date
  adopt: boolean
  exclude?: readonly string[]
  /** The full read of an adoption candidate (tests slow it down). Default readCheckpointRecord. */
  readRecord?: (idb: IDBFactory, learnerId: string) => Promise<unknown>
}): Promise<SweepResult> {
  const { idb, tryLock } = o
  const readRecord = o.readRecord ?? readCheckpointRecord
  const verdicts: Record<string, OrphanVerdict> = {}
  const t = o.now().getTime()
  const { entries, final } = await checkpointIndex(idb)

  /** Locks this sweep hands on (the adopted one's): not let go. */
  const kept = new Set<Held>()
  /** Runs `fn` under the learner's lock if it is free and nobody waits for it; the lock is let go afterwards unless `keep`. */
  const underLock = async <T>(learnerId: string, fn: (held: Held) => Promise<T>): Promise<T | "locked" | "wanted"> => {
    const held = await tryLock(learnerId)
    if (held === null) return "locked"
    try {
      if (await held.wanted()) return "wanted"
      return await fn(held)
    } finally {
      if (!kept.has(held)) held.release()
    }
  }
  /** Deletes the checkpoint (we hold its lock, nobody waits for it). */
  const remove = async (learnerId: string, why: OrphanVerdict) => {
    await deleteCheckpoint(idb, learnerId)
    return why
  }

  const eligible: { learnerId: string; savedAt: number }[] = []
  for (const { learnerId } of entries) {
    if (o.exclude?.includes(learnerId)) continue
    const v = await underLock(learnerId, async () => {
      // Under the lock: this checkpoint as it is now (its tab may have saved it last thing before closing).
      const { exists, meta } = await checkpointEntry(idb, learnerId)
      if (!exists) return null // gone meanwhile (its life ended, or another tab swept it)
      const savedAt = meta ? Date.parse(meta.savedAt) : Number.NaN
      if (!meta) return remove(learnerId, "invalid")
      if (meta.version !== LEARNER_STATE_VERSION) return remove(learnerId, "version")
      if (!Number.isFinite(savedAt) || t - savedAt >= ORPHAN_MAX_AGE_MS) return remove(learnerId, "stale")
      if (final.has(learnerId)) return remove(learnerId, "final")
      eligible.push({ learnerId, savedAt })
      return null
    })
    if (v !== null) verdicts[learnerId] = v
  }
  eligible.sort((a, b) => b.savedAt - a.savedAt)

  let adopted: SweepResult["adopted"] = null
  while (o.adopt && !adopted && eligible.length > 0) {
    const e = eligible.shift()!
    const v = await underLock(e.learnerId, async held => {
      const record = await readRecord(idb, e.learnerId)
      if (record === undefined) return null
      try {
        checkCheckpoint(record, e.learnerId)
      } catch {
        return remove(e.learnerId, "invalid")
      }
      if (await held.wanted()) return "wanted" as const // its tab asked for it while we read
      kept.add(held)
      adopted = { learnerId: e.learnerId, release: held.release, record }
      return "adopted" as const
    })
    if (v !== null) verdicts[e.learnerId] = v
  }
  const room = MAX_ORPHANS - (adopted ? 1 : 0)
  for (const e of eligible.slice(0, room)) verdicts[e.learnerId] = "kept"
  for (const e of eligible.slice(room)) {
    const v = await underLock(e.learnerId, () => remove(e.learnerId, "surplus"))
    verdicts[e.learnerId] = v
  }
  return { adopted, verdicts }
}
