// Orphaned checkpoints: learners whose tab was closed mid-life (spec §10, "Liveness and adoption"). At startup a tab
// adopts the newest one it may, and deletes the ones nobody will resume, so they do not fill the visitor's disk.
import { LEARNER_STATE_VERSION } from "@blanktab/learner"
import { checkCheckpoint, checkpointIds, checkpointMetas, deleteCheckpoint, readCheckpointRecord } from "./checkpoint"
import { openQueue } from "./queue"

/** An orphan checkpointed this long ago or more is not adopted, but deleted (its logs still go out). */
export const ORPHAN_MAX_AGE_MS = 24 * 60 * 60 * 1000
/** At most this many orphans are kept for later tabs to adopt; older ones are deleted. */
export const MAX_ORPHANS = 3
/** How long a tab waits for its own learner's lock (a sweep in another tab may hold it for a moment). */
export const OWN_LOCK_WAIT_MS = 3_000

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
  /** The orphan this tab adopted, its lock held (hand `release` to its life), or null. */
  adopted: { learnerId: string; release: () => void } | null
  /** What became of each checkpoint looked at. */
  verdicts: Record<string, OrphanVerdict>
}

/**
 * Looks at every checkpoint but those in `exclude` (this tab's own learner, from its sessionStorage), judging each by
 * its small meta record. One whose lock another tab holds, or waits for, is alive and left alone. Every other one is
 * an orphan, and under its lock:
 *  - deleted if its meta record is missing, of another LEARNER_STATE_VERSION, checkpointed ORPHAN_MAX_AGE_MS or more
 *    ago, or its learner's latest log file is final;
 *  - otherwise eligible. With `adopt`, the newest eligible one whose full record is well formed (checkCheckpoint) is
 *    adopted, its lock left held; malformed ones met on the way are deleted. Of the rest, the newest MAX_ORPHANS
 *    (counting the adopted one) are kept and the others deleted.
 * Only a record about to be adopted is read in full (a GPU checkpoint is tens of MB).
 */
export async function sweepCheckpoints(o: { idb: IDBFactory; tryLock: TryLock; now: () => Date; adopt: boolean; exclude?: readonly string[] }): Promise<SweepResult> {
  const { idb, tryLock } = o
  const queue = await openQueue(idb)
  const verdicts: Record<string, OrphanVerdict> = {}
  const eligible: { learnerId: string; savedAt: number; held: Held }[] = []
  /** Deletes an orphan, unless another tab has asked for it meanwhile. Always lets the lock go. */
  const drop = async (learnerId: string, held: Held, why: OrphanVerdict) => {
    try {
      if (await held.wanted()) {
        verdicts[learnerId] = "wanted"
        return
      }
      await deleteCheckpoint(idb, learnerId)
      verdicts[learnerId] = why
    } finally {
      held.release()
    }
  }
  const t = o.now().getTime()
  try {
    return await sweep()
  } catch (e) {
    for (const x of eligible) x.held.release() // releasing twice is harmless
    throw e
  }

  async function sweep(): Promise<SweepResult> {
    const metas = new Map((await checkpointMetas(idb)).map(m => [m.learnerId, m]))
    for (const learnerId of await checkpointIds(idb)) {
      if (o.exclude?.includes(learnerId)) continue
      const held = await tryLock(learnerId)
      if (held === null) {
        verdicts[learnerId] = "locked"
        continue
      }
      try {
        const meta = metas.get(learnerId)
        const savedAt = meta ? Date.parse(meta.savedAt) : Number.NaN
        if (!meta) await drop(learnerId, held, "invalid")
        else if (meta.version !== LEARNER_STATE_VERSION) await drop(learnerId, held, "version")
        else if (!Number.isFinite(savedAt) || t - savedAt >= ORPHAN_MAX_AGE_MS) await drop(learnerId, held, "stale")
        else if ((await queue.latest(learnerId))?.final) await drop(learnerId, held, "final")
        else eligible.push({ learnerId, savedAt, held })
      } catch (e) {
        held.release()
        throw e
      }
    }
    eligible.sort((a, b) => b.savedAt - a.savedAt)
    let adopted: (typeof eligible)[number] | undefined
    if (o.adopt) {
      while (eligible.length > 0 && !adopted) {
        const e = eligible[0]!
        let ok = true
        try {
          checkCheckpoint(await readCheckpointRecord(idb, e.learnerId), e.learnerId)
        } catch {
          ok = false
        }
        if (ok && !(await e.held.wanted())) {
          adopted = e
          break
        }
        eligible.shift()
        if (ok) { // another tab is waiting for it: its own learner
          verdicts[e.learnerId] = "wanted"
          e.held.release()
        } else {
          await drop(e.learnerId, e.held, "invalid")
        }
      }
    }
    for (const e of eligible.slice(MAX_ORPHANS)) await drop(e.learnerId, e.held, "surplus")
    for (const e of eligible.slice(0, MAX_ORPHANS)) {
      verdicts[e.learnerId] = e === adopted ? "adopted" : "kept"
      if (e !== adopted) e.held.release()
    }
    return { adopted: adopted ? { learnerId: adopted.learnerId, release: adopted.held.release } : null, verdicts }
  }
}
