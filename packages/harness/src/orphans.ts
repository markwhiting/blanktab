// Orphaned checkpoints: learners whose tab was closed mid-life (spec §10, "Liveness and adoption"). At startup a tab
// adopts the newest one it may, and deletes the ones nobody will resume, so they do not fill the visitor's disk.
import { LEARNER_STATE_VERSION } from "@blanktab/learner"
import { checkCheckpoint, checkpointIds, deleteCheckpoint, readCheckpointRecord } from "./checkpoint"
import { openQueue } from "./queue"

/** An orphan checkpointed this long ago or more is not adopted, but deleted (its logs still go out). */
export const ORPHAN_MAX_AGE_MS = 24 * 60 * 60 * 1000
/** At most this many orphans are kept for later tabs to adopt; older ones are deleted. */
export const MAX_ORPHANS = 3

/** The Web Lock a live learner holds for its whole life. */
export const learnerLockName = (learnerId: string) => `blanktab:learner:${learnerId}`

/**
 * Takes the learner's lock if no tab holds it, without waiting: a release function, or null if another tab holds it
 * (its learner is alive there).
 */
export type TryLock = (learnerId: string) => Promise<(() => void) | null>

/** tryLock on the Web Locks API: the lock is held until the release function is called. */
export function webLock(locks: LockManager): TryLock {
  return id => new Promise(resolve => {
    void locks.request(learnerLockName(id), { ifAvailable: true }, lock => {
      if (lock === null) { resolve(null); return }
      return new Promise<void>(release => resolve(release))
    })
  })
}

export type OrphanVerdict = "adopted" | "kept" | "locked" | "stale" | "invalid" | "version" | "final" | "surplus"

export interface SweepResult {
  /** The orphan this tab adopted, its lock held (hand `release` to its life), or null. */
  adopted: { learnerId: string; release: () => void } | null
  /** What became of each checkpoint looked at. */
  verdicts: Record<string, OrphanVerdict>
}

/**
 * Looks at every checkpoint but those in `exclude` (this tab's own learner). One whose lock another tab holds is
 * alive and left alone. Every other one is an orphan, and under its lock:
 *  - deleted if it is malformed (checkCheckpoint), of another LEARNER_STATE_VERSION, checkpointed ORPHAN_MAX_AGE_MS
 *    or more ago, or its learner's latest log file is final;
 *  - otherwise eligible: the newest MAX_ORPHANS eligible ones are kept, the rest deleted.
 * With `adopt`, the newest eligible one is adopted (one of the kept), its lock left held. Records are read one at a
 * time (a GPU checkpoint is tens of MB).
 */
export async function sweepCheckpoints(o: { idb: IDBFactory; tryLock: TryLock; now: () => Date; adopt: boolean; exclude?: readonly string[] }): Promise<SweepResult> {
  const { idb, tryLock } = o
  const queue = await openQueue(idb)
  const verdicts: Record<string, OrphanVerdict> = {}
  const eligible: { learnerId: string; savedAt: number; release: () => void }[] = []
  const drop = async (learnerId: string, release: () => void, why: OrphanVerdict) => {
    try {
      await deleteCheckpoint(idb, learnerId)
      verdicts[learnerId] = why
    } finally {
      release()
    }
  }
  const t = o.now().getTime()
  try {
    return await sweep()
  } catch (e) {
    for (const x of eligible) x.release() // releasing twice is harmless
    throw e
  }

  async function sweep(): Promise<SweepResult> {
    for (const learnerId of await checkpointIds(idb)) {
      if (o.exclude?.includes(learnerId)) continue
      const release = await tryLock(learnerId)
      if (release === null) {
        verdicts[learnerId] = "locked"
        continue
      }
      try {
        const rec = await readCheckpointRecord(idb, learnerId)
        if (rec === undefined) { // gone meanwhile (its life ended, or another tab swept it)
          release()
          continue
        }
        try {
          checkCheckpoint(rec, learnerId)
        } catch {
          await drop(learnerId, release, "invalid")
          continue
        }
        const savedAt = Date.parse(rec.savedAt)
        const version = (rec.learnerState as { version?: unknown } | null)?.version
        if (version !== LEARNER_STATE_VERSION) await drop(learnerId, release, "version")
        else if (!Number.isFinite(savedAt) || t - savedAt >= ORPHAN_MAX_AGE_MS) await drop(learnerId, release, "stale")
        else if ((await queue.latest(learnerId))?.final) await drop(learnerId, release, "final")
        else eligible.push({ learnerId, savedAt, release })
      } catch (e) {
        release()
        throw e
      }
    }
    eligible.sort((a, b) => b.savedAt - a.savedAt)
    for (const e of eligible.slice(MAX_ORPHANS)) await drop(e.learnerId, e.release, "surplus")
    const kept = eligible.slice(0, MAX_ORPHANS)
    const adopt = o.adopt ? kept[0] : undefined
    for (const e of kept) {
      verdicts[e.learnerId] = e === adopt ? "adopted" : "kept"
      if (e !== adopt) e.release()
    }
    return { adopted: adopt ? { learnerId: adopt.learnerId, release: adopt.release } : null, verdicts }
  }
}
