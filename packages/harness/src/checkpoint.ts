// Learner checkpoints: the IndexedDB store "checkpoints", holding the latest checkpoint per learnerId.
import type { LearnerState } from "@blanktab/learner"
import { CHECKPOINTS, done, openDb, result } from "./db"
import type { CurvePoint } from "./log"

/**
 * Where a learner's life stands: enough to resume it after a reload. Every log file is queued together with the
 * checkpoint of that moment (one transaction), so a checkpoint is never behind the learner's latest log file.
 */
export interface Checkpoint<S = LearnerState> {
  learnerId: string
  /** ISO time. */
  savedAt: string
  steps: number
  cu: number
  /** The seq the next log file gets. */
  seq: number
  /** Curve points since the latest log file (they go into the next one). */
  curveTail: CurvePoint[]
  learnerState: S
}

const isNonNegInt = (v: unknown) => Number.isInteger(v) && (v as number) >= 0
const isNonNeg = (v: unknown) => typeof v === "number" && Number.isFinite(v) && v >= 0

/** Throws unless `c` has a checkpoint's shape for `learnerId` (the learner state is checked by the learner). */
export function checkCheckpoint(c: unknown, learnerId: string): asserts c is Checkpoint<unknown> {
  const fail = (what: string) => { throw new RangeError(`checkpoint for ${learnerId}: ${what}`) }
  if (typeof c !== "object" || c === null) return fail("not an object")
  const r = c as Record<string, unknown>
  if (r.learnerId !== learnerId) fail(`belongs to learner ${String(r.learnerId)}`)
  if (typeof r.savedAt !== "string") fail("savedAt must be a string")
  if (!isNonNegInt(r.steps) || !isNonNegInt(r.seq)) fail("steps and seq must be non-negative integers")
  if (!isNonNeg(r.cu)) fail("cu must be a finite number >= 0")
  if (!Array.isArray(r.curveTail)) fail("curveTail must be an array")
  if (!("learnerState" in r)) fail("no learnerState")
}

export async function saveCheckpoint<S>(idb: IDBFactory, c: Checkpoint<S>): Promise<void> {
  const tx = (await openDb(idb)).transaction(CHECKPOINTS, "readwrite")
  const finished = done(tx)
  tx.objectStore(CHECKPOINTS).put(c)
  await finished
}

/** The learner's latest checkpoint, or undefined. Throws RangeError for a malformed record. */
export async function loadCheckpoint<S = LearnerState>(idb: IDBFactory, learnerId: string): Promise<Checkpoint<S> | undefined> {
  const db = await openDb(idb)
  const c = await result(db.transaction(CHECKPOINTS).objectStore(CHECKPOINTS).get(learnerId) as IDBRequest<unknown>)
  if (c === undefined) return undefined
  checkCheckpoint(c, learnerId)
  return c as Checkpoint<S>
}

export async function deleteCheckpoint(idb: IDBFactory, learnerId: string): Promise<void> {
  const tx = (await openDb(idb)).transaction(CHECKPOINTS, "readwrite")
  const finished = done(tx)
  tx.objectStore(CHECKPOINTS).delete(learnerId)
  await finished
}
