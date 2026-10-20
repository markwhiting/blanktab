// Learner checkpoints: the IndexedDB store "checkpoints", holding the latest checkpoint per learnerId.
import type { LearnerState } from "@blanktab/learner"
import { CHECKPOINTS, done, openDb, result } from "./db"
import { KNOWLEDGE_SHA_RE, type CurvePoint } from "./log"
import { packLearnerState, unpackLearnerState } from "./pack"

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
  /** The life's restores so far (LogFileV2.restores), kept here so they survive a reload. */
  restores: number
  /** CU thrown away by those restores (LogFileV2.lostCU). */
  lostCU: number
  /** Skipped updates thrown away by those restores (added to the learner's own count in LogFileV2.skippedUpdates). */
  lostSkips: number
  /**
   * Birth facts the log files repeat (LogFileV2.knowledgeSha, sessionsSeenAtBirth), so a life resumed before its first
   * log keeps them. Absent in checkpoints written before they were added.
   */
  knowledgeSha?: string
  sessionsSeenAtBirth?: number
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
  if (!isNonNeg(r.cu) || !isNonNeg(r.lostCU)) fail("cu and lostCU must be finite numbers >= 0")
  if (!isNonNegInt(r.restores) || !isNonNegInt(r.lostSkips)) fail("restores and lostSkips must be non-negative integers")
  if (!Array.isArray(r.curveTail)) fail("curveTail must be an array")
  if (r.knowledgeSha !== undefined && !(typeof r.knowledgeSha === "string" && KNOWLEDGE_SHA_RE.test(r.knowledgeSha))) fail("knowledgeSha must be 64 lowercase hex chars")
  if (r.sessionsSeenAtBirth !== undefined && !isNonNegInt(r.sessionsSeenAtBirth)) fail("sessionsSeenAtBirth must be a non-negative integer")
  if (!("learnerState" in r)) fail("no learnerState")
}

/**
 * The record a checkpoint is stored as: its learner state packed (pack.ts: the replay frames gzip-compressed).
 * Done before the transaction opens, since an IndexedDB transaction does not survive an await on anything else.
 */
export async function toRecord<S>(c: Checkpoint<S>): Promise<Checkpoint<unknown>> {
  return { ...c, learnerState: await packLearnerState(c.learnerState) }
}

export async function saveCheckpoint<S>(idb: IDBFactory, c: Checkpoint<S>): Promise<void> {
  const record = await toRecord(c)
  const tx = (await openDb(idb)).transaction(CHECKPOINTS, "readwrite")
  const finished = done(tx)
  tx.objectStore(CHECKPOINTS).put(record)
  await finished
}

/** A learner's checkpoint record as stored (learner state still packed), or undefined. Not checked. */
export async function readCheckpointRecord(idb: IDBFactory, learnerId: string): Promise<unknown> {
  const db = await openDb(idb)
  return result(db.transaction(CHECKPOINTS).objectStore(CHECKPOINTS).get(learnerId) as IDBRequest<unknown>)
}

/** The learnerIds that have a checkpoint (without reading any). */
export async function checkpointIds(idb: IDBFactory): Promise<string[]> {
  const db = await openDb(idb)
  return (await result(db.transaction(CHECKPOINTS).objectStore(CHECKPOINTS).getAllKeys())).filter((k): k is string => typeof k === "string")
}

/** The learner's latest checkpoint, or undefined. Throws RangeError for a malformed record (or undecodable frames). */
export async function loadCheckpoint<S = LearnerState>(idb: IDBFactory, learnerId: string): Promise<Checkpoint<S> | undefined> {
  const c = await readCheckpointRecord(idb, learnerId)
  if (c === undefined) return undefined
  checkCheckpoint(c, learnerId)
  let learnerState: unknown
  try {
    learnerState = await unpackLearnerState(c.learnerState)
  } catch (e) {
    throw new RangeError(`checkpoint for ${learnerId}: ${e instanceof Error ? e.message : String(e)}`, { cause: e })
  }
  return { ...c, learnerState } as Checkpoint<S>
}

/** Whether the learner has a checkpoint, without reading it (a GPU learner's is tens of MB). */
export async function hasCheckpoint(idb: IDBFactory, learnerId: string): Promise<boolean> {
  const db = await openDb(idb)
  return (await result(db.transaction(CHECKPOINTS).objectStore(CHECKPOINTS).count(learnerId))) > 0
}

export async function deleteCheckpoint(idb: IDBFactory, learnerId: string): Promise<void> {
  const tx = (await openDb(idb)).transaction(CHECKPOINTS, "readwrite")
  const finished = done(tx)
  tx.objectStore(CHECKPOINTS).delete(learnerId)
  await finished
}
