// Learner checkpoints: the IndexedDB store "checkpoints", holding the latest checkpoint per learnerId.
import type { LearnerState } from "@blanktab/learner"
import { CHECKPOINTS, CHECKPOINT_META, CHECKPOINT_STORES, OUTBOX, done, metaOf, openDb, result, type CheckpointMeta } from "./db"
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

/** Puts a checkpoint record and its meta record, in `tx` (which must cover CHECKPOINT_STORES). */
export function putCheckpoint(tx: IDBTransaction, record: Checkpoint<unknown>): void {
  tx.objectStore(CHECKPOINTS).put(record)
  tx.objectStore(CHECKPOINT_META).put(metaOf(record))
}

/** Removes a learner's checkpoint and its meta record, in `tx` (which must cover CHECKPOINT_STORES). */
export function removeCheckpoint(tx: IDBTransaction, learnerId: string): void {
  tx.objectStore(CHECKPOINTS).delete(learnerId)
  tx.objectStore(CHECKPOINT_META).delete(learnerId)
}

export async function saveCheckpoint<S>(idb: IDBFactory, c: Checkpoint<S>): Promise<void> {
  const record = await toRecord(c)
  const tx = (await openDb(idb)).transaction(CHECKPOINT_STORES, "readwrite")
  const finished = done(tx)
  try {
    putCheckpoint(tx, record)
  } catch (e) {
    finished.catch(() => {})
    tx.abort()
    throw e
  }
  await finished
}

/** Every checkpoint's meta record (small: read these, not the checkpoints, to judge them). */
export async function checkpointMetas(idb: IDBFactory): Promise<CheckpointMeta[]> {
  const db = await openDb(idb)
  return result(db.transaction(CHECKPOINT_META).objectStore(CHECKPOINT_META).getAll() as IDBRequest<CheckpointMeta[]>)
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

/**
 * Every checkpoint's id with its meta record (undefined if it has none), and the learners whose latest log file is
 * final, all read in one transaction: a checkpoint saved meanwhile is seen with its meta record or not at all.
 */
export async function checkpointIndex(idb: IDBFactory): Promise<{ entries: { learnerId: string; meta: CheckpointMeta | undefined }[]; final: Set<string> }> {
  const tx = (await openDb(idb)).transaction([...CHECKPOINT_STORES, OUTBOX])
  const [ids, metas, rows] = await Promise.all([
    result(tx.objectStore(CHECKPOINTS).getAllKeys()),
    result(tx.objectStore(CHECKPOINT_META).getAll() as IDBRequest<CheckpointMeta[]>),
    result(tx.objectStore(OUTBOX).getAll() as IDBRequest<{ learnerId: string; seq: number; file: { final: boolean } }[]>),
  ])
  const byId = new Map(metas.map(m => [m.learnerId, m]))
  const latest = new Map<string, { seq: number; final: boolean }>()
  for (const r of rows) if ((latest.get(r.learnerId)?.seq ?? -1) < r.seq) latest.set(r.learnerId, { seq: r.seq, final: r.file.final })
  return {
    entries: ids.filter((k): k is string => typeof k === "string").map(learnerId => ({ learnerId, meta: byId.get(learnerId) })),
    final: new Set([...latest].filter(([, v]) => v.final).map(([k]) => k)),
  }
}

/** One learner's checkpoint meta record and whether its checkpoint exists, read together (one transaction). */
export async function checkpointEntry(idb: IDBFactory, learnerId: string): Promise<{ exists: boolean; meta: CheckpointMeta | undefined }> {
  const tx = (await openDb(idb)).transaction(CHECKPOINT_STORES)
  const [n, meta] = await Promise.all([
    result(tx.objectStore(CHECKPOINTS).count(learnerId)),
    result(tx.objectStore(CHECKPOINT_META).get(learnerId) as IDBRequest<CheckpointMeta | undefined>),
  ])
  return { exists: n > 0, meta }
}

/** The learner's latest checkpoint, or undefined. Throws RangeError for a malformed record (or undecodable frames). */
export async function loadCheckpoint<S = LearnerState>(idb: IDBFactory, learnerId: string): Promise<Checkpoint<S> | undefined> {
  const c = await readCheckpointRecord(idb, learnerId)
  if (c === undefined) return undefined
  return fromRecord<S>(c, learnerId)
}

/** A checkpoint from its stored record (as readCheckpointRecord returns it). Throws RangeError as loadCheckpoint does. */
export async function fromRecord<S = LearnerState>(c: unknown, learnerId: string): Promise<Checkpoint<S>> {
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
  const tx = (await openDb(idb)).transaction(CHECKPOINT_STORES, "readwrite")
  const finished = done(tx)
  removeCheckpoint(tx, learnerId)
  await finished
}
