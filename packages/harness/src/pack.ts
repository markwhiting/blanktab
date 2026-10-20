// How a learner's state is stored in a checkpoint: its replay frames (~15 kB each, ~2000 of them) gzip-compressed
// into one buffer, everything else as is. Lossless, so a resumed learner is bit-identical to one that never stopped.
import type { Action, Observation } from "@blanktab/interface"
import type { LearnerState } from "@blanktab/learner"

/** Every distinct observation of the state, frames concatenated (periphery then fovea, per observation). */
export interface PackedFrames {
  /** "raw" where CompressionStream is missing. */
  codec: "gzip" | "raw"
  count: number
  peripheryLen: number
  foveaLen: number
  data: Uint8Array
  /** count × (x, y). */
  cursors: Float64Array
  /** count × 0 | 1. */
  buttons: Uint8Array
}

interface PackedTransition { obs: number; hIn: Float32Array; action: Action; next: number }

/** A LearnerState with its observations replaced by indices into `frames`. */
export type PackedLearnerState = Omit<LearnerState, "replay" | "last"> & {
  frames: PackedFrames
  replay: PackedTransition[]
  last: { obs: number; action: Action; hIn: Float32Array } | null
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isObs = (v: unknown): v is Observation =>
  isRecord(v) && v.periphery instanceof Uint8Array && v.fovea instanceof Uint8Array && isRecord(v.cursor) && typeof v.buttonHeld === "boolean"

/** Whether a stored learner state is packed (checkpoints written before packing, and test stubs, are not). */
export function isPacked(s: unknown): s is PackedLearnerState {
  return isRecord(s) && isRecord(s.frames) && (s.frames.codec === "gzip" || s.frames.codec === "raw")
}

async function pipe(data: Uint8Array, through: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const out = new Blob([data as Uint8Array<ArrayBuffer>]).stream().pipeThrough(through as TransformStream<Uint8Array, Uint8Array>)
  return new Uint8Array(await new Response(out).arrayBuffer())
}

/**
 * Packs a learner state for storage. Anything that is not a learner state with well-formed observations of one size
 * (a stub, a state the learner will reject on load) is returned as is, for the learner to judge.
 * `compress: false` (or no CompressionStream) stores the frames raw.
 */
export async function packLearnerState(state: unknown, o: { compress?: boolean } = {}): Promise<unknown> {
  if (!isRecord(state) || isPacked(state) || !Array.isArray(state.replay) || !("last" in state)) return state
  const s = state as unknown as LearnerState
  const last = s.last as LearnerState["last"] | unknown
  if (last !== null && !(isRecord(last) && isObs(last.obs))) return state
  if (!s.replay.every(tr => isRecord(tr) && isObs(tr.obs) && isObs(tr.next))) return state
  // Distinct observations, by identity: consecutive transitions share one (`next` of one is `obs` of the next).
  const index = new Map<Observation, number>()
  const all: Observation[] = []
  const add = (ob: Observation): number => {
    let i = index.get(ob)
    if (i === undefined) {
      i = all.length
      index.set(ob, i)
      all.push(ob)
    }
    return i
  }
  const replay: PackedTransition[] = s.replay.map(tr => ({ obs: add(tr.obs), hIn: tr.hIn, action: tr.action, next: add(tr.next) }))
  const packedLast = s.last === null ? null : { obs: add(s.last.obs), action: s.last.action, hIn: s.last.hIn }
  const peripheryLen = all[0]?.periphery.length ?? 0
  const foveaLen = all[0]?.fovea.length ?? 0
  if (!all.every(ob => ob.periphery.length === peripheryLen && ob.fovea.length === foveaLen)) return state
  const frame = peripheryLen + foveaLen
  const raw = new Uint8Array(all.length * frame)
  const cursors = new Float64Array(all.length * 2)
  const buttons = new Uint8Array(all.length)
  all.forEach((ob, i) => {
    raw.set(ob.periphery, i * frame)
    raw.set(ob.fovea, i * frame + peripheryLen)
    cursors[2 * i] = ob.cursor.x
    cursors[2 * i + 1] = ob.cursor.y
    buttons[i] = ob.buttonHeld ? 1 : 0
  })
  const gzip = (o.compress ?? true) && typeof CompressionStream === "function"
  const data = gzip ? await pipe(raw, new CompressionStream("gzip")) : raw
  return {
    ...s, replay, last: packedLast,
    frames: { codec: gzip ? "gzip" : "raw", count: all.length, peripheryLen, foveaLen, data, cursors, buttons },
  } satisfies PackedLearnerState
}

/** The learner state a packed one stands for (shared observations shared again). Unpacked states are returned as is. */
export async function unpackLearnerState(stored: unknown): Promise<unknown> {
  if (!isPacked(stored)) return stored
  const { frames: f, replay, last, ...rest } = stored
  const fail = (what: string): never => { throw new RangeError(`packed learner state: ${what}`) }
  if (!(f.data instanceof Uint8Array && f.cursors instanceof Float64Array && f.buttons instanceof Uint8Array)) fail("malformed frames")
  if (![f.count, f.peripheryLen, f.foveaLen].every(n => Number.isInteger(n) && n >= 0)) fail("malformed frame sizes")
  if (f.codec === "gzip" && typeof DecompressionStream !== "function") fail("gzip frames, but this browser has no DecompressionStream")
  let raw: Uint8Array
  try {
    raw = f.codec === "gzip" ? await pipe(f.data, new DecompressionStream("gzip")) : f.data
  } catch (e) {
    return fail(`the frames do not decompress (${e instanceof Error ? e.message : String(e)})`)
  }
  const frame = f.peripheryLen + f.foveaLen
  if (raw.length !== f.count * frame || f.cursors.length !== 2 * f.count || f.buttons.length !== f.count) fail("frame data does not match its sizes")
  const all: Observation[] = Array.from({ length: f.count }, (_, i) => ({
    periphery: raw.slice(i * frame, i * frame + f.peripheryLen),
    fovea: raw.slice(i * frame + f.peripheryLen, (i + 1) * frame),
    cursor: { x: f.cursors[2 * i]!, y: f.cursors[2 * i + 1]! },
    buttonHeld: f.buttons[i] === 1,
  }))
  const at = (i: unknown): Observation => (Number.isInteger(i) && all[i as number] !== undefined ? all[i as number]! : fail(`no frame ${String(i)}`))
  if (!Array.isArray(replay)) fail("replay is not an array")
  return {
    ...rest,
    replay: replay.map(tr => ({ obs: at(tr.obs), hIn: tr.hIn, action: tr.action, next: at(tr.next) })),
    last: last === null ? null : { obs: at(last.obs), action: last.action, hIn: last.hIn },
  } satisfies LearnerState
}
