import { BUTTON_OPTIONS, FOVEA, KEYS, isValidAction, MOVE_OPTIONS, PERIPHERY, PRED_FOVEA, PRED_PERIPHERY, SCROLL_OPTIONS } from "@blanktab/interface"
import type { Action, Learner, Observation } from "@blanktab/interface"
import {
  Adam, add, addScalar, backward, composite, createRng, disposeGraph, exp, gatherRows, logSoftmax, matmul, mse, mul,
  neg, noGrad, reshape, scale, square, sum, tensor,
} from "@blanktab/tensor"
import type { AdamState, Backend, Tensor } from "@blanktab/tensor"
import { ACTION_DIM, actionOneHot, copyLastBaseline, obsToInputs, targetsFromObs } from "./encode"
import type { Inputs, Targets } from "./encode"
import { createModel } from "./model"
import type { Widths } from "./model"
import { ReplayBuffer } from "./replay"
import type { Transition } from "./replay"
import { LEARNER_STATE_VERSION } from "./version"
/** The checkpoint format version (in ./version, a leaf module, so plain-Node build tools can read it). */
export { LEARNER_STATE_VERSION }

export interface LearnerSettings {
  lr: number
  entropyWeight: number
  curiosityWeight: number
  bufferSize: number
  replayPerSleep: number
  /** The predictor's residual output layers train at lr × headLrScale (everything else at lr). */
  headLrScale: number
}
export const DEFAULT_SETTINGS: LearnerSettings = Object.freeze({
  lr: 1e-3, entropyWeight: 0.01, curiosityWeight: 1, bufferSize: 2000, replayPerSleep: 64, headLrScale: 0.1,
})
export interface LearnerStats {
  steps: number
  cu: number
  predLoss: number
  baselineLoss: number
  /** Updates skipped for a non-finite loss: online (at most one per step) and replayed (during sleep). */
  skippedUpdates: number
  /** Skipped updates since the last applied one (reset by any applied update): the harness restores the last checkpoint after N. */
  consecutiveSkips: number
  sleeps: number
}

/**
 * Everything a learner carries between steps, as plain typed arrays and JSON (structured-clone friendly, for
 * IndexedDB). createLearner({ state }) resumes from it exactly: on CPU, the same observation stream then gives
 * bit-identical actions and losses to a learner that never stopped.
 */
export interface LearnerState {
  version: typeof LEARNER_STATE_VERSION
  seed: number
  widths: Widths
  /** Already clamped. */
  settings: LearnerSettings
  /** Model.params values, in that order. */
  params: Float32Array[]
  /** The two optimizers: [main parameters at lr, the residual output layers at lr × headLrScale]. */
  adam: [AdamState, AdamState]
  /** The action-sampling / replay-sampling RNG. */
  rng: number
  /** h_{t-1}, the recurrent state the next step starts from, [hidden]. */
  h: Float32Array
  /** The previous step (null before the first step): its observation, the action taken, and h_{t-2}. */
  last: { obs: Observation; action: Action; hIn: Float32Array } | null
  /** The replay buffer, oldest first. */
  replay: Transition[]
  stats: LearnerStats
}

/** What one online update did, for instrumentation and tests (not part of the learner's inputs). */
export interface UpdateInfo {
  step: number
  /** Prediction loss on the last transition before the update (L_before). NaN if non-finite. */
  lossBefore: number
  /** The same loss with the updated weights (L_after); NaN if not measured. */
  lossAfter: number
  /** Curiosity: clamp((L_before − L_after) / max(predLoss EMA before this step, 1e-6), −1, 1), 0 when non-finite. */
  reward: number
  /** TD advantage curiosityWeight·reward + γ·V(h_t) − V(h_{t-1}); NaN if the actor-critic update did not run. */
  advantage: number
  skipped: boolean
}

export interface TrainableLearner extends Learner {
  stats(): LearnerStats
  /**
   * Consolidation condition A: replayPerSleep prediction updates, each on one transition sampled uniformly from
   * the replay buffer with the learner's RNG (core from the stored h_{t-2} and obs, predict with the stored action
   * against the next frame; size-weighted MSE, Adam). Counts one sleep, even on an empty buffer (no updates).
   */
  sleep(): Promise<void>
  /** A snapshot of the whole learner (independent of it: later steps do not change it, nor it them). */
  save(): Promise<LearnerState>
  /**
   * Mean prediction loss over every buffered transition with the current weights (NaN if empty). Measurement only:
   * no update, no RNG draw, not counted in stats.cu.
   */
  replayLoss(): Promise<number>
  /** Every parameter's values, in checkpoint order (Model.params). */
  readParams(): Promise<Float32Array[]>
  /** The FIFO replay buffer (raw frames; do not mutate). */
  readonly replay: ReplayBuffer
  dispose(): void
}

/** The learning rate is clamped into this range (Review Focus 5). */
export const LR_RANGE = [1e-5, 1e-2] as const
/** headLrScale is clamped into this range. */
export const HEAD_LR_SCALE_RANGE = [0.01, 1] as const
/** TD discount. */
export const GAMMA = 0.9
/** Smoothing of the predLoss / baselineLoss moving averages. */
export const STATS_ALPHA = 0.01

const PRED_P = PRED_PERIPHERY.width * PRED_PERIPHERY.height * PRED_PERIPHERY.channels
const PRED_F = PRED_FOVEA.width * PRED_FOVEA.height * PRED_FOVEA.channels
/** Both outputs weighted by size: the MSE over their concatenation, comparable with copyLastBaseline. */
const W_P = PRED_P / (PRED_P + PRED_F)
const W_F = PRED_F / (PRED_P + PRED_F)

const FACTORS = [MOVE_OPTIONS, BUTTON_OPTIONS, SCROLL_OPTIONS, KEYS.length] as const
const OFFSETS = FACTORS.map((_, f) => FACTORS.slice(0, f).reduce((a, n) => a + n, 0))
const FACTOR_ROWS = FACTORS.map((n, f) => Int32Array.from({ length: n }, (_, k) => OFFSETS[f]! + k))

const isPosInt = (v: number) => Number.isInteger(v) && v >= 1
const isNonNeg = (v: number) => Number.isFinite(v) && v >= 0

/** Validates settings and clamps lr into LR_RANGE. Throws RangeError for values that cannot be clamped. */
export function clampSettings(s: LearnerSettings): LearnerSettings {
  if (!Number.isFinite(s.lr)) throw new RangeError(`lr must be finite, got ${s.lr}`)
  if (!isNonNeg(s.entropyWeight)) throw new RangeError(`entropyWeight must be finite and >= 0, got ${s.entropyWeight}`)
  if (!isNonNeg(s.curiosityWeight)) throw new RangeError(`curiosityWeight must be finite and >= 0, got ${s.curiosityWeight}`)
  if (!isPosInt(s.bufferSize)) throw new RangeError(`bufferSize must be an integer >= 1, got ${s.bufferSize}`)
  if (!isPosInt(s.replayPerSleep)) throw new RangeError(`replayPerSleep must be an integer >= 1, got ${s.replayPerSleep}`)
  if (!Number.isFinite(s.headLrScale)) throw new RangeError(`headLrScale must be finite, got ${s.headLrScale}`)
  const clamp = (v: number, [lo, hi]: readonly [number, number]) => Math.min(hi, Math.max(lo, v))
  return { ...s, lr: clamp(s.lr, LR_RANGE), headLrScale: clamp(s.headLrScale, HEAD_LR_SCALE_RANGE) }
}

/** Floor of the curiosity scale, so progress on a near-zero loss cannot blow up. */
export const CURIOSITY_SCALE_FLOOR = 1e-6

/**
 * Relative learning progress on one transition: clamp((L_before − L_after) / max(scale, 1e-6), −1, 1), where
 * `scale` is the running prediction loss (the predLoss EMA). Relative, so the reward keeps steering the policy as
 * the loss shrinks. 0 if anything is non-finite.
 */
export function curiosityReward(lossBefore: number, lossAfter: number, scale: number): number {
  const d = (lossBefore - lossAfter) / Math.max(scale, CURIOSITY_SCALE_FLOOR)
  return Number.isFinite(d) ? Math.min(1, Math.max(-1, d)) : 0
}

/** Samples one index from softmax(logits[off .. off+n)); a factor with any non-finite logit is sampled uniformly. */
function sampleFactor(logits: Float32Array, off: number, n: number, rng: () => number): number {
  let mx = -Infinity
  let finite = true
  for (let k = 0; k < n; k++) {
    const v = logits[off + k]!
    if (!Number.isFinite(v)) finite = false
    if (v > mx) mx = v
  }
  const u = rng()
  if (!finite) return Math.min(n - 1, Math.floor(u * n))
  let total = 0
  const w = new Float64Array(n)
  for (let k = 0; k < n; k++) total += (w[k] = Math.exp(logits[off + k]! - mx))
  let acc = 0
  const x = u * total
  for (let k = 0; k < n; k++) {
    acc += w[k]!
    if (x < acc) return k
  }
  return n - 1
}

const isUint32 = (v: unknown) => Number.isInteger(v) && (v as number) >= 0 && (v as number) < 2 ** 32
const PERIPHERY_LEN = PERIPHERY.width * PERIPHERY.height * PERIPHERY.channels
const FOVEA_LEN = FOVEA.width * FOVEA.height * FOVEA.channels
/** Number of trailing Model.params on the second optimizer (Model.outputParams = params.slice(-4)). */
const OUTPUT_PARAMS = 4

/**
 * Checks everything createLearner({ state }) relies on that createModel does not (createModel checks the parameter
 * count and sizes against the widths), so a bad checkpoint throws RangeError before anything is allocated.
 */
function validateState(s: LearnerState): void {
  try {
    checkState(s)
  } catch (e) {
    // A missing or mistyped field surfaces as a TypeError somewhere inside the checks: it is still a bad checkpoint.
    if (e instanceof RangeError) throw e
    throw new RangeError(`LearnerState: malformed (${e instanceof Error ? e.message : String(e)})`, { cause: e })
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const allFinite = (a: Float32Array) => {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i]!)) return false
  return true
}
const STATE_KEYS = ["version", "seed", "widths", "settings", "params", "adam", "rng", "h", "last", "replay", "stats"] as const

function checkState(s: LearnerState): void {
  const fail = (what: string) => { throw new RangeError(`LearnerState: ${what}`) }
  if (!isRecord(s)) fail("not an object")
  for (const k of STATE_KEYS) if (!(k in s)) fail(`missing ${k}`)
  const isF32 = (v: unknown) => v instanceof Float32Array
  if (!isRecord(s.widths) || !isRecord(s.settings) || !isRecord(s.stats)) fail("widths, settings and stats must be objects")
  if (!Array.isArray(s.params) || !s.params.every(isF32)) fail("params must be Float32Arrays")
  if (!Array.isArray(s.adam) || !s.adam.every(a => isRecord(a) && Array.isArray(a.slots) && a.slots.every(sl => isRecord(sl) && isF32(sl.m) && isF32(sl.v)))) {
    fail("adam must be two optimizer states of Float32Array moments")
  }
  if (!isF32(s.h)) fail("h must be a Float32Array")
  if (s.last !== null && !(isRecord(s.last) && isRecord(s.last.obs) && isRecord(s.last.action) && isF32(s.last.hIn))) fail("last is malformed")
  if (!Array.isArray(s.replay) || !s.replay.every(tr => isRecord(tr) && isRecord(tr.obs) && isRecord(tr.next) && isRecord(tr.action) && isF32(tr.hIn))) {
    fail("replay is malformed")
  }
  if (s.version !== LEARNER_STATE_VERSION) fail(`version ${String(s.version)}, expected ${LEARNER_STATE_VERSION}`)
  if (!isUint32(s.seed)) fail(`seed must be a uint32, got ${s.seed}`)
  for (const k of ["conv1", "conv2", "enc", "hidden"] as const) if (!isPosInt(s.widths[k])) fail(`widths.${k} must be an integer >= 1`)
  const clamped = clampSettings(s.settings)
  for (const k of Object.keys(DEFAULT_SETTINGS) as (keyof LearnerSettings)[]) {
    if (clamped[k] !== s.settings[k]) fail(`settings.${k} = ${s.settings[k]} is not a clamped setting`)
  }
  const n = s.params.length
  if (n <= OUTPUT_PARAMS) fail(`${n} parameters`)
  const split = [s.params.slice(0, n - OUTPUT_PARAMS), s.params.slice(n - OUTPUT_PARAMS)]
  if (s.adam.length !== 2) fail(`${s.adam.length} optimizer states, expected 2`)
  split.forEach((ps, k) => {
    const slots = s.adam[k]!.slots
    if (slots.length !== ps.length) fail(`optimizer ${k} has ${slots.length} slots for ${ps.length} parameters`)
    slots.forEach((sl, i) => {
      const len = ps[i]!.length
      if (sl.m.length !== len || sl.v.length !== len) fail(`optimizer ${k} slot ${i} does not match its parameter's ${len} values`)
      if (!(Number.isInteger(sl.t) && sl.t >= 0)) fail(`optimizer ${k} slot ${i} step count ${sl.t}`)
      if (!allFinite(sl.m) || !allFinite(sl.v)) fail(`optimizer ${k} slot ${i} has a non-finite moment`)
    })
  })
  // A non-finite weight would poison every later update (Review Focus 4): such a checkpoint is no checkpoint.
  s.params.forEach((p, i) => { if (!allFinite(p)) fail(`params[${i}] has a non-finite value`) })
  if (!isUint32(s.rng)) fail(`rng state must be a uint32, got ${s.rng}`)
  const hidden = s.widths.hidden
  const checkObs = (o: Observation, where: string) => {
    if (o.periphery.length !== PERIPHERY_LEN || o.fovea.length !== FOVEA_LEN) fail(`${where}: malformed frame`)
  }
  if (s.h.length !== hidden) fail(`h has ${s.h.length} values, hidden is ${hidden}`)
  if (!allFinite(s.h)) fail("h has a non-finite value")
  if (s.last) {
    checkObs(s.last.obs, "last.obs")
    if (!isValidAction(s.last.action)) fail("last.action is not a valid action")
    if (s.last.hIn.length !== hidden) fail(`last.hIn has ${s.last.hIn.length} values, hidden is ${hidden}`)
    if (!allFinite(s.last.hIn)) fail("last.hIn has a non-finite value")
  }
  if (s.replay.length > s.settings.bufferSize) fail(`${s.replay.length} transitions exceed bufferSize ${s.settings.bufferSize}`)
  s.replay.forEach((tr, i) => {
    checkObs(tr.obs, `replay[${i}].obs`)
    checkObs(tr.next, `replay[${i}].next`)
    if (!isValidAction(tr.action)) fail(`replay[${i}].action is not a valid action`)
    if (tr.hIn.length !== hidden) fail(`replay[${i}].hIn has ${tr.hIn.length} values, hidden is ${hidden}`)
    if (!allFinite(tr.hIn)) fail(`replay[${i}].hIn has a non-finite value`)
  })
  for (const k of ["steps", "skippedUpdates", "consecutiveSkips", "sleeps"] as const) {
    if (!(Number.isInteger(s.stats[k]) && s.stats[k] >= 0)) fail(`stats.${k} = ${s.stats[k]}`)
  }
  if (!(Number.isFinite(s.stats.cu) && s.stats.cu >= 0)) fail(`stats.cu = ${s.stats.cu}`)
  if ((s.last === null) !== (s.stats.steps === 0)) fail("last must be null exactly when no step has been taken")
}

const copyObs = (o: Observation): Observation => ({
  periphery: o.periphery.slice(), fovea: o.fovea.slice(), cursor: { x: o.cursor.x, y: o.cursor.y }, buttonHeld: o.buttonHeld,
})

interface InputTensors { periphery: Tensor; fovea: Tensor; proprio: Tensor }

interface TargetTensors { periphery: Tensor; fovea: Tensor }

/** The previous step: what the next step learns from (all caller-owned). */
interface Last {
  obs: Observation
  inputs: InputTensors
  action: Action
  targets: Targets
  /** Its targets on the device: the residual predictor's base for the next transition. */
  targetTensors: TargetTensors
}

/**
 * The online learner, condition A (one set of weights). Each `step(obs_t)`:
 *  1. learns to predict y_t from the last transition: h_{t-1} = core(inputs_{t-1}, h_{t-2}) is recomputed with
 *     gradient from the stored, detached h_{t-2} (truncated BPTT of length 1, no graph lives across steps),
 *     L_before = mse(predict(h_{t-1}, a_{t-1}, y_{t-1}), y_t) over both outputs (residual on the current frame),
 *     then backward and one Adam step;
 *  2. measures curiosity r = clamp((L_before − L_after) / max(predLoss EMA before this step, 1e-6), −1, 1), L_after
 *     being the same loss with the new weights;
 *  3. acts: h_t = core(inputs_t, h_{t-1}) and the heads, under noGrad; one 77-float read (logits, V(h_t)); a_t is sampled
 *     per factor with the learner's seeded RNG;
 *  4. updates the actor-critic on a_{t-1}: A = curiosityWeight·r + γ·V(h_t) − V(h_{t-1}),
 *     loss = −logπ(a_{t-1})·stopgrad(A) + A² − entropyWeight·H(π), one Adam step;
 *  5. stores (obs_{t-1}, h_{t-2}, a_{t-1}, obs_t) in the FIFO replay buffer and carries h ← h_t (detached).
 *
 * Steps 1 and 4 are separate backward passes because r depends on the prediction update. They touch disjoint
 * parameters — the policy/value heads read the detached h_{t-1}, so the actor-critic gradient does not enter
 * the core — hence every parameter still gets exactly one Adam step per learner step, as for a single
 * backward(L_before + actorCritic). V(h_{t-1}) uses the carried h_{t-1}, which equals the recomputed one unless
 * a sleep() changed the core weights in between (both are then valid detached states; the carried one is used).
 *
 * sleep() replays buffered transitions (condition A); save() / createLearner({ state }) checkpoint and resume exactly.
 *
 * The predictor's residual output layers train at lr × headLrScale (a second Adam), all other parameters at lr.
 *
 * Guards: lr is clamped to LR_RANGE, headLrScale to HEAD_LR_SCALE_RANGE; a non-finite loss skips its update (counted in skippedUpdates, never
 * applied, never thrown). A rejected read() is fatal and propagates (packages/tensor/README.md rule 5).
 */
export async function createLearner(opts: {
  backend: Backend
  onUpdate?: (u: UpdateInfo) => void
} & ({ seed: number; widths: Widths; settings: LearnerSettings; state?: undefined } | { state: LearnerState })): Promise<TrainableLearner> {
  const { backend } = opts
  // The learner owns its copy. Any state given (even a falsy one) is a checkpoint and is validated.
  const saved = opts.state !== undefined ? structuredClone(opts.state) : null
  if (opts.state !== undefined) validateState(saved as LearnerState)
  const seed = saved ? saved.seed : (opts as { seed: number }).seed
  const widths: Widths = saved ? saved.widths : (opts as { widths: Widths }).widths
  const settings = saved ? saved.settings : clampSettings((opts as { settings: LearnerSettings }).settings)
  const model = createModel(backend, seed, widths, saved?.params)
  // Two optimizers over disjoint parameters: the residual output layers at lr × headLrScale (they start at zero and
  // see near-zero gradients over static background, where Adam's normalized step at full lr is pure jitter that
  // outweighs what they learn), everything else at lr. Each step runs both: one Adam step per parameter.
  // The checkpoint stores the two optimizers by position: [all but the last OUTPUT_PARAMS params, the last ones].
  if (model.outputParams.length !== OUTPUT_PARAMS || model.outputParams.some((p, i) => p !== model.params[model.params.length - OUTPUT_PARAMS + i])) {
    model.dispose()
    throw new Error("learner: Model.outputParams must be the last OUTPUT_PARAMS of Model.params (checkpoint format)")
  }
  const outputSet = new Set(model.outputParams)
  // Action sampling has its own stream, independent of the initializer's.
  const rng = createRng((seed ^ 0x9e3779b9) >>> 0)
  const replay = new ReplayBuffer(settings.bufferSize)
  const stats: LearnerStats = { steps: 0, cu: 0, predLoss: Number.NaN, baselineLoss: Number.NaN, skippedUpdates: 0, consecutiveSkips: 0, sleeps: 0 }

  // Every tensor made by `t` while the learner is being constructed is registered (see `owned` below).
  let constructing: ((x: Tensor) => void) | null = null
  const t = (data: Float32Array, shape: number[]) => {
    const x = tensor(data, shape, { backend })
    constructing?.(x)
    return x
  }
  const upload = (i: Inputs): InputTensors => ({
    periphery: t(i.periphery, [PERIPHERY.channels, PERIPHERY.height, PERIPHERY.width]),
    fovea: t(i.fovea, [FOVEA.channels, FOVEA.height, FOVEA.width]),
    proprio: t(i.proprio, [1, 3]),
  })
  const disposeLast = (l: Last) => {
    for (const x of [l.inputs.periphery, l.inputs.fovea, l.inputs.proprio, l.targetTensors.periphery, l.targetTensors.fovea]) x.dispose()
  }
  const scalar = async (x: Tensor) => (await x.read())[0]!
  const hTensor = (d: Float32Array) => t(d, [1, widths.hidden])
  const targetTensors = (y: Targets): TargetTensors => ({ periphery: t(y.periphery, [1, PRED_P]), fovea: t(y.fovea, [1, PRED_F]) })
  /** Every update outcome goes through here (online and replayed). */
  const countUpdate = (applied: boolean) => {
    if (applied) stats.consecutiveSkips = 0
    else {
      stats.skippedUpdates++
      stats.consecutiveSkips++
    }
  }

  // Everything construction allocates is registered here and freed if it fails part-way (e.g. the device refusing a
  // buffer while a checkpoint is restored), so a failed createLearner leaks nothing.
  const owned: { dispose(): void }[] = [model]
  const own = <T extends { dispose(): void }>(x: T): T => { owned.push(x); return x }
  constructing = own
  const opts_: Adam[] = []
  // Recurrent state: h = h_{t-1} (output of the last core call) and hIn = h_{t-2} (its input), detached.
  let h: Tensor
  let hData: Float32Array = new Float32Array(widths.hidden)
  let hIn: Tensor | null = null
  let hInData: Float32Array = new Float32Array(widths.hidden)
  let last: Last | null = null
  let busy = false
  let disposed = false
  // concat(logits [1,76], value [1,1]) -> [1,77] as two exact 0/1 selection matmuls (no concat op in @blanktab/tensor),
  // so acting reads the heads once. Caller-owned constants, freed in dispose().
  let selL: Tensor
  let selV: Tensor
  try {
    opts_.push(own(new Adam(model.params.filter((p) => !outputSet.has(p)), { lr: settings.lr })))
    opts_.push(own(new Adam([...model.outputParams], { lr: settings.lr * settings.headLrScale })))
    const selLogits = new Float32Array(ACTION_DIM * (ACTION_DIM + 1))
    for (let k = 0; k < ACTION_DIM; k++) selLogits[k * (ACTION_DIM + 1) + k] = 1
    const selValue = new Float32Array(ACTION_DIM + 1)
    selValue[ACTION_DIM] = 1
    selL = t(selLogits, [ACTION_DIM, ACTION_DIM + 1])
    selV = t(selValue, [1, ACTION_DIM + 1])
    if (saved) {
      // validateState checked every shape; the model already holds the parameters.
      opts_[0]!.setState(saved.adam[0])
      opts_[1]!.setState(saved.adam[1])
      rng.setState(saved.rng)
      for (const tr of saved.replay) replay.push(tr)
      Object.assign(stats, saved.stats)
      h = hTensor(saved.h)
      hData = saved.h
      if (saved.last) {
        const { obs, action } = saved.last
        const y = targetsFromObs(obs)
        hIn = hTensor(saved.last.hIn)
        hInData = saved.last.hIn
        last = { obs, inputs: upload(obsToInputs(obs)), action, targets: y, targetTensors: targetTensors(y) }
      }
    } else {
      h = own(model.zeroState())
    }
  } catch (e) {
    for (const x of owned.reverse()) x.dispose()
    throw e
  } finally {
    constructing = null
  }
  owned.length = 0
  const opt = {
    step: () => { for (const o of opts_) o.step() },
    zeroGrad: () => { for (const o of opts_) o.zeroGrad() },
    dispose: () => { for (const o of opts_) o.dispose() },
  }
  const ema = (prev: number, v: number) => (Number.isFinite(prev) ? prev + STATS_ALPHA * (v - prev) : v)

  /**
   * Size-weighted MSE of predict(core(inputs, hIn), action, base) against y. `base` is the targets of the frame
   * `inputs` came from (residual predictor). Builds a graph unless under noGrad.
   */
  const predictionLoss = (inputs: InputTensors, hIn: Tensor, action: Tensor, base: TargetTensors, y: TargetTensors) => {
    const p = model.predict(model.core(inputs, hIn), action, base)
    return add(scale(mse(p.periphery, y.periphery), W_P), scale(mse(p.fovea, y.fovea), W_F))
  }

  /** −logπ(a)·A + (target − V)² − β·H(π) on the heads at the detached h. `adv` is stopgrad(target − V). */
  const actorCriticLoss = (hPrev: Tensor, action: Action, target: number) => {
    const { logits, value } = model.heads(hPrev)
    const adv = noGrad(() => composite(() => addScalar(neg(value), target)))
    const flat = reshape(logits, [ACTION_DIM, 1])
    const chosen = [action.move, action.button, action.scroll, action.key]
    let logp: Tensor | null = null
    let negEntropy: Tensor | null = null
    for (let f = 0; f < FACTORS.length; f++) {
      const n = FACTORS[f]!
      const lp = logSoftmax(reshape(gatherRows(flat, FACTOR_ROWS[f]!), [1, n]))
      const lpa = gatherRows(reshape(lp, [n, 1]), Int32Array.of(chosen[f]!))
      const ne = sum(mul(exp(lp), lp))
      logp = logp ? add(logp, lpa) : lpa
      negEntropy = negEntropy ? add(negEntropy, ne) : ne
    }
    const actor = neg(sum(mul(logp!, adv)))
    const critic = sum(square(addScalar(neg(value), target)))
    return { loss: add(add(actor, critic), scale(negEntropy!, settings.entropyWeight)), adv }
  }

  /** Builds the replay of one transition: its prediction loss (a graph unless under noGrad) and the constants it used. */
  const replayed = (tr: Transition) => {
    const inputs = upload(obsToInputs(tr.obs))
    const base = targetTensors(targetsFromObs(tr.obs))
    const y = targetTensors(targetsFromObs(tr.next))
    const hT = hTensor(tr.hIn)
    const aT = t(actionOneHot(tr.action), [1, ACTION_DIM])
    const free = () => { for (const x of [inputs.periphery, inputs.fovea, inputs.proprio, base.periphery, base.fovea, y.periphery, y.fovea, hT, aT]) x.dispose() }
    return { loss: () => predictionLoss(inputs, hT, aT, base, y), free }
  }

  async function sleepNow(): Promise<void> {
    const flops0 = backend.stats.flops
    for (let k = 0; k < settings.replayPerSleep && replay.size > 0; k++) {
      const tr = replay.at(Math.min(replay.size - 1, Math.floor(rng() * replay.size)))
      const r = replayed(tr)
      const loss = r.loss()
      const l = await scalar(loss)
      const ok = Number.isFinite(l)
      if (ok) {
        backward(loss)
        opt.step()
        opt.zeroGrad()
      }
      disposeGraph(loss, [])
      r.free()
      countUpdate(ok)
    }
    stats.sleeps++
    stats.cu += (backend.stats.flops - flops0) / 1e9
  }

  async function replayLossNow(): Promise<number> {
    let total = 0
    for (const tr of replay.toArray()) {
      const r = replayed(tr)
      const loss = noGrad(() => composite(r.loss))
      total += await scalar(loss)
      loss.dispose()
      r.free()
    }
    return replay.size > 0 ? total / replay.size : Number.NaN
  }

  async function saveNow(): Promise<LearnerState> {
    const [params, adam0, adam1] = await Promise.all([Promise.all(model.params.map((p) => p.read())), opts_[0]!.state(), opts_[1]!.state()])
    // structuredClone: a snapshot that shares nothing with the learner (frames shared between consecutive
    // transitions stay shared within the snapshot).
    return structuredClone({
      version: LEARNER_STATE_VERSION,
      seed,
      widths: { ...widths },
      settings: { ...settings },
      params,
      adam: [adam0, adam1],
      rng: rng.getState(),
      h: hData,
      last: last && { obs: last.obs, action: last.action, hIn: hInData },
      replay: replay.toArray(),
      stats: { ...stats },
    } satisfies LearnerState)
  }

  /** Runs `fn` with the learner to itself: step, sleep, save and replayLoss never interleave. */
  async function exclusive<T>(fn: () => Promise<T>): Promise<T> {
    if (disposed) throw new Error("learner: used after dispose")
    if (busy) throw new Error("learner: a step, sleep or save is already in progress (await it first)")
    busy = true
    try {
      return await fn()
    } finally {
      busy = false
    }
  }

  async function learnAndAct(o: Observation): Promise<Action> {
    const flops0 = backend.stats.flops
    const obs = copyObs(o)
    const y = targetsFromObs(obs) // validates the frames (RangeError) before any state changes
    const inputs = upload(obsToInputs(obs))
    const yT = targetTensors(y)
    stats.steps++

    let info: UpdateInfo | null = null
    let aT: Tensor | null = null
    if (last && hIn) {
      const prev: Last = last
      const hPrev2: Tensor = hIn
      aT = t(actionOneHot(prev.action), [1, ACTION_DIM])
      const a1 = aT
      info = { step: stats.steps, lossBefore: Number.NaN, lossAfter: Number.NaN, reward: 0, advantage: Number.NaN, skipped: true }

      // 1. Prediction update (TBPTT-1 through the core); the residual base is y_{t-1}.
      const lossBefore = predictionLoss(prev.inputs, hPrev2, a1, prev.targetTensors, yT)
      const lb = await scalar(lossBefore)
      if (Number.isFinite(lb)) {
        backward(lossBefore)
        opt.step()
        opt.zeroGrad()
      }
      disposeGraph(lossBefore, [])
      // 2. Curiosity: the same loss with the updated weights, relative to the running loss.
      if (Number.isFinite(lb)) {
        info.lossBefore = lb
        const after = noGrad(() => composite(() => predictionLoss(prev.inputs, hPrev2, a1, prev.targetTensors, yT)))
        const la = await scalar(after)
        after.dispose()
        if (Number.isFinite(la)) {
          info.lossAfter = la
          // Scaled by the EMA from before this loss is folded in; the very first update has no scale yet (r = 0).
          info.reward = curiosityReward(lb, la, stats.predLoss)
        }
        stats.predLoss = ema(stats.predLoss, lb)
      }
      const base = copyLastBaseline(prev.targets, y)
      if (Number.isFinite(base)) stats.baselineLoss = ema(stats.baselineLoss, base)
    }

    // 3. Act: h_t and the heads (one [1,77] read: 76 logits then V(h_t)), no graph.
    const hNext = noGrad(() => composite(() => model.core(inputs, h)))
    const headsT = noGrad(() => composite(() => {
      const { logits, value } = model.heads(hNext)
      return add(matmul(logits, selL), matmul(value, selV))
    }))
    const [headsOut, hNextData] = await Promise.all([headsT.read(), hNext.read()])
    const logits = headsOut.subarray(0, ACTION_DIM)
    const valueNext = headsOut[ACTION_DIM]!
    const action: Action = {
      move: sampleFactor(logits, OFFSETS[0]!, FACTORS[0], rng),
      button: sampleFactor(logits, OFFSETS[1]!, FACTORS[1], rng),
      scroll: sampleFactor(logits, OFFSETS[2]!, FACTORS[2], rng),
      key: sampleFactor(logits, OFFSETS[3]!, FACTORS[3], rng),
    }

    // 4. Actor-critic on a_{t-1}, only when the curiosity reward was measured.
    if (last && info && Number.isFinite(info.lossAfter)) {
      const target = settings.curiosityWeight * info.reward + GAMMA * valueNext
      const { loss, adv } = actorCriticLoss(h, last.action, target)
      const [lac, advantage] = await Promise.all([scalar(loss), scalar(adv)])
      if (Number.isFinite(lac)) {
        backward(loss)
        opt.step()
        opt.zeroGrad()
        info.advantage = advantage
        info.skipped = false
      }
      disposeGraph(loss, [])
      adv.dispose()
    }
    if (info) countUpdate(!info.skipped)

    // 5. Store the transition, carry the state, free the rest.
    if (last) replay.push({ obs: last.obs, hIn: hInData, action: last.action, next: obs })
    aT?.dispose()
    headsT.dispose()
    if (last) disposeLast(last)
    hIn?.dispose()
    hIn = h
    hInData = hData
    h = hNext
    hData = hNextData
    last = { obs, inputs, action, targets: y, targetTensors: yT }

    stats.cu += (backend.stats.flops - flops0) / 1e9
    if (info) opts.onUpdate?.(info)
    return action
  }

  return {
    replay,

    step: (o: Observation) => exclusive(() => learnAndAct(o)),
    sleep: () => exclusive(sleepNow),
    save: () => exclusive(saveNow),
    replayLoss: () => exclusive(replayLossNow),

    stats: () => ({ ...stats }),

    readParams: () => Promise.all(model.params.map((p) => p.read())),

    dispose() {
      if (disposed) return
      disposed = true
      if (last) disposeLast(last)
      last = null
      selL.dispose()
      selV.dispose()
      h.dispose()
      hIn?.dispose()
      hIn = null
      opt.dispose()
      model.dispose()
    },
  }
}
