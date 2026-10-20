import { BUTTON_OPTIONS, FOVEA, KEYS, MOVE_OPTIONS, PERIPHERY, PRED_FOVEA, PRED_PERIPHERY, SCROLL_OPTIONS } from "@blanktab/interface"
import type { Action, Learner, Observation } from "@blanktab/interface"
import {
  Adam, add, addScalar, backward, composite, createRng, disposeGraph, exp, gatherRows, logSoftmax, matmul, mse, mul,
  neg, noGrad, reshape, scale, square, sum, tensor,
} from "@blanktab/tensor"
import type { Backend, Tensor } from "@blanktab/tensor"
import { ACTION_DIM, actionOneHot, copyLastBaseline, obsToInputs, targetsFromObs } from "./encode"
import type { Inputs, Targets } from "./encode"
import { createModel } from "./model"
import type { Widths } from "./model"
import { ReplayBuffer } from "./replay"

export interface LearnerSettings { lr: number; entropyWeight: number; curiosityWeight: number; bufferSize: number; replayPerSleep: number }
export const DEFAULT_SETTINGS: LearnerSettings = Object.freeze({ lr: 1e-3, entropyWeight: 0.01, curiosityWeight: 1, bufferSize: 2000, replayPerSleep: 64 })
export interface LearnerStats { steps: number; cu: number; predLoss: number; baselineLoss: number; skippedUpdates: number; sleeps: number }

/** What one online update did, for instrumentation and tests (not part of the learner's inputs). */
export interface UpdateInfo {
  step: number
  /** Prediction loss on the last transition before the update (L_before). NaN if non-finite. */
  lossBefore: number
  /** The same loss with the updated weights (L_after); NaN if not measured. */
  lossAfter: number
  /** Curiosity: clamp((L_before − L_after) / max(predLoss EMA, 1e-6), −1, 1), 0 when non-finite. */
  reward: number
  /** TD advantage curiosityWeight·reward + γ·V(h_t) − V(h_{t-1}); NaN if the actor-critic update did not run. */
  advantage: number
  skipped: boolean
}

export interface TrainableLearner extends Learner {
  stats(): LearnerStats
  /** Every parameter's values, in checkpoint order (Model.params). */
  readParams(): Promise<Float32Array[]>
  /** The FIFO replay buffer (raw frames; do not mutate). */
  readonly replay: ReplayBuffer
  dispose(): void
}

/** The learning rate is clamped into this range (Review Focus 5). */
export const LR_RANGE = [1e-5, 1e-2] as const
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
  return { ...s, lr: Math.min(LR_RANGE[1], Math.max(LR_RANGE[0], s.lr)) }
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
 *  2. measures curiosity r = clamp((L_before − L_after) / max(predLoss EMA, 1e-6), −1, 1), L_after being the same
 *     loss with the new weights;
 *  3. acts: h_t = core(inputs_t, h_{t-1}) and the heads, under noGrad; one 77-float read (logits, V(h_t)); a_t is sampled
 *     per factor with the learner's seeded RNG;
 *  4. updates the actor-critic on a_{t-1}: A = curiosityWeight·r + γ·V(h_t) − V(h_{t-1}),
 *     loss = −logπ(a_{t-1})·stopgrad(A) + A² − entropyWeight·H(π), one Adam step;
 *  5. stores (obs_{t-1}, h_{t-2}, a_{t-1}, obs_t) in the FIFO replay buffer and carries h ← h_t (detached).
 *
 * Steps 1 and 4 are separate backward passes because r depends on the prediction update. They touch disjoint
 * parameters — the policy/value heads read the detached h_{t-1}, so the actor-critic gradient does not enter
 * the core — hence every parameter still gets exactly one Adam step per learner step, as for a single
 * backward(L_before + actorCritic). V(h_{t-1}) uses the carried h_{t-1}, which equals the recomputed one: the
 * core weights do not change between acting at t−1 and learning at t.
 *
 * Guards: lr is clamped to LR_RANGE; a non-finite loss skips its update (counted in skippedUpdates, never
 * applied, never thrown). A rejected read() is fatal and propagates (packages/tensor/README.md rule 5).
 */
export async function createLearner(opts: {
  backend: Backend
  seed: number
  widths: Widths
  settings: LearnerSettings
  onUpdate?: (u: UpdateInfo) => void
}): Promise<TrainableLearner> {
  const { backend, seed, widths } = opts
  const settings = clampSettings(opts.settings)
  const model = createModel(backend, seed, widths)
  const opt = new Adam([...model.params], { lr: settings.lr })
  // Action sampling has its own stream, independent of the initializer's.
  const rng = createRng((seed ^ 0x9e3779b9) >>> 0)
  const replay = new ReplayBuffer(settings.bufferSize)
  const stats: LearnerStats = { steps: 0, cu: 0, predLoss: Number.NaN, baselineLoss: Number.NaN, skippedUpdates: 0, sleeps: 0 }

  // Recurrent state: h = h_{t-1} (output of the last core call) and hIn = h_{t-2} (its input), detached.
  let h = model.zeroState()
  let hData: Float32Array = new Float32Array(widths.hidden)
  let hIn: Tensor | null = null
  let hInData: Float32Array = new Float32Array(widths.hidden)
  let last: Last | null = null
  let busy = false
  let disposed = false

  const t = (data: Float32Array, shape: number[]) => tensor(data, shape, { backend })
  const upload = (i: Inputs): InputTensors => ({
    periphery: t(i.periphery, [PERIPHERY.channels, PERIPHERY.height, PERIPHERY.width]),
    fovea: t(i.fovea, [FOVEA.channels, FOVEA.height, FOVEA.width]),
    proprio: t(i.proprio, [1, 3]),
  })
  const disposeLast = (l: Last) => {
    for (const x of [l.inputs.periphery, l.inputs.fovea, l.inputs.proprio, l.targetTensors.periphery, l.targetTensors.fovea]) x.dispose()
  }
  // concat(logits [1,76], value [1,1]) -> [1,77] as two exact 0/1 selection matmuls (no concat op in @blanktab/tensor),
  // so acting reads the heads once. Caller-owned constants, freed in dispose().
  const selLogits = new Float32Array(ACTION_DIM * (ACTION_DIM + 1))
  for (let k = 0; k < ACTION_DIM; k++) selLogits[k * (ACTION_DIM + 1) + k] = 1
  const selValue = new Float32Array(ACTION_DIM + 1)
  selValue[ACTION_DIM] = 1
  const selL = t(selLogits, [ACTION_DIM, ACTION_DIM + 1])
  const selV = t(selValue, [1, ACTION_DIM + 1])
  const scalar = async (x: Tensor) => (await x.read())[0]!
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

  async function learnAndAct(o: Observation): Promise<Action> {
    const flops0 = backend.stats.flops
    const obs = copyObs(o)
    const y = targetsFromObs(obs) // validates the frames (RangeError) before any state changes
    const inputs = upload(obsToInputs(obs))
    const yT: TargetTensors = { periphery: t(y.periphery, [1, PRED_P]), fovea: t(y.fovea, [1, PRED_F]) }
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
        stats.predLoss = ema(stats.predLoss, lb)
        const after = noGrad(() => composite(() => predictionLoss(prev.inputs, hPrev2, a1, prev.targetTensors, yT)))
        const la = await scalar(after)
        after.dispose()
        if (Number.isFinite(la)) {
          info.lossAfter = la
          info.reward = curiosityReward(lb, la, stats.predLoss)
        }
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
    if (info?.skipped) stats.skippedUpdates++

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

    async step(o: Observation): Promise<Action> {
      if (disposed) throw new Error("learner: step after dispose")
      if (busy) throw new Error("learner: step already in progress (await the previous step)")
      busy = true
      try {
        return await learnAndAct(o)
      } finally {
        busy = false
      }
    },

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
