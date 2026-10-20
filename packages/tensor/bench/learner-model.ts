import { Adam, add, backward, conv2d, createRng, detach, disposeGraph, gruCell, linear, logSoftmax, mean, mse, mul, neg, randn, relu, reshape, tensor } from "../src/index"
import type { Backend, GruParams, Tensor } from "../src/index"

/**
 * Spike 2 stand-in for the learner (spec §5.5) at default width. Not the learner: it has the same
 * layer shapes and per-step work (forward, backward, Adam) so its throughput predicts the learner's.
 *
 *  periphery [3,50,80] -conv 3→c1 k5 s2 p2- relu -conv c1→c2 k3 s2 p1- relu -linear→enc
 *  fovea     [3,32,32] -conv 3→c1 k3 s2 p1- relu -conv c1→c2 k3 s2 p1- relu -linear→enc
 *  core input = periphery + fovea + linear(proprio 3→enc);  GRU enc→hidden (state carried across steps)
 *  heads: periphery prediction hidden→40·25·3, fovea prediction hidden→16·16·3, policy hidden→25+3+3+45, value hidden→1
 */
export interface BenchWidths {
  conv1: number
  conv2: number
  enc: number
  hidden: number
}

export const DEFAULT_WIDTHS: BenchWidths = { conv1: 16, conv2: 32, enc: 128, hidden: 256 }

const PERIPHERY = [3, 50, 80] as const
const FOVEA = [3, 32, 32] as const
const PROPRIO = 3
const PRED_PERIPHERY = 40 * 25 * 3
const PRED_FOVEA = 16 * 16 * 3
const POLICY = 25 + 3 + 3 + 45
/** Distinct host-side frames cycled through, so every step uploads fresh inputs and targets as the learner will. */
const FRAMES = 4

const outDim = (n: number, k: number, s: number, p: number) => Math.floor((n + 2 * p - k) / s) + 1

export interface BenchModel {
  readonly paramCount: number
  /** Every trainable parameter, in creation order (stable across backends for the same seed). */
  readonly params: readonly Tensor[]
  /**
   * One training step: forward, losses, backward, Adam.step, zeroGrad, disposeGraph. Reads nothing
   * back unless `readLoss`, in which case it resolves the step's loss (and waits for the GPU).
   * `inspectGrads`, if given, is awaited after backward and before Adam.step, with every parameter's
   * `.grad` populated (tests compare them across backends).
   */
  step(readLoss?: boolean, inspectGrads?: (params: readonly Tensor[]) => Promise<void>): Promise<number | null>
  /** Frees every buffer the model owns (params, optimizer moments, carried state, constants). */
  dispose(): void
}

interface Frame {
  periphery: Float32Array
  fovea: Float32Array
  proprio: Float32Array
  targetPeriphery: Float32Array
  targetFovea: Float32Array
  targetValue: Float32Array
}

export function createBenchModel(backend: Backend, widths: BenchWidths = DEFAULT_WIDTHS, seed = 1): BenchModel {
  const { conv1, conv2, enc, hidden } = widths
  const rng = createRng(seed)
  const params: Tensor[] = []
  /** Trainable, uniform in ±1/sqrt(fanIn). */
  const p = (shape: number[], fanIn: number): Tensor => {
    const lim = 1 / Math.sqrt(fanIn)
    const n = shape.reduce((a, d) => a * d, 1)
    const t = tensor(Float32Array.from({ length: n }, () => (2 * rng() - 1) * lim), shape, { requiresGrad: true, backend })
    params.push(t)
    return t
  }
  const dense = (i: number, o: number) => ({ w: p([i, o], i), b: p([o], i) })
  const conv = (ci: number, co: number, k: number) => ({ w: p([co, ci * k * k], ci * k * k), b: p([co, 1], ci * k * k) })

  // Periphery: 50x80 -> 25x40 -> 13x20. Fovea: 32x32 -> 16x16 -> 8x8.
  const pH = outDim(outDim(PERIPHERY[1], 5, 2, 2), 3, 2, 1)
  const pW = outDim(outDim(PERIPHERY[2], 5, 2, 2), 3, 2, 1)
  const fH = outDim(outDim(FOVEA[1], 3, 2, 1), 3, 2, 1)
  const fW = outDim(outDim(FOVEA[2], 3, 2, 1), 3, 2, 1)
  const pc1 = conv(3, conv1, 5)
  const pc2 = conv(conv1, conv2, 3)
  const pLin = dense(conv2 * pH * pW, enc)
  const fc1 = conv(3, conv1, 3)
  const fc2 = conv(conv1, conv2, 3)
  const fLin = dense(conv2 * fH * fW, enc)
  const proprioLin = dense(PROPRIO, enc)
  const g = (i: number, o: number) => ({ w: p([i, o], o), u: p([o, o], o), b: p([o], o) })
  const [gz, gr, gh] = [g(enc, hidden), g(enc, hidden), g(enc, hidden)]
  const gru: GruParams = { wz: gz.w, uz: gz.u, bz: gz.b, wr: gr.w, ur: gr.u, br: gr.b, wh: gh.w, uh: gh.u, bh: gh.b }
  const headP = dense(hidden, PRED_PERIPHERY)
  const headF = dense(hidden, PRED_FOVEA)
  const headPolicy = dense(hidden, POLICY)
  const headValue = dense(hidden, 1)

  const paramCount = params.reduce((a, t) => a + t.size, 0)
  const opt = new Adam(params, { lr: 1e-3 })

  // Constants live for the model's lifetime (disposeGraph treats them as leaves and never frees them).
  const policyTarget = tensor(new Float32Array(POLICY).fill(1 / POLICY), [1, POLICY], { backend })
  let h = tensor(new Float32Array(hidden), [1, hidden], { backend })

  const frames: Frame[] = Array.from({ length: FRAMES }, () => ({
    periphery: randn(rng, PERIPHERY[0] * PERIPHERY[1] * PERIPHERY[2]).map((v) => 0.5 + 0.25 * v),
    fovea: randn(rng, FOVEA[0] * FOVEA[1] * FOVEA[2]).map((v) => 0.5 + 0.25 * v),
    proprio: Float32Array.from({ length: PROPRIO }, () => rng()),
    targetPeriphery: Float32Array.from({ length: PRED_PERIPHERY }, () => rng()),
    targetFovea: Float32Array.from({ length: PRED_FOVEA }, () => rng()),
    targetValue: Float32Array.of(rng()),
  }))
  let frameIndex = 0
  let disposed = false

  return {
    paramCount,
    params,

    async step(readLoss = false, inspectGrads) {
      if (disposed) throw new Error("BenchModel: step after dispose")
      const f = frames[frameIndex++ % FRAMES]!
      // Per-step inputs: uploaded fresh (as the learner's frames will be) and freed at the end of the step.
      const xp = tensor(f.periphery, [...PERIPHERY], { backend })
      const xf = tensor(f.fovea, [...FOVEA], { backend })
      const xq = tensor(f.proprio, [1, PROPRIO], { backend })
      const tp = tensor(f.targetPeriphery, [1, PRED_PERIPHERY], { backend })
      const tf = tensor(f.targetFovea, [1, PRED_FOVEA], { backend })
      const tv = tensor(f.targetValue, [1, 1], { backend })
      const inputs = [xp, xf, xq, tp, tf, tv]

      const ep1 = relu(conv2d(xp, pc1.w, pc1.b, { kh: 5, kw: 5, stride: 2, pad: 2 }))
      const ep2 = relu(conv2d(ep1, pc2.w, pc2.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ep = linear(reshape(ep2, [1, -1]), pLin.w, pLin.b)
      const ef1 = relu(conv2d(xf, fc1.w, fc1.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ef2 = relu(conv2d(ef1, fc2.w, fc2.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ef = linear(reshape(ef2, [1, -1]), fLin.w, fLin.b)
      const coreIn = add(add(ep, ef), linear(xq, proprioLin.w, proprioLin.b))
      const h2 = gruCell(coreIn, h, gru)

      const predLoss = add(mse(linear(h2, headP.w, headP.b), tp), mse(linear(h2, headF.w, headF.b), tf))
      // Dummy policy/value losses: cross-entropy against a fixed uniform target, MSE on the value.
      const policyLoss = neg(mean(mul(logSoftmax(linear(h2, headPolicy.w, headPolicy.b)), policyTarget)))
      const valueLoss = mse(linear(h2, headValue.w, headValue.b), tv)
      const loss = add(add(predLoss, policyLoss), valueLoss)

      backward(loss)
      if (inspectGrads) await inspectGrads(params)
      opt.step()
      opt.zeroGrad()
      // Carried state, no gradient through time: copy h2 out of the graph, then free the graph.
      const next = detach(h2)
      disposeGraph(loss, readLoss ? [loss] : [])
      for (const t of inputs) t.dispose()
      h.dispose()
      h = next

      if (!readLoss) return null
      try {
        return (await loss.read())[0]!
      } finally {
        loss.dispose()
      }
    },

    dispose() {
      if (disposed) return
      disposed = true
      opt.dispose()
      for (const t of [...params, policyTarget, h]) t.dispose()
    },
  }
}
