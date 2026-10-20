import { FOVEA, PERIPHERY, PRED_FOVEA, PRED_PERIPHERY } from "@blanktab/interface"
import { add, conv2d, createRng, gruCell, linear, relu, reshape, tanh, tensor, zeros } from "@blanktab/tensor"
import type { Backend, GruParams, Tensor } from "@blanktab/tensor"
import { ACTION_DIM } from "./encode"

export interface Widths { conv1: number; conv2: number; enc: number; hidden: number }
/** WebGPU widths (spec §5.5). */
export const GPU_WIDTHS: Widths = { conv1: 16, conv2: 32, enc: 128, hidden: 256 }
/** CPU fallback: half width. */
export const CPU_WIDTHS: Widths = { conv1: 8, conv2: 16, enc: 64, hidden: 128 }

/**
 * The learner's network. `core`, `heads` and `predict` build autograd graph nodes (every parameter
 * requires grad), so a training step frees them with `disposeGraph`; acting/inference must run inside
 * `noGrad(() => composite(...))` (or dispose every result), per packages/tensor/README.md.
 */
export interface Model {
  /**
   * Every trainable tensor, in creation order. This order is the checkpoint format: a saved state is
   * these tensors' data in this order, so never reorder, insert or remove parameters without a format bump.
   */
  readonly params: readonly Tensor[]
  readonly paramCount: number
  readonly widths: Widths
  /**
   * Encode the observation and advance the GRU: returns h' [1,hidden].
   * Inputs: periphery [3,50,80] and fovea [3,32,32] (CHW, 0..1, see obsToInputs), proprio [1,3]; h [1,hidden].
   */
  core(inp: { periphery: Tensor; fovea: Tensor; proprio: Tensor }, h: Tensor): Tensor
  /** Policy logits [1,76] (the four action factors concatenated, see ACTION_DIM) and value [1,1]. */
  heads(h: Tensor): { logits: Tensor; value: Tensor }
  /**
   * Residual next-frame prediction conditioned on the chosen action one-hot [1,76]: for each output,
   * base + tanh(linear(z)), where `base` is the current frame's targets (targetsFromObs of the observation
   * the core just saw): periphery [1,3000], fovea [1,768] (HWC, 0..1). The two output layers start at zero,
   * so a fresh model predicts exactly copy-last (no change) and learns only the change.
   */
  predict(h: Tensor, action: Tensor, base: { periphery: Tensor; fovea: Tensor }): { periphery: Tensor; fovea: Tensor }
  /** [1,hidden] zeros, caller-owned. */
  zeroState(): Tensor
  /** Frees every parameter (and any gradient on it). The model must not be used afterwards. */
  dispose(): void
}

const PROPRIO = 3
const outDim = (n: number, k: number, s: number, p: number) => Math.floor((n + 2 * p - k) / s) + 1
const PRED_P = PRED_PERIPHERY.width * PRED_PERIPHERY.height * PRED_PERIPHERY.channels
const PRED_F = PRED_FOVEA.width * PRED_FOVEA.height * PRED_FOVEA.channels

export function createModel(backend: Backend, seed: number, widths: Widths): Model {
  const { conv1, conv2, enc, hidden } = widths
  const rng = createRng(seed)
  const params: Tensor[] = []
  /** Trainable, uniform in ±1/sqrt(fanIn), drawn in creation order. */
  const p = (shape: number[], fanIn: number): Tensor => {
    const lim = 1 / Math.sqrt(fanIn)
    const n = shape.reduce((a, d) => a * d, 1)
    const t = tensor(Float32Array.from({ length: n }, () => (2 * rng() - 1) * lim), shape, { requiresGrad: true, backend })
    params.push(t)
    return t
  }
  /** Trainable, all zeros; draws nothing from the rng. */
  const p0 = (shape: number[]): Tensor => {
    const t = zeros(shape, { requiresGrad: true, backend })
    params.push(t)
    return t
  }
  const dense = (i: number, o: number) => ({ w: p([i, o], i), b: p([o], i) })
  const conv = (ci: number, co: number, k: number) => ({ w: p([co, ci * k * k], ci * k * k), b: p([co, 1], ci * k * k) })

  // Periphery: 50x80 -> 25x40 -> 13x20. Fovea: 32x32 -> 16x16 -> 8x8.
  const pH = outDim(outDim(PERIPHERY.height, 5, 2, 2), 3, 2, 1)
  const pW = outDim(outDim(PERIPHERY.width, 5, 2, 2), 3, 2, 1)
  const fH = outDim(outDim(FOVEA.height, 3, 2, 1), 3, 2, 1)
  const fW = outDim(outDim(FOVEA.width, 3, 2, 1), 3, 2, 1)
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
  const headPolicy = dense(hidden, ACTION_DIM)
  const headValue = dense(hidden, 1)
  const predH = dense(hidden, enc)
  const predA = dense(ACTION_DIM, enc)
  // Residual output layers, zero-initialised (weights and biases): the predictor starts at copy-last.
  const headP = { w: p0([enc, PRED_P]), b: p0([PRED_P]) }
  const headF = { w: p0([enc, PRED_F]), b: p0([PRED_F]) }

  return {
    params,
    paramCount: params.reduce((a, t) => a + t.size, 0),
    widths,

    core(inp, h) {
      const ep1 = relu(conv2d(inp.periphery, pc1.w, pc1.b, { kh: 5, kw: 5, stride: 2, pad: 2 }))
      const ep2 = relu(conv2d(ep1, pc2.w, pc2.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ep = linear(reshape(ep2, [1, -1]), pLin.w, pLin.b)
      const ef1 = relu(conv2d(inp.fovea, fc1.w, fc1.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ef2 = relu(conv2d(ef1, fc2.w, fc2.b, { kh: 3, kw: 3, stride: 2, pad: 1 }))
      const ef = linear(reshape(ef2, [1, -1]), fLin.w, fLin.b)
      const coreIn = add(add(ep, ef), linear(inp.proprio, proprioLin.w, proprioLin.b))
      return gruCell(coreIn, h, gru)
    },

    heads(h) {
      return { logits: linear(h, headPolicy.w, headPolicy.b), value: linear(h, headValue.w, headValue.b) }
    },

    predict(h, action, base) {
      const z = relu(add(linear(h, predH.w, predH.b), linear(action, predA.w, predA.b)))
      return {
        periphery: add(base.periphery, tanh(linear(z, headP.w, headP.b))),
        fovea: add(base.fovea, tanh(linear(z, headF.w, headF.b))),
      }
    },

    zeroState() {
      return zeros([1, hidden], { backend })
    },

    dispose() {
      for (const t of params) {
        t.grad?.dispose()
        t.grad = null
        t.dispose()
      }
    },
  }
}
