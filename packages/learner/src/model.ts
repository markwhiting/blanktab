import { FOVEA, PERIPHERY, PRED_FOVEA, PRED_PERIPHERY } from "@blanktab/interface"
import { add, composite, conv2d, createRng, gatherRows, gruCell, linear, matmul, mul, relu, reshape, softmax, sub, tanh, tensor, zeros } from "@blanktab/tensor"
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
  /** The predictor's zero-initialised residual output layers (headP w/b, headF w/b): the last four of `params`. */
  readonly outputParams: readonly Tensor[]
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
   * the core just saw): periphery [1,3000], fovea [1,768] (HWC, 0..1). The fovea also gets an action-conditioned
   * translation of its base: + g(a)·(shift(base; k(a)) − base), see `shiftFovea`. The output layers and the gate
   * g start at zero, so a fresh model predicts exactly copy-last (no change) and learns only the change.
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

/*
 * Fovea translation (docs/research/2026-10-21-why-not-learning.md). The fovea is a window centred on the cursor, so
 * a move translates its content, which a linear readout of z cannot express. This architectural prior (translation
 * of the current image, as in CDNA, Finn et al. 2016) predicts the next fovea as a convex, separable, action-chosen
 * blend of shifted copies of the current one: kernel kx ⊗ ky, kx = softmax(a·Wx + c), ky = softmax(a·Wy + c),
 * applied as two banded matmuls with zero padding, then gated by g = a·Wg (zero-initialised). No knowledge of the
 * world or of which action moves the cursor: the action → kernel map is learned.
 */
/** Kernel taps per axis: ±8 target pixels, the largest move (16 world px). */
export const SHIFT_TAPS = 17
const SHIFT_R = (SHIFT_TAPS - 1) / 2
/** Initial logit of the centre tap: e^4 / (e^4 + 16) ≈ 0.77 of each axis's mass stays put before learning. */
const SHIFT_CENTRE_LOGIT = 4
const FW = PRED_FOVEA.width, FH = PRED_FOVEA.height, FC = PRED_FOVEA.channels
const ROWS = FC * FH // the fovea as a [FC*FH, FW] plane stack
/** gather HWC -> CHW: the HWC index of each CHW element. */
const HWC_TO_CHW = Int32Array.from({ length: PRED_F }, (_, i) => (i % (FW * FH)) * FC + Math.floor(i / (FW * FH)))
/** gather CHW -> HWC: the CHW index of each HWC element. */
const CHW_TO_HWC = Int32Array.from({ length: PRED_F }, (_, i) => (i % FC) * FW * FH + Math.floor(i / FC))
/** [SHIFT_TAPS, FW*FW] 0/1: kx · this = Bx [FW, FW], Bx[x', x] = kx[x' − x + R] (horizontal correlation). */
const BAND_X = (() => {
  const m = new Float32Array(SHIFT_TAPS * FW * FW)
  for (let xp = 0; xp < FW; xp++) for (let x = 0; x < FW; x++) {
    const j = xp - x + SHIFT_R
    if (j >= 0 && j < SHIFT_TAPS) m[j * FW * FW + xp * FW + x] = 1
  }
  return m
})()
/** [SHIFT_TAPS, ROWS*ROWS] 0/1: ky · this = By [ROWS, ROWS], block-diagonal per channel, By[y, y'] = ky[y' − y + R]. */
const BAND_Y = (() => {
  const m = new Float32Array(SHIFT_TAPS * ROWS * ROWS)
  for (let c = 0; c < FC; c++) for (let y = 0; y < FH; y++) for (let yp = 0; yp < FH; yp++) {
    const j = yp - y + SHIFT_R
    if (j >= 0 && j < SHIFT_TAPS) m[j * ROWS * ROWS + (c * FH + y) * ROWS + c * FH + yp] = 1
  }
  return m
})()

/**
 * Builds the model with every weight drawn from `seed`, or, given `initial` (a checkpoint's parameter values in
 * Model.params order), with copies of those values. A checkpoint that does not fit these widths throws RangeError
 * and leaks nothing.
 */
export function createModel(backend: Backend, seed: number, widths: Widths, initial?: readonly Float32Array[]): Model {
  const params: Tensor[] = []
  try {
    const model = buildModel(backend, seed, widths, initial, params)
    if (initial && initial.length !== params.length) throw new RangeError(`createModel: ${initial.length} saved parameters, the model has ${params.length}`)
    return model
  } catch (e) {
    for (const t of params) t.dispose()
    throw e
  }
}

function buildModel(backend: Backend, seed: number, widths: Widths, initial: readonly Float32Array[] | undefined, params: Tensor[]): Model {
  const { conv1, conv2, enc, hidden } = widths
  const rng = createRng(seed)
  /** The saved values for the next parameter (a copy is uploaded), or null when drawing fresh weights. */
  const saved = (shape: number[]): Float32Array | null => {
    if (!initial) return null
    const i = params.length
    const n = shape.reduce((a, d) => a * d, 1)
    const v = initial[i]
    if (!v || v.length !== n) throw new RangeError(`createModel: saved parameter ${i} has ${v?.length ?? "no"} values, shape [${shape.join(",")}] needs ${n}`)
    return v
  }
  /** Trainable, uniform in ±1/sqrt(fanIn), drawn in creation order. */
  const p = (shape: number[], fanIn: number): Tensor => {
    const lim = 1 / Math.sqrt(fanIn)
    const n = shape.reduce((a, d) => a * d, 1)
    const data = saved(shape) ?? Float32Array.from({ length: n }, () => (2 * rng() - 1) * lim)
    const t = tensor(data, shape, { requiresGrad: true, backend })
    params.push(t)
    return t
  }
  /** Trainable, all zeros; draws nothing from the rng. */
  const p0 = (shape: number[]): Tensor => {
    const v = saved(shape)
    const t = v ? tensor(v, shape, { requiresGrad: true, backend }) : zeros(shape, { requiresGrad: true, backend })
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
  // Fovea translation (see SHIFT_TAPS): action -> kernel logits per axis and gate, all zero-initialised.
  const shiftX = p0([ACTION_DIM, SHIFT_TAPS])
  const shiftY = p0([ACTION_DIM, SHIFT_TAPS])
  const shiftGate = p0([ACTION_DIM, 1])
  const predH = dense(hidden, enc)
  const predA = dense(ACTION_DIM, enc)
  // Residual output layers, zero-initialised (weights and biases): the predictor starts at copy-last.
  const headP = { w: p0([enc, PRED_P]), b: p0([PRED_P]) }
  const headF = { w: p0([enc, PRED_F]), b: p0([PRED_F]) }

  // Constants (caller-owned, freed in dispose).
  const centre = new Float32Array(SHIFT_TAPS)
  centre[SHIFT_R] = SHIFT_CENTRE_LOGIT
  const consts: Tensor[] = []
  const constant = (d: Float32Array, shape: number[]) => {
    const t = tensor(d, shape, { backend })
    consts.push(t)
    return t
  }
  let centreT: Tensor, bandX: Tensor, bandY: Tensor
  try {
    centreT = constant(centre, [1, SHIFT_TAPS])
    bandX = constant(BAND_X, [SHIFT_TAPS, FW * FW])
    bandY = constant(BAND_Y, [SHIFT_TAPS, ROWS * ROWS])
  } catch (e) {
    for (const t of consts) t.dispose()
    throw e
  }

  /** The fovea's translation term g(a)·(shift(base; kx(a) ⊗ ky(a)) − base), for base [1, PRED_F] HWC. */
  const shiftFovea = (action: Tensor, base: Tensor): Tensor => composite(() => {
    // composite: the gathers of the (constant) base have no graph node and would otherwise leak (tensor README rule 2).
    const kx = softmax(add(matmul(action, shiftX), centreT))
    const ky = softmax(add(matmul(action, shiftY), centreT))
    const bx = reshape(matmul(kx, bandX), [FW, FW])
    const by = reshape(matmul(ky, bandY), [ROWS, ROWS])
    const planes = reshape(gatherRows(reshape(base, [PRED_F, 1]), HWC_TO_CHW), [ROWS, FW])
    const moved = matmul(by, matmul(planes, bx))
    const movedHwc = reshape(gatherRows(reshape(moved, [PRED_F, 1]), CHW_TO_HWC), [1, PRED_F])
    return mul(sub(movedHwc, base), matmul(action, shiftGate))
  })

  return {
    params,
    outputParams: [headP.w, headP.b, headF.w, headF.b],
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
        fovea: add(add(base.fovea, shiftFovea(action, base.fovea)), tanh(linear(z, headF.w, headF.b))),
      }
    },

    zeroState() {
      return zeros([1, hidden], { backend })
    },

    dispose() {
      for (const t of consts) t.dispose()
      for (const t of params) {
        t.grad?.dispose()
        t.grad = null
        t.dispose()
      }
    },
  }
}
