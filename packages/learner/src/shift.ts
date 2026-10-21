import { PRED_FOVEA } from "@blanktab/interface"
import { add, composite, gatherRows, matmul, mul, reshape, softmax, sub, tensor } from "@blanktab/tensor"
import type { Backend, Tensor } from "@blanktab/tensor"

/*
 * The fovea-shift prior (decision 0015; docs/research/2026-10-21-why-not-learning.md). The fovea is a window centred
 * on the cursor, so a move translates its content, which a linear readout of the predictor's hidden layer cannot
 * express. This architectural prior (translation of the current image, as in CDNA, Finn et al. 2016) predicts the next
 * fovea as a convex, separable, action-chosen blend of shifted copies of the current one: kernel kx ⊗ ky, with
 * kx = softmax(a·Wx + c) and ky = softmax(a·Wy + c), applied as two banded matmuls with zero padding, then gated by
 * g = a·Wg. Nothing here knows the world or which action moves the cursor: the action → kernel map is learned.
 */

/** Kernel taps per axis: ±8 target pixels, the largest move (16 world px). */
export const SHIFT_TAPS = 17
const R = (SHIFT_TAPS - 1) / 2
/** Initial logit of the centre tap: e^4 / (e^4 + 16) ≈ 0.77 of each axis's mass stays put before learning. */
const CENTRE_LOGIT = 4
const FW = PRED_FOVEA.width, FH = PRED_FOVEA.height, FC = PRED_FOVEA.channels
const PF = FW * FH * FC
const ROWS = FC * FH // the fovea as a [FC*FH, FW] stack of planes
/** gather HWC -> CHW: the HWC index of each CHW element. */
const HWC_TO_CHW = Int32Array.from({ length: PF }, (_, i) => (i % (FW * FH)) * FC + Math.floor(i / (FW * FH)))
/** gather CHW -> HWC: the CHW index of each HWC element. */
const CHW_TO_HWC = Int32Array.from({ length: PF }, (_, i) => (i % FC) * FW * FH + Math.floor(i / FC))
/** [SHIFT_TAPS, FW*FW] 0/1: kx · this = Bx [FW, FW], Bx[x', x] = kx[x' − x + R]. */
const BAND_X = (() => {
  const m = new Float32Array(SHIFT_TAPS * FW * FW)
  for (let xp = 0; xp < FW; xp++) for (let x = 0; x < FW; x++) {
    const j = xp - x + R
    if (j >= 0 && j < SHIFT_TAPS) m[j * FW * FW + xp * FW + x] = 1
  }
  return m
})()
/** [SHIFT_TAPS, ROWS*ROWS] 0/1: ky · this = By [ROWS, ROWS], block-diagonal per channel, By[y, y'] = ky[y' − y + R]. */
const BAND_Y = (() => {
  const m = new Float32Array(SHIFT_TAPS * ROWS * ROWS)
  for (let c = 0; c < FC; c++) for (let y = 0; y < FH; y++) for (let yp = 0; yp < FH; yp++) {
    const j = yp - y + R
    if (j >= 0 && j < SHIFT_TAPS) m[j * ROWS * ROWS + (c * FH + y) * ROWS + c * FH + yp] = 1
  }
  return m
})()

/** The shift's trainable parameters (owned by the model, in its params): kernel logits per axis and the gate. */
export interface ShiftParams {
  /** [ACTION_DIM, SHIFT_TAPS] */ x: Tensor
  /** [ACTION_DIM, SHIFT_TAPS] */ y: Tensor
  /** [ACTION_DIM, 1] */ gate: Tensor
}

/**
 * The fovea-shift operator on one backend. It owns only constants (the centre prior and the band matrices), which
 * `dispose` frees. Every method is a composite (packages/tensor/README.md rule 3): only its result survives.
 */
export interface FoveaShift {
  /** One axis's kernel for `action` [1,ACTION_DIM]: softmax(action·w + centre prior), [1,SHIFT_TAPS], sums to 1. */
  kernel(action: Tensor, w: Tensor): Tensor
  /**
   * `base` [1,PRED_F] (HWC) translated by the separable kernel kx ⊗ ky [1,SHIFT_TAPS] each, zero-filled at the edges:
   * out[y,x,c] = Σ_i Σ_j ky[i]·kx[j]·base[y + i − R, x + j − R, c]. A one-hot kernel at tap R + k shifts by exactly k.
   */
  translate(base: Tensor, kx: Tensor, ky: Tensor): Tensor
  /** The prediction term g(a)·(translate(base; kernel(a, p.x), kernel(a, p.y)) − base), [1,PRED_F]; 0 while g(a) = 0. */
  term(action: Tensor, base: Tensor, p: ShiftParams): Tensor
  dispose(): void
}

export function createFoveaShift(backend: Backend): FoveaShift {
  const consts: Tensor[] = []
  const constant = (d: Float32Array, shape: number[]) => {
    const t = tensor(d, shape, { backend })
    consts.push(t)
    return t
  }
  let centre: Tensor, bandX: Tensor, bandY: Tensor
  try {
    const c = new Float32Array(SHIFT_TAPS)
    c[R] = CENTRE_LOGIT
    centre = constant(c, [1, SHIFT_TAPS])
    bandX = constant(BAND_X, [SHIFT_TAPS, FW * FW])
    bandY = constant(BAND_Y, [SHIFT_TAPS, ROWS * ROWS])
  } catch (e) {
    for (const t of consts) t.dispose()
    throw e
  }

  const kernel = (action: Tensor, w: Tensor): Tensor => composite(() => softmax(add(matmul(action, w), centre)))
  // composite: the gathers of a constant base have no graph node and would otherwise leak (tensor README rule 2).
  const translate = (base: Tensor, kx: Tensor, ky: Tensor): Tensor => composite(() => {
    const bx = reshape(matmul(kx, bandX), [FW, FW])
    const by = reshape(matmul(ky, bandY), [ROWS, ROWS])
    const planes = reshape(gatherRows(reshape(base, [PF, 1]), HWC_TO_CHW), [ROWS, FW])
    const moved = matmul(by, matmul(planes, bx))
    return reshape(gatherRows(reshape(moved, [PF, 1]), CHW_TO_HWC), [1, PF])
  })

  return {
    kernel,
    translate,
    term: (action, base, p) => composite(() => {
      const moved = translate(base, kernel(action, p.x), kernel(action, p.y))
      return mul(sub(moved, base), matmul(action, p.gate))
    }),
    dispose() {
      for (const t of consts) t.dispose()
    },
  }
}
