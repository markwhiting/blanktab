import { ShapeError } from "./backend"
import { add, addScalar, div, im2col, matmul, mean, mul, reshape, scale, sigmoid, square, sqrt, sub, sumLast, tanh } from "./ops"
import { param } from "./tensor"
import type { Tensor } from "./tensor"

// Composites are built purely from differentiable ops: they have no backward code of their own.

/** x [B,in] · w [in,out] (+ b [out]) -> [B,out]. */
export function linear(x: Tensor, w: Tensor, b?: Tensor): Tensor {
  const y = matmul(x, w)
  return b ? add(y, b) : y
}

/** x [C,H,W]; w [O, C*kh*kw]; b [O,1]; out [O,outH,outW]. */
export function conv2d(
  x: Tensor,
  w: Tensor,
  b: Tensor | undefined,
  opts: { kh: number; kw: number; stride: number; pad: number },
): Tensor {
  const { kh, kw, stride, pad } = opts
  const cols = im2col(x, kh, kw, stride, pad)
  const y = matmul(w, cols) // [O, outH*outW]
  const outH = Math.floor((x.shape[1]! + 2 * pad - kh) / stride) + 1
  const outW = Math.floor((x.shape[2]! + 2 * pad - kw) / stride) + 1
  return reshape(b ? add(y, b) : y, [w.shape[0]!, outH, outW])
}

export interface GruParams {
  wz: Tensor; uz: Tensor; bz: Tensor
  wr: Tensor; ur: Tensor; br: Tensor
  wh: Tensor; uh: Tensor; bh: Tensor
}

/** Trainable GRU parameters, uniform in ±1/sqrt(hidDim). */
export function initGru(rng: () => number, inDim: number, hidDim: number): GruParams {
  const lim = 1 / Math.sqrt(hidDim)
  const u = (shape: number[]) => {
    const n = shape.reduce((p, d) => p * d, 1)
    return param(Float32Array.from({ length: n }, () => (2 * rng() - 1) * lim), shape)
  }
  return {
    wz: u([inDim, hidDim]), uz: u([hidDim, hidDim]), bz: u([hidDim]),
    wr: u([inDim, hidDim]), ur: u([hidDim, hidDim]), br: u([hidDim]),
    wh: u([inDim, hidDim]), uh: u([hidDim, hidDim]), bh: u([hidDim]),
  }
}

/** z=σ(xWz+hUz+bz), r=σ(xWr+hUr+br), ĥ=tanh(xWh+(r⊙h)Uh+bh), h'=(1−z)⊙h+z⊙ĥ. x [B,in], h [B,hid]. */
export function gruCell(x: Tensor, h: Tensor, p: GruParams): Tensor {
  const z = sigmoid(add(add(matmul(x, p.wz), matmul(h, p.uz)), p.bz))
  const r = sigmoid(add(add(matmul(x, p.wr), matmul(h, p.ur)), p.br))
  const cand = tanh(add(add(matmul(x, p.wh), matmul(mul(r, h), p.uh)), p.bh))
  // (1−z)⊙h + z⊙ĥ = h + z⊙(ĥ − h)
  return add(h, mul(z, sub(cand, h)))
}

/** Normalizes over the last axis, then scales by gamma and shifts by beta. */
export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
  const n = x.shape[x.shape.length - 1]
  if (n === undefined || n === 0) throw new ShapeError(`layerNorm: needs a non-empty last axis, got [${x.shape.join(",")}]`)
  const keep = [...x.shape.slice(0, -1), 1]
  const mu = reshape(scale(sumLast(x), 1 / n), keep)
  const centered = sub(x, mu)
  const variance = reshape(scale(sumLast(square(centered)), 1 / n), keep)
  const normed = div(centered, sqrt(addScalar(variance, eps)))
  return add(mul(normed, gamma), beta)
}

/** Mean of squared differences -> scalar. */
export function mse(pred: Tensor, target: Tensor): Tensor {
  return mean(square(sub(pred, target)))
}
