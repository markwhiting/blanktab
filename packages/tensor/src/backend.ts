export class ShapeError extends Error {}
export class ResourceError extends Error {}

export type UnaryOp = "neg" | "relu" | "step" | "tanh" | "sigmoid" | "exp" | "log" | "recip" | "sqrt" | "square"
export type BinaryOp = "add" | "sub" | "mul" | "div"

export interface Storage {
  readonly length: number
  readonly backend: Backend
}

export interface Backend {
  readonly name: "cpu" | "webgpu"
  readonly stats: { flops: number; liveBuffers: number }
  upload(data: Float32Array): Storage
  zeros(length: number): Storage
  read(s: Storage): Promise<Float32Array>
  dispose(s: Storage): void
  /** A new storage holding the same values (0 FLOPs; a copy, not arithmetic). */
  copy(s: Storage): Storage
  unary(op: UnaryOp, a: Storage): Storage
  /** a op c elementwise */
  scalar(op: BinaryOp, a: Storage, c: number): Storage
  /** NumPy broadcasting */
  binary(op: BinaryOp, a: Storage, aShape: number[], b: Storage, bShape: number[], outShape: number[]): Storage
  /** inverse of broadcast */
  sumToShape(a: Storage, fromShape: number[], toShape: number[]): Storage
  matmul(a: Storage, b: Storage, m: number, k: number, n: number, transA: boolean, transB: boolean): Storage
  /** -> length rows */
  reduceLast(op: "sum" | "max", a: Storage, rows: number, cols: number): Storage
  /** x is CHW; output [c*kh*kw, outH*outW] row-major, zero padding */
  im2col(x: Storage, c: number, h: number, w: number, kh: number, kw: number, stride: number, pad: number): Storage
  /** adjoint of im2col (sums overlaps) */
  col2im(cols: Storage, c: number, h: number, w: number, kh: number, kw: number, stride: number, pad: number): Storage
  gatherRows(a: Storage, rows: number, cols: number, idx: Int32Array): Storage
  /** adjoint of gatherRows */
  scatterAddRows(grad: Storage, rows: number, cols: number, idx: Int32Array): Storage
  /** in-place, bias-corrected; `decay` (default 0) is decoupled weight decay: p ← (1 − decay)·p − step */
  adam(p: Storage, g: Storage, m: Storage, v: Storage, lr: number, b1: number, b2: number, eps: number, t: number, decay?: number): void
}

const fmt = (s: number[]) => `[${s.join(",")}]`

/**
 * Output size [oh, ow] of im2col/col2im, shared by every backend so geometry rules cannot drift.
 * Throws ShapeError unless stride >= 1, kh, kw >= 1, pad >= 0 (all integers) and the output is non-empty.
 */
export function im2colGeometry(what: string, c: number, h: number, w: number, kh: number, kw: number, stride: number, pad: number): [number, number] {
  const dims = { c, h, w, kh, kw, stride, pad }
  for (const [name, v] of Object.entries(dims)) {
    if (!Number.isInteger(v)) throw new ShapeError(`${what}: ${name} must be an integer, got ${v}`)
  }
  if (stride < 1) throw new ShapeError(`${what}: stride must be >= 1, got ${stride}`)
  if (kh < 1 || kw < 1) throw new ShapeError(`${what}: kernel ${kh}x${kw} must be at least 1x1`)
  if (pad < 0 || c < 0 || h < 0 || w < 0) throw new ShapeError(`${what}: negative size in ${c}x${h}x${w} pad ${pad}`)
  const oh = Math.floor((h + 2 * pad - kh) / stride) + 1
  const ow = Math.floor((w + 2 * pad - kw) / stride) + 1
  if (oh < 1 || ow < 1) {
    throw new ShapeError(`${what}: kernel ${kh}x${kw} stride ${stride} does not fit ${c}x${h}x${w} with pad ${pad} (output ${oh}x${ow})`)
  }
  return [oh, ow]
}

/** NumPy broadcasting; throws ShapeError naming both shapes on mismatch. */
export function broadcastShape(a: number[], b: number[]): number[] {
  const n = Math.max(a.length, b.length)
  const out: number[] = new Array<number>(n)
  for (let i = 0; i < n; i++) {
    const da = i < n - a.length ? 1 : a[i - (n - a.length)]!
    const db = i < n - b.length ? 1 : b[i - (n - b.length)]!
    if (da !== db && da !== 1 && db !== 1) {
      throw new ShapeError(`cannot broadcast shapes ${fmt(a)} and ${fmt(b)}`)
    }
    out[i] = da === 1 ? db : da
  }
  return out
}
