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
  /** in-place, bias-corrected */
  adam(p: Storage, g: Storage, m: Storage, v: Storage, lr: number, b1: number, b2: number, eps: number, t: number): void
}

const fmt = (s: number[]) => `[${s.join(",")}]`

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
