import { ResourceError, ShapeError, broadcastShape } from "./backend"
import type { BinaryOp, Storage, UnaryOp } from "./backend"
import { Tensor, isGradEnabled } from "./tensor"

const fmt = (s: readonly number[]) => `[${s.join(",")}]`

const notImplemented = (op: string) => (): Tensor[] => {
  throw new Error(`backward for "${op}" is not implemented yet`)
}

/** Attach a graph node to `out` iff grad mode is on and any parent requires grad. */
function record(out: Tensor, parents: Tensor[], op: string): Tensor {
  if (isGradEnabled() && parents.some((p) => p.requiresGrad)) {
    out.requiresGrad = true
    out._node = { parents, backward: notImplemented(op) }
  }
  return out
}

function sameBackend(op: string, a: Tensor, b: Tensor): void {
  if (a.backend !== b.backend) throw new ResourceError(`${op}: tensors live on different backends`)
}

function binaryOp(op: BinaryOp, a: Tensor, b: Tensor): Tensor {
  sameBackend(op, a, b)
  const outShape = broadcastShape([...a.shape], [...b.shape])
  const s = a.backend.binary(op, a.storage, [...a.shape], b.storage, [...b.shape], outShape)
  return record(new Tensor(s, outShape), [a, b], op)
}

export const add = (a: Tensor, b: Tensor): Tensor => binaryOp("add", a, b)
export const sub = (a: Tensor, b: Tensor): Tensor => binaryOp("sub", a, b)
export const mul = (a: Tensor, b: Tensor): Tensor => binaryOp("mul", a, b)
export const div = (a: Tensor, b: Tensor): Tensor => binaryOp("div", a, b)

function scalarOp(op: BinaryOp, name: string, a: Tensor, c: number): Tensor {
  return record(new Tensor(a.backend.scalar(op, a.storage, c), a.shape), [a], name)
}

export const scale = (a: Tensor, c: number): Tensor => scalarOp("mul", "scale", a, c)
export const addScalar = (a: Tensor, c: number): Tensor => scalarOp("add", "addScalar", a, c)

function unaryOp(op: UnaryOp, a: Tensor): Tensor {
  return record(new Tensor(a.backend.unary(op, a.storage), a.shape), [a], op)
}

export const neg = (a: Tensor): Tensor => unaryOp("neg", a)
export const relu = (a: Tensor): Tensor => unaryOp("relu", a)
export const tanh = (a: Tensor): Tensor => unaryOp("tanh", a)
export const sigmoid = (a: Tensor): Tensor => unaryOp("sigmoid", a)
export const exp = (a: Tensor): Tensor => unaryOp("exp", a)
export const log = (a: Tensor): Tensor => unaryOp("log", a)
export const sqrt = (a: Tensor): Tensor => unaryOp("sqrt", a)
export const square = (a: Tensor): Tensor => unaryOp("square", a)

/** 2-D only: [m,k]·[k,n] -> [m,n]. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  sameBackend("matmul", a, b)
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[0]) {
    throw new ShapeError(`matmul: incompatible shapes ${fmt(a.shape)} and ${fmt(b.shape)}`)
  }
  const [m, k] = a.shape as [number, number]
  const n = b.shape[1]!
  const s = a.backend.matmul(a.storage, b.storage, m, k, n, false, false)
  return record(new Tensor(s, [m, n]), [a, b], "matmul")
}

/** Sum of all elements -> scalar (shape []). */
export function sum(a: Tensor): Tensor {
  const s = a.backend.reduceLast("sum", a.storage, 1, a.size)
  return record(new Tensor(s, []), [a], "sum")
}

export function mean(a: Tensor): Tensor {
  if (a.size === 0) throw new ShapeError(`mean: empty tensor of shape ${fmt(a.shape)}`)
  const total = a.backend.reduceLast("sum", a.storage, 1, a.size)
  const s = a.backend.scalar("mul", total, 1 / a.size)
  a.backend.dispose(total)
  return record(new Tensor(s, []), [a], "mean")
}

function lastAxis(op: string, a: Tensor): { lead: number[]; n: number; rows: number } {
  if (a.shape.length === 0) throw new ShapeError(`${op}: needs at least 1 dimension, got shape []`)
  const n = a.shape[a.shape.length - 1]!
  return { lead: a.shape.slice(0, -1), n, rows: n === 0 ? 0 : a.size / n }
}

/** [..., n] -> [...]. */
export function sumLast(a: Tensor): Tensor {
  const { lead, n, rows } = lastAxis("sumLast", a)
  return record(new Tensor(a.backend.reduceLast("sum", a.storage, rows, n), lead), [a], "sumLast")
}

/** Shifted logits (a - rowmax) plus keepdims shape; caller owns the returned storage. */
function shifted(a: Tensor, op: string) {
  const { lead, n, rows } = lastAxis(op, a)
  if (n === 0) throw new ShapeError(`${op}: last axis is empty in shape ${fmt(a.shape)}`)
  const be = a.backend
  const keep = [...lead, 1]
  const mx = be.reduceLast("max", a.storage, rows, n)
  const sh = be.binary("sub", a.storage, [...a.shape], mx, keep, [...a.shape])
  be.dispose(mx)
  return { be, keep, n, rows, sh }
}

/** Softmax over the last axis (max-subtracted for stability). */
export function softmax(a: Tensor): Tensor {
  const { be, keep, n, rows, sh } = shifted(a, "softmax")
  const e = be.unary("exp", sh)
  const z = be.reduceLast("sum", e, rows, n)
  const out = be.binary("div", e, [...a.shape], z, keep, [...a.shape])
  for (const s of [sh, e, z]) be.dispose(s)
  return record(new Tensor(out, a.shape), [a], "softmax")
}

/** log-softmax over the last axis: (a - max) - log(sum(exp(a - max))). */
export function logSoftmax(a: Tensor): Tensor {
  const { be, keep, n, rows, sh } = shifted(a, "logSoftmax")
  const e = be.unary("exp", sh)
  const z = be.reduceLast("sum", e, rows, n)
  const lz = be.unary("log", z)
  const out = be.binary("sub", sh, [...a.shape], lz, keep, [...a.shape])
  for (const s of [sh, e, z, lz]) be.dispose(s)
  return record(new Tensor(out, a.shape), [a], "logSoftmax")
}

/** Reshape sharing storage; one dimension may be -1 (inferred). */
export function reshape(a: Tensor, shape: number[]): Tensor {
  const unknown = shape.filter((d) => d === -1).length
  if (unknown > 1) throw new ShapeError(`reshape: at most one -1 allowed in ${fmt(shape)}`)
  let target = shape
  if (unknown === 1) {
    const known = shape.reduce((p, d) => (d === -1 ? p : p * d), 1)
    if (known === 0 || a.size % known !== 0) {
      throw new ShapeError(`reshape: cannot reshape ${fmt(a.shape)} to ${fmt(shape)}`)
    }
    target = shape.map((d) => (d === -1 ? a.size / known : d))
  }
  if (target.reduce((p, d) => p * d, 1) !== a.size) {
    throw new ShapeError(`reshape: cannot reshape ${fmt(a.shape)} to ${fmt(shape)}`)
  }
  return record(new Tensor(a.storage, target), [a], "reshape")
}

/** a: [rows, cols] -> [idx.length, cols]. */
export function gatherRows(a: Tensor, idx: Int32Array): Tensor {
  if (a.shape.length !== 2) throw new ShapeError(`gatherRows: needs a 2-D tensor, got ${fmt(a.shape)}`)
  const [rows, cols] = a.shape as [number, number]
  const s: Storage = a.backend.gatherRows(a.storage, rows, cols, idx)
  return record(new Tensor(s, [idx.length, cols]), [a], "gatherRows")
}
