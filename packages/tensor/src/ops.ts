import { ResourceError, ShapeError, broadcastShape } from "./backend"
import type { BinaryOp, Storage, UnaryOp } from "./backend"
import { Tensor, isGradEnabled } from "./tensor"
import type { GradNode } from "./tensor"

const fmt = (s: readonly number[]) => `[${s.join(",")}]`
const sameShape = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((d, i) => d === b[i])

/** Attach a graph node to `out` iff grad mode is on and any parent requires grad. */
function record(out: Tensor, parents: Tensor[], backward: GradNode["backward"]): Tensor {
  if (isGradEnabled() && parents.some((p) => p.requiresGrad)) {
    out.requiresGrad = true
    out._node = { parents, backward }
  }
  return out
}

// ---- backward helpers (storage-level; never record) ----

/**
 * Runs `fn` with a `tmp` registrar; every tensor passed to `tmp` is disposed afterwards
 * unless its storage is part of the result. Keeps the backward rules leak-free.
 */
function scoped<T extends (Tensor | null)[]>(fn: (tmp: (t: Tensor) => Tensor) => T): T {
  const temps: Tensor[] = []
  const out = fn((t) => (temps.push(t), t))
  const kept = new Set(out.map((t) => t?.storage))
  for (const t of temps) if (!kept.has(t.storage)) t.dispose()
  return out
}

const scoped1 = (fn: (tmp: (t: Tensor) => Tensor) => Tensor): Tensor => scoped((tmp) => [fn(tmp)])[0]!

const un = (op: UnaryOp, a: Tensor) => new Tensor(a.backend.unary(op, a.storage), a.shape)
const sc = (op: BinaryOp, a: Tensor, c: number) => new Tensor(a.backend.scalar(op, a.storage, c), a.shape)
/** Elementwise with broadcasting; `aShape`/`bShape` default to the tensors' own shapes. */
function bin(op: BinaryOp, a: Tensor, b: Tensor, aShape: readonly number[] = a.shape, bShape: readonly number[] = b.shape): Tensor {
  const outShape = broadcastShape([...aShape], [...bShape])
  return new Tensor(a.backend.binary(op, a.storage, [...aShape], b.storage, [...bShape], outShape), outShape)
}
/** Sum a broadcast gradient back to `shape` (returns `g` itself when no reduction is needed). */
const reduceTo = (g: Tensor, shape: readonly number[]): Tensor =>
  sameShape(g.shape, shape) ? g : new Tensor(g.backend.sumToShape(g.storage, [...g.shape], [...shape]), shape)
/** Broadcast `g` (viewed as `gShape`) up to `shape`. */
function expandTo(g: Tensor, gShape: readonly number[], shape: readonly number[]): Tensor {
  const be = g.backend
  const z = be.zeros(shape.reduce((p, d) => p * d, 1))
  const s = be.binary("add", z, [...shape], g.storage, [...gShape], [...shape])
  be.dispose(z)
  return new Tensor(s, shape)
}

// ---- elementwise binary ----

function sameBackend(op: string, a: Tensor, b: Tensor): void {
  if (a.backend !== b.backend) throw new ResourceError(`${op}: tensors live on different backends`)
}

type BinaryGrad = (g: Tensor, a: Tensor, b: Tensor, out: Tensor, needs: readonly boolean[]) => (Tensor | null)[]

const binaryGrads: Record<BinaryOp, BinaryGrad> = {
  add: (g, a, b, _o, [na, nb]) => [na ? reduceTo(g, a.shape) : null, nb ? reduceTo(g, b.shape) : null],
  sub: (g, a, b, _o, [na, nb]) =>
    scoped((tmp) => [na ? reduceTo(g, a.shape) : null, nb ? reduceTo(tmp(un("neg", g)), b.shape) : null]),
  mul: (g, a, b, _o, [na, nb]) =>
    scoped((tmp) => [na ? reduceTo(tmp(bin("mul", g, b)), a.shape) : null, nb ? reduceTo(tmp(bin("mul", g, a)), b.shape) : null]),
  // d/da = g/b; d/db = -g*a/b^2 = -(g*out)/b
  div: (g, a, b, out, [na, nb]) =>
    scoped((tmp) => [
      na ? reduceTo(tmp(bin("div", g, b)), a.shape) : null,
      nb ? reduceTo(tmp(sc("mul", tmp(bin("div", tmp(bin("mul", g, out)), b)), -1)), b.shape) : null,
    ]),
}

function binaryOp(op: BinaryOp, a: Tensor, b: Tensor): Tensor {
  sameBackend(op, a, b)
  const outShape = broadcastShape([...a.shape], [...b.shape])
  const s = a.backend.binary(op, a.storage, [...a.shape], b.storage, [...b.shape], outShape)
  const out = new Tensor(s, outShape)
  return record(out, [a, b], (g, needs) => binaryGrads[op](g, a, b, out, needs))
}

export const add = (a: Tensor, b: Tensor): Tensor => binaryOp("add", a, b)
export const sub = (a: Tensor, b: Tensor): Tensor => binaryOp("sub", a, b)
export const mul = (a: Tensor, b: Tensor): Tensor => binaryOp("mul", a, b)
export const div = (a: Tensor, b: Tensor): Tensor => binaryOp("div", a, b)

// ---- scalar ----

export function scale(a: Tensor, c: number): Tensor {
  return record(sc("mul", a, c), [a], (g) => [sc("mul", g, c)])
}

export function addScalar(a: Tensor, c: number): Tensor {
  return record(sc("add", a, c), [a], (g) => [g])
}

// ---- unary ----

type PublicUnary = Exclude<UnaryOp, "step" | "recip">

/** d(loss)/d(a) given g, the input a and the output y. */
const unaryGrads: Record<PublicUnary, (g: Tensor, a: Tensor, y: Tensor) => Tensor> = {
  neg: (g) => un("neg", g),
  relu: (g, a) => scoped1((tmp) => bin("mul", g, tmp(un("step", a)))),
  // 1 - y^2
  tanh: (g, _a, y) => scoped1((tmp) => bin("mul", g, tmp(sc("add", tmp(sc("mul", tmp(un("square", y)), -1)), 1)))),
  // y(1 - y) = y - y^2
  sigmoid: (g, _a, y) => scoped1((tmp) => bin("mul", g, tmp(bin("sub", y, tmp(un("square", y)))))),
  exp: (g, _a, y) => bin("mul", g, y),
  log: (g, a) => scoped1((tmp) => bin("mul", g, tmp(un("recip", a)))),
  // 1 / (2 sqrt(a)) = 0.5 / y
  sqrt: (g, _a, y) => scoped1((tmp) => bin("div", tmp(sc("mul", g, 0.5)), y)),
  square: (g, a) => scoped1((tmp) => bin("mul", g, tmp(sc("mul", a, 2)))),
}

function unaryOp(op: PublicUnary, a: Tensor): Tensor {
  const out = un(op, a)
  return record(out, [a], (g) => [unaryGrads[op](g, a, out)])
}

export const neg = (a: Tensor): Tensor => unaryOp("neg", a)
export const relu = (a: Tensor): Tensor => unaryOp("relu", a)
export const tanh = (a: Tensor): Tensor => unaryOp("tanh", a)
export const sigmoid = (a: Tensor): Tensor => unaryOp("sigmoid", a)
export const exp = (a: Tensor): Tensor => unaryOp("exp", a)
export const log = (a: Tensor): Tensor => unaryOp("log", a)
export const sqrt = (a: Tensor): Tensor => unaryOp("sqrt", a)
export const square = (a: Tensor): Tensor => unaryOp("square", a)

// ---- matmul ----

/** 2-D only: [m,k]·[k,n] -> [m,n]. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  sameBackend("matmul", a, b)
  if (a.shape.length !== 2 || b.shape.length !== 2 || a.shape[1] !== b.shape[0]) {
    throw new ShapeError(`matmul: incompatible shapes ${fmt(a.shape)} and ${fmt(b.shape)}`)
  }
  const [m, k] = a.shape as [number, number]
  const n = b.shape[1]!
  const be = a.backend
  const out = new Tensor(be.matmul(a.storage, b.storage, m, k, n, false, false), [m, n])
  return record(out, [a, b], (g, [na, nb]) => [
    // dA = g·Bᵀ : [m,n]·[n,k]
    na ? new Tensor(be.matmul(g.storage, b.storage, m, n, k, false, true), [m, k]) : null,
    // dB = Aᵀ·g : [k,m]·[m,n]
    nb ? new Tensor(be.matmul(a.storage, g.storage, k, m, n, true, false), [k, n]) : null,
  ])
}

// ---- reductions ----

/** Sum of all elements -> scalar (shape []). */
export function sum(a: Tensor): Tensor {
  const out = new Tensor(a.backend.reduceLast("sum", a.storage, 1, a.size), [])
  return record(out, [a], (g) => [expandTo(g, [], a.shape)])
}

export function mean(a: Tensor): Tensor {
  if (a.size === 0) throw new ShapeError(`mean: empty tensor of shape ${fmt(a.shape)}`)
  const total = a.backend.reduceLast("sum", a.storage, 1, a.size)
  const s = a.backend.scalar("mul", total, 1 / a.size)
  a.backend.dispose(total)
  return record(new Tensor(s, []), [a], (g) => scoped((tmp) => [expandTo(tmp(sc("mul", g, 1 / a.size)), [], a.shape)]))
}

/** Leading dims, last-axis length, and row count (product of leading dims, even when n is 0). */
function lastAxis(op: string, a: Tensor): { lead: number[]; n: number; rows: number } {
  if (a.shape.length === 0) throw new ShapeError(`${op}: needs at least 1 dimension, got shape []`)
  const lead = a.shape.slice(0, -1)
  return { lead, n: a.shape[a.shape.length - 1]!, rows: lead.reduce((p, d) => p * d, 1) }
}

/** [..., n] -> [...]. */
export function sumLast(a: Tensor): Tensor {
  const { lead, n, rows } = lastAxis("sumLast", a)
  const out = new Tensor(a.backend.reduceLast("sum", a.storage, rows, n), lead)
  return record(out, [a], (g) => [expandTo(g, [...lead, 1], a.shape)])
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
  const out = new Tensor(be.binary("div", e, [...a.shape], z, keep, [...a.shape]), a.shape)
  for (const s of [sh, e, z]) be.dispose(s)
  // dx = y * (g - sum_last(g*y))
  return record(out, [a], (g) =>
    scoped((tmp) => {
      const gy = tmp(bin("mul", g, out))
      const s = tmp(new Tensor(be.reduceLast("sum", gy.storage, rows, n), keep))
      return [bin("mul", out, tmp(bin("sub", g, s)))]
    }),
  )
}

/** log-softmax over the last axis: (a - max) - log(sum(exp(a - max))). */
export function logSoftmax(a: Tensor): Tensor {
  const { be, keep, n, rows, sh } = shifted(a, "logSoftmax")
  const e = be.unary("exp", sh)
  const z = be.reduceLast("sum", e, rows, n)
  const lz = be.unary("log", z)
  const out = new Tensor(be.binary("sub", sh, [...a.shape], lz, keep, [...a.shape]), a.shape)
  for (const s of [sh, e, z, lz]) be.dispose(s)
  // dx = g - softmax(x) * sum_last(g), softmax(x) = exp(out)
  return record(out, [a], (g) =>
    scoped((tmp) => {
      const s = tmp(new Tensor(be.reduceLast("sum", g.storage, rows, n), keep))
      return [bin("sub", g, tmp(bin("mul", tmp(un("exp", out)), s)))]
    }),
  )
}

// ---- shape / indexing ----

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
  // The gradient view shares g's storage; autograd reference-counts it.
  return record(new Tensor(a.storage, target), [a], (g) => [new Tensor(g.storage, a.shape)])
}

/** a: [rows, cols] -> [idx.length, cols]. */
export function gatherRows(a: Tensor, idx: Int32Array): Tensor {
  if (a.shape.length !== 2) throw new ShapeError(`gatherRows: needs a 2-D tensor, got ${fmt(a.shape)}`)
  const [rows, cols] = a.shape as [number, number]
  const be = a.backend
  const s: Storage = be.gatherRows(a.storage, rows, cols, idx)
  return record(new Tensor(s, [idx.length, cols]), [a], (g) => [new Tensor(be.scatterAddRows(g.storage, rows, cols, idx), a.shape)])
}
