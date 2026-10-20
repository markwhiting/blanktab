import { describe, expect, it } from "vitest"
import {
  Adam, ShapeError, Tensor, add, addScalar, backward, createRng, disposeGraph, div, exp, gatherRows, getDefaultBackend,
  gradCheck, log, logSoftmax, matmul, mean, mul, neg, noGrad, param, relu, reshape, scale, sigmoid, softmax, sqrt,
  square, sub, sum, sumLast, tanh, tensor,
} from "../src/index"

const rng = createRng(1234)
/** Values in ±[0.2, 1.2] (away from relu's kink) or, if positive, in [0.2, 1.2]. */
const rand = (shape: number[], positive = false) => {
  const n = shape.reduce((p, d) => p * d, 1)
  const data = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const v = 0.2 + rng()
    data[i] = positive || rng() < 0.5 ? v : -v
  }
  return { data, shape }
}

type F = (...xs: Tensor[]) => Tensor
const unary: [string, F, boolean][] = [
  ["neg", (a) => neg(a), false],
  ["relu", (a) => relu(a), false],
  ["tanh", (a) => tanh(a), false],
  ["sigmoid", (a) => sigmoid(a), false],
  ["exp", (a) => exp(a), false],
  ["log", (a) => log(a), true],
  ["sqrt", (a) => sqrt(a), true],
  ["square", (a) => square(a), false],
  ["scale", (a) => scale(a, -2.5), false],
  ["addScalar", (a) => addScalar(a, 3), false],
  ["sum", (a) => sum(a), false],
  ["mean", (a) => mean(a), false],
  ["sumLast", (a) => sumLast(a), false],
  ["softmax", (a) => softmax(a), false],
  ["logSoftmax", (a) => logSoftmax(a), false],
  ["reshape", (a) => reshape(a, [2, 6]), false],
  ["gatherRows", (a) => gatherRows(a, new Int32Array([2, 0, 2, 1])), false],
]

describe("gradCheck on every op (3x4 inputs)", () => {
  for (const [name, f, positive] of unary) {
    it(name, () => {
      expect(gradCheck(f, [rand([3, 4], positive)])).toBeLessThan(1e-3)
    })
  }
  for (const [name, f] of [["add", add], ["sub", sub], ["mul", mul]] as [string, F][]) {
    it(name, () => {
      expect(gradCheck(f, [rand([3, 4]), rand([3, 4])])).toBeLessThan(1e-3)
    })
  }
  it("div (positive denominator)", () => {
    expect(gradCheck(div, [rand([3, 4]), rand([3, 4], true)])).toBeLessThan(1e-3)
  })
  it("broadcasting add [3,4]+[4]", () => {
    expect(gradCheck(add, [rand([3, 4]), rand([4])])).toBeLessThan(1e-3)
  })
  it("broadcasting mul and div [3,4]x[3,1]", () => {
    expect(gradCheck(mul, [rand([3, 4]), rand([3, 1])])).toBeLessThan(1e-3)
    expect(gradCheck(div, [rand([3, 1]), rand([4], true)])).toBeLessThan(1e-3)
  })
  it("matmul [3,4]·[4,2]", () => {
    expect(gradCheck(matmul, [rand([3, 4]), rand([4, 2])])).toBeLessThan(1e-3)
  })
  it("logSoftmax followed by gatherRows", () => {
    const f = (a: Tensor) => gatherRows(reshape(logSoftmax(a), [12, 1]), new Int32Array([1, 6, 11]))
    expect(gradCheck(f, [rand([3, 4])])).toBeLessThan(1e-3)
  })
  it("a small MLP composition", () => {
    const f = (x: Tensor, w: Tensor, b: Tensor) => mean(square(tanh(add(matmul(x, w), b))))
    expect(gradCheck(f, [rand([3, 4]), rand([4, 2]), rand([2])])).toBeLessThan(1e-3)
  })
  it("detects a wrong gradient", () => {
    // Forward of a+a, but gradient flows only once: relative error ~ 0.5.
    const f = (a: Tensor) => {
      const out = add(a, a)
      out._node!.backward = (g) => [g, null]
      return out
    }
    expect(gradCheck(f, [rand([3, 4])])).toBeGreaterThan(0.1)
  })
})

describe("backward", () => {
  it("accumulates a tensor used twice, across calls, and zeroGrad resets it (Review Focus 5)", async () => {
    const x = param(new Float32Array([3]), [1])
    const opt = new Adam([x], { lr: 0.1 })
    backward(sum(mul(x, x)))
    expect(Array.from(await x.grad!.read())).toEqual([6])
    backward(sum(mul(x, x)))
    expect(Array.from(await x.grad!.read())).toEqual([12])
    opt.zeroGrad()
    expect(x.grad).toBeNull()
  })
  it("requires a scalar loss", () => {
    const x = param(new Float32Array([1, 2]), [2])
    expect(() => backward(mul(x, x))).toThrow(ShapeError)
  })
  it("leaves without requiresGrad get no grad", () => {
    const x = param(new Float32Array([1, 2]), [2])
    const c = tensor([5, 7], [2])
    backward(sum(mul(x, c)))
    expect(c.grad).toBeNull()
    expect(x.grad).not.toBeNull()
  })
  it("gives two leaves that receive the same upstream gradient independent grad storage", async () => {
    const a = param(new Float32Array([1, 2]), [2])
    const b = param(new Float32Array([3, 4]), [2])
    backward(sum(add(a, b)))
    expect(a.grad!.storage).not.toBe(b.grad!.storage)
    a.grad!.dispose()
    expect(Array.from(await b.grad!.read())).toEqual([1, 1])
  })
  it("a leaf reached only through a reshape keeps a live grad", async () => {
    const x = param(new Float32Array([1, 2, 3, 4]), [2, 2])
    const loss = sum(mul(reshape(x, [4]), tensor([1, 2, 3, 4], [4])))
    backward(loss)
    disposeGraph(loss, [])
    expect(Array.from(await x.grad!.read())).toEqual([1, 2, 3, 4])
    expect(x.grad!.shape).toEqual([2, 2])
  })
})

describe("noGrad", () => {
  it("records no graph node", () => {
    const x = param(new Float32Array([1, 2]), [2])
    const y = noGrad(() => mul(x, x))
    expect(y._node).toBeUndefined()
    expect(y.requiresGrad).toBe(false)
  })
})

describe("disposeGraph", () => {
  it("returns liveBuffers to the pre-forward value after backward + zeroGrad", async () => {
    const be = getDefaultBackend()
    const w = param(new Float32Array([0.5, -1, 2, 0.25, 1, -0.5]), [3, 2])
    const b = param(new Float32Array([0.1, -0.2]), [2])
    const x = tensor([1, 2, 3, 4, 5, 6], [2, 3])
    const opt = new Adam([w, b], { lr: 0.01 })
    const before = be.stats.liveBuffers
    const h = reshape(relu(add(matmul(x, w), b)), [4])
    const loss = mean(square(sub(logSoftmax(reshape(h, [2, 2])), tensor([0, 0, 0, 0], [2, 2]))))
    backward(loss)
    opt.step()
    disposeGraph(loss, [])
    opt.zeroGrad()
    expect(be.stats.liveBuffers).toBe(before + 1) // the target constant created inside the expression
    await expect(w.read()).resolves.toBeInstanceOf(Float32Array)
    await expect(x.read()).resolves.toBeInstanceOf(Float32Array)
  })
  it("returns liveBuffers exactly to the recorded value for a pure-param forward", () => {
    const be = getDefaultBackend()
    const w = param(new Float32Array([1, 2, 3, 4]), [2, 2])
    const opt = new Adam([w], { lr: 0.01 })
    const before = be.stats.liveBuffers
    const loss = sum(sigmoid(matmul(w, reshape(w, [2, 2]))))
    backward(loss)
    disposeGraph(loss, [])
    opt.zeroGrad()
    expect(be.stats.liveBuffers).toBe(before)
  })
  it("never disposes a leaf through a reshaped view of it, and spares kept tensors", async () => {
    const x = param(new Float32Array([1, 2, 3, 4]), [4])
    const r = reshape(x, [2, 2])
    const keepMe = tanh(r)
    const kr = reshape(keepMe, [4])
    const loss = sum(kr)
    backward(loss)
    disposeGraph(loss, [keepMe])
    await expect(x.read()).resolves.toBeInstanceOf(Float32Array)
    await expect(keepMe.read()).resolves.toBeInstanceOf(Float32Array)
    await expect(loss.read()).rejects.toThrow()
  })
})
