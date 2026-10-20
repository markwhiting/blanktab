import { describe, expect, it } from "vitest"
import { Adam, add, backward, createRng, disposeGraph, getDefaultBackend, matmul, mean, param, square, sub, tensor } from "../src/index"

describe("Adam", () => {
  it("fits y = 3x - 2 on 64 seeded points within 500 steps at lr 0.05 (MSE < 1e-3)", async () => {
    const rng = createRng(42)
    const xs = new Float32Array(64)
    const ys = new Float32Array(64)
    for (let i = 0; i < 64; i++) {
      xs[i] = rng() * 2 - 1
      ys[i] = 3 * xs[i]! - 2
    }
    const X = tensor(xs, [64, 1])
    const Y = tensor(ys, [64, 1])
    const w = param(new Float32Array([0]), [1, 1])
    const b = param(new Float32Array([0]), [1])
    const opt = new Adam([w, b], { lr: 0.05 })
    const be = getDefaultBackend()
    const live = be.stats.liveBuffers
    let mse = Infinity
    for (let step = 0; step < 500 && mse >= 1e-3; step++) {
      opt.zeroGrad()
      const loss = mean(square(sub(add(matmul(X, w), b), Y)))
      backward(loss)
      mse = (await loss.read())[0]!
      opt.step()
      disposeGraph(loss, [])
    }
    opt.zeroGrad()
    expect(mse).toBeLessThan(1e-3)
    expect(be.stats.liveBuffers).toBe(live)
    expect((await w.read())[0]).toBeCloseTo(3, 1)
    expect((await b.read())[0]).toBeCloseTo(-2, 1)
  })

  it("skips params whose grad is null and leaves them unchanged", async () => {
    const a = param(new Float32Array([1]), [1])
    const b = param(new Float32Array([1]), [1])
    const opt = new Adam([a, b], { lr: 0.1 })
    backward(mean(square(a)))
    opt.step()
    expect((await a.read())[0]).toBeCloseTo(0.9, 5) // first Adam step moves by lr
    expect(Array.from(await b.read())).toEqual([1])
  })

  it("setLr changes the step size", async () => {
    const a = param(new Float32Array([1]), [1])
    const opt = new Adam([a], { lr: 0.1 })
    opt.setLr(0.5)
    backward(mean(square(a)))
    opt.step()
    expect((await a.read())[0]).toBeCloseTo(0.5, 5)
  })

  it("dispose releases optimizer state", () => {
    const be = getDefaultBackend()
    const a = param(new Float32Array([1, 2]), [2])
    const before = be.stats.liveBuffers
    const opt = new Adam([a], { lr: 0.1 })
    expect(be.stats.liveBuffers).toBe(before + 2)
    opt.dispose()
    expect(be.stats.liveBuffers).toBe(before)
  })
})
