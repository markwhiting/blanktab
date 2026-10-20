import { describe, expect, it } from "vitest"
import { Adam, add, backward, createCpuBackend, createRng, disposeGraph, getDefaultBackend, matmul, mean, mul, param, square, sub, tensor } from "../src/index"
import type { Tensor } from "../src/index"

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

  it("state() / setState() round-trips moments and per-parameter step counts exactly", async () => {
    const be = createCpuBackend()
    const mk = (d: number[]) => tensor(Float32Array.from(d), [d.length], { requiresGrad: true, backend: be })
    const train = (opt: Adam, ps: Tensor[], useB: boolean) => {
      const x = tensor(Float32Array.from([0.3, -1.7, 2.1]), [3], { backend: be })
      const loss = add(mean(square(mul(ps[0]!, x))), useB ? mean(square(sub(ps[1]!, x))) : mean(square(ps[0]!)))
      backward(loss)
      opt.step()
      opt.zeroGrad()
      disposeGraph(loss, [])
      x.dispose()
    }
    const a = [mk([1, 2, 3]), mk([-1, 0.5, 4])]
    const optA = new Adam(a, { lr: 0.05 })
    for (let i = 0; i < 7; i++) train(optA, a, i % 3 !== 0) // b gets a grad on 4 of 7 steps: step counts differ
    const st = await optA.state()
    expect(st.slots.map((s) => s.t)).toEqual([7, 4])
    expect(st.slots[0]!.m).toBeInstanceOf(Float32Array)
    expect(structuredClone(st)).toEqual(st)

    const b = await Promise.all(a.map(async (p) => mk(Array.from(await p.read()))))
    const optB = new Adam(b, { lr: 0.05 })
    const live = be.stats.liveBuffers
    optB.setState(structuredClone(st))
    expect(be.stats.liveBuffers).toBe(live) // old moments freed
    for (let i = 0; i < 5; i++) {
      train(optA, a, i % 2 === 0)
      train(optB, b, i % 2 === 0)
    }
    for (let k = 0; k < 2; k++) expect(await b[k]!.read()).toEqual(await a[k]!.read())
    expect(await optB.state()).toEqual(await optA.state())
    optA.dispose()
    optB.dispose()
  })

  it("state() is a copy: later steps and caller mutation do not alias it", async () => {
    const be = createCpuBackend()
    const p = tensor(Float32Array.from([1, 2]), [2], { requiresGrad: true, backend: be })
    const opt = new Adam([p], { lr: 0.1 })
    backward(mean(square(p)))
    opt.step()
    opt.zeroGrad()
    const st = await opt.state()
    const m0 = st.slots[0]!.m.slice()
    opt.setState(st)
    st.slots[0]!.m.fill(99) // mutating the saved state must not reach the optimizer
    expect((await opt.state()).slots[0]!.m).toEqual(m0)
    opt.dispose()
    p.dispose()
  })

  it("setState rejects a state that does not match the parameters", async () => {
    const be = createCpuBackend()
    const p = tensor(Float32Array.from([1, 2]), [2], { requiresGrad: true, backend: be })
    const opt = new Adam([p], { lr: 0.1 })
    const ok = await opt.state()
    const live = be.stats.liveBuffers
    const z = (n: number) => new Float32Array(n)
    for (const bad of [
      { slots: [] },
      { slots: [ok.slots[0]!, ok.slots[0]!] },
      { slots: [{ m: z(3), v: z(2), t: 0 }] },
      { slots: [{ m: z(2), v: z(1), t: 0 }] },
      { slots: [{ m: z(2), v: z(2), t: -1 }] },
      { slots: [{ m: z(2), v: z(2), t: 1.5 }] },
    ]) expect(() => opt.setState(bad)).toThrow(RangeError)
    expect(be.stats.liveBuffers).toBe(live)
    opt.dispose()
    p.dispose()
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
