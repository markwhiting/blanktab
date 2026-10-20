import { describe, it, expect } from "vitest"
import { NOOP_ACTION, emptyObservation } from "@blanktab/interface"
import { ReplayBuffer } from "../src/replay"
import type { Transition } from "../src/replay"

const tr = (i: number): Transition => {
  const obs = emptyObservation()
  obs.periphery[0] = i
  return { obs, hIn: Float32Array.of(i), action: { ...NOOP_ACTION, move: i % 25 }, next: emptyObservation() }
}

describe("ReplayBuffer", () => {
  it("evicts the oldest entry first and never exceeds its capacity", () => {
    const b = new ReplayBuffer(3)
    expect(b.size).toBe(0)
    for (let i = 0; i < 10; i++) {
      b.push(tr(i))
      expect(b.size).toBe(Math.min(i + 1, 3))
      expect(b.size).toBeLessThanOrEqual(b.capacity)
      // Oldest first: at(0) is the earliest surviving transition, at(size-1) the newest.
      const ids = Array.from({ length: b.size }, (_, k) => b.at(k).hIn[0])
      expect(ids).toEqual(Array.from({ length: b.size }, (_, k) => Math.max(0, i - 2) + k))
    }
  })

  it("rejects an invalid capacity and out-of-range reads", () => {
    for (const c of [0, -1, 1.5, Number.NaN]) expect(() => new ReplayBuffer(c)).toThrow(RangeError)
    const b = new ReplayBuffer(2)
    b.push(tr(0))
    expect(() => b.at(1)).toThrow(RangeError)
    expect(() => b.at(-1)).toThrow(RangeError)
  })
})
