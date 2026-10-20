import { describe, it, expect } from "vitest"
import { FLUSH_EVERY_MS, LIFETIME_STEPS, STEPS_PER_SECOND, flushPolicy } from "../src/schedule"

describe("pacing", () => {
  it("the placeholder learner lives at 2 steps/s: a 3000-step life takes 25 minutes", () => {
    expect(STEPS_PER_SECOND).toBe(2)
    expect(LIFETIME_STEPS).toBe(3000)
    expect(LIFETIME_STEPS / STEPS_PER_SECOND).toBe(25 * 60)
  })
})

describe("flush cadence", () => {
  const setup = () => {
    let flushes = 0
    const timers: { fn: () => void; ms: number }[] = []
    const p = flushPolicy(() => { flushes++ }, (fn, ms) => { timers.push({ fn, ms }) })
    return { p, timers, flushes: () => flushes }
  }
  it("does not flush after a non-final log is queued", () => {
    const t = setup()
    t.p.queued(false)
    t.p.queued(false)
    expect(t.flushes()).toBe(0)
  })
  it("flushes when the final log is queued", () => {
    const t = setup()
    t.p.queued(true)
    expect(t.flushes()).toBe(1)
  })
  it("otherwise flushes every 10 minutes", () => {
    const t = setup()
    expect(FLUSH_EVERY_MS).toBe(600_000)
    expect(t.timers.map(x => x.ms)).toEqual([600_000])
    t.timers[0]!.fn()
    expect(t.flushes()).toBe(1)
  })
  it("flushes when a token is saved", () => {
    const t = setup()
    t.p.tokenSaved()
    expect(t.flushes()).toBe(1)
  })
})

it("a non-final queued log updates the waiting count instead of sending", () => {
  let flushes = 0, reports = 0
  const p = flushPolicy(() => { flushes++ }, () => {}, () => { reports++ })
  p.queued(false)
  expect([flushes, reports]).toEqual([0, 1])
  p.queued(true)
  expect([flushes, reports]).toEqual([1, 1])
})
