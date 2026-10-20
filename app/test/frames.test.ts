import { describe, it, expect } from "vitest"
import { frameGate } from "../src/frames"

describe("frames to the page", () => {
  it("at most one every minMs", () => {
    const g = frameGate(33)
    expect([0, 10, 32, 33, 40, 70].map(t => g.due(t))).toEqual([true, false, false, true, false, true])
  })
  it("none while the page is hidden; they resume when it is shown again", () => {
    const g = frameGate(33)
    expect(g.due(0)).toBe(true)
    g.setHidden(true)
    expect([100, 200, 300].map(t => g.due(t))).toEqual([false, false, false])
    g.setHidden(false)
    expect(g.due(301)).toBe(true)
  })
  it("a page hidden at the start posts nothing until shown", () => {
    const g = frameGate(33, true)
    expect(g.due(0)).toBe(false)
    g.setHidden(false)
    expect(g.due(1)).toBe(true)
  })
})
