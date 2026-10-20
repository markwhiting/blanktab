import { describe, it, expect } from "vitest"
import type { CurvePoint } from "@blanktab/harness"
import { ratioText, sparkline, appendPoints, MAX_SHOWN_POINTS } from "../src/curve"

const pt = (cu: number, predLoss: number, baselineLoss: number, steps = Math.round(cu * 100)): CurvePoint => ({ cu, steps, predLoss, baselineLoss })

describe("sparkline", () => {
  it("is empty without points", () => {
    expect(sparkline([], 100, 40)).toEqual({ pred: [], base: [] })
  })
  it("maps compute lived to x (left to right) and both losses to one shared y scale (higher loss = higher up)", () => {
    const s = sparkline([pt(0, 0.4, 0.2), pt(1, 0.3, 0.2), pt(2, 0.1, 0.2)], 100, 40, 2)
    expect(s.pred.map(p => p[0])).toEqual([2, 50, 98])
    expect(s.base.map(p => p[0])).toEqual([2, 50, 98])
    // Shared scale: min 0.1 -> bottom (y = 38), max 0.4 -> top (y = 2).
    expect(s.pred[0]![1]).toBeCloseTo(2)
    expect(s.pred[2]![1]).toBeCloseTo(38)
    expect(s.base[0]![1]).toBeCloseTo(38 - (0.1 / 0.3) * 36)
    expect(s.pred[1]![1]).toBeCloseTo(38 - (0.2 / 0.3) * 36)
  })
  it("x starts at birth (0 CU), not at the first point", () => {
    const s = sparkline([pt(5, 0.4, 0.2), pt(10, 0.3, 0.2)], 100, 40, 2)
    expect(s.pred.map(p => p[0])).toEqual([50, 98])
  })
  it("one point sits at its CU, flat losses mid-height, all without dividing by zero", () => {
    const one = sparkline([pt(3, 0.2, 0.2)], 100, 40, 2)
    expect(one.pred).toEqual([[98, 20]])
    expect(one.base).toEqual([[98, 20]])
    const flat = sparkline([pt(1, 0.2, 0.2), pt(1, 0.2, 0.2)], 100, 40, 2)
    for (const [x, y] of [...flat.pred, ...flat.base]) { expect(x).toBe(98); expect(y).toBe(20) }
    expect(sparkline([pt(0, 0.2, 0.3)], 100, 40, 2).pred[0]![0]).toBe(50)
  })
})

describe("ratioText", () => {
  it("says nothing until there is a point", () => {
    expect(ratioText([])).toBe("no guesses measured yet")
  })
  it("gives the last point's predLoss / baselineLoss to 3 significant digits, in plain words", () => {
    expect(ratioText([pt(1, 0.5, 0.5), pt(2, 0.44, 0.5)])).toBe("0.880 (its guesses are 12% less wrong than the lazy guess)")
    expect(ratioText([pt(2, 0.6, 0.5)])).toBe("1.20 (its guesses are 20% more wrong than the lazy guess)")
    expect(ratioText([pt(2, 0.5, 0.5)])).toBe("1.00 (its guesses are about as wrong as the lazy guess)")
    expect(ratioText([pt(2, 0.4985, 0.5)])).toBe("0.997 (its guesses are about as wrong as the lazy guess)")
    expect(ratioText([pt(2, 0.495, 0.5)])).toBe("0.990 (its guesses are 1% less wrong than the lazy guess)")
  })
  it("survives a zero baseline", () => {
    expect(ratioText([pt(2, 0.1, 0)])).toBe("no guesses measured yet")
  })
})

describe("appendPoints", () => {
  it("adds newer points, ignores ones not ahead of the last, and keeps at most MAX_SHOWN_POINTS (thinned, newest kept)", () => {
    let c = appendPoints([], [pt(1, 1, 1, 10), pt(2, 1, 1, 20)])
    c = appendPoints(c, [pt(2, 1, 1, 20), pt(1.5, 1, 1, 15), pt(3, 1, 1, 30)])
    expect(c.map(p => p.steps)).toEqual([10, 20, 30])
    const many = appendPoints([], Array.from({ length: MAX_SHOWN_POINTS * 3 }, (_, i) => pt(i, 1, 1, i)))
    expect(many.length).toBeLessThanOrEqual(MAX_SHOWN_POINTS)
    expect(many.at(-1)!.steps).toBe(MAX_SHOWN_POINTS * 3 - 1)
    expect(many[0]!.steps).toBeLessThan(4) // the oldest survivor is still from the start of the life
  })
})
