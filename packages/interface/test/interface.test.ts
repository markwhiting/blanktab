import { describe, it, expect } from "vitest"
import {
  KEYS, MOVE_OPTIONS, NOOP_ACTION, decodeMove, isValidAction, emptyObservation,
  SCHEMA_VERSION, BUTTON_OPTIONS, SCROLL_OPTIONS,
} from "../src/index"

describe("interface contract", () => {
  it("action space sizes", () => {
    expect(KEYS).toHaveLength(45); expect(KEYS[0]).toBe(""); expect(MOVE_OPTIONS).toBe(25)
    expect(SCHEMA_VERSION).toBe(1); expect(BUTTON_OPTIONS).toBe(3); expect(SCROLL_OPTIONS).toBe(3)
    expect(KEYS.at(-1)).toBe("ArrowDown"); expect(new Set(KEYS).size).toBe(45)
  })
  it("decodeMove", () => {
    expect(decodeMove(0)).toEqual({ dx: 0, dy: 0 })
    expect(decodeMove(1)).toEqual({ dx: 1, dy: 0 })
    expect(decodeMove(3)).toEqual({ dx: 16, dy: 0 })
    expect(decodeMove(1 + 2 * 3 + 1)).toEqual({ dx: 0, dy: -4 })
    expect(decodeMove(1 + 1 * 3 + 0)).toEqual({ dx: 1, dy: -1 })
  })
  it("decodeMove covers 24 distinct nonzero moves", () => {
    const seen = new Set<string>()
    for (let m = 1; m < MOVE_OPTIONS; m++) { const { dx, dy } = decodeMove(m); expect(dx !== 0 || dy !== 0).toBe(true); seen.add(`${dx},${dy}`) }
    expect(seen.size).toBe(24)
  })
  it("isValidAction", () => {
    expect(isValidAction(NOOP_ACTION)).toBe(true)
    expect(isValidAction({ move: 25, button: 0, scroll: 0, key: 0 })).toBe(false)
    expect(isValidAction({ move: 1.5, button: 0, scroll: 0, key: 0 })).toBe(false)
    expect(isValidAction({ move: -1, button: 0, scroll: 0, key: 0 })).toBe(false)
    expect(isValidAction({ move: 0, button: 3, scroll: 0, key: 0 })).toBe(false)
    expect(isValidAction({ move: 0, button: 0, scroll: 3, key: 0 })).toBe(false)
    expect(isValidAction({ move: 0, button: 0, scroll: 0, key: 45 })).toBe(false)
    expect(isValidAction({ move: 0, button: 0, scroll: 0, key: 44 })).toBe(true)
  })
  it("observation carries pixels only, no clock", () => {
    const o = emptyObservation()
    expect(o.periphery.length).toBe(80 * 50 * 3); expect(o.fovea.length).toBe(32 * 32 * 3); expect("tick" in o).toBe(false)
  })
})
