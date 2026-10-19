// Canvas stand-in: Node tests use `@napi-rs/canvas` (prebuilt binaries, devDependency of
// packages/world) in place of OffscreenCanvas. Vitest browser mode is not used.
import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { NOOP_ACTION, WORLD, PERIPHERY, FOVEA, type Action } from "@blanktab/interface"
import { createWorld, layoutCrib, type World } from "../src/index"

const mk = (seed: number): World => createWorld({ seed, canvas: createCanvas(WORLD.width, WORLD.height) })
const act = (move: number, button = 0): Action => ({ ...NOOP_ACTION, move, button })
// move codes: 1 + dirIndex*3 + magIndex; dirs: E,NE,N,NW,W,SW,S,SE
const code = (dir: number, mag: number) => 1 + dir * 3 + mag
const E16 = code(0, 2) // east, 16px

/** Drive the cursor to (tx, ty) using 1px/4px/16px moves. */
function goTo(w: World, tx: number, ty: number) {
  for (let guard = 0; guard < 400; guard++) {
    const { cursor } = w.observe()
    const dx = tx - Math.round(cursor.x * WORLD.width)
    const dy = ty - Math.round(cursor.y * WORLD.height)
    if (dx === 0 && dy === 0) return
    const mag = Math.max(Math.abs(dx), Math.abs(dy)) >= 16 ? 2 : Math.max(Math.abs(dx), Math.abs(dy)) >= 4 ? 1 : 0
    // axis-aligned moves only, so overshoot is impossible
    if (dx !== 0 && (Math.abs(dx) >= Math.abs(dy) || dy === 0)) {
      const m = Math.min(mag, Math.abs(dx) >= 16 ? 2 : Math.abs(dx) >= 4 ? 1 : 0)
      w.act(act(code(dx > 0 ? 0 : 4, m)))
    } else {
      const m = Math.min(mag, Math.abs(dy) >= 16 ? 2 : Math.abs(dy) >= 4 ? 1 : 0)
      w.act(act(code(dy > 0 ? 6 : 2, m)))
    }
  }
  throw new Error("goTo did not converge")
}

describe("world", () => {
  it("is deterministic per seed and differs across seeds", () => {
    const a = mk(1).observe(), b = mk(1).observe(), c = mk(2).observe()
    expect(a.periphery.length).toBe(PERIPHERY.width * PERIPHERY.height * PERIPHERY.channels)
    expect(a.fovea.length).toBe(FOVEA.width * FOVEA.height * FOVEA.channels)
    expect(Array.from(a.periphery)).toEqual(Array.from(b.periphery))
    expect(Array.from(a.fovea)).toEqual(Array.from(b.fovea))
    expect(Array.from(a.periphery)).not.toEqual(Array.from(c.periphery))
  })

  it("starts the cursor at the centre and moves it by decoded amounts", () => {
    const w = mk(1)
    expect(w.observe().cursor).toEqual({ x: 0.5, y: 0.5 })
    const before = w.observe().cursor.x
    w.act(act(E16))
    expect(w.observe().cursor.x).toBeCloseTo(before + 16 / 320, 10)
    expect(w.tick).toBe(1)
  })

  it("clamps the cursor to the screen", () => {
    const w = mk(1)
    for (let i = 0; i < 40; i++) w.act(act(code(4, 2))) // west
    for (let i = 0; i < 40; i++) w.act(act(code(2, 2))) // north
    expect(w.observe().cursor).toEqual({ x: 0, y: 0 })
    for (let i = 0; i < 60; i++) w.act(act(code(7, 2))) // south-east
    const c = w.observe().cursor
    expect(c.x).toBeLessThanOrEqual(1)
    expect(c.y).toBeLessThanOrEqual(1)
    expect(c.x).toBeGreaterThan(0.9)
  })

  it("lays out 6-10 shapes clear of the taskbar", () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const { shapes } = layoutCrib(seed)
      expect(shapes.length).toBeGreaterThanOrEqual(6)
      expect(shapes.length).toBeLessThanOrEqual(10)
      for (const s of shapes) expect(s.y + s.size).toBeLessThanOrEqual(WORLD.height - 16)
    }
  })

  it("pressing on a shape emits a respond event and changes its pixels", () => {
    const w = mk(1)
    const s = layoutCrib(1).shapes[0]!
    const cx = Math.round(s.x + s.size / 2), cy = Math.round(s.y + s.size / 2)
    // sample upper-left of the shape (the cursor arrow extends down-right from its tip)
    const px = Math.floor((cx - s.size * 0.25) / 4), py = Math.floor((cy - s.size * 0.25) / 4)
    const at = (p: Uint8Array) => Array.from(p.slice((py * PERIPHERY.width + px) * 3, (py * PERIPHERY.width + px) * 3 + 3))
    goTo(w, cx, cy)
    w.drainTrace()
    const before = at(w.observe().periphery)
    w.act(act(0, 1))
    expect(w.observe().buttonHeld).toBe(true)
    const trace = w.drainTrace()
    expect(trace.some(e => e.kind === "respond" && e.elementId === s.id)).toBe(true)
    expect(trace.some(e => e.kind === "click" && e.elementId === s.id)).toBe(true)
    expect(at(w.observe().periphery)).not.toEqual(before)
  })

  it("emits hover when the cursor enters a shape, and drainTrace empties", () => {
    const w = mk(1)
    const s = layoutCrib(1).shapes[0]!
    goTo(w, Math.round(s.x + s.size / 2), Math.round(s.y + s.size / 2))
    const first = w.drainTrace()
    expect(first.some(e => e.kind === "hover" && e.elementId === s.id)).toBe(true)
    expect(w.drainTrace()).toEqual([])
  })

  it("pressing on empty background emits nothing", () => {
    const w = mk(1)
    goTo(w, 2, 2)
    w.drainTrace()
    w.act(act(0, 1))
    expect(w.drainTrace()).toEqual([])
  })

  it("fovea is padded black outside the screen", () => {
    const w = mk(1)
    goTo(w, 0, 0)
    const f = w.observe().fovea
    expect(Array.from(f.slice(0, 3))).toEqual([0, 0, 0]) // top-left of crop is off-screen
  })
})
