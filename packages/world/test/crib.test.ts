import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { FOVEA, PERIPHERY } from "@blanktab/interface"
import { cribRoom, layoutCrib, ROOM_VIEW, type Shape } from "../src/index"
import { createGfx } from "../src/gfx"
import { hashSeed } from "../src/rng"
import { WORDS } from "../src/words"
import { PRESS, makeWorld, moveTo, untimed, pixelAt } from "./helpers"

const crib = (seed: number) => makeWorld(seed, { rooms: ["crib"] })
const shapesOf = (seed: number) => layoutCrib(hashSeed(seed, "crib")).shapes
const centre = (s: Shape) => ({ x: Math.round(s.x + s.size / 2), y: Math.round(s.y + s.size / 2) })

describe("crib layout", () => {
  it("lays out 6-10 shapes inside the room viewport", () => {
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const { shapes } = layoutCrib(seed)
      expect(shapes.length).toBeGreaterThanOrEqual(6)
      expect(shapes.length).toBeLessThanOrEqual(10)
      for (const s of shapes) {
        expect(s.x).toBeGreaterThanOrEqual(0)
        expect(s.x + s.size).toBeLessThanOrEqual(ROOM_VIEW.width)
        expect(s.y + s.size).toBeLessThanOrEqual(ROOM_VIEW.height)
      }
    }
  })
})

describe("crib in the desktop", () => {
  it("is deterministic per seed and differs across seeds", () => {
    const a = crib(1).observe(), b = crib(1).observe(), c = crib(2).observe()
    expect(a.periphery.length).toBe(PERIPHERY.width * PERIPHERY.height * PERIPHERY.channels)
    expect(a.fovea.length).toBe(FOVEA.width * FOVEA.height * FOVEA.channels)
    expect(Array.from(a.periphery)).toEqual(Array.from(b.periphery))
    expect(Array.from(a.fovea)).toEqual(Array.from(b.fovea))
    expect(Array.from(a.periphery)).not.toEqual(Array.from(c.periphery))
  })

  it("pressing on a shape emits click and respond and changes its pixels", () => {
    const w = crib(1)
    const s = shapesOf(1)[0]!
    const c = centre(s)
    // sample upper-left of the shape (the cursor arrow extends down-right from its tip)
    const sx = Math.round(c.x - s.size * 0.25), sy = Math.round(c.y - s.size * 0.25)
    moveTo(w, c.x, c.y)
    w.drainTrace()
    const before = pixelAt(w.observe(), sx, sy)
    w.act(PRESS)
    expect(w.observe().buttonHeld).toBe(true)
    const trace = w.drainTrace()
    expect(trace).toContainEqual({ tick: w.tick, room: "crib", kind: "click", elementId: s.id, interactive: true, responsive: true })
    expect(trace).toContainEqual({ tick: w.tick, room: "crib", kind: "respond", elementId: s.id })
    expect(pixelAt(w.observe(), sx, sy)).not.toEqual(before)
  })

  it("emits hover when the cursor enters a shape, reports it in state(), and drainTrace empties", () => {
    const w = crib(1)
    const s = shapesOf(1)[0]!
    moveTo(w, centre(s).x, centre(s).y)
    expect(w.drainTrace()).toContainEqual(expect.objectContaining({ kind: "hover", elementId: s.id, room: "crib" }))
    expect(w.state().hovered).toEqual({ id: s.id, interactive: true, responsive: true })
    expect(w.drainTrace()).toEqual([])
  })

  it("pressing on empty background emits only the shell click, with no element", () => {
    const w = crib(1)
    moveTo(w, 1, 1)
    w.drainTrace()
    w.act(PRESS)
    expect(w.drainTrace()).toEqual([{ tick: w.tick, room: "crib", kind: "click", interactive: false, responsive: false }])
  })

  it("hover fires on entering a shape, on leaving it for empty space, and on coming back, with flags", () => {
    const w = crib(1)
    const s = shapesOf(1)[0]!
    const c = centre(s)
    const out = { x: s.x - 2, y: c.y } // shapes keep a 4 px gap, so this is background
    const hovers = () => w.drainTrace().filter(e => e.kind === "hover").map(untimed)
    moveTo(w, c.x, c.y)
    expect(hovers().at(-1)).toEqual({ room: "crib", kind: "hover", elementId: s.id, interactive: true, responsive: true })
    moveTo(w, out.x, out.y) // a straight horizontal path: inside s, then out
    expect(hovers()).toEqual([{ room: "crib", kind: "hover", interactive: false, responsive: false }])
    expect(w.state().hovered).toBeNull()
    moveTo(w, c.x, c.y)
    const back = hovers()
    expect(back).toHaveLength(1)
    expect(back[0]).toMatchObject({ room: "crib", kind: "hover", elementId: s.id, interactive: true, responsive: true })
    moveTo(w, c.x + 1, c.y) // still on the same shape: nothing new
    expect(hovers()).toEqual([])
  })

  it("fovea is padded black outside the screen", () => {
    const w = crib(1)
    moveTo(w, 0, 0)
    expect(Array.from(w.observe().fovea.slice(0, 3))).toEqual([0, 0, 0])
  })
})

describe("cribRoom", () => {
  const room = () => cribRoom(hashSeed(1, "crib"), WORDS)
  const draw = (r: ReturnType<typeof room>) => {
    const c = createCanvas(ROOM_VIEW.width, ROOM_VIEW.height)
    r.draw(createGfx(c.getContext("2d")))
    return Array.from(c.getContext("2d").getImageData(0, 0, ROOM_VIEW.width, ROOM_VIEW.height).data)
  }

  it("reports every shape as interactive and responsive, background as nothing", () => {
    const r = room()
    for (const s of shapesOf(1)) expect(r.inspect(centre(s).x, centre(s).y)).toEqual({ id: s.id, interactive: true, responsive: true })
    expect(r.inspect(0, 0)).toBeNull()
    expect(r.inspect(-5, 500)).toBeNull()
    expect(r.kind).toBe("crib")
    expect(r.scrollState()).toBeNull()
    expect(r.page()).toBeNull()
  })

  it("areas are the shapes' share of the viewport, all responsive", () => {
    const r = room()
    const a = r.areas()
    let px = 0
    for (let y = 0; y < ROOM_VIEW.height; y++) for (let x = 0; x < ROOM_VIEW.width; x++) if (r.inspect(x, y)) px++
    expect(a.interactive).toBeCloseTo(px / (ROOM_VIEW.width * ROOM_VIEW.height), 10)
    expect(a.responsive).toBe(a.interactive)
    expect(a.unresponsive).toBeCloseTo(1 - a.responsive, 12) // the background is the unresponsive side
    expect(a.unresponsive).toBeGreaterThan(0.5)
  })

  it("highlights the hovered shape, and drops the highlight when the cursor leaves the viewport", () => {
    const r = room()
    const plain = draw(r)
    const c = centre(shapesOf(1)[0]!)
    r.move(c.x, c.y, false)
    expect(draw(r)).not.toEqual(plain)
    r.move(c.x, ROOM_VIEW.height + 2, false) // the shell's exit move: outside the viewport
    expect(draw(r)).toEqual(plain)
  })

  it("a press on a shape returns respond only (the shell emits click); elsewhere nothing", () => {
    const r = room()
    const s = shapesOf(1)[0]!
    expect(r.press(centre(s).x, centre(s).y)).toEqual([{ kind: "respond", elementId: s.id }])
    expect(r.press(0, 0)).toEqual([])
    expect(r.release(0, 0)).toEqual([])
    expect(r.step(1)).toEqual([])
  })
})
