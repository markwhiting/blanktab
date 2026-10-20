import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { WORLD } from "@blanktab/interface"
import { layoutToys, toysRoom, ROOM_VIEW, type ToyWidget } from "../src/index"
import { createGfx } from "../src/gfx"
import { hashSeed } from "../src/rng"
import { WORDS } from "../src/words"
import { PRESS, RELEASE, makeWorld, moveTo, rgbAt } from "./helpers"

const roomSeed = (seed: number) => hashSeed(seed, "toys")
const layout = (seed: number) => layoutToys(roomSeed(seed))
const first = (seed: number, kind: ToyWidget["kind"]) => layout(seed).widgets.find(w => w.kind === kind)!
const toys = (seed = 1) => makeWorld(seed, { rooms: ["toys"] })
const mid = (w: ToyWidget) => ({ x: Math.round(w.x + w.w / 2), y: Math.round(w.y + w.h / 2) })
const knobMid = (w: ToyWidget) => ({ x: w.x + Math.floor(w.knobW! / 2), y: w.y + Math.floor(w.h / 2) })
const AWAY = { x: 310, y: 178 }

/** Sum of channel values over a rectangle of the full-resolution screen. */
function sum(s: Uint8ClampedArray, x: number, y: number, w: number, h: number) {
  let t = 0
  for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) t += rgbAt(s, i, j).reduce((a, b) => a + b, 0)
  return t
}
const region = (w: ToyWidget) => [w.x, w.y, w.w, w.h] as const

const freshRoom = (seed = 1) => toysRoom(roomSeed(seed), WORDS)

describe("toys layout", () => {
  it("has 4-7 non-overlapping widgets (one of each kind at least) and 2-4 labels inside the viewport", () => {
    for (let seed = 1; seed <= 30; seed++) {
      const { widgets, labels } = layout(seed)
      expect(widgets.length).toBeGreaterThanOrEqual(4)
      expect(widgets.length).toBeLessThanOrEqual(7)
      expect(labels.length).toBeGreaterThanOrEqual(2)
      expect(labels.length).toBeLessThanOrEqual(4)
      for (const k of ["button", "toggle", "slider", "block"] as const) expect(widgets.some(w => w.kind === k)).toBe(true)
      const all = [...widgets, ...labels]
      all.forEach((a, i) => {
        expect(a.x).toBeGreaterThanOrEqual(0)
        expect(a.y).toBeGreaterThanOrEqual(0)
        expect(a.x + a.w).toBeLessThanOrEqual(ROOM_VIEW.width)
        expect(a.y + a.h).toBeLessThanOrEqual(ROOM_VIEW.height)
        all.slice(i + 1).forEach(b => {
          expect(a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h).toBe(false)
        })
      })
      labels.forEach(l => expect(WORDS).toContain(l.text.split(" ")[0]))
    }
  })
  it("is deterministic per seed and differs across seeds", () => {
    expect(layout(1)).toEqual(layout(1))
    expect(layout(1)).not.toEqual(layout(2))
    const a = toys(1).observe(), b = toys(1).observe(), c = toys(2).observe()
    expect(Array.from(a.periphery)).toEqual(Array.from(b.periphery))
    expect(Array.from(a.periphery)).not.toEqual(Array.from(c.periphery))
  })
})

describe("toys inspect and areas", () => {
  it("reports widgets as interactive+responsive and labels as neither", () => {
    const r = freshRoom()
    const { widgets, labels } = layout(1)
    for (const w of widgets) {
      const p = w.kind === "slider" ? knobMid(w) : mid(w)
      expect(r.inspect(p.x, p.y)).toEqual({ id: w.id, interactive: true, responsive: true })
    }
    for (const l of labels) expect(r.inspect(l.x + 1, l.y + 1)).toEqual({ id: l.id, interactive: false, responsive: false })
    expect(r.inspect(-1, -1)).toBeNull()
    expect(r.inspect(5, ROOM_VIEW.height + 3)).toBeNull()
  })
  it("area fractions are inside (0,1) and responsive + unresponsive <= 1", () => {
    for (let seed = 1; seed <= 20; seed++) {
      const a = freshRoom(seed).areas()
      for (const v of Object.values(a)) { expect(v).toBeGreaterThan(0); expect(v).toBeLessThan(1) }
      expect(a.responsive + a.unresponsive).toBeLessThanOrEqual(1)
      expect(a.interactive).toBeCloseTo(a.responsive, 10)
    }
  })
})

describe("toys button", () => {
  it("looks pressed until release, counts once per press (not per held tick), emits widget press", () => {
    const b = first(1, "button")
    const w = toys()
    const c = mid(b)
    moveTo(w, c.x, c.y)
    w.drainTrace()
    moveTo(w, AWAY.x, AWAY.y)
    const idle = sum(w.screen(), ...region(b))
    moveTo(w, c.x, c.y)
    w.act(PRESS)
    expect(w.drainTrace()).toContainEqual({ tick: w.tick, room: "toys", kind: "widget", elementId: b.id, value: "press" })
    for (let i = 0; i < 5; i++) w.act({ ...PRESS, move: 0 }) // held ticks: no further events
    expect(w.drainTrace().filter(e => e.kind === "widget")).toEqual([])
    const down = sum(w.screen(), ...region(b))
    w.act(RELEASE)
    moveTo(w, AWAY.x, AWAY.y)
    const up = sum(w.screen(), ...region(b))
    expect(down).not.toBe(up) // pressed look is gone after release
    expect(up).not.toBe(idle) // but the counter now reads 1
    moveTo(w, c.x, c.y)
    w.act(PRESS); w.act(RELEASE)
    moveTo(w, AWAY.x, AWAY.y)
    const two = sum(w.screen(), ...region(b))
    expect(two).not.toBe(up) // counter 2
  })
})

describe("toys toggle (Review Focus 2)", () => {
  it("a press flips state and pixels, emits 1 then 0, and the state survives a room switch", () => {
    const t = first(1, "toggle")
    const w = makeWorld(1, { rooms: ["crib", "toys"], start: "toys" })
    const c = mid(t)
    moveTo(w, AWAY.x, AWAY.y)
    const off = Array.from(w.screen())
    moveTo(w, c.x, c.y)
    w.drainTrace()
    w.act(PRESS); w.act(RELEASE)
    expect(w.drainTrace().filter(e => e.kind === "widget")).toEqual([{ tick: w.tick - 1, room: "toys", kind: "widget", elementId: t.id, value: 1 }])
    moveTo(w, AWAY.x, AWAY.y)
    const on = Array.from(w.screen())
    expect(sum(Uint8ClampedArray.from(on), ...region(t))).not.toBe(sum(Uint8ClampedArray.from(off), ...region(t)))
    w.bringToFront("crib")
    w.bringToFront("toys")
    expect(Array.from(w.screen())).toEqual(on)
    moveTo(w, c.x, c.y)
    w.drainTrace()
    w.act(PRESS); w.act(RELEASE)
    expect(w.drainTrace().filter(e => e.kind === "widget").map(e => e.value)).toEqual([0])
    moveTo(w, AWAY.x, AWAY.y)
    expect(Array.from(w.screen())).toEqual(off)
  })
})

describe("toys slider", () => {
  it("drags end to end through the world: 0 at the left, 100 at the right, emitted on release", () => {
    const s = first(1, "slider")
    const w = toys()
    const k = knobMid(s)
    moveTo(w, k.x, k.y)
    w.act(PRESS)
    moveTo(w, s.x + s.w - 1, k.y) // far right end of the track
    w.drainTrace()
    w.act(RELEASE)
    expect(w.drainTrace().filter(e => e.kind === "widget")).toEqual([{ tick: w.tick, room: "toys", kind: "widget", elementId: s.id, value: 100 }])
    const k2 = { x: s.x + s.w - s.knobW! + Math.floor(s.knobW! / 2), y: k.y }
    moveTo(w, k2.x, k2.y)
    w.act(PRESS)
    moveTo(w, s.x, k.y)
    w.drainTrace()
    w.act(RELEASE)
    expect(w.drainTrace().filter(e => e.kind === "widget").map(e => e.value)).toEqual([0])
  })
  it("a release far past the track end clamps, and the drag follows the cursor until release", () => {
    const s = first(1, "slider")
    const r = freshRoom()
    const k = knobMid(s)
    expect(r.press(k.x, k.y)).toEqual([{ kind: "respond", elementId: s.id }])
    r.move(s.x + s.w + 60, k.y, true)
    expect(r.release(s.x + s.w + 60, k.y + 40)).toEqual([{ kind: "widget", elementId: s.id, value: 100 }])
    const k2 = { x: s.x + s.w - 2, y: k.y }
    r.press(k2.x, k2.y)
    expect(r.release(-50, k.y)).toEqual([{ kind: "widget", elementId: s.id, value: 0 }])
  })
  it("a press off the knob does not start a drag", () => {
    const s = first(1, "slider")
    const r = freshRoom()
    r.press(s.x + s.w - 2, s.y + 1)
    r.move(s.x + 10, s.y + 1, true)
    expect(r.release(s.x + 10, s.y + 1)).toEqual([])
  })
})

describe("toys block (Review Focus 3)", () => {
  it("a 40px single-step move beyond the block edge still drags it; release drops it", () => {
    const b = first(1, "block")
    const r = freshRoom()
    const c = mid(b)
    r.press(c.x, c.y)
    r.move(c.x + 40, c.y, true)
    r.move(c.x + 40, c.y, true)
    expect(r.inspect(c.x + 40, c.y)).toEqual({ id: b.id, interactive: true, responsive: true })
    expect(r.release(c.x + 40, c.y)).toEqual([{ kind: "widget", elementId: b.id, value: "drop" }])
    r.move(c.x + 80, c.y, false) // after the drop, plain moves do nothing
    expect(r.inspect(c.x + 80, c.y)?.id).not.toBe(b.id)
  })
  it("clamps the block to the viewport", () => {
    const b = first(1, "block")
    const r = freshRoom()
    const c = mid(b)
    r.press(c.x, c.y)
    r.move(-100, -100, true)
    expect(r.inspect(1, 1)?.id).toBe(b.id)
    r.move(1000, 1000, true)
    expect(r.inspect(ROOM_VIEW.width - 1, ROOM_VIEW.height - 1)?.id).toBe(b.id)
    r.release(1000, 1000)
  })
  it("a press on empty space then a move over a block does not move it", () => {
    const b = first(1, "block")
    const r = freshRoom()
    const used = [...layout(1).widgets, ...layout(1).labels]
    let ex = -1, ey = -1
    for (let y = 2; y < ROOM_VIEW.height && ex < 0; y += 3) for (let x = 2; x < ROOM_VIEW.width; x += 3) {
      if (!used.some(u => x >= u.x - 2 && x < u.x + u.w + 2 && y >= u.y - 2 && y < u.y + u.h + 2)) { ex = x; ey = y; break }
    }
    expect(ex).toBeGreaterThanOrEqual(0)
    r.press(ex, ey)
    const c = mid(b)
    r.move(c.x, c.y, true)
    r.move(c.x + 30, c.y + 5, true)
    expect(r.inspect(c.x, c.y)?.id).toBe(b.id)
    expect(r.release(c.x + 30, c.y + 5)).toEqual([])
  })
  it("a held move with no prior press in this room is never a grab", () => {
    const b = first(1, "block"), s = first(1, "slider")
    const r = freshRoom()
    const c = mid(b), k = knobMid(s)
    r.move(c.x, c.y, true)
    r.move(c.x + 30, c.y, true)
    expect(r.inspect(c.x, c.y)?.id).toBe(b.id)
    r.move(k.x, k.y, true)
    r.move(k.x + 30, k.y, true)
    expect(r.release(k.x + 30, k.y)).toEqual([])
  })
  it("a drag started in the world and released over the taskbar still drops", () => {
    const b = first(1, "block")
    const w = toys()
    const c = mid(b)
    moveTo(w, c.x, c.y)
    w.act(PRESS)
    moveTo(w, c.x, WORLD.height - 4)
    w.drainTrace()
    w.act(RELEASE)
    expect(w.drainTrace().filter(e => e.kind === "widget")).toEqual([{ tick: w.tick, room: "toys", kind: "widget", elementId: b.id, value: "drop" }])
  })
})

describe("toys rendering", () => {
  it("draws without throwing on a Node canvas", () => {
    const canvas = createCanvas(WORLD.width, WORLD.height)
    const ctx = canvas.getContext("2d")
    expect(() => freshRoom().draw(createGfx(ctx))).not.toThrow()
  })
})

describe("toys: respond, shell click, persistence", () => {
  it("room presses return respond (and widget for button/toggle); never click", () => {
    const r = freshRoom()
    const b = first(1, "button"), t = first(1, "toggle"), s = first(1, "slider"), k = first(1, "block")
    expect(r.press(mid(b).x, mid(b).y)).toEqual([{ kind: "respond", elementId: b.id }, { kind: "widget", elementId: b.id, value: "press" }])
    r.release(0, 0)
    expect(r.press(mid(t).x, mid(t).y)).toEqual([{ kind: "respond", elementId: t.id }, { kind: "widget", elementId: t.id, value: 1 }])
    r.release(0, 0)
    expect(r.press(knobMid(s).x, knobMid(s).y)).toEqual([{ kind: "respond", elementId: s.id }])
    r.release(knobMid(s).x, knobMid(s).y)
    expect(r.press(mid(k).x, mid(k).y)).toEqual([{ kind: "respond", elementId: k.id }])
    r.release(mid(k).x, mid(k).y)
  })
  it("the shell click carries the inspect id: widget, label (no respond), empty space (no id)", () => {
    const w = toys()
    const b = first(1, "button"), l = layout(1).labels[0]!
    moveTo(w, mid(b).x, mid(b).y); w.drainTrace()
    w.act(PRESS)
    let t = w.drainTrace()
    expect(t).toContainEqual({ tick: w.tick, room: "toys", kind: "click", elementId: b.id })
    expect(t.some(e => e.kind === "respond" && e.elementId === b.id)).toBe(true)
    w.act(RELEASE)
    moveTo(w, l.x + 1, l.y + 1); w.drainTrace()
    w.act(PRESS)
    t = w.drainTrace()
    expect(t).toContainEqual({ tick: w.tick, room: "toys", kind: "click", elementId: l.id })
    expect(t.some(e => e.kind === "respond")).toBe(false)
    w.act(RELEASE)
    moveTo(w, AWAY.x, AWAY.y); w.drainTrace()
    w.act(PRESS)
    expect(w.drainTrace().filter(e => e.kind === "click")).toEqual([{ tick: w.tick, room: "toys", kind: "click" }])
  })
  it("a caregiver switch emits no click", () => {
    const w = makeWorld(1, { rooms: ["crib", "toys"] })
    w.drainTrace()
    w.bringToFront("toys")
    expect(w.drainTrace().some(e => e.kind === "click")).toBe(false)
  })
  it("a dropped block position and a slider value persist across a room switch", () => {
    const b = first(1, "block"), s = first(1, "slider")
    const w = makeWorld(1, { rooms: ["crib", "toys"], start: "toys" })
    moveTo(w, mid(b).x, mid(b).y); w.act(PRESS)
    moveTo(w, mid(b).x, Math.min(150, mid(b).y + 30) ); w.act(RELEASE)
    const k = knobMid(s)
    moveTo(w, k.x, k.y); w.act(PRESS)
    moveTo(w, s.x + 50, k.y); w.act(RELEASE)
    moveTo(w, AWAY.x, AWAY.y)
    const before = Array.from(w.screen())
    w.bringToFront("crib"); w.bringToFront("toys")
    expect(Array.from(w.screen())).toEqual(before)
    expect(Array.from(w.screen())).not.toEqual(Array.from(makeWorld(1, { rooms: ["toys"] }).screen()))
  })
  it("layoutToys gives up with an error rather than looping forever", () => {
    expect(() => layoutToys(1, [])).toThrow()
  })
})
