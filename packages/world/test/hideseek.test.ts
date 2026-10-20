import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { WORLD } from "@blanktab/interface"
import {
  flowHideseek, hideseekRoom, layoutHideseek, ROOM_VIEW, HS_HEADER_H, HS_TAB_H,
  type Hideseek, type Room,
} from "../src/index"
import { createGfx } from "../src/gfx"
import { hashSeed } from "../src/rng"
import { WORDS } from "../src/words"
import { NOOP, PRESS, RELEASE, act, makeWorld, moveTo, rgbAt } from "./helpers"

const roomSeed = (seed: number) => hashSeed(seed, "hideseek")
const layout = (seed = 1) => layoutHideseek(roomSeed(seed), WORDS)
const freshRoom = (seed = 1) => hideseekRoom(roomSeed(seed), WORDS)
const AWAY = { x: WORLD.width - 1, y: WORLD.height - 1 } // taskbar corner, not an icon

/** Draws a room alone onto a canvas and returns the RGBA viewport. */
function render(r: Room): Uint8ClampedArray {
  const canvas = createCanvas(ROOM_VIEW.width, ROOM_VIEW.height)
  const ctx = canvas.getContext("2d")
  r.draw(createGfx(ctx))
  return ctx.getImageData(0, 0, ROOM_VIEW.width, ROOM_VIEW.height).data
}
const px = (d: Uint8ClampedArray, x: number, y: number) => {
  const i = (y * ROOM_VIEW.width + x) * 4
  return [d[i]!, d[i + 1]!, d[i + 2]!]
}
function crop(d: Uint8ClampedArray, x: number, y: number, w: number, h: number) {
  const out: number[] = []
  for (let j = y; j < y + h; j++) for (let i = x; i < x + w; i++) out.push(...px(d, i, j))
  return out
}
const frac = (r: Room) => r.scrollState()!.revealedFraction
const offset = (r: Room) => r.scrollState()!.offset

/** Items displayed with all sections expanded and the given tab active (what pure scrolling can reach). */
function reachable(h: Hideseek, tab = 0) {
  return new Set(flowHideseek(h, new Set(), tab).items.keys())
}

/** Scroll the room (no ticks, so no pop-up) until the content row `y` is inside the viewport. */
function scrollToShow(r: Room, y: number, hgt: number) {
  for (let guard = 0; guard < 200; guard++) {
    const o = offset(r)
    if (y >= o && y + hgt <= o + ROOM_VIEW.height) return o
    r.scroll(y < o ? "up" : "down")
  }
  throw new Error("could not scroll to the row")
}

describe("hideseek layout", () => {
  it("has 12-40 items, 2-4 sections, 2-3 tabs, content 500-1500 px tall, a pop-up of 30-50% at tick 20-200", () => {
    for (let seed = 1; seed <= 40; seed++) {
      const h = layout(seed)
      expect(h.items.length).toBeGreaterThanOrEqual(12)
      expect(h.items.length).toBeLessThanOrEqual(40)
      expect(new Set(h.items.map(i => i.kind))).toEqual(new Set(["line", "block", "card"]))
      expect(new Set(h.items.map(i => i.id)).size).toBe(h.items.length)
      expect(h.sections.length).toBeGreaterThanOrEqual(2)
      expect(h.sections.length).toBeLessThanOrEqual(4)
      for (const s of h.sections) expect(s.items.length).toBeGreaterThan(0)
      expect(h.tabs.length).toBeGreaterThanOrEqual(2)
      expect(h.tabs.length).toBeLessThanOrEqual(3)
      for (const t of h.tabs) expect(t.items.length).toBeGreaterThan(0)
      const f = flowHideseek(h, new Set(), 0)
      expect(f.height).toBeGreaterThanOrEqual(500)
      expect(f.height).toBeLessThanOrEqual(1500)
      expect(f.tabY).toBeLessThan(ROOM_VIEW.height / 2) // near the top
      const p = h.popup
      const area = (p.w * p.h) / (ROOM_VIEW.width * ROOM_VIEW.height)
      expect(area).toBeGreaterThanOrEqual(0.3)
      expect(area).toBeLessThanOrEqual(0.5)
      expect(p.x).toBeGreaterThanOrEqual(0)
      expect(p.y).toBeGreaterThanOrEqual(0)
      expect(p.x + p.w).toBeLessThanOrEqual(ROOM_VIEW.width)
      expect(p.y + p.h).toBeLessThanOrEqual(ROOM_VIEW.height)
      expect(p.tick).toBeGreaterThanOrEqual(20)
      expect(p.tick).toBeLessThanOrEqual(200)
      expect([h.close.w, h.close.h]).toEqual([9, 9])
      expect(h.close.x >= p.x && h.close.y >= p.y && h.close.x + 9 <= p.x + p.w && h.close.y + 9 <= p.y + p.h).toBe(true)
      for (const i of h.items) {
        expect(i.x).toBeGreaterThanOrEqual(0)
        expect(i.x + i.w).toBeLessThanOrEqual(ROOM_VIEW.width)
        if (i.text) for (const word of i.text.split(" ")) expect(WORDS).toContain(word)
      }
    }
  })
  it("is deterministic per seed and differs across seeds", () => {
    expect(layout(1)).toEqual(layout(1))
    expect(layout(1)).not.toEqual(layout(2))
  })
  it("gives up with an error rather than looping forever", () => {
    expect(() => layoutHideseek(1, [])).toThrow()
  })
})

describe("hideseek reveal and scroll (Review Focus 4)", () => {
  it("at tick 0 the revealed fraction is strictly between 0 and 1", () => {
    for (let seed = 1; seed <= 20; seed++) {
      const f = frac(freshRoom(seed))
      expect(f).toBeGreaterThan(0)
      expect(f).toBeLessThan(1)
    }
  })
  it("scrolling to the end and back raises the fraction monotonically to exactly the reachable items", () => {
    for (let seed = 1; seed <= 10; seed++) {
      const h = layout(seed)
      const r = freshRoom(seed)
      const ok = reachable(h)
      const start = frac(r)
      const revealed: string[] = []
      let last = start
      const track = (events: ReturnType<Room["scroll"]>) => {
        for (const e of events) if (e.kind === "reveal") revealed.push(e.elementId!)
        const f = frac(r)
        expect(f).toBeGreaterThanOrEqual(last)
        expect(f).toBeLessThanOrEqual(1)
        last = f
      }
      const { max } = r.scrollState()!
      expect(max).toBe(flowHideseek(h, new Set(), 0).height - ROOM_VIEW.height)
      for (let i = 0; i < 120; i++) track(r.scroll("down"))
      expect(offset(r)).toBe(max)
      for (let i = 0; i < 120; i++) track(r.scroll("up"))
      expect(offset(r)).toBe(0)
      expect(frac(r)).toBeCloseTo(ok.size / h.items.length, 10)
      expect(new Set(revealed).size).toBe(revealed.length) // each reveal once
      for (const id of revealed) expect(ok.has(id)).toBe(true)
      // reveal events cover every reachable item not already visible at tick 0
      expect(revealed.length).toBe(Math.round((frac(r) - start) * h.items.length))
    }
  })
  it("scroll moves 16 px per action and emits the new offset; at the ends it clamps without throwing", () => {
    const r = freshRoom()
    expect(() => r.scroll("up")).not.toThrow()
    expect(offset(r)).toBe(0)
    expect(r.scroll("up").filter(e => e.kind === "scroll")).toEqual([]) // nothing moved
    expect(r.scroll("down").filter(e => e.kind === "scroll")).toEqual([{ kind: "scroll", value: 16 }])
    expect(offset(r)).toBe(16)
    const { max } = r.scrollState()!
    let before = frac(r)
    for (let i = 0; i < 200; i++) {
      expect(() => r.scroll("down")).not.toThrow()
      expect(frac(r)).toBeGreaterThanOrEqual(before)
      expect(frac(r)).toBeLessThanOrEqual(1)
      before = frac(r)
    }
    expect(offset(r)).toBe(max)
    expect(r.scroll("down").filter(e => e.kind === "scroll")).toEqual([])
    expect(offset(r)).toBe(max)
    for (let i = 0; i < 200; i++) r.scroll("up")
    expect(offset(r)).toBe(0)
    expect(frac(r)).toBe(before)
  })
  it("clamps through the world too: scroll past both ends, fraction stays in (0,1]", () => {
    const w = makeWorld(1, { rooms: ["hideseek"] })
    let last = w.state().scroll!.revealedFraction
    for (const dir of [1, 2, 1]) {
      for (let i = 0; i < 110; i++) {
        w.act(act(0, 0, dir))
        const s = w.state().scroll!
        expect(s.offset).toBeGreaterThanOrEqual(0)
        expect(s.offset).toBeLessThanOrEqual(s.max)
        expect(s.revealedFraction).toBeGreaterThanOrEqual(last)
        expect(s.revealedFraction).toBeLessThanOrEqual(1)
        last = s.revealedFraction
      }
    }
    const scrolls = w.drainTrace().filter(e => e.kind === "scroll")
    expect(scrolls.length).toBeGreaterThan(0)
    expect(scrolls.every(e => e.room === "hideseek" && typeof e.value === "number")).toBe(true)
  })
  it("scrollState returns a fresh object each call", () => {
    const r = freshRoom()
    const a = r.scrollState()!
    a.offset = 999; a.revealedFraction = 5
    expect(r.scrollState()!.offset).toBe(0)
    expect(r.scrollState()).not.toBe(r.scrollState())
  })
  it("an item is revealed once any part is visible and not under the open pop-up", () => {
    for (let seed = 1; seed <= 10; seed++) {
      const h = layout(seed)
      const r = freshRoom(seed)
      r.step(h.popup.tick) // the pop-up opens before any scrolling
      const f = flowHideseek(h, new Set(), 0)
      const byId = new Map(h.items.map(i => [i.id, i]))
      const seen = new Set<string>()
      const p = h.popup
      const check = () => {
        const o = offset(r)
        for (const [id, y] of f.items) {
          const it = byId.get(id)!
          const top = Math.max(0, y - o), bot = Math.min(ROOM_VIEW.height, y - o + it.h)
          if (bot <= top) continue
          const covered = it.x >= p.x && it.x + it.w <= p.x + p.w && top >= p.y && bot <= p.y + p.h
          if (!covered) seen.add(id)
        }
      }
      const tick0 = new Set(visibleAt(h, f, 0)) // the room was built before the pop-up opened
      for (const id of tick0) seen.add(id)
      check()
      let events: string[] = []
      for (let i = 0; i < 120; i++) {
        events = events.concat(r.scroll("down").filter(e => e.kind === "reveal").map(e => e.elementId!))
        check()
      }
      expect(frac(r)).toBeCloseTo(seen.size / h.items.length, 10)
      expect(new Set(events)).toEqual(new Set([...seen].filter(id => !tick0.has(id))))
    }
  })
})

describe("hideseek sections", () => {
  it("collapsing a section hides its items' pixels and re-flows the content below; expanding restores it", () => {
    const h = layout(1)
    const r = freshRoom(1)
    const s = h.sections[0]!
    const open = flowHideseek(h, new Set(), 0)
    const shut = flowHideseek(h, new Set([s.id]), 0)
    const hy = open.headers.get(s.id)!
    const o = scrollToShow(r, hy, HS_HEADER_H + 30)
    const first = s.items[0]!
    const fy = open.items.get(first.id)! - o
    expect(r.inspect(first.x + 1, fy + 1)?.id).toBe(first.id)
    const before = render(r)
    const area = [first.x, Math.max(0, fy), first.w, Math.min(first.h, ROOM_VIEW.height - fy)] as const

    const ev = r.press(10, hy - o + 4)
    expect(ev[0]).toEqual({ kind: "respond", elementId: s.id })
    expect(ev.slice(1).every(e => e.kind === "reveal")).toBe(true) // content below moved up into view
    r.release(10, hy - o + 4)
    r.move(AWAY.x, AWAY.y, false) // drop hover
    const collapsed = render(r)
    expect(crop(collapsed, ...area)).not.toEqual(crop(before, ...area))
    expect(r.inspect(first.x + 1, fy + 1)?.id).not.toBe(first.id)
    for (const it of s.items) expect(shut.items.has(it.id)).toBe(false)
    // whatever came after the section moved up by exactly the section body
    const next = h.sections[1]!
    const shift = open.headers.get(next.id)! - shut.headers.get(next.id)!
    expect(shift).toBeGreaterThan(0)
    expect(open.height - shut.height).toBe(shift)
    expect(r.scrollState()!.max).toBe(Math.max(0, shut.height - ROOM_VIEW.height))
    const ny = shut.headers.get(next.id)! - offset(r)
    if (ny >= 0 && ny + HS_HEADER_H <= ROOM_VIEW.height) expect(r.inspect(10, ny + 4)?.id).toBe(next.id)

    r.press(10, hy - o + 4); r.release(10, hy - o + 4)
    r.move(AWAY.x, AWAY.y, false)
    expect(Array.from(render(r))).toEqual(Array.from(before))
    expect(r.inspect(first.x + 1, fy + 1)?.id).toBe(first.id)
  })
  it("items inside a collapsed section are never revealed by scrolling", () => {
    for (let seed = 1; seed <= 10; seed++) {
      const h = layout(seed)
      const r = freshRoom(seed)
      const last = h.sections[h.sections.length - 1]!
      const open = flowHideseek(h, new Set(), 0)
      const hy = open.headers.get(last.id)!
      const o = scrollToShow(r, hy, HS_HEADER_H)
      const expected = new Set<string>()
      for (let off = 0; off <= o; off += 16) for (const id of visibleAt(h, open, off)) expected.add(id)
      r.press(10, hy - o + 4)
      r.release(10, hy - o + 4)
      const shut = flowHideseek(h, new Set([last.id]), 0)
      for (const id of shut.items.keys()) expected.add(id) // all of these are reachable by scrolling
      for (let i = 0; i < 120; i++) r.scroll("down")
      for (let i = 0; i < 120; i++) r.scroll("up")
      expect(frac(r)).toBeCloseTo(expected.size / h.items.length, 10)
      expect(last.items.some(i => !expected.has(i.id))).toBe(true) // the section really hid something
    }
  })
})

/** Ids of the items in `flow` with at least one row inside the viewport at scroll offset `off` (no pop-up). */
function visibleAt(h: Hideseek, flow: ReturnType<typeof flowHideseek>, off: number) {
  const byId = new Map(h.items.map(i => [i.id, i]))
  return [...flow.items].filter(([id, y]) => y - off < ROOM_VIEW.height && y - off + byId.get(id)!.h > 0).map(([id]) => id)
}

describe("hideseek tabs", () => {
  it("clicking a tab swaps the panel; the tab's items become visible and revealed", () => {
    const h = layout(1)
    const r = freshRoom(1)
    const f0 = flowHideseek(h, new Set(), 0)
    const t0 = h.tabs[0]!, t1 = h.tabs[1]!
    const a = t0.items[0]!, b = t1.items[0]!
    const ay = f0.items.get(a.id)!
    expect(ay).toBeLessThan(ROOM_VIEW.height)
    expect(r.inspect(a.x + 1, ay + 1)?.id).toBe(a.id)
    const panel = [0, f0.panelY, ROOM_VIEW.width - 8, Math.min(f0.panelH, ROOM_VIEW.height - f0.panelY)] as const
    const before = crop(render(r), ...panel)
    const f1 = flowHideseek(h, new Set(), 1)
    expect(f1.items.has(b.id)).toBe(true)
    expect(f1.items.has(a.id)).toBe(false)
    expect(f1.height).toBe(f0.height) // the panel keeps its height: no re-flow below
    const was = frac(r)

    expect(r.press(t1.x + 2, f0.tabY + 4)).toEqual(expect.arrayContaining([{ kind: "respond", elementId: t1.id }]))
    r.release(t1.x + 2, f0.tabY + 4)
    r.move(AWAY.x, AWAY.y, false)
    expect(crop(render(r), ...panel)).not.toEqual(before)
    const by = f1.items.get(b.id)!
    expect(r.inspect(b.x + 1, by + 1)?.id).toBe(b.id)
    expect(r.inspect(a.x + 1, ay + 1)?.id).not.toBe(a.id)
    expect(frac(r)).toBeGreaterThan(was)
    // pressing the active tab again changes nothing and is not a response
    expect(r.press(t1.x + 2, f0.tabY + 4)).toEqual([])
  })
})

describe("hideseek pop-up", () => {
  it("appears at its seeded tick, covers its rect, blocks what is under it, and the close button removes it", () => {
    const h = layout(1)
    const p = h.popup
    const w = makeWorld(1, { rooms: ["hideseek"] })
    moveTo(w, AWAY.x, AWAY.y)
    while (w.tick < p.tick - 1) w.act(NOOP)
    const clear = Array.from(w.screen())
    const clearFrac = w.state().scroll!.revealedFraction
    w.act(NOOP)
    expect(w.tick).toBe(p.tick)
    const shown = w.screen()
    expect(rgbAt(shown, p.x, p.y)).not.toEqual(rgbAt(Uint8ClampedArray.from(clear), p.x, p.y))
    // the pop-up is opaque and fixed: every pixel of its rect is one of its own colours, and scrolling leaves it alone
    const rect = (s: Uint8ClampedArray) => { const out: number[] = []; for (let y = p.y; y < p.y + p.h; y++) for (let x = p.x; x < p.x + p.w; x++) out.push(...rgbAt(s, x, y)); return out }
    const atOpen = rect(shown)
    moveTo(w, Math.round(p.x + p.w / 2), Math.round(p.y + p.h / 2))
    expect(w.state().hovered).toEqual({ id: "popup", interactive: false, responsive: false })
    w.drainTrace()
    w.act(PRESS); w.act(RELEASE)
    expect(w.drainTrace().filter(e => e.kind === "respond")).toEqual([]) // the body is inert
    w.act(act(0, 0, 2)); w.act(act(0, 0, 2))
    expect(w.state().scroll!.offset).toBe(32)
    moveTo(w, AWAY.x, AWAY.y)
    expect(rect(w.screen())).toEqual(atOpen)
    moveTo(w, Math.round(p.x + p.w / 2), Math.round(p.y + p.h / 2))
    w.act(act(0, 0, 1)); w.act(act(0, 0, 1))
    expect(w.state().scroll!.offset).toBe(0)

    const cx = h.close.x + 4, cy = h.close.y + 4
    moveTo(w, cx, cy)
    expect(w.state().hovered).toEqual({ id: "popup-close", interactive: true, responsive: true })
    w.drainTrace()
    w.act(PRESS)
    const t = w.drainTrace()
    expect(t).toContainEqual({ tick: w.tick, room: "hideseek", kind: "click", elementId: "popup-close" })
    expect(t).toContainEqual({ tick: w.tick, room: "hideseek", kind: "respond", elementId: "popup-close" })
    w.act(RELEASE)
    moveTo(w, AWAY.x, AWAY.y)
    expect(Array.from(w.screen())).toEqual(clear) // gone, page exactly as before it appeared
    expect(w.state().scroll!.revealedFraction).toBeGreaterThanOrEqual(clearFrac)
    for (let i = 0; i < 300; i++) w.act(NOOP) // never comes back
    expect(Array.from(w.screen())).toEqual(clear)
  })
})

describe("hideseek inspect and areas", () => {
  it("headers, tabs and the close button are interactive+responsive; items are neither", () => {
    const h = layout(1)
    const r = freshRoom(1)
    const f = flowHideseek(h, new Set(), 0)
    const t = h.tabs[1]!
    expect(r.inspect(t.x + 1, f.tabY + 1)).toEqual({ id: t.id, interactive: true, responsive: true })
    const a = h.tabs[0]!.items[0]!
    expect(r.inspect(a.x + 1, f.items.get(a.id)! + 1)).toEqual({ id: a.id, interactive: false, responsive: false })
    const s = h.sections[0]!
    const o = scrollToShow(r, f.headers.get(s.id)!, HS_HEADER_H)
    expect(r.inspect(10, f.headers.get(s.id)! - o + 2)).toEqual({ id: s.id, interactive: true, responsive: true })
    expect(r.inspect(-1, 5)).toBeNull()
    expect(r.inspect(5, ROOM_VIEW.height)).toBeNull()
    r.step(h.popup.tick)
    expect(r.inspect(h.close.x, h.close.y)).toEqual({ id: "popup-close", interactive: true, responsive: true })
    expect(r.inspect(h.popup.x, h.popup.y + h.popup.h - 1)).toEqual({ id: "popup", interactive: false, responsive: false })
    expect(r.page()).toBeNull()
  })
  it("area fractions are in [0,1], responsive + unresponsive <= 1, and track scrolling", () => {
    for (let seed = 1; seed <= 10; seed++) {
      const r = freshRoom(seed)
      const a = r.areas()
      for (const v of Object.values(a)) { expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThanOrEqual(1) }
      expect(a.responsive + a.unresponsive).toBeLessThanOrEqual(1)
      expect(a.interactive).toBeGreaterThan(0) // the tab strip is near the top
      expect(a.unresponsive).toBeGreaterThan(0)
      expect(a.interactive).toBeCloseTo(a.responsive, 10)
    }
    const r = freshRoom(1)
    const a0 = r.areas()
    r.scroll("down"); r.scroll("down"); r.scroll("down")
    expect(r.areas()).not.toEqual(a0)
  })
  it("areas agree with inspect() pixel by pixel, with and without the pop-up", () => {
    const scan = (r: Room) => {
      let on = 0, off = 0
      for (let y = 0; y < ROOM_VIEW.height; y++) for (let x = 0; x < ROOM_VIEW.width; x++) {
        const i = r.inspect(x, y)
        if (i?.responsive) on++
        else if (i) off++
      }
      const n = ROOM_VIEW.width * ROOM_VIEW.height
      return { interactive: on / n, responsive: on / n, unresponsive: off / n }
    }
    for (const seed of [1, 2, 3]) {
      const r = freshRoom(seed)
      expect(r.areas()).toEqual(scan(r))
      for (let i = 0; i < 7; i++) r.scroll("down")
      expect(r.areas()).toEqual(scan(r))
      r.step(layout(seed).popup.tick)
      expect(r.areas()).toEqual(scan(r))
    }
  })
  it("tab strip and header are drawn at their flow positions (header ink differs from the page)", () => {
    const h = layout(1)
    const r = freshRoom(1)
    const f = flowHideseek(h, new Set(), 0)
    const d = render(r)
    const bg = px(d, ROOM_VIEW.width - 9, 1)
    expect(px(d, h.tabs[0]!.x + 1, f.tabY + HS_TAB_H - 2)).not.toEqual(bg)
  })
})

describe("hideseek persistence (Review Focus 2) and determinism", () => {
  it("the scroll offset, collapsed sections and tab survive a room switch", () => {
    const h = layout(1)
    const w = makeWorld(1, { rooms: ["crib", "hideseek"], start: "hideseek" })
    const f = flowHideseek(h, new Set(), 0)
    moveTo(w, h.tabs[1]!.x + 2, f.tabY + 4)
    w.act(PRESS); w.act(RELEASE)
    moveTo(w, 160, 100)
    for (let i = 0; i < 5; i++) w.act(act(0, 0, 2))
    expect(w.state().scroll!.offset).toBe(80)
    moveTo(w, AWAY.x, AWAY.y)
    const before = Array.from(w.screen())
    const st = w.state().scroll
    w.bringToFront("crib")
    expect(w.state().scroll).toBeNull()
    w.bringToFront("hideseek")
    expect(w.state().scroll).toEqual(st)
    expect(Array.from(w.screen())).toEqual(before)
  })
  it("the same seed and actions give identical observations and traces; another seed differs", () => {
    const run = (seed: number) => {
      const w = makeWorld(seed, { rooms: ["hideseek"] })
      const obs: number[][] = []
      for (let i = 0; i < 260; i++) {
        const a = i % 7 === 0 ? act(0, 0, i % 3 === 0 ? 1 : 2) : i % 11 === 0 ? PRESS : i % 11 === 1 ? RELEASE : act(1 + (i % 24))
        w.act(a)
        if (i % 20 === 0) obs.push(Array.from(w.observe().periphery))
      }
      return { obs, trace: w.drainTrace(), state: w.state() }
    }
    expect(run(1)).toEqual(run(1))
    expect(run(1).obs).not.toEqual(run(2).obs)
  })
  it("is a default room after toys", () => {
    expect(makeWorld(1).rooms()).toEqual(["crib", "toys", "hideseek"])
  })
})
