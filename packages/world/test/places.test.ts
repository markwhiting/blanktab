import { describe, it, expect } from "vitest"
import { placesRoom, ROOM_VIEW, type PlacesRoom, type Room } from "../src/index"
import { hashSeed } from "../src/rng"
import { WORDS } from "../src/words"
import { makeWorld, clickAt } from "./helpers"

const make = (seed = 1) => placesRoom(hashSeed(seed, "places"), WORDS) as PlacesRoom
const draw = (r: Room) => {
  const px = new Map<string, string>()
  const g = {
    rect: (x: number, y: number, w: number, h: number, c: string) => px.set(`${x},${y},${w},${h}`, c),
    circle: () => {},
    text: (s: string, x: number, y: number, c: string) => px.set(`t${x},${y},${s}`, c),
    clip: (_x: number, _y: number, _w: number, _h: number, d: () => void) => d(),
  }
  r.draw(g)
  return [...px.entries()].join("|")
}
/** Find a point where inspect reports the given element id. */
function find(r: Room, id: string) {
  for (let y = 0; y < ROOM_VIEW.height; y++) for (let x = 0; x < ROOM_VIEW.width; x++) if (r.inspect(x, y)?.id === id) return { x, y }
  throw new Error(`no ${id}`)
}
const linkId = (t: string) => `link-${t}`

describe("places graph", () => {
  it("has 6-20 pages p0.., every page reachable from p0, out-degree 2-4, no self links (seeds 1-50)", () => {
    for (let seed = 1; seed <= 50; seed++) {
      const { pages, links } = make(seed).graph()
      expect(pages.length).toBeGreaterThanOrEqual(6)
      expect(pages.length).toBeLessThanOrEqual(20)
      expect(pages).toEqual(pages.map((_, i) => `p${i}`))
      const seen = new Set(["p0"]), q = ["p0"]
      while (q.length) for (const t of links[q.shift()!]!) if (!seen.has(t)) { seen.add(t); q.push(t) }
      expect(seen.size).toBe(pages.length)
      for (const p of pages) {
        const l = links[p]!
        expect(l.length).toBeGreaterThanOrEqual(2)
        expect(l.length).toBeLessThanOrEqual(4)
        expect(new Set(l).size).toBe(l.length)
        expect(l).not.toContain(p)
        l.forEach(t => expect(pages).toContain(t))
      }
    }
  })
  it("link words are distinct per page; link boxes are inside the viewport and pairwise disjoint (seeds 1-50)", () => {
    for (let seed = 1; seed <= 50; seed++) {
      const links = make(seed).links()
      for (const [page, ls] of Object.entries(links)) {
        const words = ls.map(l => l.word)
        expect(new Set(words).size, `seed ${seed} ${page}`).toBe(words.length)
        for (const [i, a] of ls.entries()) {
          expect(a.x).toBeGreaterThanOrEqual(0)
          expect(a.y).toBeGreaterThanOrEqual(0)
          expect(a.x + a.w).toBeLessThanOrEqual(ROOM_VIEW.width)
          expect(a.y + a.h).toBeLessThanOrEqual(ROOM_VIEW.height)
          for (const b of ls.slice(i + 1)) {
            const apart = a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y
            expect(apart, `seed ${seed} ${page} ${a.target}/${b.target}`).toBe(true)
          }
        }
      }
    }
  })
  it("link words are distinct across target pages and the word pool is deduplicated and guarded", () => {
    // A pool with duplicates still yields distinct link words per page when enough unique words exist.
    const dup = [...WORDS, ...WORDS]
    const r = placesRoom(hashSeed(5, "places"), dup) as PlacesRoom
    for (const ls of Object.values(r.links())) expect(new Set(ls.map(l => l.word)).size).toBe(ls.length)
    // Fewer unique words than pages: fail loudly rather than reuse a word.
    const few = ["cat", "dog", "cat", "dog", "cat"]
    expect(() => placesRoom(hashSeed(1, "places"), few)).toThrow(RangeError)
  })
  it("is deterministic per seed and differs across seeds", () => {
    expect(make(3).graph()).toEqual(make(3).graph())
    expect(draw(make(3))).toBe(draw(make(3)))
    expect(draw(make(3))).not.toBe(draw(make(4)))
  })
  it("returns fresh objects from graph(), inspect() and scrollState()", () => {
    const r = make()
    const g = r.graph()
    g.links.p0!.push("zzz")
    expect(r.graph().links.p0).not.toContain("zzz")
    const t = r.graph().links.p0![0]!
    const { x, y } = find(r, linkId(t))
    expect(r.inspect(x, y)).not.toBe(r.inspect(x, y))
    expect(r.scrollState()).toBeNull()
  })
})

describe("places navigation", () => {
  it("starts on p0; clicking a link changes page, pixels, and emits respond + navigate", () => {
    const r = make()
    expect(r.page()).toBe("p0")
    const before = draw(r)
    const t = r.graph().links.p0![0]!
    const { x, y } = find(r, linkId(t))
    expect(r.inspect(x, y)).toMatchObject({ interactive: true, responsive: true })
    const ev = r.press(x, y)
    expect(ev).toEqual([{ kind: "respond", elementId: linkId(t) }, { kind: "navigate", value: t }])
    expect(r.page()).toBe(t)
    expect(draw(r)).not.toBe(before)
    expect(r.release(x, y)).toEqual([])
  })
  it("Back returns to the previous page; Back at the start does nothing", () => {
    const r = make()
    const b = find(r, "back")
    expect(r.inspect(b.x, b.y)).toMatchObject({ interactive: false, responsive: false })
    const snap = draw(r)
    expect(r.press(b.x, b.y)).toEqual([])
    expect(r.page()).toBe("p0")
    expect(draw(r)).toBe(snap)
    const t = r.graph().links.p0![0]!
    const l = find(r, linkId(t))
    r.press(l.x, l.y)
    expect(r.inspect(b.x, b.y)).toMatchObject({ interactive: true, responsive: true })
    expect(r.press(b.x, b.y)).toEqual([{ kind: "respond", elementId: "back" }, { kind: "navigate", value: "p0" }])
    expect(r.page()).toBe("p0")
    expect(r.press(b.x, b.y)).toEqual([])
  })
  it("back is 11x11 at top-left", () => {
    const r = make()
    let n = 0
    for (let y = 0; y < ROOM_VIEW.height; y++) for (let x = 0; x < ROOM_VIEW.width; x++) if (r.inspect(x, y)?.id === "back") n++
    expect(n).toBe(121)
    expect(find(r, "back")).toEqual({ x: 4, y: 4 })
  })
  it("a visited link is drawn in a different colour", () => {
    const r = make()
    const [a] = r.graph().links.p0!
    const colorOf = (id: string) => {
      const p = find(r, linkId(id))
      return draw(r).split("|").filter(e => e.startsWith(`t${p.x},`))
    }
    const unvisited = colorOf(a!)
    const l = find(r, linkId(a!))
    r.press(l.x, l.y)
    const bk = find(r, "back")
    r.press(bk.x, bk.y)
    expect(colorOf(a!)).not.toEqual(unvisited)
  })
  it("text is neither interactive nor responsive; areas sum sensibly", () => {
    const r = make()
    const a = r.areas()
    expect(a.interactive).toBeGreaterThan(0)
    expect(a.unresponsive).toBeGreaterThan(0)
    expect(a.interactive).toBe(a.responsive)
    expect(a.interactive + a.unresponsive).toBeLessThan(1)
    expect(r.areas()).not.toBe(r.areas())
  })
  it("ignores scroll and keys; out-of-view points hit nothing", () => {
    const r = make()
    expect(r.scroll("down")).toEqual([])
    expect(r.key("a")).toEqual([])
    expect(r.inspect(-1, 5)).toBeNull()
    expect(r.press(400, 400)).toEqual([])
  })
})

describe("places in the desktop", () => {
  it("the current page persists across room switches, and navigation is traced", () => {
    const w = makeWorld(1, { rooms: ["crib", "places"], start: "places" })
    expect(w.state().page).toBe("p0")
    const r = make(1)
    const t = r.graph().links.p0![0]!
    const p = find(r, linkId(t))
    clickAt(w, p.x, p.y)
    expect(w.state().page).toBe(t)
    w.bringToFront("crib")
    expect(w.state().page).toBeNull()
    w.bringToFront("places")
    expect(w.state().page).toBe(t)
  })
  it("pages differ in background tint", () => {
    const r = make()
    const bg = (rm: Room) => draw(rm).split("|")[0]
    const first = bg(r)
    const t = r.graph().links.p0![0]!
    const p = find(r, linkId(t))
    r.press(p.x, p.y)
    expect(bg(r)).not.toBe(first)
  })
})
