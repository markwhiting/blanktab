import { mulberry32 } from "../rng"
import { GLYPH_H, textWidth } from "../font"
import { ROOM_VIEW, inView, type ElementInfo, type Room, type RoomFactory } from "../room"

export interface PlacesGraph { pages: string[]; links: Record<string, string[]> }
export interface PlacesLink { target: string; word: string; x: number; y: number; w: number; h: number }
/** The Places room; `graph()` and `links()` are test-only accessors (fresh copies each call). */
export type PlacesRoom = Room & { graph(): PlacesGraph; links(): Record<string, PlacesLink[]> }

type Box = { x: number; y: number; w: number; h: number }
interface Link extends Box { target: string }
interface Page {
  id: string
  tint: [number, number, number]
  title: string
  titleBox: Box
  lines: { text: string; box: Box }[]
  links: Link[]
}

const BACK: Box = { x: 4, y: 4, w: 11, h: 11 }
const INK = "rgb(30,30,40)", BODY_INK = "rgb(70,70,80)"
const LINK_NEW = "rgb(20,60,200)", LINK_SEEN = "rgb(130,40,150)"
const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`
const within = (b: Box, x: number, y: number) => x >= b.x && y >= b.y && x < b.x + b.w && y < b.y + b.h

function build(seed: number, words: readonly string[]) {
  const rnd = mulberry32(seed)
  const int = (lo: number, hi: number) => lo + Math.floor(rnd() * (hi - lo + 1))
  const n = int(6, 20)
  const ids = Array.from({ length: n }, (_, i) => `p${i}`)

  // Random spanning tree rooted at p0 (parent -> child), at most 4 children per page.
  const out: number[][] = ids.map(() => [])
  for (let i = 1; i < n; i++) {
    const open = out.slice(0, i).map((c, j) => (c.length < 4 ? j : -1)).filter(j => j >= 0)
    out[open[int(0, open.length - 1)]!]!.push(i)
  }
  // Extra links up to a random out-degree of 2-4.
  for (let i = 0; i < n; i++) {
    const want = int(2, 4)
    while (out[i]!.length < want) {
      const t = int(0, n - 1)
      if (t !== i && !out[i]!.includes(t)) out[i]!.push(t)
    }
  }
  // A distinct link word per target page.
  const pool = [...new Set(words)]
  if (pool.length < n) throw new RangeError(`places needs ${n} distinct words for its links, got ${pool.length}`)
  for (let i = pool.length - 1; i > 0; i--) { const j = int(0, i); [pool[i], pool[j]] = [pool[j]!, pool[i]!] }
  const linkWord = pool.slice(0, n)
  const word = () => words[int(0, words.length - 1)]!

  const pages: Page[] = ids.map((id, i) => {
    const tint: [number, number, number] = [int(215, 250), int(215, 250), int(215, 250)]
    const title = word()
    const titleBox = { x: 20, y: 4, w: textWidth(title, 2), h: GLYPH_H * 2 }
    const lines: Page["lines"] = []
    for (let l = int(2, 5), y = 26; l > 0; l--, y += 12) {
      let text = word()
      for (let k = int(2, 4); k > 0; k--) {
        const next = `${text} ${word()}`
        if (textWidth(next) <= ROOM_VIEW.width - 16) text = next
      }
      lines.push({ text, box: { x: 8, y, w: textWidth(text), h: GLYPH_H } })
    }
    const order = out[i]!.map(t => ({ t, k: rnd() })).sort((a, b) => a.k - b.k).map(o => o.t)
    const links: Link[] = []
    let x = 8, y = 100
    for (const t of order) {
      const w = textWidth(linkWord[t]!)
      if (x + w > ROOM_VIEW.width - 8) { x = 8; y += 20 }
      links.push({ target: ids[t]!, x, y, w, h: GLYPH_H + 2 })
      x += w + 20 + int(0, 12)
    }
    return { id, tint, title, titleBox, lines, links }
  })
  return { ids, pages, linkWord, out }
}

/**
 * A small seeded linked site: pages with a title, a few lines of text, underlined links and a Back button.
 * Navigation happens on press. Session state (current page, history, visited set) lives for the life of the room.
 */
export const placesRoom: RoomFactory = (seed, words) => {
  const { ids, pages, linkWord, out } = build(seed, words)
  const byId = new Map(pages.map(p => [p.id, p]))
  const visited = new Set(["p0"])
  const history: string[] = []
  let current = "p0"
  let areaCache: ReturnType<Room["areas"]> | null = null

  const cur = () => byId.get(current)!
  const linkAt = (x: number, y: number) => (inView(x, y) ? cur().links.find(l => within(l, x, y)) : undefined)
  const backLive = () => history.length > 0
  const go = (target: string, push: boolean) => {
    if (push) history.push(current)
    current = target
    visited.add(target)
    areaCache = null
  }

  const room: PlacesRoom = {
    kind: "places",
    draw(g) {
      const p = cur()
      g.rect(0, 0, ROOM_VIEW.width, ROOM_VIEW.height, rgb(p.tint))
      g.rect(BACK.x, BACK.y, BACK.w, BACK.h, backLive() ? INK : "rgb(150,150,160)")
      g.rect(BACK.x + 1, BACK.y + 1, BACK.w - 2, BACK.h - 2, "rgb(250,250,250)")
      g.text("<", BACK.x + 3, BACK.y + 2, backLive() ? INK : "rgb(150,150,160)")
      g.text(p.title, p.titleBox.x, p.titleBox.y, INK, 2)
      for (const l of p.lines) g.text(l.text, l.box.x, l.box.y, BODY_INK)
      for (const l of p.links) {
        const c = visited.has(l.target) ? LINK_SEEN : LINK_NEW
        g.text(linkWord[ids.indexOf(l.target)]!, l.x, l.y, c)
        g.rect(l.x, l.y + GLYPH_H + 1, l.w, 1, c)
      }
    },
    press(x, y) {
      if (!inView(x, y)) return []
      if (within(BACK, x, y)) {
        if (!backLive()) return []
        const prev = history.pop()!
        go(prev, false)
        return [{ kind: "respond", elementId: "back" }, { kind: "navigate", value: prev }]
      }
      const l = linkAt(x, y)
      if (!l) return []
      go(l.target, true)
      return [{ kind: "respond", elementId: `link-${l.target}` }, { kind: "navigate", value: l.target }]
    },
    move: () => [],
    release: () => [],
    scroll: () => [],
    key: () => [],
    step: () => [],
    inspect(x, y): ElementInfo | null {
      if (!inView(x, y)) return null
      if (within(BACK, x, y)) return { id: "back", interactive: backLive(), responsive: backLive() }
      const l = linkAt(x, y)
      if (l) return { id: `link-${l.target}`, interactive: true, responsive: true }
      const p = cur()
      if (within(p.titleBox, x, y)) return { id: "title", interactive: false, responsive: false }
      const i = p.lines.findIndex(t => within(t.box, x, y))
      return i >= 0 ? { id: `line-${i}`, interactive: false, responsive: false } : null
    },
    areas() {
      if (!areaCache) {
        let live = 0, inert = 0
        for (let y = 0; y < ROOM_VIEW.height; y++) {
          for (let x = 0; x < ROOM_VIEW.width; x++) {
            const info = room.inspect(x, y)
            if (info?.interactive) live++
            else if (info) inert++
          }
        }
        const total = ROOM_VIEW.width * ROOM_VIEW.height
        areaCache = { interactive: live / total, responsive: live / total, unresponsive: inert / total }
      }
      return { ...areaCache }
    },
    scrollState: () => null,
    page: () => current,
    links: () => Object.fromEntries(pages.map(p => [p.id, p.links.map(l => ({
      target: l.target, word: linkWord[ids.indexOf(l.target)]!, x: l.x, y: l.y, w: l.w, h: l.h,
    }))])),
    graph: () => ({
      pages: [...ids],
      links: Object.fromEntries(ids.map((id, i) => [id, out[i]!.map(t => ids[t]!)])),
    }),
  }
  return room
}
