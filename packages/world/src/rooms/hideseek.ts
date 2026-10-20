import { mulberry32 } from "../rng"
import { WORDS } from "../words"
import { GLYPH_H, textWidth } from "../font"
import { ROOM_VIEW, inView, type ElementInfo, type Room, type RoomEvent, type RoomFactory } from "../room"
import type { Gfx } from "../gfx"

export const HS_BACKGROUND = "rgb(244,240,230)"
export const HS_HEADER_H = 14
export const HS_TAB_H = 14
export const HS_SCROLL_STEP = 16
export const HS_CLOSE = 9

type Rgb = [number, number, number]
type Box = { x: number; y: number; w: number; h: number }

/** One piece of page content. `y` is not stored: it comes from the flow (sections fold, tabs swap). */
export interface HsItem {
  id: string
  kind: "line" | "block" | "card"
  x: number
  w: number
  h: number
  gap: number // space below the item
  text?: string // line / card words
  scale?: number // line text scale
  color: Rgb
}
export interface HsSection { id: string; title: string; color: Rgb; items: HsItem[] }
export interface HsTab { id: string; label: string; x: number; w: number; items: HsItem[] }
export type HsNode = { type: "item"; item: HsItem } | { type: "tabs" } | { type: "section"; section: HsSection }
export interface HsPopup extends Box { tick: number; title: string; lines: string[]; color: Rgb }
export interface Hideseek {
  nodes: HsNode[] // document order
  tabs: HsTab[]
  sections: HsSection[]
  items: HsItem[] // every item, wherever it lives
  popup: HsPopup // fixed in the viewport
  close: Box // the pop-up's close button, viewport coords
}

/** Where everything sits for one fold/tab state. Content coordinates (add -offset for the viewport). */
export interface HsFlow {
  height: number
  headers: Map<string, number> // section id -> header y
  tabY: number
  panelY: number
  panelH: number
  items: Map<string, number> // displayed item id -> y (collapsed sections and other tabs are absent)
}

const TOP = 6, BOTTOM = 8, LEFT = 4, RIGHT = 312, PANEL_PAD = 6, SECTION_GAP = 6
const LINE_H = GLYPH_H, CARD_H = 21
const HEIGHT = { min: 500, max: 1500 } as const

/** The page plus how it folds. Pure function of seed; retries until the page is 500-1500 px tall. */
export function layoutHideseek(seed: number, words: readonly string[] = WORDS): Hideseek {
  if (words.length === 0) throw new RangeError("layoutHideseek: the word list is empty")
  const rnd = mulberry32(seed)
  const int = (lo: number, hi: number) => lo + Math.floor(rnd() * (hi - lo + 1))
  const word = () => words[Math.floor(rnd() * words.length)]!
  const color = (lo: number, hi: number): Rgb => [int(lo, hi), int(lo, hi), int(lo, hi)]
  const words2 = (lo: number, hi: number, maxW: number, scale = 1) => {
    const out = [word()]
    for (let n = int(lo, hi) - 1; n > 0; n--) {
      const w = word()
      if (textWidth([...out, w].join(" "), scale) > maxW) break
      out.push(w)
    }
    return out.join(" ")
  }

  for (let round = 0; round < 200; round++) {
    let count = 0
    const id = () => `item-${count++}`
    const at = (w: number) => int(LEFT + 4, RIGHT - 4 - w)
    const item = (): HsItem => {
      const r = rnd()
      if (r < 0.4) {
        const text = words2(2, 6, 280)
        const w = textWidth(text)
        return { id: id(), kind: "line", x: at(w), w, h: LINE_H, gap: int(6, 30), text, scale: 1, color: color(20, 70) }
      }
      if (r < 0.7) {
        const w = int(30, 220)
        return { id: id(), kind: "block", x: at(w), w, h: int(12, 40), gap: int(6, 30), color: color(80, 220) }
      }
      const text = words2(1, 2, 120)
      const w = textWidth(text) + 16
      return { id: id(), kind: "card", x: at(w), w, h: CARD_H, gap: int(6, 30), text, color: color(110, 230) }
    }

    const total = int(12, 40)
    const title = words2(1, 2, 260, 2)
    const titleItem: HsItem = {
      id: id(), kind: "line", x: LEFT + 4, w: textWidth(title, 2), h: LINE_H * 2, gap: 8, text: title, scale: 2, color: [30, 30, 40],
    }
    const nTabs = int(2, 3), nSections = int(2, 4)
    const tabSizes = Array.from({ length: nTabs }, () => int(1, 3))
    const sectionSizes = Array.from({ length: nSections }, () => int(2, 6))
    const loose = total - 1 - tabSizes.reduce((a, b) => a + b, 0) - sectionSizes.reduce((a, b) => a + b, 0)
    if (loose < 0) continue

    let tx = LEFT + 4
    const tabs: HsTab[] = tabSizes.map((n, i) => {
      const label = words2(1, 1, 80)
      const tab: HsTab = { id: `tab-${i}`, label, x: tx, w: textWidth(label) + 10, items: [] }
      tx += tab.w + 2
      for (let k = 0; k < n; k++) tab.items.push(item())
      return tab
    })
    const sections: HsSection[] = sectionSizes.map((n, i) => {
      const s: HsSection = { id: `section-${i}`, title: words2(1, 3, 200), color: color(120, 200), items: [] }
      for (let k = 0; k < n; k++) s.items.push(item())
      return s
    })
    // loose items go into the gaps around the sections (slot 0 is right after the tab panel)
    const slots: HsItem[][] = Array.from({ length: nSections + 1 }, () => [])
    for (let k = 0; k < loose; k++) slots[int(0, nSections)]!.push(item())
    const nodes: HsNode[] = [{ type: "item", item: titleItem }, { type: "tabs" }]
    slots.forEach((slot, i) => {
      for (const it of slot) nodes.push({ type: "item", item: it })
      if (i < nSections) nodes.push({ type: "section", section: sections[i]! })
    })
    const items = [titleItem, ...tabs.flatMap(t => t.items), ...sections.flatMap(s => s.items), ...slots.flat()]
      .sort((a, b) => Number(a.id.slice(5)) - Number(b.id.slice(5)))

    const pw = int(150, 300)
    const ph = Math.round((0.3 + rnd() * 0.2) * ROOM_VIEW.width * ROOM_VIEW.height / pw)
    const area = (pw * ph) / (ROOM_VIEW.width * ROOM_VIEW.height)
    if (ph > ROOM_VIEW.height - 4 || area < 0.3 || area > 0.5) continue
    const popup: HsPopup = {
      x: int(2, ROOM_VIEW.width - pw - 2), y: int(2, ROOM_VIEW.height - ph - 2), w: pw, h: ph,
      tick: int(20, 200), title: words2(1, 2, pw - 30), lines: [], color: color(150, 235),
    }
    for (let ly = 22; ly + LINE_H <= ph - 6; ly += LINE_H + 5) popup.lines.push(words2(1, 4, pw - 16))
    const close = { x: popup.x + pw - HS_CLOSE - 3, y: popup.y + 3, w: HS_CLOSE, h: HS_CLOSE }

    const h: Hideseek = { nodes, tabs, sections, items, popup, close }
    const height = flowHideseek(h, new Set(), 0).height
    if (height < HEIGHT.min || height > HEIGHT.max) continue
    if (new Set(items.map(i => i.kind)).size < 3) continue
    return h
  }
  throw new Error("layoutHideseek: could not build a page of the right size")
}

/** Positions for a fold/tab state. The tab panel is as tall as its tallest tab, so switching never re-flows. */
export function flowHideseek(h: Hideseek, collapsed: ReadonlySet<string>, activeTab: number): HsFlow {
  const stack = (list: readonly HsItem[], y: number) => list.reduce((yy, it) => yy + it.h + it.gap, y)
  const panelH = Math.max(...h.tabs.map(t => stack(t.items, PANEL_PAD)))
  const out: HsFlow = { height: 0, headers: new Map(), tabY: 0, panelY: 0, panelH, items: new Map() }
  const place = (list: readonly HsItem[], y: number) => {
    for (const it of list) { out.items.set(it.id, y); y += it.h + it.gap }
    return y
  }
  let y = TOP
  for (const n of h.nodes) {
    if (n.type === "item") y = place([n.item], y)
    else if (n.type === "tabs") {
      out.tabY = y
      out.panelY = y + HS_TAB_H
      place(h.tabs[activeTab]!.items, out.panelY + PANEL_PAD)
      y = out.panelY + panelH + 10
    } else {
      out.headers.set(n.section.id, y)
      y += HS_HEADER_H + SECTION_GAP
      if (!collapsed.has(n.section.id)) y = place(n.section.items, y)
      y += 4
    }
  }
  out.height = y + BOTTOM
  return out
}

const lighten = (c: readonly number[]) => c.map(v => Math.min(255, Math.round(v + (255 - v) * 0.4)))
const darken = (c: readonly number[]) => c.map(v => Math.round(v * 0.7))
const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const within = (b: Box, x: number, y: number) => x >= b.x && y >= b.y && x < b.x + b.w && y < b.y + b.h
const INK = "rgb(30,30,40)", PANEL = "rgb(252,252,250)", TAB_OFF = "rgb(200,200,205)", TRACK = "rgb(220,216,206)"

type Hit =
  | { kind: "close" | "popup"; id: string }
  | { kind: "header"; id: string; index: number }
  | { kind: "tab"; id: string; index: number }
  | { kind: "item"; id: string }

/**
 * A long page seen through the viewport: a tab strip near the top, folding sections, a fixed pop-up that opens
 * on a seeded tick. Privileged tracking of which items have been seen: an item is revealed once any of its rows
 * is inside the viewport, displayed (not folded away, not on another tab) and not wholly under the open pop-up.
 * Items visible at construction count as revealed but emit no `reveal` (nothing was uncovered).
 */
export const hideseekRoom: RoomFactory = (seed, words) => {
  const h = layoutHideseek(seed, words)
  const { popup, close } = h
  const byId = new Map(h.items.map(i => [i.id, i]))
  const collapsed = new Set<string>()
  let activeTab = 0
  let flow = flowHideseek(h, collapsed, activeTab)
  let offset = 0
  let popupOpen = false, popupDone = false
  const revealed = new Set<string>()
  let hovered: string | null = null
  let cursor: { x: number; y: number } | null = null
  let areaCache: ReturnType<Room["areas"]> | null = null

  const max = () => Math.max(0, flow.height - ROOM_VIEW.height)
  const headerBox = (y: number): Box => ({ x: LEFT, y, w: RIGHT - LEFT, h: HS_HEADER_H })
  const tabBox = (t: HsTab): Box => ({ x: t.x, y: flow.tabY, w: t.w, h: HS_TAB_H })

  const hitAt = (x: number, y: number): Hit | null => {
    if (!inView(x, y)) return null
    if (popupOpen && within(popup, x, y)) return within(close, x, y) ? { kind: "close", id: "popup-close" } : { kind: "popup", id: "popup" }
    const cy = y + offset
    for (const [i, s] of h.sections.entries()) {
      if (within(headerBox(flow.headers.get(s.id)!), x, cy)) return { kind: "header", id: s.id, index: i }
    }
    for (const [i, t] of h.tabs.entries()) if (within(tabBox(t), x, cy)) return { kind: "tab", id: t.id, index: i }
    for (const [id, iy] of flow.items) {
      const it = byId.get(id)!
      if (within({ x: it.x, y: iy, w: it.w, h: it.h }, x, cy)) return { kind: "item", id }
    }
    return null
  }
  const live = (k: Hit["kind"]) => k === "close" || k === "header" || k === "tab"

  /** Marks newly visible items; returns their `reveal` events. */
  const reveal = (): RoomEvent[] => {
    const out: RoomEvent[] = []
    for (const [id, iy] of flow.items) {
      if (revealed.has(id)) continue
      const it = byId.get(id)!
      const top = Math.max(0, iy - offset), bottom = Math.min(ROOM_VIEW.height, iy - offset + it.h)
      if (bottom <= top) continue
      const shown = { x: it.x, y: top, w: it.w, h: bottom - top }
      const hidden = popupOpen && shown.x >= popup.x && shown.x + shown.w <= popup.x + popup.w &&
        shown.y >= popup.y && shown.y + shown.h <= popup.y + popup.h
      if (hidden) continue
      revealed.add(id)
      out.push({ kind: "reveal", elementId: id })
    }
    return out
  }
  /** Content moved under a still cursor: recompute hover so the next frame is not stale. */
  const changed = () => {
    areaCache = null
    const hit = cursor ? hitAt(cursor.x, cursor.y) : null
    hovered = hit && live(hit.kind) ? hit.id : null
  }
  reveal() // what is on screen from the start counts, silently

  const drawItem = (g: Gfx, it: HsItem, y: number) => {
    if (it.kind === "line") g.text(it.text!, it.x, y, rgb(it.color), it.scale)
    else if (it.kind === "block") g.rect(it.x, y, it.w, it.h, rgb(it.color))
    else {
      g.rect(it.x, y, it.w, it.h, INK)
      g.rect(it.x + 1, y + 1, it.w - 2, it.h - 2, rgb(it.color))
      g.text(it.text!, it.x + 8, y + 7, INK)
    }
  }
  const drawPlus = (g: Gfx, x: number, y: number, open: boolean) => {
    g.rect(x, y + 3, 7, 1, INK)
    if (!open) g.rect(x + 3, y, 1, 7, INK)
  }

  const room: Room = {
    kind: "hideseek",
    draw(g) {
      g.rect(0, 0, ROOM_VIEW.width, ROOM_VIEW.height, HS_BACKGROUND)
      const vis = (y: number, hgt: number) => y - offset < ROOM_VIEW.height && y - offset + hgt > 0
      // tab strip and panel
      const ty = flow.tabY - offset, py = flow.panelY - offset
      if (vis(flow.tabY, HS_TAB_H + flow.panelH)) {
        g.rect(LEFT, py, RIGHT - LEFT, flow.panelH, INK)
        g.rect(LEFT + 1, py + 1, RIGHT - LEFT - 2, flow.panelH - 2, PANEL)
        for (const [i, t] of h.tabs.entries()) {
          const on = i === activeTab
          g.rect(t.x, ty, t.w, HS_TAB_H + (on ? 1 : 0), INK)
          const face = on ? [252, 252, 250] : [200, 200, 205]
          g.rect(t.x + 1, ty + 1, t.w - 2, HS_TAB_H - 1 + (on ? 1 : 0), hovered === t.id ? rgb(lighten(face)) : on ? PANEL : TAB_OFF)
          g.text(t.label, t.x + 5, ty + 3, INK)
        }
      }
      for (const s of h.sections) {
        const hy = flow.headers.get(s.id)!
        if (!vis(hy, HS_HEADER_H)) continue
        const y = hy - offset
        g.rect(LEFT, y, RIGHT - LEFT, HS_HEADER_H, rgb(darken(s.color)))
        g.rect(LEFT + 1, y + 1, RIGHT - LEFT - 2, HS_HEADER_H - 2, rgb(hovered === s.id ? lighten(s.color) : s.color))
        drawPlus(g, LEFT + 5, y + 4, !collapsed.has(s.id))
        g.text(s.title, LEFT + 18, y + 3, INK)
      }
      for (const [id, iy] of flow.items) {
        const it = byId.get(id)!
        if (vis(iy, it.h)) drawItem(g, it, iy - offset)
      }
      // scroll bar: where the viewport sits on the page
      const trackH = ROOM_VIEW.height - 4
      const thumbH = Math.max(8, Math.round((trackH * ROOM_VIEW.height) / flow.height))
      const thumbY = 2 + (max() === 0 ? 0 : Math.round(((trackH - thumbH) * offset) / max()))
      g.rect(ROOM_VIEW.width - 5, 2, 3, trackH, TRACK)
      g.rect(ROOM_VIEW.width - 5, thumbY, 3, thumbH, "rgb(140,136,128)")
      if (popupOpen) {
        g.rect(popup.x, popup.y, popup.w, popup.h, INK)
        g.rect(popup.x + 1, popup.y + 1, popup.w - 2, popup.h - 2, rgb(popup.color))
        g.rect(popup.x + 1, popup.y + 1, popup.w - 2, 14, rgb(darken(popup.color)))
        g.text(popup.title, popup.x + 5, popup.y + 4, INK)
        popup.lines.forEach((l, i) => g.text(l, popup.x + 8, popup.y + 22 + i * (LINE_H + 5), INK))
        g.rect(close.x, close.y, close.w, close.h, INK)
        g.rect(close.x + 1, close.y + 1, close.w - 2, close.h - 2, hovered === "popup-close" ? "rgb(255,190,190)" : PANEL)
        g.text("x", close.x + 2, close.y + 1, INK)
      }
    },
    press(x, y) {
      const hit = hitAt(x, y)
      if (!hit) return []
      if (hit.kind === "close") {
        popupOpen = false
        changed()
        return [{ kind: "respond", elementId: hit.id }, ...reveal()]
      }
      if (hit.kind === "header") {
        if (collapsed.has(hit.id)) collapsed.delete(hit.id)
        else collapsed.add(hit.id)
      } else if (hit.kind === "tab" && hit.index !== activeTab) {
        activeTab = hit.index
      } else return [] // items, the pop-up body, the tab already showing: nothing reacts
      flow = flowHideseek(h, collapsed, activeTab)
      offset = clamp(offset, 0, max())
      changed()
      return [{ kind: "respond", elementId: hit.id }, ...reveal()]
    },
    move(x, y) {
      cursor = { x, y }
      const hit = hitAt(x, y)
      hovered = hit && live(hit.kind) ? hit.id : null
      return []
    },
    release: () => [],
    scroll(dir) {
      const next = clamp(offset + (dir === "down" ? HS_SCROLL_STEP : -HS_SCROLL_STEP), 0, max())
      if (next === offset) return [] // at the end already: nothing moves
      offset = next
      changed()
      return [{ kind: "scroll", value: offset }, ...reveal()]
    },
    key: () => [],
    step(tick) {
      if (popupDone || tick < popup.tick) return []
      popupDone = true
      popupOpen = true
      changed()
      return [] // covering things never reveals them
    },
    inspect(x, y): ElementInfo | null {
      const hit = hitAt(x, y)
      if (!hit) return null
      const on = live(hit.kind)
      return { id: hit.id, interactive: on, responsive: on }
    },
    areas() {
      if (!areaCache) {
        // paint each element's on-screen rect into a mask (1 inert, 2 live); the pop-up paints last, on top
        const mask = new Uint8Array(ROOM_VIEW.width * ROOM_VIEW.height)
        const paint = (b: Box, v: number) => {
          const x0 = clamp(b.x, 0, ROOM_VIEW.width), x1 = clamp(b.x + b.w, 0, ROOM_VIEW.width)
          const y0 = clamp(b.y, 0, ROOM_VIEW.height), y1 = clamp(b.y + b.h, 0, ROOM_VIEW.height)
          for (let y = y0; y < y1; y++) mask.fill(v, y * ROOM_VIEW.width + x0, y * ROOM_VIEW.width + x1)
        }
        for (const [id, iy] of flow.items) {
          const it = byId.get(id)!
          paint({ x: it.x, y: iy - offset, w: it.w, h: it.h }, 1)
        }
        for (const s of h.sections) paint(headerBox(flow.headers.get(s.id)! - offset), 2)
        for (const t of h.tabs) paint({ ...tabBox(t), y: flow.tabY - offset }, 2)
        if (popupOpen) { paint(popup, 1); paint(close, 2) }
        let on = 0, off = 0
        for (const v of mask) { if (v === 2) on++; else if (v === 1) off++ }
        const total = mask.length
        areaCache = { interactive: on / total, responsive: on / total, unresponsive: off / total }
      }
      return { ...areaCache }
    },
    scrollState: () => ({ offset, max: max(), revealedFraction: revealed.size / h.items.length }),
    page: () => null,
  }
  return room
}
