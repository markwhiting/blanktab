import { mulberry32 } from "../rng"
import { WORDS } from "../words"
import { GLYPH_H, textWidth } from "../font"
import { ROOM_VIEW, inView, type ElementInfo, type Room, type RoomEvent, type RoomFactory } from "../room"

export const TOYS_BACKGROUND = "rgb(214,226,222)"

export interface ToyWidget {
  id: string
  kind: "button" | "toggle" | "slider" | "block"
  x: number // bounding box top-left, room-local (a block's start position)
  y: number
  w: number
  h: number
  label?: string // button caption
  knobW?: number // slider knob width
  color: [number, number, number]
}

export interface ToyLabel { id: string; text: string; x: number; y: number; w: number; h: number; scale: number }
export interface Toys { widgets: ToyWidget[]; labels: ToyLabel[] }

const PAD = 6
const BUTTON_H = 18, TOGGLE = { w: 32, h: 16 }, SLIDER = { w: 96, h: 12, knobW: 8 }, BLOCK = 24

type Box = { x: number; y: number; w: number; h: number }
const overlaps = (a: Box, b: Box, pad: number) =>
  a.x < b.x + b.w + pad && b.x < a.x + a.w + pad && a.y < b.y + b.h + pad && b.y < a.y + a.h + pad

/** 4-7 widgets (at least one of each kind) and 2-4 decorative labels, non-overlapping. Pure function of seed. */
export function layoutToys(seed: number, words: readonly string[] = WORDS): Toys {
  const rnd = mulberry32(seed)
  const int = (lo: number, hi: number) => lo + Math.floor(rnd() * (hi - lo + 1))
  const word = () => words[Math.floor(rnd() * words.length)]!
  const KINDS = ["button", "toggle", "slider", "block"] as const
  for (let round = 0; round < 200; round++) {
    const extra = int(0, 3)
    const kinds: ToyWidget["kind"][] = [...KINDS]
    for (let i = 0; i < extra; i++) kinds.push(KINDS[int(0, 3)]!)
    const nLabels = int(2, 4)
    const placed: Box[] = []
    const place = (w: number, h: number): Box | null => {
      for (let attempt = 0; attempt < 300; attempt++) {
        const b = { x: int(4, ROOM_VIEW.width - w - 4), y: int(4, ROOM_VIEW.height - h - 4), w, h }
        if (!placed.some(p => overlaps(p, b, PAD))) { placed.push(b); return b }
      }
      return null
    }
    const widgets: ToyWidget[] = []
    const labels: ToyLabel[] = []
    let ok = true
    for (const kind of kinds) {
      const color: [number, number, number] = [int(90, 220), int(90, 220), int(90, 220)]
      const label = kind === "button" ? word() : undefined
      const w = kind === "button" ? textWidth(label!) + 8 + 16 : kind === "toggle" ? TOGGLE.w : kind === "slider" ? SLIDER.w : BLOCK
      const h = kind === "button" ? BUTTON_H : kind === "toggle" ? TOGGLE.h : kind === "slider" ? SLIDER.h : BLOCK
      const b = place(w, h)
      if (!b) { ok = false; break }
      const widget: ToyWidget = { id: `${kind}-${widgets.length}`, kind, ...b, color }
      if (label) widget.label = label
      if (kind === "slider") widget.knobW = SLIDER.knobW
      widgets.push(widget)
    }
    for (let i = 0; ok && i < nLabels; i++) {
      const text = word(), scale = int(1, 2)
      const b = place(textWidth(text, scale), GLYPH_H * scale)
      if (!b) ok = false
      else labels.push({ id: `label-${i}`, text, ...b, scale })
    }
    if (ok) return { widgets, labels }
  }
  throw new Error("layoutToys: could not fit the widgets")
}

const lighten = (c: readonly number[]) => c.map(v => Math.min(255, Math.round(v + (255 - v) * 0.4)))
const darken = (c: readonly number[]) => c.map(v => Math.round(v * 0.6))
const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
const WHITE = "rgb(250,250,250)", INK = "rgb(30,30,40)", LABEL_INK = "rgb(100,110,120)"

type Drag = { id: string; dx: number; dy: number }

/**
 * Buttons, toggles, sliders and draggable blocks whose state persists for the life of the room, plus a few
 * inert word labels. Drags start only in `press` and follow the cursor until `release` (or a move that says
 * the button is no longer held), however fast the cursor leaves the element.
 */
export const toysRoom: RoomFactory = (seed, words) => {
  const { widgets, labels } = layoutToys(seed, words)
  const pos = new Map(widgets.map(w => [w.id, { x: w.x, y: w.y }]))
  const counts = new Map<string, number>()
  const on = new Set<string>()
  const knob = new Map(widgets.filter(w => w.kind === "slider").map(w => [w.id, 0]))
  let pressedId: string | null = null // the button held down
  let hovered: string | null = null
  let drag: Drag | null = null
  let areaCache: ReturnType<Room["areas"]> | null = null

  const slideMax = (w: ToyWidget) => w.w - w.knobW!
  const rectOf = (w: ToyWidget): Box => {
    const p = pos.get(w.id)!
    if (w.kind === "slider") return { x: p.x + knob.get(w.id)!, y: p.y, w: w.knobW!, h: w.h } // only the knob is live
    return { x: p.x, y: p.y, w: w.w, h: w.h }
  }
  const within = (b: Box, x: number, y: number) => x >= b.x && y >= b.y && x < b.x + b.w && y < b.y + b.h
  const blocks = widgets.filter(w => w.kind === "block")
  const others = widgets.filter(w => w.kind !== "block")

  /** Topmost widget under a point; a grabbed block is always on top. */
  const widgetAt = (x: number, y: number): ToyWidget | undefined => {
    if (!inView(x, y)) return undefined
    const grabbed = blocks.find(b => b.id === drag?.id)
    if (grabbed && within(rectOf(grabbed), x, y)) return grabbed
    for (let i = blocks.length - 1; i >= 0; i--) if (within(rectOf(blocks[i]!), x, y)) return blocks[i]
    return others.find(w => within(rectOf(w), x, y))
  }
  const labelAt = (x: number, y: number) => (inView(x, y) ? labels.find(l => within(l, x, y)) : undefined)

  const sliderValue = (w: ToyWidget) => Math.round((knob.get(w.id)! / slideMax(w)) * 100)

  const follow = (x: number, y: number) => {
    if (!drag) return
    const w = widgets.find(v => v.id === drag!.id)!
    if (w.kind === "slider") {
      knob.set(w.id, clamp(Math.round(x - drag.dx - pos.get(w.id)!.x), 0, slideMax(w)))
    } else {
      pos.set(w.id, {
        x: clamp(Math.round(x - drag.dx), 0, ROOM_VIEW.width - w.w),
        y: clamp(Math.round(y - drag.dy), 0, ROOM_VIEW.height - w.h),
      })
    }
    areaCache = null
  }

  const room: Room = {
    kind: "toys",
    draw(g) {
      g.rect(0, 0, ROOM_VIEW.width, ROOM_VIEW.height, TOYS_BACKGROUND)
      for (const l of labels) g.text(l.text, l.x, l.y, LABEL_INK, l.scale)
      const draws = [...others, ...blocks.filter(b => b.id !== drag?.id), ...blocks.filter(b => b.id === drag?.id)]
      for (const w of draws) {
        const { x, y } = pos.get(w.id)!
        const hot = hovered === w.id
        if (w.kind === "button") {
          const down = pressedId === w.id
          const face = down ? darken(w.color) : hot ? lighten(w.color) : w.color
          g.rect(x, y, w.w, w.h, down ? WHITE : INK)
          g.rect(x + (down ? 1 : 0), y + (down ? 1 : 0), w.w - 1, w.h - 1, down ? INK : WHITE)
          g.rect(x + 1, y + 1, w.w - 2, w.h - 2, rgb(face))
          const o = down ? 1 : 0
          g.text(w.label!, x + 4 + o, y + 4 + o, INK)
          const n = Math.min(99, counts.get(w.id) ?? 0)
          g.rect(x + w.w - 16 + o, y + 3 + o, 13, 12, WHITE)
          g.text(String(n), x + w.w - 14 + o, y + 4 + o, INK)
        } else if (w.kind === "toggle") {
          const isOn = on.has(w.id)
          const track = isOn ? [60, 170, 90] : [140, 140, 150]
          const c = rgb(hot ? lighten(track) : track)
          g.rect(x + w.h / 2, y, w.w - w.h, w.h, c)
          g.circle(x + w.h / 2, y + w.h / 2, w.h / 2, c)
          g.circle(x + w.w - w.h / 2, y + w.h / 2, w.h / 2, c)
          g.circle(isOn ? x + w.w - w.h / 2 : x + w.h / 2, y + w.h / 2, w.h / 2 - 2, WHITE)
        } else if (w.kind === "slider") {
          g.rect(x, y + w.h / 2 - 2, w.w, 4, "rgb(80,85,95)")
          const kx = x + knob.get(w.id)!
          g.rect(kx, y, w.knobW!, w.h, INK)
          g.rect(kx + 1, y + 1, w.knobW! - 2, w.h - 2, rgb(hot || drag?.id === w.id ? lighten(w.color) : w.color))
        } else {
          const grabbed = drag?.id === w.id
          g.rect(x - 1, y - 1, w.w + 2, w.h + 2, grabbed ? WHITE : INK)
          g.rect(x, y, w.w, w.h, rgb(hot || grabbed ? lighten(w.color) : w.color))
        }
      }
    },
    press(x, y) {
      const w = widgetAt(x, y)
      if (!w) return []
      if (w.kind === "button") {
        pressedId = w.id
        counts.set(w.id, (counts.get(w.id) ?? 0) + 1)
        return [{ kind: "respond", elementId: w.id }, { kind: "widget", elementId: w.id, value: "press" }]
      }
      if (w.kind === "toggle") {
        if (on.has(w.id)) on.delete(w.id)
        else on.add(w.id)
        return [{ kind: "respond", elementId: w.id }, { kind: "widget", elementId: w.id, value: on.has(w.id) ? 1 : 0 }]
      }
      const r = rectOf(w)
      drag = { id: w.id, dx: x - r.x, dy: y - r.y } // grab offset inside the knob / block
      return [{ kind: "respond", elementId: w.id }] // the grab is the visible reaction
    },
    move(x, y, held) {
      hovered = widgetAt(x, y)?.id ?? null
      if (drag && !held) drag = null // the release was missed; never keep dragging an unheld cursor
      follow(x, y)
      return []
    },
    release(x, y) {
      pressedId = null
      if (!drag) return []
      follow(x, y)
      const w = widgets.find(v => v.id === drag!.id)!
      drag = null
      areaCache = null
      const value = w.kind === "slider" ? sliderValue(w) : "drop"
      return [{ kind: "widget", elementId: w.id, value } satisfies RoomEvent]
    },
    scroll: () => [],
    key: () => [],
    step: () => [],
    inspect(x, y): ElementInfo | null {
      const w = widgetAt(x, y)
      if (w) return { id: w.id, interactive: true, responsive: true }
      const l = labelAt(x, y)
      return l ? { id: l.id, interactive: false, responsive: false } : null
    },
    areas() {
      if (!areaCache) {
        let live = 0
        for (let y = 0; y < ROOM_VIEW.height; y++) for (let x = 0; x < ROOM_VIEW.width; x++) if (widgetAt(x, y)) live++
        const total = ROOM_VIEW.width * ROOM_VIEW.height
        // unresponsive: labels and bare background alike
        areaCache = { interactive: live / total, responsive: live / total, unresponsive: (total - live) / total }
      }
      return { ...areaCache }
    },
    scrollState: () => null,
    page: () => null,
  }
  return room
}
