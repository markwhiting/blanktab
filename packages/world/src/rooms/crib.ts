import { mulberry32 } from "../rng"
import { ROOM_VIEW, inView, type ElementInfo, type Room, type RoomFactory } from "../room"

export const BACKGROUND = "rgb(236,232,220)"

export interface Shape {
  id: string
  kind: "circle" | "square"
  x: number // bounding box top-left, room-local
  y: number
  size: number // bounding box side
  color: [number, number, number]
}

export interface Crib { shapes: Shape[] }

const overlaps = (a: Shape, b: Shape, pad: number) =>
  a.x < b.x + b.size + pad && b.x < a.x + a.size + pad && a.y < b.y + b.size + pad && b.y < a.y + a.size + pad

/** 6-10 non-overlapping shapes inside the room viewport. Pure function of seed. */
export function layoutCrib(seed: number): Crib {
  const rnd = mulberry32(seed)
  const count = 6 + Math.floor(rnd() * 5)
  const shapes: Shape[] = []
  let attempts = 0
  while (shapes.length < count && attempts++ < 5000) {
    const size = 28 + Math.floor(rnd() * 17)
    const cand: Shape = {
      id: `shape-${shapes.length}`,
      kind: rnd() < 0.5 ? "circle" : "square",
      x: 4 + Math.floor(rnd() * (ROOM_VIEW.width - size - 8)),
      y: 4 + Math.floor(rnd() * (ROOM_VIEW.height - size - 8)),
      size,
      color: [40 + Math.floor(rnd() * 170), 40 + Math.floor(rnd() * 170), 40 + Math.floor(rnd() * 170)],
    }
    if (!shapes.some(s => overlaps(s, cand, 4))) shapes.push(cand)
  }
  return { shapes }
}

export function shapeContains(s: Shape, px: number, py: number): boolean {
  if (px < s.x || py < s.y || px >= s.x + s.size || py >= s.y + s.size) return false
  if (s.kind === "square") return true
  const r = s.size / 2
  return (px - (s.x + r)) ** 2 + (py - (s.y + r)) ** 2 <= r * r
}

const lighten = (c: readonly number[]) => c.map(v => Math.min(255, Math.round(v + (255 - v) * 0.4)))
const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`

/** Sparse shapes that lighten on hover and invert colour on press. Every shape is interactive and responsive. */
export const cribRoom: RoomFactory = seed => {
  const { shapes } = layoutCrib(seed)
  const colors = shapes.map(s => [...s.color])
  let hovered: string | null = null

  const shapeAt = (x: number, y: number) => (inView(x, y) ? shapes.find(s => shapeContains(s, x, y)) : undefined)

  // Exact pixel count of the shapes (they never overlap); constant because shapes never move.
  let covered = 0
  for (let y = 0; y < ROOM_VIEW.height; y++) for (let x = 0; x < ROOM_VIEW.width; x++) if (shapeAt(x, y)) covered++
  const fraction = covered / (ROOM_VIEW.width * ROOM_VIEW.height)

  const room: Room = {
    kind: "crib",
    draw(g) {
      g.rect(0, 0, ROOM_VIEW.width, ROOM_VIEW.height, BACKGROUND)
      shapes.forEach((s, i) => {
        const c = rgb(hovered === s.id ? lighten(colors[i]!) : colors[i]!)
        if (s.kind === "square") g.rect(s.x, s.y, s.size, s.size, c)
        else g.circle(s.x + s.size / 2, s.y + s.size / 2, s.size / 2, c)
      })
    },
    press(x, y) {
      const s = shapeAt(x, y)
      if (!s) return []
      const i = shapes.indexOf(s)
      colors[i] = colors[i]!.map(v => 255 - v)
      return [{ kind: "click", elementId: s.id }, { kind: "respond", elementId: s.id }]
    },
    move(x, y) {
      hovered = shapeAt(x, y)?.id ?? null
      return []
    },
    release: () => [],
    scroll: () => [],
    key: () => [],
    step: () => [],
    inspect(x, y): ElementInfo | null {
      const s = shapeAt(x, y)
      return s ? { id: s.id, interactive: true, responsive: true } : null
    },
    areas: () => ({ interactive: fraction, responsive: fraction, unresponsive: 0 }),
    scrollState: () => null,
    page: () => null,
  }
  return room
}
