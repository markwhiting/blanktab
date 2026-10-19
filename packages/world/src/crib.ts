import { WORLD } from "@blanktab/interface"
import { mulberry32 } from "./rng"

export const TASKBAR_HEIGHT = 16
export const BACKGROUND = "rgb(236,232,220)"

export interface Shape {
  id: string
  kind: "circle" | "square"
  x: number // bounding box top-left
  y: number
  size: number // bounding box side
  color: [number, number, number]
}

export interface Crib {
  shapes: Shape[]
  icon: { id: string; x: number; y: number; w: number; h: number }
}

const overlaps = (a: Shape, b: Shape, pad: number) =>
  a.x < b.x + b.size + pad && b.x < a.x + a.size + pad && a.y < b.y + b.size + pad && b.y < a.y + a.size + pad

/** 6-10 non-overlapping shapes above the taskbar, plus one taskbar icon. Pure function of seed. */
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
      x: 4 + Math.floor(rnd() * (WORLD.width - size - 8)),
      y: 4 + Math.floor(rnd() * (WORLD.height - TASKBAR_HEIGHT - size - 8)),
      size,
      color: [40 + Math.floor(rnd() * 170), 40 + Math.floor(rnd() * 170), 40 + Math.floor(rnd() * 170)],
    }
    if (!shapes.some(s => overlaps(s, cand, 4))) shapes.push(cand)
  }
  return { shapes, icon: { id: "taskbar-icon", x: 4, y: WORLD.height - TASKBAR_HEIGHT + 2, w: 12, h: 12 } }
}

export function shapeContains(s: Shape, px: number, py: number): boolean {
  if (px < s.x || py < s.y || px >= s.x + s.size || py >= s.y + s.size) return false
  if (s.kind === "square") return true
  const r = s.size / 2
  return (px - (s.x + r)) ** 2 + (py - (s.y + r)) ** 2 <= r * r
}

export function iconContains(i: Crib["icon"], px: number, py: number): boolean {
  return px >= i.x && py >= i.y && px < i.x + i.w && py < i.y + i.h
}
