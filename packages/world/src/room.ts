import { WORLD } from "@blanktab/interface"
import type { Gfx } from "./gfx"

export type RoomKind = "crib" | "toys" | "hideseek" | "places"
/** Every kind, in default taskbar order. */
export const ROOM_KINDS: readonly RoomKind[] = ["crib", "toys", "hideseek", "places"]

export const TASKBAR_HEIGHT = 16
/** The room viewport: the screen above the taskbar, at (0,0). Room-local coords equal screen coords. */
export const ROOM_VIEW = { width: 320, height: 184 } as const
if (ROOM_VIEW.width !== WORLD.width || ROOM_VIEW.height !== WORLD.height - TASKBAR_HEIGHT) {
  throw new Error("ROOM_VIEW must be the screen above the taskbar")
}

export const inView = (x: number, y: number) => x >= 0 && y >= 0 && x < ROOM_VIEW.width && y < ROOM_VIEW.height

/** Privileged: what the harness knows about an element. Never reaches the learner. */
export interface ElementInfo { id: string; interactive: boolean; responsive: boolean }

export interface RoomEvent {
  kind: "click" | "respond" | "scroll" | "reveal" | "navigate" | "widget"
  elementId?: string
  value?: number | string
}

/**
 * A pluggable room. All coordinates are room-local (viewport ROOM_VIEW).
 * The shell calls `move` every tick the cursor is in the viewport, plus once on the tick it leaves
 * (with the outside point), so rooms can drop hover state; points outside ROOM_VIEW mean "not over the room".
 * `release` may also arrive with an outside point when a drag ends over the taskbar.
 */
export interface Room {
  readonly kind: RoomKind
  draw(g: Gfx): void
  press(x: number, y: number): RoomEvent[]
  move(x: number, y: number, held: boolean): RoomEvent[]
  release(x: number, y: number): RoomEvent[]
  scroll(dir: "up" | "down"): RoomEvent[]
  key(k: string): RoomEvent[]
  step(tick: number): RoomEvent[]
  inspect(x: number, y: number): ElementInfo | null
  areas(): { interactive: number; responsive: number; unresponsive: number }
  scrollState(): { offset: number; max: number; revealedFraction: number } | null
  page(): string | null
}

export type RoomFactory = (seed: number, words: readonly string[]) => Room
