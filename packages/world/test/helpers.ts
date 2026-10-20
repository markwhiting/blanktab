// Canvas stand-in: Node tests use `@napi-rs/canvas` (prebuilt binaries, devDependency of
// packages/world) in place of OffscreenCanvas. Vitest browser mode is not used.
import { createCanvas } from "@napi-rs/canvas"
import { NOOP_ACTION, PERIPHERY, WORLD, type Action, type Observation } from "@blanktab/interface"
import { createWorld, type TraceEvent, type World } from "../src/index"

export type WorldOpts = Omit<Parameters<typeof createWorld>[0], "seed" | "canvas">
export type TestWorld = World & { screen(): Uint8ClampedArray }

/** A world on a Node canvas; `screen()` renders and returns the full-resolution RGBA screen. */
export function makeWorld(seed: number, opts: WorldOpts = {}): TestWorld {
  const canvas = createCanvas(WORLD.width, WORLD.height)
  const w = createWorld({ seed, canvas, ...opts })
  return Object.assign(w, {
    screen() {
      w.observe()
      return canvas.getContext("2d").getImageData(0, 0, WORLD.width, WORLD.height).data
    },
  })
}

export const act = (move: number, button = 0, scroll = 0, key = 0): Action => ({ ...NOOP_ACTION, move, button, scroll, key })
// move codes: 1 + dirIndex*3 + magIndex; dirs: E,NE,N,NW,W,SW,S,SE
export const code = (dir: number, mag: number) => 1 + dir * 3 + mag
export const PRESS = act(0, 1), RELEASE = act(0, 2), NOOP = act(0)

const magFor = (d: number) => (d >= 16 ? 2 : d >= 4 ? 1 : 0)

/** Drive the cursor to screen pixel (tx, ty) with axis-aligned 1/4/16 px moves (never overshoots). */
export function moveTo(w: World, tx: number, ty: number) {
  for (let guard = 0; guard < 400; guard++) {
    const { x, y } = w.state().cursor
    const dx = tx - x, dy = ty - y
    if (dx === 0 && dy === 0) return
    if (dx !== 0 && Math.abs(dx) >= Math.abs(dy)) w.act(act(code(dx > 0 ? 0 : 4, magFor(Math.abs(dx)))))
    else w.act(act(code(dy > 0 ? 6 : 2, magFor(Math.abs(dy)))))
  }
  throw new Error("moveTo did not converge")
}

export function clickAt(w: World, x: number, y: number) {
  moveTo(w, x, y)
  w.act(PRESS)
  w.act(RELEASE)
}

/** RGB of the periphery cell covering screen pixel (x, y). */
export function pixelAt(obs: Observation, x: number, y: number): number[] {
  const d = WORLD.width / PERIPHERY.width
  const o = (Math.floor(y / d) * PERIPHERY.width + Math.floor(x / d)) * 3
  return Array.from(obs.periphery.slice(o, o + 3))
}

/** RGB of full-resolution screen pixel (x, y) from `TestWorld.screen()`. */
export const rgbAt = (s: Uint8ClampedArray, x: number, y: number) => {
  const i = (y * WORLD.width + x) * 4
  return [s[i]!, s[i + 1]!, s[i + 2]!]
}

/** A trace event without its tick, for comparing events whose exact tick a multi-step move makes awkward. */
export function untimed(e: TraceEvent): Omit<TraceEvent, "tick"> {
  const { tick, ...rest } = e
  void tick
  return rest
}
