import { FOVEA, PERIPHERY, WORLD, decodeMove, emptyObservation } from "@blanktab/interface"
import { BUTTON_OPTIONS, KEYS, MOVE_OPTIONS, SCROLL_OPTIONS } from "@blanktab/interface"
import type { Action, Observation } from "@blanktab/interface"
import { createRng } from "@blanktab/tensor"
import type { HeldOutSet } from "../src/learner"

/** Side of the bright block, in screen (world) pixels: 2x2 periphery pixels, one 40x25 target cell. */
export const BLOCK = 8

/**
 * A synthetic world for learner tests (the world package is behind the wall): a plain background and a
 * solid BLOCK x BLOCK bright block at the cursor. The block moves by the action's dx/dy (world pixels,
 * clamped so it stays on screen), so the next frame is a fixed function of the current frame and the
 * action, and the motion is visible at the prediction-target scale. The screen is rendered at world
 * resolution; the periphery is its 4x4 box average, the fovea a 32x32 native window centred on the
 * cursor (black outside the world).
 */
export function createSyntheticStream(opts: { background?: number; block?: number; start?: { x: number; y: number } } = {}) {
  const bg = opts.background ?? 40
  const bright = opts.block ?? 240
  const half = BLOCK / 2
  const clampX = (v: number) => Math.min(WORLD.width - half, Math.max(half, v))
  const clampY = (v: number) => Math.min(WORLD.height - half, Math.max(half, v))
  let x = clampX(opts.start?.x ?? WORLD.width / 2)
  let y = clampY(opts.start?.y ?? WORLD.height / 2)
  const screen = new Uint8Array(WORLD.width * WORLD.height)
  const s = WORLD.width / PERIPHERY.width // 4

  const observe = (): Observation => {
    screen.fill(bg)
    for (let yy = y - half; yy < y + half; yy++) screen.fill(bright, yy * WORLD.width + x - half, yy * WORLD.width + x + half)
    const o = emptyObservation()
    for (let py = 0; py < PERIPHERY.height; py++) {
      for (let px = 0; px < PERIPHERY.width; px++) {
        let acc = 0
        for (let dy = 0; dy < s; dy++) for (let dx = 0; dx < s; dx++) acc += screen[(py * s + dy) * WORLD.width + px * s + dx]!
        o.periphery.fill(Math.round(acc / (s * s)), (py * PERIPHERY.width + px) * 3, (py * PERIPHERY.width + px) * 3 + 3)
      }
    }
    const fh = FOVEA.width / 2
    for (let fy = 0; fy < FOVEA.height; fy++) {
      for (let fx = 0; fx < FOVEA.width; fx++) {
        const wx = x - fh + fx
        const wy = y - fh + fy
        const v = wx >= 0 && wy >= 0 && wx < WORLD.width && wy < WORLD.height ? screen[wy * WORLD.width + wx]! : 0
        o.fovea.fill(v, (fy * FOVEA.width + fx) * 3, (fy * FOVEA.width + fx) * 3 + 3)
      }
    }
    o.cursor = { x: x / (WORLD.width - 1), y: y / (WORLD.height - 1) }
    return o
  }

  const act = (a: Action) => {
    const { dx, dy } = decodeMove(a.move)
    x = clampX(x + dx)
    y = clampY(y + dy)
  }

  return { observe, act }
}

/** A held-out trajectory of the synthetic stream under uniformly random actions (decision 0015 §3); shared by the GPU parity and golden tests. */
export function heldOutSet(): HeldOutSet {
  const w = createSyntheticStream({ start: { x: 60, y: 60 } })
  const rnd = createRng(5)
  const pick = (k: number) => Math.min(k - 1, Math.floor(rnd() * k))
  const obs = [w.observe()]
  const actions: Action[] = []
  for (let i = 0; i < 16; i++) {
    const a = { move: pick(MOVE_OPTIONS), button: pick(BUTTON_OPTIONS), scroll: pick(SCROLL_OPTIONS), key: pick(KEYS.length) }
    w.act(a); actions.push(a); obs.push(w.observe())
  }
  return { obs, actions }
}
