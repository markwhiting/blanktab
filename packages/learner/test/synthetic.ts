import { FOVEA, PERIPHERY, WORLD, decodeMove, emptyObservation } from "@blanktab/interface"
import type { Action, Observation } from "@blanktab/interface"

/**
 * A synthetic world for learner tests (the world package is behind the wall): a plain background and a
 * cursor sprite. The cursor moves by the action's dx/dy (world pixels, clamped to the world), so the
 * next frame is a fixed function of the current frame and the action. Periphery is the world at 1/4
 * scale; the fovea is a 32x32 native-resolution window centred on the cursor (black outside the world).
 */
export function createSyntheticStream(opts: { background?: number; sprite?: number; start?: { x: number; y: number } } = {}) {
  const bg = opts.background ?? 60
  const sprite = opts.sprite ?? 230
  let x = opts.start?.x ?? WORLD.width / 2
  let y = opts.start?.y ?? WORLD.height / 2
  const scale = WORLD.width / PERIPHERY.width // 4

  const observe = (): Observation => {
    const o = emptyObservation()
    o.periphery.fill(bg)
    const px = Math.floor(x / scale)
    const py = Math.floor(y / scale)
    for (let yy = py - 1; yy <= py + 1; yy++) {
      for (let xx = px - 1; xx <= px + 1; xx++) {
        if (xx < 0 || yy < 0 || xx >= PERIPHERY.width || yy >= PERIPHERY.height) continue
        for (let c = 0; c < 3; c++) o.periphery[(yy * PERIPHERY.width + xx) * 3 + c] = sprite
      }
    }
    const half = FOVEA.width / 2
    for (let fy = 0; fy < FOVEA.height; fy++) {
      for (let fx = 0; fx < FOVEA.width; fx++) {
        const wx = Math.floor(x) - half + fx
        const wy = Math.floor(y) - half + fy
        const inside = wx >= 0 && wy >= 0 && wx < WORLD.width && wy < WORLD.height
        const onSprite = Math.abs(fx - half) <= 2 && Math.abs(fy - half) <= 2
        const v = !inside ? 0 : onSprite ? sprite : bg
        for (let c = 0; c < 3; c++) o.fovea[(fy * FOVEA.width + fx) * 3 + c] = v
      }
    }
    o.cursor = { x: x / (WORLD.width - 1), y: y / (WORLD.height - 1) }
    return o
  }

  const act = (a: Action) => {
    const { dx, dy } = decodeMove(a.move)
    x = Math.min(WORLD.width - 1, Math.max(0, x + dx))
    y = Math.min(WORLD.height - 1, Math.max(0, y + dy))
  }

  return { observe, act }
}
