import {
  FOVEA, PERIPHERY, WORLD, isValidAction, decodeMove,
  type Action, type Observation, type WorldView,
} from "@blanktab/interface"
import type { Context2DLike } from "./gfx"
import { BACKGROUND, TASKBAR_HEIGHT, iconContains, layoutCrib, shapeContains, type Crib } from "./crib"

export interface TraceEvent { tick: number; kind: "hover" | "click" | "respond"; elementId: string }

export interface World extends WorldView {
  readonly tick: number
  drainTrace(): TraceEvent[] // privileged; harness only
}

export type { Context2DLike }

export interface CanvasLike {
  width: number
  height: number
  getContext(type: "2d"): Context2DLike | null
}

const DOWN = WORLD.width / PERIPHERY.width // 4
const lighten = (c: readonly number[]) => c.map(v => Math.min(255, Math.round(v + (255 - v) * 0.4)))
const rgb = (c: readonly number[]) => `rgb(${c[0]},${c[1]},${c[2]})`

export function createWorld(opts: { seed: number; canvas: CanvasLike }): World {
  const { canvas } = opts
  canvas.width = WORLD.width
  canvas.height = WORLD.height
  const ctx = canvas.getContext("2d")
  if (!ctx) throw new Error("2d context unavailable")

  const crib: Crib = layoutCrib(opts.seed)
  const colors = crib.shapes.map(s => [...s.color])
  let cx = Math.floor(WORLD.width / 2)
  let cy = Math.floor(WORLD.height / 2)
  let held = false
  let tick = 0
  let hovered: string | null = null
  let trace: TraceEvent[] = []

  const elementAt = (): string | null => {
    for (const s of crib.shapes) if (shapeContains(s, cx, cy)) return s.id
    return iconContains(crib.icon, cx, cy) ? crib.icon.id : null
  }

  function render() {
    ctx!.fillStyle = BACKGROUND
    ctx!.fillRect(0, 0, WORLD.width, WORLD.height)
    crib.shapes.forEach((s, i) => {
      const c = colors[i]!
      ctx!.fillStyle = rgb(hovered === s.id ? lighten(c) : c)
      if (s.kind === "square") ctx!.fillRect(s.x, s.y, s.size, s.size)
      else {
        ctx!.beginPath()
        ctx!.arc(s.x + s.size / 2, s.y + s.size / 2, s.size / 2, 0, Math.PI * 2)
        ctx!.fill()
      }
    })
    ctx!.fillStyle = "rgb(60,60,70)"
    ctx!.fillRect(0, WORLD.height - TASKBAR_HEIGHT, WORLD.width, TASKBAR_HEIGHT)
    const ic = crib.icon
    ctx!.fillStyle = hovered === ic.id ? "rgb(255,230,120)" : "rgb(200,170,60)"
    ctx!.fillRect(ic.x, ic.y, ic.w, ic.h)
    // cursor arrow, tip at (cx, cy)
    ctx!.fillStyle = "rgb(0,0,0)"
    ctx!.beginPath()
    ctx!.moveTo(cx, cy)
    ctx!.lineTo(cx, cy + 10)
    ctx!.lineTo(cx + 3, cy + 7)
    ctx!.lineTo(cx + 6, cy + 7)
    ctx!.closePath()
    ctx!.fill()
  }

  return {
    get tick() { return tick },

    act(a: Action) {
      if (!isValidAction(a)) throw new RangeError("invalid action")
      tick++
      const { dx, dy } = decodeMove(a.move)
      cx = Math.min(WORLD.width - 1, Math.max(0, cx + dx))
      cy = Math.min(WORLD.height - 1, Math.max(0, cy + dy))
      const over = elementAt()
      if (over !== hovered) {
        hovered = over
        if (over) trace.push({ tick, kind: "hover", elementId: over })
      }
      if (a.button === 1 && !held) {
        held = true
        if (over) {
          trace.push({ tick, kind: "click", elementId: over })
          const i = crib.shapes.findIndex(s => s.id === over)
          if (i >= 0) {
            colors[i] = colors[i]!.map(v => 255 - v)
            trace.push({ tick, kind: "respond", elementId: over })
          }
        }
      } else if (a.button === 2) held = false
    },

    observe(): Observation {
      render()
      const { data } = ctx.getImageData(0, 0, WORLD.width, WORLD.height)
      const periphery = new Uint8Array(PERIPHERY.width * PERIPHERY.height * 3)
      const n = DOWN * DOWN
      for (let py = 0; py < PERIPHERY.height; py++) {
        for (let px = 0; px < PERIPHERY.width; px++) {
          let r = 0, g = 0, b = 0
          for (let yy = 0; yy < DOWN; yy++) {
            for (let xx = 0; xx < DOWN; xx++) {
              const i = ((py * DOWN + yy) * WORLD.width + px * DOWN + xx) * 4
              r += data[i]!; g += data[i + 1]!; b += data[i + 2]!
            }
          }
          const o = (py * PERIPHERY.width + px) * 3
          periphery[o] = Math.round(r / n); periphery[o + 1] = Math.round(g / n); periphery[o + 2] = Math.round(b / n)
        }
      }
      const fovea = new Uint8Array(FOVEA.width * FOVEA.height * 3) // zero = black padding
      const x0 = cx - FOVEA.width / 2, y0 = cy - FOVEA.height / 2
      for (let fy = 0; fy < FOVEA.height; fy++) {
        for (let fx = 0; fx < FOVEA.width; fx++) {
          const sx = x0 + fx, sy = y0 + fy
          if (sx < 0 || sy < 0 || sx >= WORLD.width || sy >= WORLD.height) continue
          const i = (sy * WORLD.width + sx) * 4, o = (fy * FOVEA.width + fx) * 3
          fovea[o] = data[i]!; fovea[o + 1] = data[i + 1]!; fovea[o + 2] = data[i + 2]!
        }
      }
      return { periphery, fovea, cursor: { x: cx / WORLD.width, y: cy / WORLD.height }, buttonHeld: held }
    },

    drainTrace() {
      const out = trace
      trace = []
      return out
    },
  }
}
