import { GLYPH_ADVANCE, GLYPH_H, GLYPH_W, glyph } from "./font"

/** The minimal 2D-context subset the world draws with (fits OffscreenCanvas and @napi-rs/canvas). */
export interface Context2DLike {
  fillStyle: string | CanvasGradient | CanvasPattern
  fillRect(x: number, y: number, w: number, h: number): void
  beginPath(): void
  arc(x: number, y: number, r: number, start: number, end: number): void
  moveTo(x: number, y: number): void
  lineTo(x: number, y: number): void
  closePath(): void
  fill(): void
  save(): void
  restore(): void
  rect(x: number, y: number, w: number, h: number): void
  clip(): void
  getImageData(sx: number, sy: number, sw: number, sh: number): { data: Uint8ClampedArray }
}

export interface Gfx {
  rect(x: number, y: number, w: number, h: number, color: string): void
  circle(cx: number, cy: number, r: number, color: string): void
  /** Unsupported chars render as a 5x7 box outline. */
  text(s: string, x: number, y: number, color: string, scale?: number): void
  /** Offset-free clipping region: coordinates inside `draw` stay absolute. */
  clip(x: number, y: number, w: number, h: number, draw: () => void): void
}

const BOX = [0b11111, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11111, 0, 0]

export function createGfx(ctx: Context2DLike): Gfx {
  const rect = (x: number, y: number, w: number, h: number, color: string) => {
    ctx.fillStyle = color
    ctx.fillRect(x, y, w, h)
  }
  return {
    rect,
    circle(cx, cy, r, color) {
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fill()
    },
    text(s, x, y, color, scale = 1) {
      ctx.fillStyle = color
      let cx = x
      for (const ch of s) {
        const rows = glyph(ch) ?? BOX
        for (let r = 0; r < GLYPH_H; r++) {
          for (let c = 0; c < GLYPH_W; c++) {
            if ((rows[r]! >> (GLYPH_W - 1 - c)) & 1) ctx.fillRect(cx + c * scale, y + r * scale, scale, scale)
          }
        }
        cx += GLYPH_ADVANCE * scale
      }
    },
    clip(x, y, w, h, draw) {
      ctx.save()
      ctx.beginPath()
      ctx.rect(x, y, w, h)
      ctx.clip()
      try { draw() } finally { ctx.restore() }
    },
  }
}
