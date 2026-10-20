import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { glyph, textWidth, GLYPH_W, GLYPH_H } from "../src/font"
import { createGfx } from "../src/gfx"
import { WORDS } from "../src/words"
import { hashSeed } from "../src/rng"

describe("font", () => {
  it("has 7-row 5-bit glyphs for the supported charset", () => {
    const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-:<>x "
    for (const ch of chars) {
      const g = glyph(ch)
      expect(g, ch).not.toBeNull()
      expect(g!.length).toBe(7)
      for (const row of g!) expect(row).toBeLessThan(32)
    }
  })
  it("returns null for unsupported chars", () => expect(glyph("é")).toBeNull())
  it("measures text width", () => {
    expect(textWidth("abc")).toBe(17)
    expect(textWidth("abc", 2)).toBe(34)
  })
  it("draws exactly glyph A's pixels", () => {
    const c = createCanvas(10, 10)
    const ctx = c.getContext("2d")
    ctx.fillStyle = "rgb(255,255,255)"
    ctx.fillRect(0, 0, 10, 10)
    createGfx(ctx).text("A", 0, 0, "rgb(0,0,0)")
    const d = ctx.getImageData(0, 0, 10, 10).data
    const mask = glyph("A")!
    for (let y = 0; y < 10; y++) for (let x = 0; x < 10; x++) {
      const on = y < GLYPH_H && x < GLYPH_W && ((mask[y]! >> (GLYPH_W - 1 - x)) & 1) === 1
      expect(d[(y * 10 + x) * 4]!, `${x},${y}`).toBe(on ? 0 : 255)
    }
  })
  it("draws unsupported chars as a box outline", () => {
    const c = createCanvas(10, 10)
    const ctx = c.getContext("2d")
    ctx.fillStyle = "rgb(255,255,255)"
    ctx.fillRect(0, 0, 10, 10)
    createGfx(ctx).text("é", 0, 0, "rgb(0,0,0)")
    const d = ctx.getImageData(0, 0, 10, 10).data
    const px = (x: number, y: number) => d[(y * 10 + x) * 4]
    expect(px(0, 0)).toBe(0); expect(px(4, 6)).toBe(0); expect(px(2, 3)).toBe(255)
  })
  it("clips drawing", () => {
    const c = createCanvas(10, 10)
    const ctx = c.getContext("2d")
    ctx.fillStyle = "rgb(255,255,255)"
    ctx.fillRect(0, 0, 10, 10)
    const g = createGfx(ctx)
    g.clip(2, 2, 3, 3, () => g.rect(0, 0, 10, 10, "rgb(0,0,0)"))
    const d = ctx.getImageData(0, 0, 10, 10).data
    expect(d[(3 * 10 + 3) * 4]).toBe(0)
    expect(d[(1 * 10 + 1) * 4]).toBe(255)
    expect(d[(6 * 10 + 6) * 4]).toBe(255)
  })
})

describe("words", () => {
  it("has >=500 unique lowercase words", () => {
    expect(new Set(WORDS).size).toBe(WORDS.length)
    expect(WORDS.length).toBeGreaterThanOrEqual(500)
    for (const w of WORDS) expect(w).toMatch(/^[a-z]{2,10}$/)
  })
})

describe("hashSeed", () => {
  it("is deterministic, salt-sensitive, uint32", () => {
    expect(hashSeed(1, "toys")).not.toBe(hashSeed(1, "crib"))
    expect(hashSeed(1, "toys")).toBe(hashSeed(1, "toys"))
    expect(hashSeed(2, "toys")).not.toBe(hashSeed(1, "toys"))
    const h = hashSeed(123456, "x")
    expect(Number.isInteger(h) && h >= 0 && h <= 0xffffffff).toBe(true)
  })
})
