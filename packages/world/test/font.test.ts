import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { glyph, textWidth, GLYPH_W, GLYPH_H, GLYPH_BASELINE } from "../src/font"
import { createGfx, type Context2DLike } from "../src/gfx"
import { WORDS, validateWords } from "../src/words"
import { hashSeed } from "../src/rng"

describe("font", () => {
  it("has GLYPH_H-row 5-bit glyphs for the supported charset", () => {
    const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-:<>x "
    for (const ch of chars) {
      const g = glyph(ch)
      expect(g, ch).not.toBeNull()
      expect(g!.length).toBe(GLYPH_H)
      for (const row of g!) expect(row).toBeLessThan(32)
    }
  })
  it("p, y, g, q, j have tails below the baseline; no other glyph uses the descender rows", () => {
    expect(GLYPH_H).toBeGreaterThan(GLYPH_BASELINE)
    const below = (ch: string) => glyph(ch)!.slice(GLYPH_BASELINE).some(r => r !== 0)
    for (const ch of "pygqj") expect(below(ch), ch).toBe(true)
    for (const ch of "abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-:<>") expect(below(ch), ch).toBe(false)
  })
  it("g differs from 9, y from Y, p from P, q from 9 (masks)", () => {
    for (const [a, b] of [["g", "9"], ["y", "Y"], ["p", "P"], ["q", "9"], ["j", "J"]] as const) {
      expect(glyph(a), `${a}/${b}`).not.toEqual(glyph(b))
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
    expect(glyph("A")!.slice(0, 7)).toEqual([0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001])
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
    expect(px(0, 0)).toBe(0); expect(px(4, 6)).toBe(0); expect(px(2, 3)).toBe(255); expect(px(0, 8)).toBe(255)
  })
  it("text sets fillStyle once per call, not once per pixel", () => {
    const c = createCanvas(40, 10)
    const real = c.getContext("2d")
    let sets = 0
    const spy = new Proxy(real, {
      set(t, k, v) { if (k === "fillStyle") sets++; return Reflect.set(t, k, v) },
      get(t, k) { const v = Reflect.get(t, k); return typeof v === "function" ? v.bind(t) : v },
    }) as unknown as Context2DLike
    createGfx(spy).text("AB?", 0, 0, "rgb(0,0,0)")
    expect(sets).toBe(1)
    const d = real.getImageData(0, 0, 40, 10).data
    expect(d[3]).toBe(0) // A's top-left pixel is off (transparent)
    expect(d[(0 * 40 + 1) * 4 + 3]).toBe(255) // A's top row pixel 1 is on
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
  it("rejects malformed entries loudly instead of filtering them", () => {
    expect(validateWords(["cat", "dog"])).toEqual(["cat", "dog"])
    for (const bad of ["Cat", "a", "elevenletter", "e-mail", "caf\u00e9", ""]) {
      expect(() => validateWords(["cat", bad]), JSON.stringify(bad)).toThrow(/word/)
    }
  })
})

describe("hashSeed", () => {
  it("is deterministic, salt-sensitive, uint32", () => {
    expect(hashSeed(1, "toys")).not.toBe(hashSeed(1, "crib"))
    expect(hashSeed(1, "toys")).toBe(hashSeed(1, "toys"))
    expect(hashSeed(2, "toys")).not.toBe(hashSeed(1, "toys"))
    // pinned: room layouts are derived from these, so a change silently reseeds every world
    expect(hashSeed(1, "crib")).toBe(1517135794)
    expect(hashSeed(42, "toys")).toBe(3142927674)
    const h = hashSeed(123456, "x")
    expect(Number.isInteger(h) && h >= 0 && h <= 0xffffffff).toBe(true)
  })
})
