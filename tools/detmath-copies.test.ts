// @blanktab/detmath cannot import world or tensor (it imports nothing), so it carries copies of their seed
// functions. These tests pin the copies to the originals.
import { describe, it, expect } from "vitest"
import { createRng, hashSeed } from "@blanktab/detmath"
import { createRng as tensorRng } from "@blanktab/tensor"
import { hashSeed as worldHashSeed } from "@blanktab/world"

describe("detmath copies of the seed functions", () => {
  it("hashSeed equals @blanktab/world hashSeed on 1,000 (seed, salt) pairs", () => {
    const rnd = tensorRng(0x51ed)
    for (let i = 0; i < 1000; i++) {
      const seed = Math.floor(rnd() * 2 ** 32)
      const salt = Array.from({ length: Math.floor(rnd() * 24) }, () => String.fromCharCode(32 + Math.floor(rnd() * 400))).join("")
      expect(hashSeed(seed, salt), `${seed} ${JSON.stringify(salt)}`).toBe(worldHashSeed(seed, salt))
    }
  })
  it("createRng streams equal @blanktab/tensor createRng for 1,000 draws on 20 seeds", () => {
    for (let s = 0; s < 20; s++) {
      const seed = s === 0 ? 0 : hashSeed(s, "copies")
      const a = createRng(seed), b = tensorRng(seed)
      for (let i = 0; i < 1000; i++) expect(a()).toBe(b())
    }
  })
})
