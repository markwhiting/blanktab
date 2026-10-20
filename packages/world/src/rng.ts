// mulberry32. Duplicated in @blanktab/learner: packages cannot share code across the wall.
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Deterministic 32-bit unsigned mix of a numeric seed and a string salt (FNV-1a over salt, then a mulberry step). */
export function hashSeed(seed: number, salt: string): number {
  let h = (0x811c9dc5 ^ (seed >>> 0)) >>> 0
  for (let i = 0; i < salt.length; i++) h = Math.imul(h ^ salt.charCodeAt(i), 0x01000193) >>> 0
  h = (h + 0x6d2b79f5) >>> 0
  let t = Math.imul(h ^ (h >>> 15), h | 1)
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
  return (t ^ (t >>> 14)) >>> 0
}
