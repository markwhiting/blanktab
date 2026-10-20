/** A seeded uniform generator on [0,1) whose position can be saved and restored (for checkpoints). */
export interface Rng {
  (): number
  /** The generator's whole state: a uint32. */
  getState(): number
  /** Restores a state from getState(); the next draws repeat the sequence from that point. */
  setState(state: number): void
}

/** mulberry32: deterministic uniform generator on [0,1). */
export function createRng(seed: number): Rng {
  let s = seed >>> 0
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0
    let t = s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  return Object.assign(next, {
    getState: () => s,
    setState(state: number) {
      if (!(Number.isInteger(state) && state >= 0 && state < 2 ** 32)) throw new RangeError(`Rng: state must be a uint32, got ${state}`)
      s = state
    },
  })
}

/** n standard-normal samples via Box-Muller. */
export function randn(rng: () => number, n: number): Float32Array {
  const out = new Float32Array(n)
  for (let i = 0; i < n; i += 2) {
    const u1 = 1 - rng() // (0,1]
    const u2 = rng()
    const r = Math.sqrt(-2 * Math.log(u1))
    out[i] = r * Math.cos(2 * Math.PI * u2)
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * u2)
  }
  return out
}
