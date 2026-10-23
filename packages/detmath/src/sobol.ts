import { createRng, hashSeed } from "./seed"

// Joe & Kuo (2008) direction numbers, file new-joe-kuo-6.21201, dimensions 2..16: [s, a, m_1..m_s].
// Dimension 1 is the van der Corput sequence (all m = 1).
const JOE_KUO: [number, number, number[]][] = [
  [1, 0, [1]],
  [2, 1, [1, 3]],
  [3, 1, [1, 3, 1]],
  [3, 2, [1, 1, 1]],
  [4, 1, [1, 1, 3, 3]],
  [4, 4, [1, 3, 5, 13]],
  [5, 2, [1, 1, 5, 5, 17]],
  [5, 4, [1, 1, 5, 5, 5]],
  [5, 7, [1, 1, 7, 11, 19]],
  [5, 11, [1, 1, 5, 1, 1]],
  [5, 13, [1, 1, 1, 3, 11]],
  [5, 14, [1, 3, 5, 5, 31]],
  [6, 1, [1, 3, 3, 9, 7, 49]],
  [6, 13, [1, 1, 1, 15, 21, 21]],
  [6, 16, [1, 3, 1, 13, 27, 49]],
]
export const SOBOL_MAX_DIMS = JOE_KUO.length + 1
const BITS = 32
const TWO32 = 4294967296

/** Direction numbers V_1..V_32 (as uint32, V_i = m_i·2^(32−i)) for dimension index d (0-based). */
function directions(d: number): number[] {
  const v = new Array<number>(BITS)
  if (d === 0) {
    for (let i = 0; i < BITS; i++) v[i] = (1 << (BITS - 1 - i)) >>> 0
    return v
  }
  const [s, a, m] = JOE_KUO[d - 1]!
  for (let i = 0; i < s; i++) v[i] = (m[i]! << (BITS - 1 - i)) >>> 0 // m_i < 2^(i+1): fits in 32 bits
  for (let i = s; i < BITS; i++) {
    let x = (v[i - s]! ^ (v[i - s]! >>> s)) >>> 0
    for (let k = 1; k < s; k++) if ((a >>> (s - 1 - k)) & 1) x = (x ^ v[i - k]!) >>> 0
    v[i] = x
  }
  return v
}

/**
 * Points 1..n (point 0, the origin, is skipped) of the Sobol sequence in `dims` ≤ 16 dimensions, Joe–Kuo
 * 6.21201 direction numbers, Gray-code order. Each dimension is digitally shifted (XOR) by one uint32 drawn from
 * createRng(hashSeed(0, salt)), in dimension order, so every salt gives a different, fixed, low-discrepancy set.
 */
export function sobolPoints(dims: number, n: number, salt: string): number[][] {
  if (!(Number.isInteger(dims) && dims >= 1 && dims <= SOBOL_MAX_DIMS))
    throw new RangeError(`sobolPoints: dims must be an integer in 1..${SOBOL_MAX_DIMS}, got ${dims}`)
  if (!(Number.isInteger(n) && n >= 0 && n < TWO32))
    throw new RangeError(`sobolPoints: n must be an integer in [0, 2^32), got ${n}`)
  const rng = createRng(hashSeed(0, salt))
  const shift = Array.from({ length: dims }, () => (rng() * TWO32) >>> 0)
  const v = Array.from({ length: dims }, (_, d) => directions(d))
  const x = new Array<number>(dims).fill(0)
  const out: number[][] = []
  for (let i = 1; i <= n; i++) {
    // Gray code: point i differs from point i−1 in direction c, the index of the lowest zero bit of i−1.
    let c = 0
    let j = i - 1
    while (j & 1) { j = Math.floor(j / 2); c++ }
    out.push(x.map((xd, d) => {
      const next = (xd ^ v[d]![c]!) >>> 0
      x[d] = next
      return ((next ^ shift[d]!) >>> 0) / TWO32
    }))
  }
  return out
}
