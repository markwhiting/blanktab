// The statistics behind the cohort evaluators (spec §2.5, §2.6): rank tests, the ICC, Fisher's exact test and Holm.
// Pure, and all maths through @blanktab/detmath (+ − × ÷ and Math.sqrt otherwise), so results do not depend on the engine.
import { chiSquareSf, dexp, lnGamma, normCdf } from "@blanktab/detmath"

/** signal-check's looks (spec §4.6): at 30, 45 and 60 calibration lives, at most 3. */
export const SIGNAL_LOOKS = [30, 45, 60] as const
/** Each look's α: 0.05 split over the three looks (Bonferroni), so repeated looks do not manufacture signal. */
export const ALPHA_LOOK = 0.05 / 3
/** The least ICC of the search score that counts as signal (spec §2.6 [guess]). */
export const ICC_MIN = 0.1

/** 1-based mid-ranks: tied values share the mean of their positions. Values must be finite. */
export function midRanks(xs: readonly number[]): number[] {
  if (!xs.every(Number.isFinite)) throw new Error("midRanks needs finite values")
  const order = xs.map((_, i) => i).sort((a, b) => xs[a]! - xs[b]! || a - b)
  const ranks = new Array<number>(xs.length)
  for (let i = 0; i < order.length;) {
    let j = i
    while (j + 1 < order.length && xs[order[j + 1]!] === xs[order[i]!]) j++
    const r = (i + j + 2) / 2
    for (let k = i; k <= j; k++) ranks[order[k]!] = r
    i = j + 1
  }
  return ranks
}

/** Σ (t³ − t) over the tie groups of `xs` (t = a tie group's size), for the tie corrections. */
function tieSum(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  let sum = 0
  for (let i = 0; i < s.length;) {
    let j = i
    while (j + 1 < s.length && s[j + 1] === s[i]) j++
    const t = j - i + 1
    sum += t * t * t - t
    i = j + 1
  }
  return sum
}

/**
 * The Kruskal–Wallis test across `groups` (each non-empty, at least 2): H = 12/(N(N+1))·Σ R_g²/n_g − 3(N+1), divided
 * by the tie correction 1 − Σ(t³ − t)/(N³ − N); p = P(χ²_{k−1} > H). All values tied: H = 0, p = 1.
 */
export function kruskalWallis(groups: readonly (readonly number[])[]): { H: number; df: number; p: number } {
  if (groups.length < 2 || groups.some(g => g.length === 0)) throw new Error("kruskalWallis needs at least 2 non-empty groups")
  const pooled = groups.flat()
  const N = pooled.length
  const ranks = midRanks(pooled)
  let sum = 0, at = 0
  for (const g of groups) {
    let R = 0
    for (let i = 0; i < g.length; i++) R += ranks[at + i]!
    at += g.length
    sum += (R * R) / g.length
  }
  const df = groups.length - 1
  const correction = 1 - tieSum(pooled) / (N * N * N - N)
  if (!(correction > 0)) return { H: 0, df, p: 1 }
  const raw = (12 / (N * (N + 1))) * sum - 3 * (N + 1)
  const H = raw > 0 ? raw / correction : 0 // rounding can leave a tiny negative where H is 0
  return { H, df, p: chiSquareSf(H, df) }
}

/**
 * ICC(1) of one-way ANOVA over `groups` (unequal sizes): (MSB − MSW)/(MSB + (n0 − 1)·MSW), with
 * n0 = (N − Σ n_g²/N)/(k − 1). A negative estimate reads as 0 (agent's call: no between-group share). Zero
 * within-group variance with groups that differ gives 1; all values equal gives 0. Needs k ≥ 2 and N > k.
 */
export function icc1(groups: readonly (readonly number[])[]): number {
  const k = groups.length
  const N = groups.reduce((s, g) => s + g.length, 0)
  if (k < 2 || groups.some(g => g.length === 0) || N <= k) throw new Error("icc1 needs at least 2 non-empty groups and more values than groups")
  if (!groups.every(g => g.every(Number.isFinite))) throw new Error("icc1 needs finite values")
  let total = 0
  for (const g of groups) for (const x of g) total += x
  const grand = total / N
  let ssb = 0, ssw = 0, sumN2 = 0
  for (const g of groups) {
    let s = 0
    for (const x of g) s += x
    const m = s / g.length
    ssb += g.length * (m - grand) * (m - grand)
    for (const x of g) ssw += (x - m) * (x - m)
    sumN2 += g.length * g.length
  }
  const msb = ssb / (k - 1), msw = ssw / (N - k)
  const n0 = (N - sumN2 / N) / (k - 1)
  const den = msb + (n0 - 1) * msw
  if (!(den > 0)) return 0
  const icc = (msb - msw) / den
  return icc > 0 ? (icc < 1 ? icc : 1) : 0
}

/** ln C(n, r). */
const lnChoose = (n: number, r: number): number => lnGamma(n + 1) - lnGamma(r + 1) - lnGamma(n - r + 1)

/**
 * One-sided Fisher exact test of the 2×2 table [[a, b], [c, d]] (rows: arm; columns: pass, non-pass): P(X ≥ a) for
 * X hypergeometric with the table's margins, i.e. the chance the first row passes at least this often by luck.
 */
export function fisherOneSided(a: number, b: number, c: number, d: number): number {
  if (![a, b, c, d].every(v => Number.isInteger(v) && v >= 0)) throw new Error("fisherOneSided needs non-negative integer counts")
  const n1 = a + b, m = a + c, N = a + b + c + d
  const hi = n1 < m ? n1 : m
  const lnDen = lnChoose(N, n1)
  let p = 0
  for (let x = a; x <= hi; x++) p += dexp(lnChoose(m, x) + lnChoose(N - m, n1 - x) - lnDen)
  return p < 1 ? p : 1
}

/**
 * Holm's step-down procedure at family level `alpha`: sorted ascending, the i-th smallest p (0-based) is rejected if
 * p ≤ alpha/(m − i) and every smaller one was; the first that is not stops it. Returns rejections in input order.
 */
export function holm(ps: readonly number[], alpha: number): boolean[] {
  const m = ps.length
  const order = ps.map((_, i) => i).sort((x, y) => ps[x]! - ps[y]! || x - y)
  const out = new Array<boolean>(m).fill(false)
  for (let i = 0; i < m; i++) {
    const j = order[i]!
    if (!(ps[j]! <= alpha / (m - i))) break
    out[j] = true
  }
  return out
}

/**
 * One-sided Mann–Whitney U: the p-value for "x tends to exceed y", by the normal approximation with the tie-corrected
 * variance and a continuity correction of 0.5 [agent]. U = R_x − n_x(n_x + 1)/2; p = Φ(−(U − n_x·n_y/2 − 0.5)/σ),
 * σ² = n_x·n_y/12·((N + 1) − Σ(t³ − t)/(N(N − 1))). Every value tied (σ = 0): p = 1.
 */
export function mannWhitneyOneSided(x: readonly number[], y: readonly number[]): number {
  if (x.length === 0 || y.length === 0) throw new Error("mannWhitneyOneSided needs two non-empty samples")
  const nx = x.length, ny = y.length, N = nx + ny
  const ranks = midRanks([...x, ...y])
  let R = 0
  for (let i = 0; i < nx; i++) R += ranks[i]!
  const U = R - (nx * (nx + 1)) / 2
  const variance = ((nx * ny) / 12) * (N + 1 - tieSum([...x, ...y]) / (N * (N - 1)))
  if (!(variance > 0)) return 1
  const z = (U - (nx * ny) / 2 - 0.5) / Math.sqrt(variance)
  return normCdf(-z)
}
