// Deterministic elementary functions. Every floating-point step is +, −, ×, ÷ or Math.sqrt, which IEEE-754
// rounds exactly in every JS engine (and JS never fuses multiply-adds), so the same inputs give the same bits in
// Node and in every browser. Math.exp, Math.log and Math.pow are not used: engines may implement them differently.
// Loops run in a fixed order with fixed iteration caps.

// ln 2 split so that k·LN2_HI is exact for |k| < 2^20 (LN2_HI has its low 32 bits zero); fdlibm's constants.
const LN2_HI = 0.6931471803691238
const LN2_LO = 1.9082149292705877e-10
const INV_LN2 = 1.4426950408889634

/** 2^k for an integer k in [−1022, 1023], by exponentiation by squaring. Every product is a power of two: exact. */
function pow2(k: number): number {
  let base = k < 0 ? 0.5 : 2
  let n = k < 0 ? -k : k
  let out = 1
  while (n > 0) {
    if (n & 1) out *= base
    base *= base
    n >>>= 1
  }
  return out
}

/** y·2^k with a single final rounding, for k in [−1074 − 60, 1024 + 60]. */
function scale(y: number, k: number): number {
  if (k > 1023) return y * pow2(k - 1023) * pow2(1023)
  if (k < -1022) return y * pow2(k + 60) * pow2(-60) // first product is exact (normal); the second rounds once
  return y * pow2(k)
}

// 1/n! for n = 0..13, each a correctly rounded quotient.
const INV_FACT = [
  1, 1, 0.5, 1 / 6, 1 / 24, 1 / 120, 1 / 720, 1 / 5040, 1 / 40320, 1 / 362880, 1 / 3628800, 1 / 39916800,
  1 / 479001600, 1 / 6227020800,
]

/** e^x. Range reduction x = k·ln2 + r with |r| ≤ ln2/2, a degree-13 Taylor series for e^r, then scaling by 2^k. */
export function dexp(x: number): number {
  if (x !== x) return Number.NaN
  if (x > 709.782712893384) return Infinity
  if (x < -745.1332191019412) return 0
  const k = Math.floor(x * INV_LN2 + 0.5)
  const hi = x - k * LN2_HI // exact: k·LN2_HI is exact and close to x
  const lo = k * LN2_LO
  const r = hi - lo
  // e^r = 1 + r + r²·(1/2! + r/3! + … + r^11/13!). The tail s is small next to 1 + r, so its rounding barely
  // matters; 1 + r is formed in double-double (h + e) so the only significant rounding is the last addition.
  let q = INV_FACT[13]!
  for (let n = 12; n >= 2; n--) q = q * r + INV_FACT[n]!
  const s = r * r * q
  const rErr = (hi - r) - lo // the part of hi − lo lost when rounding r
  const h = 1 + r
  const e = (1 - h) + r
  return scale(h + (e + (s + rErr)), k)
}

const SQRT2 = 1.4142135623730951
const TWO32 = 4294967296

/** ln x for x > 0 (NaN for x < 0 or NaN, −Infinity at 0, Infinity at Infinity). */
export function dlog(x: number): number {
  if (!(x > 0)) return x === 0 ? -Infinity : Number.NaN
  if (x === Infinity) return Infinity
  // x = m·2^e with m ∈ [√½, √2): scale by powers of two (exact), coarse steps first.
  let m = x
  let e = 0
  while (m >= TWO32) { m /= TWO32; e += 32 }
  while (m < 1 / TWO32) { m *= TWO32; e -= 32 }
  while (m >= SQRT2) { m /= 2; e += 1 }
  while (m < SQRT2 / 2) { m *= 2; e -= 1 }
  // ln m = 2·atanh(s) with s = f/(2+f) = (m−1)/(m+1), f = m − 1 (exact). Written as fdlibm does,
  // ln(1+f) = f − (f²/2 − s·(f²/2 + R)) with R = Σ_{j≥1} 2·s^{2j}/(2j+1), so the large part f stays exact.
  const f = m - 1
  const s = f / (2 + f)
  const z = s * s
  let R = 0
  for (let j = 13; j >= 1; j--) R = (R + 2 / (2 * j + 1)) * z
  const hfsq = 0.5 * f * f
  return e * LN2_HI - ((hfsq - (s * (hfsq + R) + e * LN2_LO)) - f)
}

// Lanczos approximation, g = 7, n = 9 (the widely used coefficient set).
const LANCZOS_G = 7
const LANCZOS = [
  0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406,
  12.507343278686905, -0.13857109526572012, 0.000009984369578019572, 1.5056327351493116e-7,
]
const HALF_LN_2PI = 0.9189385332046728

/** ln Γ(x) for x > 0 (NaN otherwise). For x < ½ it uses ln Γ(x) = ln Γ(x+1) − ln x (no reflection: no sin). */
export function lnGamma(x: number): number {
  if (!(x > 0)) return Number.NaN
  if (x === Infinity) return Infinity
  if (x < 0.5) return lnGamma(x + 1) - dlog(x)
  const y = x - 1
  let a = LANCZOS[0]!
  for (let i = 1; i < LANCZOS.length; i++) a += LANCZOS[i]! / (y + i)
  const t = y + LANCZOS_G + 0.5
  return HALF_LN_2PI + (y + 0.5) * dlog(t) - t + dlog(a)
}

const GAMMA_EPS = 2.220446049250313e-16
const GAMMA_MAX_ITER = 10_000
const FPMIN = 1e-300

/** x^a·e^−x/Γ(a), the common prefactor of both incomplete-gamma expansions, computed in log space. */
function gammaPrefactor(a: number, x: number): number {
  return dexp(a * dlog(x) - x - lnGamma(a))
}

/** Regularized upper incomplete gamma Q(a, x) = Γ(a, x)/Γ(a), for a > 0, x ≥ 0. */
function gammaQ(a: number, x: number): number {
  if (x === 0) return 1
  if (x === Infinity) return 0
  if (x < a + 1) {
    // Series: P(a, x) = x^a e^−x / Γ(a+1) · Σ x^n / ((a+1)…(a+n)).
    let ap = a
    let del = 1 / a
    let sum = del
    for (let n = 0; n < GAMMA_MAX_ITER; n++) {
      ap += 1
      del *= x / ap
      sum += del
      if (Math.abs(del) < Math.abs(sum) * GAMMA_EPS) break
    }
    return 1 - sum * gammaPrefactor(a, x)
  }
  // Continued fraction (modified Lentz), Numerical Recipes gcf.
  let b = x + 1 - a
  let c = 1 / FPMIN
  let d = 1 / b
  let h = d
  for (let i = 1; i <= GAMMA_MAX_ITER; i++) {
    const an = -i * (i - a)
    b += 2
    d = an * d + b
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = b + an / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    const del = d * c
    h *= del
    if (Math.abs(del - 1) < GAMMA_EPS) break
  }
  return gammaPrefactor(a, x) * h
}

/** P(χ²_df > x): the regularized upper gamma Q(df/2, x/2). 1 for x ≤ 0; NaN for df ≤ 0 or NaN inputs. */
export function chiSquareSf(x: number, df: number): number {
  if (x !== x || !(df > 0)) return Number.NaN
  if (x <= 0) return 1
  return gammaQ(df / 2, x / 2)
}

/** Φ(x), the standard normal CDF: Φ(−|x|) = ½·Q(½, x²/2), computed on the tail side so small values keep
 *  their relative precision. */
export function normCdf(x: number): number {
  if (x !== x) return Number.NaN
  const tail = 0.5 * gammaQ(0.5, 0.5 * x * x) // Φ(−|x|)
  return x < 0 ? tail : 1 - tail
}

// Wichura (1988), Algorithm AS 241, PPND16: Φ⁻¹ to about 1e-16 relative.
const A = [
  3.3871328727963665, 133.14166789178438, 1971.5909503065513, 13731.69376550946,
  45921.95393154987, 67265.7709270087, 33430.57558358813, 2509.0809287301227,
]
const B = [
  1, 42.31333070160091, 687.1870074920579, 5394.196021424751,
  21213.794301586597, 39307.89580009271, 28729.085735721943, 5226.495278852854,
]
const C = [
  1.4234371107496835, 4.630337846156546, 5.769497221460691, 3.6478483247632045,
  1.2704582524523684, 0.2417807251774506, 0.022723844989269184, 0.0007745450142783414,
]
const D = [
  1, 2.053191626637759, 1.6763848301838038, 0.6897673349851,
  0.14810397642748008, 0.015198666563616457, 0.0005475938084995345, 1.0507500716444169e-9,
]
const E = [
  6.657904643501103, 5.463784911164114, 1.7848265399172913, 0.29656057182850487,
  0.026532189526576124, 0.0012426609473880784, 0.000027115555687434876, 2.0103343992922881e-7,
]
const F = [
  1, 0.599832206555888, 0.1369298809227358, 0.014875361290850615,
  0.0007868691311456133, 0.000018463183175100548, 1.421511758316446e-7, 2.0442631033899397e-15,
]

/** c[0] + c[1]·r + … + c[7]·r^7 by Horner. */
function poly(c: number[], r: number): number {
  let v = c[7]!
  for (let i = 6; i >= 0; i--) v = v * r + c[i]!
  return v
}

/** Φ⁻¹(p) for p ∈ (0, 1); NaN otherwise. */
export function normInv(p: number): number {
  if (!(p > 0 && p < 1)) return Number.NaN
  const q = p - 0.5
  if (Math.abs(q) <= 0.425) {
    const r = 0.180625 - q * q
    return q * poly(A, r) / poly(B, r)
  }
  let r = Math.sqrt(-dlog(q < 0 ? p : 1 - p))
  let v: number
  if (r <= 5) { r -= 1.6; v = poly(C, r) / poly(D, r) }
  else { r -= 5; v = poly(E, r) / poly(F, r) }
  return q < 0 ? -v : v
}
