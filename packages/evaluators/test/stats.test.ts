import { describe, it, expect } from "vitest"
import {
  ALPHA_LOOK, ICC_MIN, SIGNAL_LOOKS, fisherOneSided, holm, icc1, kruskalWallis, mannWhitneyOneSided, midRanks,
} from "../src/stats"

describe("stats (spec §2.5, §2.6)", () => {
  it("the constants: three looks at 30, 45 and 60 lives, each at 0.05/3; ICC ≥ 0.1", () => {
    expect([...SIGNAL_LOOKS]).toEqual([30, 45, 60])
    expect(ALPHA_LOOK).toBe(0.05 / 3)
    expect(ICC_MIN).toBe(0.1)
  })

  it("midRanks: ties share the mean of their positions; non-finite values throw", () => {
    expect(midRanks([3, 1, 3, 2])).toEqual([3.5, 1, 3.5, 2])
    expect(midRanks([])).toEqual([])
    expect(() => midRanks([1, Number.NaN])).toThrow(/finite/)
  })

  it("kruskalWallis on the textbook example (Hollander & Wolfe 6.1: H ≈ 6.4, p ≈ 0.04) to 1e-3", () => {
    // Hollander & Wolfe Example 6.1 itself (mucociliary clearance half-times; normal, obstructive airway disease,
    // asbestosis) has no ties and gives H = 0.771, p = 0.680, not the plan's "H ≈ 6.4, p ≈ 0.04". Both are checked here.
    // Hand calculation: pooled ranks 2.0→1 … 4.0→14; R = 36, 36, 33; H = 12/(14·15)·(36²/5 + 36²/4 + 33²/5) − 45
    // = (12/210)·801 − 45 = 27/35 = 0.771429; with df 2, p = e^(−H/2) = 0.679965.
    const hw = kruskalWallis([[2.9, 3.0, 2.5, 2.6, 3.2], [3.8, 2.7, 4.0, 2.4], [2.8, 3.4, 3.7, 2.2, 2.0]])
    expect(hw.df).toBe(2)
    expect(hw.H).toBeCloseTo(27 / 35, 12)
    expect(hw.p).toBeCloseTo(0.679965, 6)
    // A tied example with H ≈ 6.4, p ≈ 0.04 (agent's data, hand calculation). Pooled sorted: 1 | 2 2 2 2 | 3 3 | 4 4 |
    // 6 6 6 | 8 | 9 with mid-ranks 1, 3.5, 6.5, 8.5, 11, 13, 14. R = (3.5+3.5+8.5+11+11) = 37.5,
    // (1+3.5+3.5+6.5) = 14.5, (6.5+8.5+11+13+14) = 53. Uncorrected H = (12/210)·(37.5²/5 + 14.5²/4 + 53²/5) − 45 = 8649/1400.
    // Ties Σ(t³ − t) = 60 + 6 + 6 + 24 = 96, correction 1 − 96/2730 = 439/455, so H = 8649·455/(1400·439) = 6.403018;
    // df 2, p = e^(−H/2) = 0.040701.
    const tied = kruskalWallis([[2, 2, 4, 6, 6], [1, 2, 2, 3], [3, 4, 6, 8, 9]])
    expect(tied.df).toBe(2)
    expect(tied.H).toBeCloseTo(6.403018, 3)
    expect(tied.H).toBeCloseTo((8649 * 455) / (1400 * 439), 12)
    expect(tied.p).toBeCloseTo(0.040701, 3)
    // All tied: no evidence.
    expect(kruskalWallis([[1, 1], [1]])).toEqual({ H: 0, df: 1, p: 1 })
    expect(() => kruskalWallis([[1, 2]])).toThrow()
    expect(() => kruskalWallis([[1, 2], []])).toThrow()
  })

  it("icc1 = 0 for identical groups' means; → 1 for zero within-group variance", () => {
    // Identical means: MSB = 0, so the estimate −MSW/((n0 − 1)MSW) is negative and reads as 0.
    expect(icc1([[1, 2, 3], [3, 2, 1], [2, 1, 3]])).toBe(0)
    expect(icc1([[5, 5], [5, 5]])).toBe(0)
    expect(icc1([[1, 1, 1], [4, 4, 4], [9, 9]])).toBe(1)
    // By hand: means 2 and 5, MSB = 13.5, MSW = 1, n0 = (6 − 18/6)/1 = 3, ICC = 12.5/15.5.
    expect(icc1([[1, 2, 3], [4, 5, 6]])).toBeCloseTo(12.5 / 15.5, 12)
    // Unequal sizes: n0 = (5 − (9 + 4)/5)/1 = 2.4. Means 2 and 5.5, grand 3.4: SSB = 3·1.96 + 2·4.41 = 14.7, MSB = 14.7;
    // SSW = 2 + 0.5 = 2.5, MSW = 2.5/3; ICC = (14.7 − 2.5/3)/(14.7 + 1.4·2.5/3).
    expect(icc1([[1, 2, 3], [5, 6]])).toBeCloseTo((14.7 - 2.5 / 3) / (14.7 + (1.4 * 2.5) / 3), 12)
    expect(() => icc1([[1, 2]])).toThrow()
    expect(() => icc1([[1], [2]])).toThrow()
  })

  it("fisherOneSided(3,0,0,3) = 0.05", () => {
    expect(fisherOneSided(3, 0, 0, 3)).toBeCloseTo(0.05, 12)
    // Exact rationals by hand (Σ C(m,x)C(N−m,n1−x)/C(N,n1)): 0.706294 for [[2,3],[4,6]]; 1 when a = 0.
    expect(fisherOneSided(2, 3, 4, 6)).toBeCloseTo(0.7062937062937062, 10)
    expect(fisherOneSided(0, 5, 0, 5)).toBe(1)
    // The confirmation's design point, 20/24 against 6/60: 1.5536e-10 (relative 1e-6).
    expect(fisherOneSided(20, 4, 6, 54) / 1.5536059923224531e-10).toBeCloseTo(1, 6)
    expect(() => fisherOneSided(1.5, 0, 0, 1)).toThrow()
  })

  it("holm rejects in order", () => {
    // Sorted: 0.005 ≤ 0.05/4 ✓, 0.01 ≤ 0.05/3 ✓, 0.03 > 0.05/2 stops it, so 0.04 is not rejected either.
    expect(holm([0.01, 0.04, 0.03, 0.005], 0.05)).toEqual([true, false, false, true])
    // 0.011 ≤ 0.05/2, then 0.02 ≤ 0.05/1.
    expect(holm([0.02, 0.011], 0.05)).toEqual([true, true])
    // A step-down: 0.026 > 0.05/2 stops it, so 0.03, which would pass alone at 0.05, is not rejected.
    expect(holm([0.03, 0.026], 0.05)).toEqual([false, false])
    expect(holm([0.05], 0.05)).toEqual([true])
    expect(holm([], 0.05)).toEqual([])
  })

  it("mannWhitneyOneSided(x≫y) < 0.01; identical samples ≈ 0.5", () => {
    const x = Array.from({ length: 12 }, (_, i) => 100 + i)
    const y = Array.from({ length: 12 }, (_, i) => i)
    expect(mannWhitneyOneSided(x, y)).toBeLessThan(0.01)
    expect(mannWhitneyOneSided(y, x)).toBeGreaterThan(0.99)
    const same = Array.from({ length: 10 }, (_, i) => i)
    expect(mannWhitneyOneSided(same, same)).toBeCloseTo(0.5, 1)
    // By hand: x = 3,4,5 against y = 1,2 gives U = 6, mean 3, σ² = 3, z = 2.5/√3, p = Φ(−z) = 0.074457.
    expect(mannWhitneyOneSided([3, 4, 5], [1, 2])).toBeCloseTo(0.07445733658938283, 6)
    // Ties: x = 2,3,3 against y = 1,2,3: U = 6.5, σ² = 9/12·(7 − 30/30) = 4.5, z = 1.5/√4.5, p = 0.239750.
    expect(mannWhitneyOneSided([2, 3, 3], [1, 2, 3])).toBeCloseTo(0.23975006109347674, 6)
    expect(mannWhitneyOneSided([1, 1], [1, 1, 1])).toBe(1)
    expect(() => mannWhitneyOneSided([], [1])).toThrow()
  })
})
