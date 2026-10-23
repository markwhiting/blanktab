// search-composite@1 (spec §2.4): a search life's score, behaviour C (checkup measures against newborns) plus prediction
// H (the held-out ratio), from its first 2,048 CU only. Changing anything here means a new version (decision 0020).
import { dlog, normInv } from "@blanktab/detmath"
import type { LogFileV2 } from "@blanktab/harness/log"
import { budgetWindow, clip, isGpuWidths, isV2, median } from "./life"
import type { EvalOutput, Evaluator, LifeInput, MilestoneKey, References } from "./types"

/** This version's own budget and levels, published as `searchComposite.budget` (config/evaluation.json must agree). */
const SEARCH_BUDGET_CU = 2048
const SEARCH_LEVELS: readonly number[] = Object.freeze([512, 1024, 2048])
/** The milestone suite this version reads (the newborn reference's suite). */
const SUITE = 1
/** C's milestones and κ_m = 0.1 × each pass threshold (decision 0016). M1 is left out of C (H covers prediction). */
const KAPPA: Readonly<Record<Exclude<MilestoneKey, "M1">, number>> = { M2: 0.2, M3: 0.2, M4: 0.06, M5: 0.15, M6: 0.3 }
const C_MILESTONES = Object.keys(KAPPA) as (keyof typeof KAPPA)[]
/** M1's κ for its diagnostic q only: 0.1 × its 0.8 threshold (agent's call, decision 0020). */
const KAPPA_M1 = 0.08
/** Every q and H are clipped to [−Q_BOUND, Q_BOUND]. */
const Q_BOUND = 2
/** A failed life's C and H (spec §2.4: ranked worst, ties included). */
const FAILED = -Q_BOUND
/** H reads held-out points with HELD_OUT_FROM < cu ≤ SEARCH_BUDGET_CU; it needs at least MIN_HELD_OUT of them. */
const HELD_OUT_FROM = 1024
const MIN_HELD_OUT = 3
const MIN_LEVELS = 2
/** If the newborn held-out median is outside 1 ± this, H is taken relative to it (spec §2.4). */
const NEWBORN_HELD_OUT_TOLERANCE = 0.01

const ID = "search-composite"
const VERSION = 1

const ineligible = (reason: string, diagnostics: Record<string, unknown> = {}): EvalOutput =>
  ({ evaluator: ID, version: VERSION, eligible: false, reason, score: null, components: { C: null, H: null }, diagnostics })
const scored = (C: number, H: number, diagnostics: Record<string, unknown>): EvalOutput =>
  ({ evaluator: ID, version: VERSION, eligible: true, score: C + H, components: { C, H }, diagnostics })

/** ln((x + κ)/(ñ + κ)) clipped; a non-positive ratio (a negative measure) clips to −Q_BOUND. */
function q(x: number, newbornMedian: number, kappa: number): number {
  const r = (x + kappa) / (newbornMedian + kappa)
  return r > 0 ? clip(dlog(r), Q_BOUND) : -Q_BOUND
}

function evaluateLife(life: LifeInput, refs: References): EvalOutput {
  const window = budgetWindow(life.files, SEARCH_BUDGET_CU)
  const metas = window.flatMap(f => (isV2(f) && f.meta ? [f.meta] : []))
  if (metas.length === 0 || !metas.every(m => m.role === "search" && m.genomeSchema === 1 && m.inheritance.filters === "none")) {
    return ineligible("not a search life")
  }
  if (metas.length !== window.length) return ineligible("no meta block before the checkup")
  const files = window as LogFileV2[] // every file has a meta block, so every file is v2
  if (!files.every(f => isGpuWidths(f.widths))) return ineligible("not gpu widths")
  const ref = refs.newborn
  if (!ref) return ineligible("no newborn reference")
  const suites = files.filter(f => (f.checkups ?? []).length > 0).map(f => f.milestoneSuiteVersion)
  if (suites.some(v => v !== SUITE)) return ineligible("wrong milestone suite", { suites })

  const measures = new Map<number, Record<string, number | null>>()
  for (const f of files) {
    for (const c of f.checkups ?? []) if (SEARCH_LEVELS.includes(c.level) && !measures.has(c.level)) measures.set(c.level, c.measures)
  }
  // The life's end. A life that failed before its budget checkup is a row, ranked worst, whatever its cu (spec §2.4);
  // one that stopped below the budget without failing is abandoned.
  const last = files[files.length - 1]!
  if (last.final && !measures.has(SEARCH_BUDGET_CU)) {
    if (last.meta!.end === "failed") return scored(FAILED, FAILED, { failed: true, cu: last.cu })
    if (last.cu < SEARCH_BUDGET_CU) return ineligible("abandoned", { cu: last.cu, end: last.meta!.end ?? null })
  }
  if (!measures.has(SEARCH_BUDGET_CU)) return ineligible("no 2048 checkup")
  const levels = SEARCH_LEVELS.filter(L => measures.has(L))
  if (levels.length < MIN_LEVELS) return ineligible("fewer than 2 of 3 levels", { levels })

  const ratios = files.flatMap(f => f.curve)
    .filter(p => p.cu > HELD_OUT_FROM && p.cu <= SEARCH_BUDGET_CU && p.heldOut !== undefined && p.heldOut.baseline > 0)
    .map(p => p.heldOut!.loss / p.heldOut!.baseline)
  if (ratios.length < MIN_HELD_OUT) return ineligible("fewer than 3 held-out points in (1024, 2048]", { heldOutPoints: ratios.length })

  // C: the mean over levels of the mean over M2–M6 of q (a null measure reads as 0).
  const qs: Record<string, Record<string, number>> = {}
  const qM1: Record<string, number | null> = {}
  const perLevel: Record<string, number> = {}
  for (const L of levels) {
    const m = measures.get(L)!
    const row: Record<string, number> = {}
    let sum = 0
    for (const id of C_MILESTONES) {
      row[id] = q(m[id] ?? 0, ref.milestones[id].median, KAPPA[id])
      sum += row[id]
    }
    qs[String(L)] = row
    perLevel[String(L)] = sum / C_MILESTONES.length
    const m1 = m.M1
    qM1[String(L)] = m1 === null || m1 === undefined ? null : q(m1, ref.milestones.M1.median, KAPPA_M1)
  }
  let C = 0
  for (const L of levels) C += perLevel[String(L)]!
  C /= levels.length
  const perMilestone: Record<string, number> = {}
  for (const id of C_MILESTONES) {
    let s = 0
    for (const L of levels) s += qs[String(L)]![id]!
    perMilestone[id] = s / levels.length
  }

  // H: −ln of the median held-out ratio, relative to the newborns' when theirs is not ≈ 1.
  const rho = median(ratios)!
  const fallback = ref.heldOutMedian < 1 - NEWBORN_HELD_OUT_TOLERANCE || ref.heldOutMedian > 1 + NEWBORN_HELD_OUT_TOLERANCE
  const H = clip(-dlog(fallback ? rho / ref.heldOutMedian : rho), Q_BOUND)

  return scored(C, H, {
    failed: false, levels, q: qs, qM1, perLevel, perMilestone, heldOutRatio: rho, heldOutPoints: ratios.length, heldOutFallback: fallback,
  })
}

/** 1-based mid-ranks: tied values share the mean of their positions. Values must be finite. */
function midRanks(xs: readonly number[]): number[] {
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

/**
 * The rank sums rank(C) + rank(H) over the eligible outputs, in order. Mid-ranks are halves, so the sums are exact.
 * Every output must be this evaluator's at this version: rows never mix evaluator versions (decision 0020).
 */
function rankSums(outs: readonly EvalOutput[]): number[] {
  const alien = outs.find(o => o.evaluator !== ID || o.version !== VERSION)
  if (alien) throw new Error(`${ID}@${VERSION} cannot rank an output of ${alien.evaluator}@${alien.version}: rows never mix evaluator versions`)
  const rows = outs.filter(o => o.eligible)
  const comp = (k: "C" | "H") => rows.map(o => {
    const v = o.components[k]
    if (typeof v !== "number") throw new Error(`an eligible ${ID} output needs a numeric ${k}`)
    return v
  })
  const rC = midRanks(comp("C")), rH = midRanks(comp("H"))
  return rC.map((r, i) => r + rH[i]!)
}

/**
 * a_i of spec §2.4 step 2, over the eligible outputs in order: (u_C + u_H)/2 with u = (mid-rank − 0.5)/n, computed as
 * (rank(C) + rank(H) − 1)/(2n) so that equal rank sums give equal a.
 */
export function rankAverage(outs: EvalOutput[]): number[] {
  const s = rankSums(outs)
  return s.map(x => (x - 1) / (2 * s.length))
}

/** Spec §2.4 steps 1–3: y_i = Φ⁻¹((rank(a_i) − 0.5)/n). Ranks a through its exact rank sums, so no tie is split by rounding. */
function targets(outs: EvalOutput[]): number[] {
  const s = rankSums(outs)
  return midRanks(s).map(r => normInv((r - 0.5) / s.length))
}

export const searchComposite: Evaluator = Object.freeze({
  id: ID, version: VERSION, kind: "life" as const, needs: Object.freeze(["newborn"]) as string[],
  budget: Object.freeze({ budgetCU: SEARCH_BUDGET_CU, levels: SEARCH_LEVELS }), evaluateLife, targets,
})
