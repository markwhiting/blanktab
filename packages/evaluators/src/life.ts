// Shared reading of a life's merged files: which files an evaluation budget covers, and small order-free statistics.
import { nextCrossing, type LogFile, type LogFileV2 } from "@blanktab/harness/log"
import { GPU_WIDTHS_CLASS, type CohortContext, type Evaluator } from "./types.ts"

export const isV2 = (f: LogFile): f is LogFileV2 => f.schemaVersion === 2

/** Whether `w` is the GPU widths class (a GPU life that fell back to the CPU keeps its widths, so it still is). */
export const isGpuWidths = (w: LogFileV2["widths"]): boolean =>
  w.conv1 === GPU_WIDTHS_CLASS.conv1 && w.conv2 === GPU_WIDTHS_CLASS.conv2 && w.enc === GPU_WIDTHS_CLASS.enc && w.hidden === GPU_WIDTHS_CLASS.hidden

/** A copy sorted by seq (the input should already be; an evaluator never relies on it). */
export const bySeq = (files: readonly LogFile[]): LogFile[] => [...files].sort((a, b) => a.seq - b.seq)

/**
 * The files an evaluation budget of `budget` CU reads, in seq order: every file up to and including the first that holds
 * a checkup at a level ≥ `budget`, or else the first whose cu reaches the log crossing after `budget`
 * (nextCrossing(budget)), or else all files. The session writes the file that crosses `budget` at whatever step crosses
 * it (its cu a fraction over), and runs the budget checkup at the next sleep, so the checkup lands in a later file: the
 * next crossing's, a reload's, or the final one. Readers filter what they read by cu (curve points ≤ `budget`) and by
 * checkup level, so the window's last file, which may run past `budget`, cannot change a score either.
 */
export function budgetWindow(files: readonly LogFile[], budget: number): LogFile[] {
  const out: LogFile[] = []
  const bound = nextCrossing(budget)
  for (const f of bySeq(files)) {
    out.push(f)
    if (isV2(f) && ((f.checkups ?? []).some(c => c.level >= budget) || f.cu >= bound)) break
  }
  return out
}

/** The median of `xs` (the mean of the middle two for an even count); null for none. Order-free. */
export function median(xs: readonly number[]): number | null {
  if (xs.length === 0) return null
  const s = [...xs].sort((a, b) => a - b)
  const h = s.length >> 1
  return s.length % 2 === 1 ? s[h]! : (s[h - 1]! + s[h]!) / 2
}

/** `v` clipped to [−bound, bound] (±Infinity clip to the bound). */
export const clip = (v: number, bound: number): number => (v > bound ? bound : v < -bound ? -bound : v)

/** A search evaluator as a cohort test uses it: a life evaluator that scores lives and ranks them. */
export type SearchEvaluator = Evaluator & Required<Pick<Evaluator, "evaluateLife" | "rankAverage">>

/** The pinned search evaluator from `ctx`, checked to have what a cohort test needs. */
export function searchOf(ctx: CohortContext | undefined, who: string): SearchEvaluator {
  const s = ctx?.search
  if (!s || s.kind !== "life" || !s.evaluateLife || !s.rankAverage) {
    throw new Error(`${who} needs the pinned search evaluator (a life evaluator with rankAverage) in its context: cohortContext(config)`)
  }
  return s as SearchEvaluator
}
