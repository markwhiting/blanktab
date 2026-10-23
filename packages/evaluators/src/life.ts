// Shared reading of a life's merged files: which files an evaluation budget covers, and small order-free statistics.
import type { LogFile, LogFileV2 } from "@blanktab/harness/log"
import { GPU_WIDTHS_CLASS } from "./types"

export const isV2 = (f: LogFile): f is LogFileV2 => f.schemaVersion === 2

/** Whether `w` is the GPU widths class (a GPU life that fell back to the CPU keeps its widths, so it still is). */
export const isGpuWidths = (w: LogFileV2["widths"]): boolean =>
  w.conv1 === GPU_WIDTHS_CLASS.conv1 && w.conv2 === GPU_WIDTHS_CLASS.conv2 && w.enc === GPU_WIDTHS_CLASS.enc && w.hidden === GPU_WIDTHS_CLASS.hidden

/** A copy sorted by seq (the input should already be; an evaluator never relies on it). */
export const bySeq = (files: readonly LogFile[]): LogFile[] => [...files].sort((a, b) => a.seq - b.seq)

/**
 * The files an evaluation budget of `budget` CU reads, in seq order: every file up to and including the first that holds
 * the checkup at level `budget`, or else the first whose cu passes `budget` (the checkup runs at the first sleep after
 * crossing it, so it lands in that file or not at all), or else all files. Files after that are never read, so a life's
 * later files cannot change its score.
 */
export function budgetWindow(files: readonly LogFile[], budget: number): LogFile[] {
  const out: LogFile[] = []
  for (const f of bySeq(files)) {
    out.push(f)
    if (isV2(f) && ((f.checkups ?? []).some(c => c.level === budget) || f.cu > budget)) break
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
