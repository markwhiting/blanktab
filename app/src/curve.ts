// The loss curve on the LEARNER tab: how wrong the learner's guesses are vs the lazy guess "nothing changes"
// (copy-last). Pure helpers, no DOM.
import type { CurvePoint } from "@blanktab/harness/log"

/** The page keeps at most this many points of a life (thinned evenly, the newest always kept). */
export const MAX_SHOWN_POINTS = 256

/** Adds the points that are ahead of the last one (in steps), then thins to MAX_SHOWN_POINTS. */
export function appendPoints(curve: readonly CurvePoint[], points: readonly CurvePoint[]): CurvePoint[] {
  const out = [...curve]
  for (const p of points) if (out.length === 0 || p.steps > out.at(-1)!.steps) out.push(p)
  let c = out
  while (c.length > MAX_SHOWN_POINTS) c = c.filter((_, i) => (c.length - 1 - i) % 2 === 0)
  return c
}

export type Polyline = [number, number][]

/**
 * Canvas coordinates of both series in a w×h box with `pad` pixels of margin: x is compute lived (CU), y one scale
 * shared by both losses (higher loss = higher up), so the gap between the lines is the learning. A single x or a flat
 * loss is drawn mid-box.
 */
export function sparkline(points: readonly CurvePoint[], w: number, h: number, pad = 2): { pred: Polyline; base: Polyline } {
  if (points.length === 0) return { pred: [], base: [] }
  const xs = points.map(p => p.cu)
  const ys = points.flatMap(p => [p.predLoss, p.baselineLoss])
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  const x = (v: number) => (x1 > x0 ? pad + ((v - x0) / (x1 - x0)) * (w - 2 * pad) : w / 2)
  const y = (v: number) => (y1 > y0 ? h - pad - ((v - y0) / (y1 - y0)) * (h - 2 * pad) : h / 2)
  return { pred: points.map(p => [x(p.cu), y(p.predLoss)]), base: points.map(p => [x(p.cu), y(p.baselineLoss)]) }
}

const round = (v: number, digits: number) => String(Math.round(v * 10 ** digits) / 10 ** digits)

/** The last point's predLoss ÷ baselineLoss, said in plain words. */
export function ratioText(points: readonly CurvePoint[]): string {
  const p = points.at(-1)
  if (!p || !(p.baselineLoss > 0) || !Number.isFinite(p.predLoss)) return "no guesses measured yet"
  const r = p.predLoss / p.baselineLoss
  const pct = Math.round(Math.abs(1 - r) * 100)
  const words = pct === 0 ? "as wrong as" : r < 1 ? `${pct}% less wrong than` : `${pct}% more wrong than`
  return `${round(r, 2)} (its guesses are ${words} the lazy guess)`
}
