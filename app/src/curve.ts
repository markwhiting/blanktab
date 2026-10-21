// The loss curve on the LEARNER tab: how wrong the learner's guesses are vs the lazy guess "nothing changes"
// (copy-last), on its fixed held-out set, which it never trains on (decision 0015 §3); its own-stream ratio is a
// secondary number (a curious learner picks what it sees, so that one can flatter it). Pure helpers, no DOM.
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

type Scored = CurvePoint & { heldOut: NonNullable<CurvePoint["heldOut"]> }
/** The points with a held-out score (a learner from before the check has none). */
const scored = (points: readonly CurvePoint[]): Scored[] => points.filter((p): p is Scored => p.heldOut !== undefined)

/**
 * Canvas coordinates of both held-out series (the learner's loss and the lazy guess's, on the same fixed transitions)
 * in a w×h box with `pad` pixels of margin: x is compute lived from birth (0 CU) to the newest scored point, y one
 * scale shared by both losses (higher loss = higher up), so the gap between the lines is the learning. Points without
 * a held-out score are skipped. Points all at 0 CU, or a flat loss, are drawn mid-box.
 */
export function sparkline(points: readonly CurvePoint[], w: number, h: number, pad = 2): { pred: Polyline; base: Polyline } {
  const ps = scored(points)
  if (ps.length === 0) return { pred: [], base: [] }
  const xs = ps.map(p => p.cu)
  const ys = ps.flatMap(p => [p.heldOut.loss, p.heldOut.baseline])
  const x0 = 0, x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys)
  const x = (v: number) => (x1 > x0 ? pad + ((v - x0) / (x1 - x0)) * (w - 2 * pad) : w / 2)
  const y = (v: number) => (y1 > y0 ? h - pad - ((v - y0) / (y1 - y0)) * (h - 2 * pad) : h / 2)
  return { pred: ps.map(p => [x(p.cu), y(p.heldOut.loss)]), base: ps.map(p => [x(p.cu), y(p.heldOut.baseline)]) }
}

/** The latest held-out score's loss ÷ baseline (3 significant digits), said in plain words. */
export function ratioText(points: readonly CurvePoint[]): string {
  const p = scored(points).at(-1)?.heldOut
  if (!p || !(p.baseline > 0) || !Number.isFinite(p.loss)) return "no guesses measured yet"
  const r = p.loss / p.baseline
  const off = Math.abs(1 - r)
  const pct = Math.max(1, Math.round(off * 100))
  const words = off < 0.01 ? "about as wrong as" : r < 1 ? `${pct}% less wrong than` : `${pct}% more wrong than`
  return `${r.toPrecision(3)} (its guesses are ${words} the lazy guess)`
}

/** The secondary number: the last point's own-stream predLoss ÷ baselineLoss, or "" without one. */
export function ownStreamText(points: readonly CurvePoint[]): string {
  const p = points.at(-1)
  if (!p || !(p.baselineLoss > 0) || !Number.isFinite(p.predLoss)) return ""
  return `On what it chose to look at (which it also trains on): ${(p.predLoss / p.baselineLoss).toPrecision(3)}`
}
