import { createWorld } from "@blanktab/world"
import type { Milestone } from "./types"

/**
 * M4 uncovers hidden content: of what was hidden at the start (1 − r0), the share revealed by the end, where r is the
 * hide-and-seek page's revealed fraction. Content visible at the start is in r0 and earns nothing.
 */
export const M4: Milestone = {
  id: "M4", version: 1,
  description: "Uncovers hidden content: reveals ≥ 60 % of a fresh long page's initially hidden content within budget",
  room: "hideseek", prerequisites: ["M2"], budgetSteps: 400, measuresLoss: false,
  makeVariant: (seed, canvas) => createWorld({ seed, canvas, rooms: ["hideseek"] }),
  measure(trace) {
    const r0 = trace.initial.scroll?.revealedFraction
    const r = (trace.states.at(-1) ?? trace.initial).scroll?.revealedFraction
    if (r0 === undefined || r === undefined || !(r0 < 1)) return NaN
    return (r - r0) / (1 - r0)
  },
  passes: (trace, stats) => M4.measure(trace, stats) >= 0.6,
}
