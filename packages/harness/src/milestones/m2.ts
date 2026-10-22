import { createWorld } from "@blanktab/world"
import { inViewStates, isTaskbarId, mean } from "./record"
import type { Milestone } from "./types"

/** A pass needs the cursor in the viewport for at least this share of the budget (a learner parked on the taskbar after a few in-view steps does not pass). */
export const M2_MIN_IN_VIEW = 0.5

/**
 * M2 moves with purpose: over the steps with the cursor in the viewport, the fraction spent over an interactive
 * element, divided by the mean fraction of the viewport those elements cover. Taskbar steps count for neither side,
 * and a pass needs at least M2_MIN_IN_VIEW of the budget in view.
 */
export const M2: Milestone = {
  id: "M2", version: 1,
  description: "Moves with purpose: fraction of in-view steps over interactive elements ≥ 2 × their area fraction, with ≥ 50 % of the budget in view",
  room: "crib", prerequisites: [], budgetSteps: 400, measuresLoss: false,
  makeVariant: (seed, canvas) => createWorld({ seed, canvas, rooms: ["crib"] }),
  measure(trace) {
    const view = inViewStates(trace.states)
    const area = mean(view.map(s => s.areas.interactive))
    if (!(area > 0)) return NaN // no in-view step, or nothing interactive
    const over = view.filter(s => s.hovered?.interactive === true && !isTaskbarId(s.hovered.id)).length
    return over / view.length / area
  },
  passes: (trace, stats) =>
    inViewStates(trace.states).length >= M2_MIN_IN_VIEW * M2.budgetSteps && M2.measure(trace, stats) >= 2,
}
