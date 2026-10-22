import { createWorld } from "@blanktab/world"
import { inViewStates, isTaskbarId, mean } from "./record"
import type { Milestone } from "./types"

const counts = (trace: Parameters<Milestone["measure"]>[0]) => {
  const clicks = trace.events.filter(e => e.kind === "click" && !isTaskbarId(e.elementId))
  const responsive = clicks.filter(e => e.responsive === true).length
  return { responsive, unresponsive: clicks.length - responsive }
}

/**
 * M3 prefers things that respond: viewport clicks on responsive elements per unit of their area, over clicks on
 * everything else per unit of its area (unresponsive = 1 − responsive; areas are means over in-view steps). At
 * least 3 responsive clicks are needed. Taskbar clicks are ignored.
 */
export const M3: Milestone = {
  id: "M3", version: 1,
  description: "Prefers things that respond: area-normalized click rate on responsive elements ≥ 2 × on unresponsive ones, with ≥ 3 responsive clicks",
  room: "toys", prerequisites: ["M2"], budgetSteps: 400, measuresLoss: false,
  makeVariant: (seed, canvas) => createWorld({ seed, canvas, rooms: ["toys"] }),
  measure(trace) {
    const resp = mean(inViewStates(trace.states).map(s => s.areas.responsive))
    if (!(resp > 0 && resp < 1)) return NaN // no in-view step, or one side has no area
    const c = counts(trace)
    return c.responsive / resp / (Math.max(c.unresponsive, 1) / (1 - resp))
  },
  passes: (trace, stats) => counts(trace).responsive >= 3 && M3.measure(trace, stats) >= 2,
}
