import { createWorld } from "@blanktab/world"
import type { Milestone } from "./types"

/** M1 predicts its world: on a fresh crib/toys desktop, frozen prediction loss < 0.8 × the copy-last baseline. */
export const M1: Milestone = {
  id: "M1", version: 1,
  description: "Predicts its world: on a fresh crib/toys variant, prediction loss < 0.8 × the copy-last-frame baseline",
  room: "desktop", prerequisites: [], budgetSteps: 300, measuresLoss: true,
  makeVariant: (seed, canvas) => createWorld({ seed, canvas, rooms: ["crib", "toys"], start: "crib" }),
  measure: (_trace, stats) =>
    Number.isFinite(stats.baselineLoss) && stats.baselineLoss > 0 ? stats.predLoss / stats.baselineLoss : NaN,
  passes: (trace, stats) => M1.measure(trace, stats) < 0.8, // NaN fails
}
