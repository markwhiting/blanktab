import { createRng } from "@blanktab/tensor"
import { createWorld, hashSeed } from "@blanktab/world"
import { uniformAction } from "../heldout"
import type { CheckupTrace, Milestone } from "./types"

const BUDGET = 400
/** Random clickers averaged for the reference: one alone visits 1–6 pages, too noisy to compare against. */
export const M5_REFERENCE_CLICKERS = 16
/** Distinct pages a pass needs besides the ratio: random clickers seldom reach 4. */
export const M5_MIN_PAGES = 4

/** Distinct pages in a trace, the start page included. */
export const distinctPages = (trace: CheckupTrace) =>
  new Set([trace.initial.page, ...trace.states.map(s => s.page)].filter(p => p !== null)).size

/** Distinct pages (start included) a uniform-random clicker driven by `rnd` visits in a fresh variant. */
function clickerPages(seed: number, canvas: Parameters<Milestone["makeVariant"]>[1], rnd: () => number): number {
  const world = M5.makeVariant(seed, canvas)
  const pages = new Set([world.state().page])
  for (let i = 0; i < BUDGET; i++) {
    world.act(uniformAction(rnd))
    world.drainTrace()
    pages.add(world.state().page)
  }
  pages.delete(null)
  return pages.size
}

/**
 * M5 explores places efficiently: distinct pages visited (the start page included) over the reference, the mean of the
 * same count for M5_REFERENCE_CLICKERS uniform-random clickers on the same variant and budget (clicker j's actions
 * from the RNG seeded hashSeed(seed, `random-clicker/${j}`)); a pass also needs at least M5_MIN_PAGES pages. The
 * reference is a pure function of the seed. It needs no observations (the clickers ignore them, and observing never
 * changes the world), so it runs synchronously.
 */
export const M5: Milestone = {
  id: "M5", version: 1,
  description: "Explores places efficiently: distinct pages visited ≥ 1.5 × the mean of 16 uniform-random clickers on the same fresh site, and ≥ 4",
  room: "places", prerequisites: ["M3"], budgetSteps: BUDGET, measuresLoss: false,
  makeVariant: (seed, canvas) => createWorld({ seed, canvas, rooms: ["places"] }),
  reference(seed, canvas) {
    let sum = 0
    for (let j = 0; j < M5_REFERENCE_CLICKERS; j++) sum += clickerPages(seed, canvas, createRng(hashSeed(seed, `random-clicker/${j}`)))
    return sum / M5_REFERENCE_CLICKERS
  },
  measure: (trace, _stats, reference) =>
    reference !== undefined && reference > 0 ? distinctPages(trace) / reference : NaN,
  passes: (trace, stats, reference) => distinctPages(trace) >= M5_MIN_PAGES && M5.measure(trace, stats, reference) >= 1.5,
}
