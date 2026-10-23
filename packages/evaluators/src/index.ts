// @blanktab/evaluators: swappable, versioned evaluators (spec §2; decision 0020). Every evaluator is pure and
// deterministic. REGISTRY keys are "id@version"; old versions stay forever, so any merged life can be rescored under
// any of them. Which ones count is pinned in config/evaluation.json, which the gate never writes.
import { newbornBaseline } from "./newborn-baseline"
import { searchComposite } from "./search-composite"
import type { Evaluator } from "./types"

export type { EvalOutput, Evaluator, LifeInput, MilestoneKey, NewbornReference, References } from "./types"
export { GPU_WIDTHS_CLASS } from "./types"
export { parseNewbornReference } from "./reference"
export { rankAverage, searchComposite, SEARCH_BUDGET_CU, SEARCH_LEVELS } from "./search-composite"
export { newbornBaseline } from "./newborn-baseline"

export const REGISTRY: Readonly<Record<string, Evaluator>> = Object.freeze({
  "search-composite@1": searchComposite,
  "newborn-baseline@1": newbornBaseline,
})

/** The evaluator registered as `idAtVersion` (e.g. "search-composite@1"); throws on an unknown one. */
export function getEvaluator(idAtVersion: string): Evaluator {
  const e = Object.hasOwn(REGISTRY, idAtVersion) ? REGISTRY[idAtVersion] : undefined
  if (!e) throw new Error(`unknown evaluator ${JSON.stringify(idAtVersion)}`)
  return e
}
