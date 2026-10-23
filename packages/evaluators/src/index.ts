// @blanktab/evaluators: swappable, versioned evaluators (spec §2; decision 0020). Every evaluator is pure and
// deterministic. REGISTRY keys are "id@version"; old versions stay forever, so any merged life can be rescored under
// any of them. Which ones count is pinned in config/evaluation.json, which the gate never writes.
import { confirmFisher } from "./confirm-fisher"
import { newbornBaseline } from "./newborn-baseline"
import { searchComposite } from "./search-composite"
import { signalCheck } from "./signal-check"
import type { CohortContext, Evaluator } from "./types"

export type {
  CohortContext, EvalBudget, EvalOutput, Evaluator, LifeInput, LifeOptions, LifeRole, MilestoneKey, NewbornProvenance, NewbornReference, ReferenceParser, References,
} from "./types"
export { GPU_WIDTHS_CLASS } from "./types"
export { parseNewbornReference, PROVENANCE_TEXT_RE, REFERENCE_PARSERS } from "./reference"
export { rankAverage, searchComposite } from "./search-composite"
export { newbornBaseline } from "./newborn-baseline"
export { signalCheck } from "./signal-check"
export { confirmFisher } from "./confirm-fisher"
export { ALPHA_LOOK, ICC_MIN, SIGNAL_LOOKS } from "./stats"

export const REGISTRY: Readonly<Record<string, Evaluator>> = Object.freeze({
  "search-composite@1": searchComposite,
  "newborn-baseline@1": newbornBaseline,
  "signal-check@1": signalCheck,
  "confirm-fisher@1": confirmFisher,
})

/** The evaluator registered as `idAtVersion` (e.g. "search-composite@1"); throws on an unknown one. */
export function getEvaluator(idAtVersion: string): Evaluator {
  const e = Object.hasOwn(REGISTRY, idAtVersion) ? REGISTRY[idAtVersion] : undefined
  if (!e) throw new Error(`unknown evaluator ${JSON.stringify(idAtVersion)}`)
  return e
}

/**
 * What the cohort evaluators run with: the pinned search evaluator, REGISTRY[pins.search] (config/evaluation.json's
 * `search`). signal-check and confirm-fisher score lives only through it, so swapping the search evaluator is a pin edit.
 */
export function cohortContext(pins: { search: string }): CohortContext {
  return { search: getEvaluator(pins.search) }
}
