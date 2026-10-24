// The page's words for a life's recipe and for the meta-learner's state (spec §6). Pure text from the meta block, the
// genome space and the posterior, so it is testable. None of it reaches the learner.
import type { Knowledge } from "@blanktab/harness/knowledge"
import type { MetaBlock, SessionRow } from "@blanktab/harness/log"
import { parseCentre, parseGenomeSpace, parsePosterior, searched, toUnit, type Genome, type GenomeSpace, type Posterior } from "@blanktab/meta"

/** Plain names for the genes (the full list behind the toggle keeps the gene names). */
export const GENE_LABELS: Readonly<Record<string, string>> = Object.freeze({
  lr: "learning rate", headLrScale: "head learning rate", residualDecay: "residual decay", entropyWeight: "entropy bonus",
  curiosityWeight: "curiosity", exploreEpsilon: "exploration", bufferSize: "memory size", replayPerSleep: "replay per sleep",
  gamma: "planning horizon", caregiverPeriod: "caregiver period", caregiverTau: "caregiver temperature",
  caregiverEpsilon: "caregiver exploration",
})
const label = (name: string) => GENE_LABELS[name] ?? name

/** How this life's genome was chosen, in words. `posterior`: the one the page has (for a calibration life's recipe number). */
export function recipeText(meta: MetaBlock, posterior: Posterior | null): string {
  switch (meta.method) {
    case "calibrate": {
      const genomes = posterior?.calibration.genomes ?? []
      const i = genomes.findIndex(g => g.genomeSha === meta.genomeSha)
      return i < 0 ? "a calibration life" : `a calibration life (recipe ${i + 1} of ${genomes.length})`
    }
    case "uniform": return "one of the 1-in-5 random explorers"
    case "centre": return "the current recipe"
    case "legacy": return "born before the meta-learner (its old settings)"
    case "newborn": return `a newborn reference (panel ${meta.evalPanel})`
    case "ts": return "drawn by the meta-learner"
    case "confirm": return "a confirmation life"
  }
}

const horizon = (v: number) => 1 / (1 - v)
const ratioText = (r: number) => String(Number(r.toPrecision(2)))

/**
 * The `n` searched genes furthest from the centre in the unit cube (ties in file order), each as its plain name and
 * its ratio to the centre's value ("learning rate ×2.6"); for gamma, the ratio of planning horizons 1/(1 − γ).
 * Genes at the centre are left out, so the centre has none.
 */
export function genomeHighlights(genome: Genome, centre: Genome, space: GenomeSpace, n = 3): string[] {
  const genes = searched(space)
  const u = toUnit(space, genome), c = toUnit(space, centre)
  return genes
    .map((g, i) => ({ g, d: Math.abs(u[i]! - c[i]!), i }))
    .filter(x => x.d > 1e-12)
    .sort((a, b) => b.d - a.d || a.i - b.i)
    .slice(0, n)
    .map(({ g }) => {
      const v = genome[g.name] as number, at = centre[g.name] as number
      const r = g.scale === "horizon" ? horizon(v) / horizon(at) : v / at
      return `${label(g.name)} ×${ratioText(r)}`
    })
}

/** The meta-learner's state, in one line (spec §6), from the posterior's page counts; null: the page could not read one. */
export function metaStateText(posterior: Posterior | null): string {
  if (posterior === null) return "Not reachable: this learner runs the current recipe."
  const { calibrationLives, calibrationTarget } = posterior.page
  switch (posterior.mode) {
    case "calibrate": return `Calibrating: ${calibrationLives} of ${calibrationTarget} lives measured, to learn whether the score can tell recipes apart.`
    case "paused": return `Paused: the score can't yet tell recipes apart; measuring more (${calibrationLives} of ${calibrationTarget}).`
    case "flat": return `Flat: after ${calibrationTarget} lives the score could not tell recipes apart. Recipes are the current one, with 1 in 5 random.`
    case "search": return "Signal found: the search can start. (Recipes are the current one or random until it does.)"
  }
}

/** The meta-learner's line without readable genome files (Review Focus 5: the page says which case applies). */
export const NO_GENOME_TEXT = "Not reachable: the genome files could not be read, so this learner runs the default settings and is not counted."

/** The meta-learner's line for the knowledge the page has: metaStateText of its posterior, or NO_GENOME_TEXT. */
export function metaLineOf(k: Pick<Knowledge, "meta">): string {
  if (!k.meta) return NO_GENOME_TEXT
  try {
    parseCentre(k.meta.genome, parseGenomeSpace(k.meta.genomeSpace))
  } catch {
    return NO_GENOME_TEXT
  }
  return metaStateText(k.meta.posteriorText === null ? null : parsePosterior(k.meta.posteriorText))
}

/** The meta-learner line in newborn reference mode (?newborns=a-b): no draw happens in this tab. */
export const NEWBORN_RUN_TEXT =
  "Newborn reference run: no draw here. Each newborn is the current recipe, untrained, checked once to set the baseline the score is measured against."

/** The recipe line of a life that logs no meta block (no genome files, or a test knob contradicts its genome). */
export const unlabelledText = (reason: string) => `the default settings, not counted (${reason})`

/** A legacy life's genes its settings never recorded (their values on the page are placeholders), or null if none. */
export function missingText(missing: readonly string[]): string | null {
  return missing.length === 0 ? null : `some settings not recorded: ${missing.map(label).join(", ")}`
}

/** This life's score under `evaluator` (id@version) once its summary row has one, e.g. "0.41 (search-composite@1)". */
export function scoreText(row: SessionRow | undefined, evaluator: string): string | null {
  const e = row?.evals?.[evaluator]
  if (!e) return null
  return e.score === null ? `not eligible (${evaluator})` : `${e.score.toFixed(2)} (${evaluator})`
}
