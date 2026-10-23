// signal-check@1 (spec §2.6, §4.6): does the search score have signal? Calibration genomes are each run several times;
// if the score's differences between genomes stand out from its repeat noise, search can begin. Changing anything here
// means a new version (decision 0020).
import { isV2, searchOf } from "./life"
import { ALPHA_LOOK, ICC_MIN, icc1, kruskalWallis } from "./stats"
import type { CohortContext, EvalOutput, Evaluator, LifeInput, References } from "./types"

const ID = "signal-check"
const VERSION = 1
/** Enough data (spec §2.6): at least MIN_GENOMES genomes with MIN_REPEATS lives or more each, and MIN_LIVES lives in all. */
const MIN_GENOMES = 6
const MIN_REPEATS = 2
const MIN_LIVES = 20

/** The life's genomeSha, if every one of its files has a meta block and they agree. */
function genomeOf(life: LifeInput): string | null {
  const shas = new Set(life.files.map(f => (isV2(f) && f.meta ? f.meta.genomeSha : null)))
  const [only] = shas
  return shas.size === 1 && typeof only === "string" ? only : null
}

/**
 * `groups` maps a calibration genomeSha to its lives. Each life is scored by the pinned search evaluator; the eligible
 * ones give a_i (its rankAverage over all of them, pooled in sorted genome then learnerId order), grouped back by genome.
 * A life listed twice, or under a genome its meta does not carry, is not counted (and is reported).
 */
function evaluateCohort(groups: Record<string, LifeInput[]>, refs: References, ctx?: CohortContext): EvalOutput {
  const search = searchOf(ctx, `${ID}@${VERSION}`)
  const searchId = `${search.id}@${search.version}`
  const seen = new Set<string>()
  let duplicates = 0, mislabelled = 0, ineligible = 0
  const rows: { genome: string; learnerId: string; out: EvalOutput }[] = []
  for (const genome of Object.keys(groups).sort()) {
    const lives = [...groups[genome]!].sort((a, b) => (a.learnerId < b.learnerId ? -1 : a.learnerId > b.learnerId ? 1 : 0))
    for (const life of lives) {
      if (seen.has(life.learnerId)) { duplicates++; continue }
      seen.add(life.learnerId)
      if (genomeOf(life) !== genome) { mislabelled++; continue }
      const out = search.evaluateLife(life, refs)
      if (!out.eligible || typeof out.score !== "number") { ineligible++; continue }
      rows.push({ genome, learnerId: life.learnerId, out })
    }
  }
  const a = search.rankAverage(rows.map(r => r.out))
  const byGenome = new Map<string, { a: number[]; score: number[]; learnerIds: string[] }>()
  rows.forEach((r, i) => {
    const g = byGenome.get(r.genome) ?? { a: [], score: [], learnerIds: [] }
    g.a.push(a[i]!)
    g.score.push(r.out.score!)
    g.learnerIds.push(r.learnerId)
    byGenome.set(r.genome, g)
  })
  const genomes = [...byGenome.keys()]
  const n = rows.length, k = genomes.length
  const repeated = genomes.filter(g => byGenome.get(g)!.a.length >= MIN_REPEATS).length
  const learnerIds = Object.fromEntries(genomes.map(g => [g, byGenome.get(g)!.learnerIds]))
  const base = { search: searchId, n, k, repeatedGenomes: repeated, duplicates, mislabelled, ineligible, learnerIds }
  const no = (reason: string): EvalOutput =>
    ({ evaluator: ID, version: VERSION, eligible: false, reason, score: null, components: { H: null, p: null, icc: null }, diagnostics: base })
  if (repeated < MIN_GENOMES) return no(`fewer than ${MIN_GENOMES} genomes with ${MIN_REPEATS} or more lives`)
  if (n < MIN_LIVES) return no(`fewer than ${MIN_LIVES} lives`)

  const kw = kruskalWallis(genomes.map(g => byGenome.get(g)!.a))
  const icc = icc1(genomes.map(g => byGenome.get(g)!.score))
  const signal = kw.p < ALPHA_LOOK && icc >= ICC_MIN
  return {
    evaluator: ID, version: VERSION, eligible: true, score: signal ? 1 : 0, components: { H: kw.H, p: kw.p, icc },
    diagnostics: { ...base, H: kw.H, df: kw.df, p: kw.p, icc, alphaLook: ALPHA_LOOK, iccMin: ICC_MIN },
  }
}

export const signalCheck: Evaluator = Object.freeze({
  id: ID, version: VERSION, kind: "cohort" as const, needs: [], evaluateCohort,
})
