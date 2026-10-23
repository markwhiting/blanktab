// The birth draw (spec §4.1): which genome a new life runs. A pure function of the learnerId, the posterior's bytes
// (through its parsed form and sha), the genome space, the centre and the widths class, so the gate can recompute
// every pick. The order of rng() calls is fixed: u0 picks the method, then the method's own draws follow.
//
// The 05b rule. 05b adds `ts` (search, gpu, u0 >= UNIFORM_SHARE) and `confirm` (u0 in [UNIFORM_SHARE, 0.7) while a
// confirmation is in flight) without touching the branches below. `ts` falls back to `centre` when
// posterior.models[widthsClass] is absent, so a 05a posterior (models {}) still draws centre for gpu search births
// with u0 >= UNIFORM_SHARE under 05b code: every 05a draw stays valid.
//
// Posterior-supplied genomes are checked against the tab's own knowledge (controller ruling, Task 4 review). A
// posterior for another genome schema, another centre, or with an invalid calibration genome gives `centre` with a
// `reason`. drawSeed and posteriorSha are still recorded, so the gate recomputes the same fallback.
import { createRng, hashSeed } from "@blanktab/detmath"
import { checkGenome, fromUnit, searched, type Genome, type GenomeSpace } from "./genome"
import type { Posterior } from "./posterior"

export type Method = "centre" | "calibrate" | "uniform" | "ts" | "confirm" | "newborn" | "legacy"
export type Role = "search" | "confirm" | "newborn" | "legacy"
export interface DrawInput {
  learnerId: string; space: GenomeSpace; centre: Genome; widthsClass: "gpu" | "cpu"
  /** genomeSha of `centre` (the tab's knowledge/genome.json): the posterior must have been built for this centre. */
  centreSha: string
  /** null together: offline, absent or malformed. */
  posterior: Posterior | null; posteriorSha: string | null
}
export interface Draw {
  role: Role; method: Method; drawSeed: number | null; posteriorSha: string | null; evalPanel: number; genome: Genome
  calibrationIndex?: number
  /** Why a posterior was present but the draw fell back to centre. */
  reason?: string
}

/** The share of births that run a uniform point of the cube, in every mode (spec §4.1; OLS §10.1). */
export const UNIFORM_SHARE = 0.2
/** The number of shared evaluation panels (spec §4.1 step 4). */
export const PANELS = 16

/** The learner's shared checkup panel, 1..PANELS. */
export function evalPanelOf(learnerId: string): number {
  return 1 + (hashSeed(0, "panel/" + learnerId) % PANELS)
}

/** The seed of the learner's draw under the posterior whose bytes hash to `posteriorSha`. */
export function drawSeedOf(learnerId: string, posteriorSha: string): number {
  return hashSeed(0, "draw/" + learnerId + "/" + posteriorSha)
}

/** Index i with probability need[i] / Σneed, or uniform when Σneed = 0, from one uniform u in [0, 1). */
function pickByNeed(need: readonly number[], u: number): number {
  let total = 0
  for (const w of need) total += w
  if (total === 0) return Math.min(need.length - 1, Math.floor(u * need.length))
  const target = u * total
  let cum = 0, last = 0
  for (let i = 0; i < need.length; i++) {
    if (need[i]! === 0) continue
    last = i
    cum += need[i]!
    if (target < cum) return i
  }
  return last // only reached through rounding at the top end
}

const HEX64 = /^[0-9a-f]{64}$/

/**
 * The genome this birth runs, and how it was chosen. Every 05a draw has role "search". Throws on a programming error:
 * `posterior` and `posteriorSha` not null together, or a sha that is not 64 lowercase hex.
 */
export function birthDraw(input: DrawInput): Draw {
  const { learnerId, space, centre, centreSha, widthsClass, posterior, posteriorSha } = input
  if ((posterior === null) !== (posteriorSha === null)) throw new TypeError("posterior and posteriorSha must be null together")
  if (posteriorSha !== null && !HEX64.test(posteriorSha)) throw new TypeError(`posteriorSha must be 64 lowercase hex, got ${JSON.stringify(posteriorSha)}`)
  if (!HEX64.test(centreSha)) throw new TypeError(`centreSha must be 64 lowercase hex, got ${JSON.stringify(centreSha)}`)
  const evalPanel = evalPanelOf(learnerId)
  if (posterior === null || posteriorSha === null)
    return { role: "search", method: "centre", drawSeed: null, posteriorSha: null, evalPanel, genome: { ...centre } }

  const drawSeed = drawSeedOf(learnerId, posteriorSha)
  const base = { role: "search" as const, drawSeed, posteriorSha, evalPanel }
  const fallback = (reason: string): Draw => ({ ...base, method: "centre", genome: { ...centre }, reason })
  if (posterior.genomeSchema !== space.schemaVersion)
    return fallback(`posterior genomeSchema ${posterior.genomeSchema} is not the space's ${space.schemaVersion}`)
  if (posterior.centre.genomeSha !== centreSha) return fallback("posterior built for another centre")

  const rng = createRng(drawSeed)
  const u0 = rng()

  if (widthsClass === "gpu" && posterior.mode === "calibrate") {
    const { genomes, need } = posterior.calibration
    const i = pickByNeed(need, rng())
    const bad = checkGenome(space, genomes[i]!.values)
    if (bad !== null) return fallback(`calibration genome ${i} is invalid: ${bad}`)
    return { ...base, method: "calibrate", genome: { ...genomes[i]!.values }, calibrationIndex: i }
  }
  // search, paused and flat; and every mode on cpu. In 05a a search posterior has no model, so it draws like paused.
  if (u0 < UNIFORM_SHARE) {
    const u = searched(space).map(() => rng())
    return { ...base, method: "uniform", genome: fromUnit(space, u, centre) }
  }
  return { ...base, method: "centre", genome: { ...centre } }
}
