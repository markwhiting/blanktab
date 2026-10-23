// The birth draw (spec §4.1): which genome a new life runs. A pure function of the learnerId, the posterior's bytes
// (through its parsed form and sha), the genome space, the centre and the widths class, so the gate can recompute
// every pick. The order of rng() calls is fixed: u0 picks the method, then the method's own draws follow. 05b adds
// `ts` (search, gpu, u0 >= UNIFORM_SHARE) and `confirm` (u0 in [UNIFORM_SHARE, 0.7) while a confirmation is in
// flight) without touching the branches below, so every 05a draw stays valid.
import { createRng, hashSeed } from "@blanktab/detmath"
import { fromUnit, searched, type Genome, type GenomeSpace } from "./genome"
import type { Posterior } from "./posterior"

export type Method = "centre" | "calibrate" | "uniform" | "ts" | "confirm" | "newborn" | "legacy"
export type Role = "search" | "confirm" | "newborn" | "legacy"
export interface DrawInput {
  learnerId: string; space: GenomeSpace; centre: Genome; widthsClass: "gpu" | "cpu"
  /** null together: offline, absent or malformed. */
  posterior: Posterior | null; posteriorSha: string | null
}
export interface Draw {
  role: Role; method: Method; drawSeed: number | null; posteriorSha: string | null; evalPanel: number; genome: Genome
  calibrationIndex?: number
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

/** The genome this birth runs, and how it was chosen. Every 05a draw has role "search". */
export function birthDraw(input: DrawInput): Draw {
  const { learnerId, space, centre, widthsClass, posterior, posteriorSha } = input
  const evalPanel = evalPanelOf(learnerId)
  if (posterior === null || posteriorSha === null)
    return { role: "search", method: "centre", drawSeed: null, posteriorSha: null, evalPanel, genome: { ...centre } }

  const drawSeed = drawSeedOf(learnerId, posteriorSha)
  const rng = createRng(drawSeed)
  const u0 = rng()
  const base = { role: "search" as const, drawSeed, posteriorSha, evalPanel }

  if (widthsClass === "gpu" && posterior.mode === "calibrate") {
    const { genomes, need } = posterior.calibration
    const i = pickByNeed(need, rng())
    return { ...base, method: "calibrate", genome: { ...genomes[i]!.values }, calibrationIndex: i }
  }
  // search, paused and flat; and every mode on cpu. In 05a a search posterior has no model, so it draws like paused.
  if (u0 < UNIFORM_SHARE) {
    const u = searched(space).map(() => rng())
    return { ...base, method: "uniform", genome: fromUnit(space, u, centre) }
  }
  return { ...base, method: "centre", genome: { ...centre } }
}
