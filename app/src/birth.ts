// A newborn's genome (Plan 05a): the knowledge's meta-learner files parsed into a draw context, the seeded birth draw
// (@blanktab/meta birthDraw, which the gate recomputes) decoded into what each component runs, and the session's meta
// block source. Pure apart from hashing, so it is testable; the worker calls it once the backend (widths class) is known.
import { DEFAULT_SETTINGS, type LearnerSettings } from "@blanktab/learner"
import { CAREGIVER_DEFAULTS, type CaregiverSettings } from "@blanktab/harness/caregiver"
import { CONFIG_BUDGET, type EvalBudget, type MetaBlock } from "@blanktab/harness/log"
import type { Knowledge } from "@blanktab/harness/knowledge"
import type { SessionMeta } from "@blanktab/harness/session"
import {
  birthDraw, decodeGenome, genomeSha, legacyGenome, parseCentre, parseGenomeSpace, parsePosterior, sha256Hex,
  type Genome, type GenomeSpace, type Posterior,
} from "@blanktab/meta"

/** What a birth draw needs from the knowledge: the space, the centre and its sha, and the posterior with the sha of its bytes. */
export interface DrawContext {
  space: GenomeSpace; centre: Genome; centreSha: string
  /** null together: absent, unreadable, or not a posterior this code reads (an unknown schemaVersion): the draw is centre. */
  posterior: Posterior | null; posteriorText: string | null; posteriorSha: string | null
}

/**
 * The draw context of `k`'s meta-learner files, or null when the genome space or the centre (knowledge/genome.json) is
 * missing or malformed, or does not decode: such a tab runs DEFAULT_SETTINGS and logs no meta block (Review Focus 5).
 * A posterior that does not parse gives posterior, posteriorText and posteriorSha null (the draw is then centre).
 */
export async function drawContextOf(k: Pick<Knowledge, "meta">): Promise<DrawContext | null> {
  if (!k.meta) return null
  let space: GenomeSpace, centre: Genome
  try {
    space = parseGenomeSpace(k.meta.genomeSpace)
    centre = parseCentre(k.meta.genome, space)
    decodeGenome(centre, space)
  } catch {
    return null
  }
  const centreSha = await genomeSha(space.schemaVersion, centre)
  const text = k.meta.posteriorText
  const posterior = text === null ? null : parsePosterior(text)
  if (posterior === null || text === null) return { space, centre, centreSha, posterior: null, posteriorText: null, posteriorSha: null }
  return { space, centre, centreSha, posterior, posteriorText: text, posteriorSha: await sha256Hex(new TextEncoder().encode(text)) }
}

/** What a newborn of a drawn genome runs, and the block its logs carry. `reason`: why a present posterior gave centre. */
export interface Birth {
  meta: MetaBlock; learnerSettings: LearnerSettings; caregiver: CaregiverSettings; maxStepsPerSec: number; sleepInterval: number
  reason?: string
}

/** Each key of `decoded` replaces the default's; a key the component does not have is an error (it would not run). */
function overlay<T extends object>(defaults: T, decoded: Record<string, number>, what: string): T {
  const out: Record<string, number> = { ...(defaults as Record<string, number>) }
  for (const [k, v] of Object.entries(decoded)) {
    if (!Object.hasOwn(defaults, k)) throw new RangeError(`the genome's ${what} gene ${k} is not a ${what} setting`)
    out[k] = v
  }
  return out as T
}

/** The meta block of a draw: its genome and sha at the space's schema, born under `budget` (decision 0020; none: legacy). */
async function block(space: GenomeSpace, d: Pick<MetaBlock, "role" | "method" | "drawSeed" | "posteriorSha" | "evalPanel" | "genome">, budget: EvalBudget | null): Promise<MetaBlock> {
  return {
    v: 1, role: d.role, method: d.method, genomeSchema: space.schemaVersion, genome: { ...d.genome }, genomeSha: await genomeSha(space.schemaVersion, d.genome),
    drawSeed: d.drawSeed, posteriorSha: d.posteriorSha, evalPanel: d.evalPanel, evalSeed: null, confirmation: null, inheritance: { filters: "none" },
    ...(budget ? { budget: { budgetCU: budget.budgetCU, levels: [...budget.levels] } } : {}),
  }
}

/**
 * A newborn's birth: birthDraw for `learnerId` in `ctx` at `widthsClass` (the gate's recomputation, spec §7.4), decoded
 * (decodeGenome): the learner and caregiver genes over DEFAULT_SETTINGS and CAREGIVER_DEFAULTS, the harness genes as the
 * speed cap and the sleep interval. The block carries `budget` (default the config's). Throws if the genome does not
 * decode to settings the components have (the worker then falls back to a centre draw).
 */
export async function birthFromDraw(ctx: DrawContext, learnerId: string, widthsClass: "gpu" | "cpu", budget: EvalBudget = CONFIG_BUDGET): Promise<Birth> {
  const d = birthDraw({
    learnerId, space: ctx.space, centre: ctx.centre, centreSha: ctx.centreSha, widthsClass, posterior: ctx.posterior, posteriorSha: ctx.posteriorSha,
  })
  const dec = decodeGenome(d.genome, ctx.space)
  return {
    meta: await block(ctx.space, d, budget),
    learnerSettings: overlay(DEFAULT_SETTINGS, dec.learner, "learner"),
    caregiver: overlay(CAREGIVER_DEFAULTS, dec.caregiver, "caregiver"),
    maxStepsPerSec: dec.harness.maxStepsPerSec, sleepInterval: dec.harness.sleepInterval,
    ...(d.reason !== undefined ? { reason: d.reason } : {}),
  }
}

/** The same context without its posterior: what a birth falls back to (a centre draw). */
export const withoutPosterior = (ctx: DrawContext): DrawContext => ({ ...ctx, posterior: null, posteriorText: null, posteriorSha: null })

/**
 * The session's meta source (SessionMeta) for a life born with `birth` in `ctx`. A resumed pre-05a life gets a legacy
 * block (legacyGenome of its logged settings: role and method legacy, no panel, its unrecorded genes in `missing`).
 */
export function sessionMetaOf(ctx: DrawContext, birth: MetaBlock, budget: EvalBudget = CONFIG_BUDGET): SessionMeta {
  return {
    space: ctx.space, birth, budget,
    async legacy(flat) {
      const { genome, missing } = legacyGenome(flat, ctx.space)
      // No budget: a legacy life runs under the v1 budget (lifeBudget), and the parser refuses one on it.
      const b = await block(ctx.space, { role: "legacy", method: "legacy", drawSeed: null, posteriorSha: null, evalPanel: null, genome }, null)
      return { ...b, ...(missing.length > 0 ? { missing } : {}) }
    },
  }
}
