// newborn-baseline@1 (spec §2.3): the newborn reference, computed from newborn lives (untrained learners of the centre
// genome, frozen checkups at 512, 1,024 and 2,048 on the panels search lives meet, and one held-out score each).
import type { LogFileV2 } from "@blanktab/harness/log"
import { isGpuWidths, isV2, median } from "./life"
import { REFERENCE_MILESTONES } from "./reference"
import type { EvalOutput, Evaluator, LifeInput, NewbornReference } from "./types"

const ID = "newborn-baseline"
const VERSION = 1
const LEVELS: readonly number[] = [512, 1024, 2048]
/** A sustained pass: passing at this many of the LEVELS or more (spec §2.3, V2 §6.3). */
const SUSTAINED = 2

/** The newborn file this version counts: role newborn, genome schema 1, suite 1, GPU widths, checkups at exactly LEVELS. */
function newbornFile(life: LifeInput): LogFileV2 | null {
  const f = life.files.find(x => isV2(x) && x.meta?.role === "newborn")
  if (!f || !isV2(f) || f.learnerId !== life.learnerId) return null
  const cks = f.checkups ?? []
  const ok = f.meta!.genomeSchema === 1 && f.milestoneSuiteVersion === 1 && isGpuWidths(f.widths)
    && cks.length === LEVELS.length && cks.every((c, i) => c.level === LEVELS[i])
  return ok ? f : null
}

function evaluateCohort(groups: Record<string, LifeInput[]>): EvalOutput {
  const counted = new Map<string, LogFileV2>()
  let ignored = 0
  for (const life of groups.newborn ?? []) {
    const f = newbornFile(life)
    if (!f || counted.has(life.learnerId)) { ignored++; continue }
    counted.set(life.learnerId, f)
  }
  const learnerIds = [...counted.keys()].sort()
  const files = learnerIds.map(id => counted.get(id)!)
  const out = (eligible: boolean, reason: string | undefined, components: EvalOutput["components"], diagnostics: Record<string, unknown>): EvalOutput =>
    ({ evaluator: ID, version: VERSION, eligible, ...(reason ? { reason } : {}), score: null, components, diagnostics })
  if (files.length === 0) return out(false, "no newborn lives", { lives: 0, heldOutMedian: null }, { ignored })
  const ratios = files.flatMap(f => (f.meta!.heldOut && f.meta!.heldOut.baseline > 0 ? [f.meta!.heldOut.loss / f.meta!.heldOut.baseline] : []))
  const heldOutMedian = median(ratios)
  if (heldOutMedian === null || !(heldOutMedian > 0)) {
    return out(false, "no newborn held-out scores", { lives: files.length, heldOutMedian: null }, { ignored, learnerIds })
  }

  const milestones = {} as NewbornReference["milestones"]
  for (const m of REFERENCE_MILESTONES) {
    const values: number[] = []
    let passes = 0, sustained = 0
    for (const f of files) {
      let lifePasses = 0
      for (const c of f.checkups!) {
        values.push(c.measures[m] ?? 0)
        if (c.passed.includes(m)) { passes++; lifePasses++ }
      }
      if (lifePasses >= SUSTAINED) sustained++
    }
    milestones[m] = { median: median(values)!, passRate: passes / values.length, sustainedPassRate: sustained / files.length }
  }
  const reference: NewbornReference = {
    evaluator: "newborn-baseline@1", suiteVersion: 1, genomeSchema: 1, widthsClass: "gpu", learnerIds, milestones, heldOutMedian,
  }
  return out(true, undefined, { lives: files.length, heldOutMedian }, { reference, ignored, heldOutScores: ratios.length })
}

export const newbornBaseline: Evaluator = Object.freeze({
  id: ID, version: VERSION, kind: "cohort" as const, needs: [], evaluateCohort,
})
