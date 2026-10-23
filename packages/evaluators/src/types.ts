// The evaluator interface (spec §2.1). Evaluators are pure and deterministic: no IO, no clock, no Math.random, and all
// maths through @blanktab/detmath, so the same input bytes give the same output bytes (plan 05a Global Constraints).
import type { LogFile } from "@blanktab/harness/log"

/** One learner's merged log files, sorted by seq. */
export interface LifeInput { learnerId: string; files: LogFile[] }

export interface EvalOutput {
  evaluator: string; version: number // e.g. "search-composite", 1
  eligible: boolean; reason?: string // why not, e.g. "no 2048 checkup"
  score: number | null // the headline number
  components: Record<string, number | null> // e.g. { C, H }
  diagnostics: Record<string, unknown> // per-milestone q's, levels used, ...
}

export type MilestoneKey = "M1" | "M2" | "M3" | "M4" | "M5" | "M6"

/** The newborn reference (spec §2.3): what `newborn-baseline@1` computes, pinned by a person under a decision. */
export interface NewbornReference {
  evaluator: "newborn-baseline@1"; suiteVersion: 1; genomeSchema: 1; widthsClass: "gpu"
  learnerIds: string[]
  milestones: Record<MilestoneKey, { median: number; passRate: number; sustainedPassRate: number }>
  heldOutMedian: number
}

/**
 * The pinned reference files, by name. An evaluator declares the names it reads in `needs`; tools/eval/config.ts loads
 * exactly those, each with its parser from REFERENCE_PARSERS. An absent one → the evaluator says it is ineligible.
 * `newborn` is the only reference in v1; a new one is a parser entry plus a typed key here.
 */
export interface References { newborn?: NewbornReference; [name: string]: unknown }

/** A reference file's parser: the parsed reference, or null if the JSON is not one. */
export type ReferenceParser = (v: unknown) => unknown

/** The evaluation budget an evaluator reads (a life evaluator's own constants; config/evaluation.json must agree). */
export interface EvalBudget { budgetCU: number; levels: readonly number[] }

export interface Evaluator {
  id: string; version: number; kind: "life" | "cohort"
  needs: string[] // reference names it reads, e.g. ["newborn"]: the source of truth for what config loads
  budget?: EvalBudget // the CU budget and checkup levels it reads (life evaluators)
  evaluateLife?(life: LifeInput, refs: References): EvalOutput // kind "life"
  evaluateCohort?(groups: Record<string, LifeInput[]>, refs: References): EvalOutput // kind "cohort": a test over arms
  targets?(outs: EvalOutput[]): number[] // search only: the GP target transform (spec §2.4)
}

/** The GPU widths class (the learner's GPU_WIDTHS; tools/eval/config.test.ts pins the equality, as the wall forbids the import). */
export const GPU_WIDTHS_CLASS = { conv1: 16, conv2: 32, enc: 128, hidden: 256 } as const
