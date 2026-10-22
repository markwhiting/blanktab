// Milestone modules (spec §8.1, amended by decision 0016). Everything here is privileged and harness only: a milestone
// reads the world's state and trace to judge behaviour; none of it ever reaches the learner, which only acts on
// observations through the policy.
import type { Action, Observation } from "@blanktab/interface"
import type { FrozenResult } from "@blanktab/learner"
import type { CanvasLike, PrivilegedState, RoomKind, TraceEvent, World } from "@blanktab/world"

export type MilestoneId = "M1" | "M2" | "M3" | "M4" | "M5" | "M6"

/** Privileged, harness only: the state before the first action, then after each one, and every trace event. */
export interface CheckupTrace { initial: PrivilegedState; states: PrivilegedState[]; events: TraceEvent[] }

/** What acts in a checkup world: in a checkup, the learner's frozen run; in tests, a scripted or random stand-in. */
export type Policy = (o: Observation) => Action | Promise<Action>

export interface Milestone {
  id: MilestoneId
  version: number
  description: string
  room: RoomKind | "desktop" // "desktop": a world of more than one room
  prerequisites: readonly MilestoneId[] // metadata for the ladder; not a gate in v1
  budgetSteps: number
  measuresLoss: boolean // M1 only
  /** A fresh world for this milestone, a pure function of `seed`, drawing on `canvas`. */
  makeVariant(seed: number, canvas: CanvasLike): World
  /** M5: the random clicker's distinct pages on the same variant. */
  reference?(seed: number, canvas: CanvasLike): number
  /** The measured value (NaN when there is nothing to measure; a checkup logs a non-finite measure as null). */
  measure(trace: CheckupTrace, stats: FrozenResult, reference?: number): number
  passes(trace: CheckupTrace, stats: FrozenResult, reference?: number): boolean
}
