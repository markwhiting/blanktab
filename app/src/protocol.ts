// Messages between the page (main thread) and the learner's worker. The token never crosses this boundary.
import type { Manifest } from "@blanktab/harness/knowledge"
import type { CurvePoint } from "@blanktab/harness/log"
import type { Widths } from "@blanktab/learner"
import type { BackendKind } from "./backend"
import type { LifeConfig } from "./life"

export interface StartMessage {
  type: "start"
  cfg: LifeConfig
  /** Test knobs (never set by the production build): skip WebGPU; a newborn's widths. */
  backend?: "cpu"
  widths?: Widths
  /** document.hidden when the life starts (then VisibilityMessage on each change): no frames are posted while hidden. */
  hidden?: boolean
  /** The learner's checkpoint record, already read by the startup sweep that adopted it (its buffers transferred). */
  checkpointRecord?: unknown
}
export interface VisibilityMessage { type: "visibility"; hidden: boolean }
/** Refreshed knowledge (never offline), as its manifest: its knowledgeSha is in force for later log files; handed to the learner. */
export interface KnowledgeMessage { type: "knowledge"; manifest: Manifest }
/**
 * A forced checkpoint (before a code reload): answered by a `checkpointed` message with the same id. With `halt` the
 * learner stops right after the save, so the reload loses no step.
 */
export interface CheckpointNowMessage { type: "checkpointNow"; id: number; halt: boolean }
export type ToWorker = StartMessage | VisibilityMessage | KnowledgeMessage | CheckpointNowMessage

export type FromWorker =
  | { type: "backend"; kind: BackendKind; label: string }
  /** The learner as it runs: its settings (clamped, with the speed cap and seed) and widths, newborn or from a checkpoint. */
  | { type: "learner"; settings: Record<string, number>; widths: Widths }
  | { type: "frame"; steps: number; cu: number; stepsPerSec: number; periphery: Uint8Array; fovea: Uint8Array }
  /** `reset`: the whole curve so far (birth or resume); otherwise points to add. */
  | { type: "curve"; points: CurvePoint[]; reset: boolean }
  | { type: "status"; message: string }
  | { type: "queued"; seq: number; steps: number; cu: number; final: boolean }
  | { type: "final"; steps: number; cu: number }
  /** The answer to CheckpointNowMessage `id`: saved, or not (skipped updates outstanding, or the life ended): ask again later. */
  | { type: "checkpointed"; id: number; saved: boolean }
  | { type: "error"; message: string }
