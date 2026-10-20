// Messages between the page (main thread) and the learner's worker. The token never crosses this boundary.
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
}
export interface VisibilityMessage { type: "visibility"; hidden: boolean }
export type ToWorker = StartMessage | VisibilityMessage

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
  | { type: "error"; message: string }
