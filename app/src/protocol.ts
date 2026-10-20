// Messages between the page (main thread) and the learner's worker. The token never crosses this boundary.
import type { SessionConfig } from "@blanktab/harness/session"

export interface StartMessage { type: "start"; cfg: SessionConfig; stepsPerSecond: number }
export type ToWorker = StartMessage

export type FromWorker =
  | { type: "frame"; steps: number; periphery: Uint8Array; fovea: Uint8Array }
  | { type: "queued"; seq: number; steps: number; final: boolean }
  | { type: "final"; steps: number }
  | { type: "error"; message: string }
