// One learner's life on the best backend this machine offers, surviving a lost GPU (Review Focus 3).
import { loadCheckpoint, saveCheckpoint } from "@blanktab/harness/checkpoint"
import type { LogFileV2 } from "@blanktab/harness/log"
import type { SessionConfig, SessionResult } from "@blanktab/harness/session"
import { ResourceError, type WebGpuBackend } from "@blanktab/tensor"
import { cpuFallback, type Chosen } from "./backend"

/** Two uint32 seeds from the learner id (lrn_ + 16 hex): the world's (its crib) and a newborn learner's weights'. */
export const seedsOf = (learnerId: string) => ({ world: parseInt(learnerId.slice(4, 12), 16), learner: parseInt(learnerId.slice(12, 20), 16) })

/** A SessionConfig before the backend is known: widths and hostSig.webgpu come from the chosen backend. */
export type LifeConfig = Omit<SessionConfig, "widths" | "hostSig"> & { hostSig: Omit<LogFileV2["hostSig"], "webgpu"> }

/** Why a run resumes after a GPU failure: shown, with the step it resumed from, in the Status line. */
export interface Restore { reason: string; again: boolean; on: Chosen["kind"] }

/**
 * Runs (or resumes) the life on one backend: runSession with a learner on `chosen.backend`. `restore` is set when this
 * run is the restore after a GPU failure, so its `resumed` event is reported as that restore (resumeStatus).
 */
export type RunLife = (cfg: SessionConfig, chosen: Chosen, restore: Restore | null) => Promise<SessionResult>

/**
 * Whether a run's error is one liveOnBestBackend recovers from (a rejected read on the WebGPU backend): pass it to
 * runSession as `recoverable`, so such a run is not ended on the record (every other error ends the life there).
 */
export const isRecoverable = (chosen: Chosen) => (e: unknown): boolean => chosen.kind === "webgpu" && e instanceof ResourceError

/** The Status line for a session's `resumed` event: a reload, or the restore after a GPU failure. */
export function resumeStatus(steps: number, restore: Restore | null): string {
  if (!restore) return `resumed from its checkpoint at step ${steps}`
  const where = restore.on === "webgpu" ? "on WebGPU" : "on the CPU at the same widths"
  return `A GPU read failed${restore.again ? " again" : ""} (${restore.reason}); restored from the checkpoint at step ${steps} ${where}`
}

export interface LifeDeps {
  /** The backend choice (chooseBackend); asked again for a fresh GPU after the first GPU failure. */
  choose(): Promise<Chosen>
  /** The CPU backend after a second GPU failure (default cpuFallback at the failed life's widths). */
  cpu?(widths: Chosen["widths"]): Chosen
  run: RunLife
  /** Holds the checkpoints. */
  idb: IDBFactory
  /** The CU the learner had lived when its run failed (the latest onTick), for the record. */
  lastCU(): number
  onBackend?(c: Chosen): void
  onStatus?(message: string): void
}

/**
 * A failed GPU run threw away the work since the checkpoint: put that on the record (LogFileV2.restores and lostCU,
 * which the checkpoint carries), as runSession does for its own restores.
 */
async function recordRestore(idb: IDBFactory, learnerId: string, cuAtFailure: number): Promise<number | null> {
  const cp = await loadCheckpoint<unknown>(idb, learnerId)
  if (!cp) return null
  await saveCheckpoint(idb, { ...cp, restores: cp.restores + 1, lostCU: cp.lostCU + Math.max(0, cuAtFailure - cp.cu) })
  return cp.steps
}

/**
 * Lives the life on `choose()`'s backend. A rejected GPU read (a ResourceError on the WebGPU backend: a lost device,
 * a GPU error) is fatal to that run (packages/tensor/README.md), so:
 *  1. the first one restores the life from its last checkpoint on a fresh WebGPU backend (or, if none is offered,
 *     on the CPU at the same widths);
 *  2. a second one restores it from the checkpoint on the CPU backend at the same widths (a GPU checkpoint is valid
 *     on the CPU, only slower), for the rest of this tab's life.
 * The failed backend is destroyed first. The restored run's `resumed` event is reported as the restore (resumeStatus).
 * Each restore is on the record (the checkpoint's restores and lostCU) and reported through onStatus. Any other error,
 * and any error on the CPU, ends the life (the page starts a newborn); `run` must pass isRecoverable(chosen) to
 * runSession, so that those, and only those, are ended on the record with a final log file.
 */
export async function liveOnBestBackend(cfg: LifeConfig, deps: LifeDeps): Promise<SessionResult> {
  let chosen = await deps.choose()
  let gpuFailures = 0
  let restore: Restore | null = null
  const cpu = deps.cpu ?? cpuFallback
  for (;;) {
    deps.onBackend?.(chosen)
    try {
      return await deps.run({ ...cfg, widths: chosen.widths, hostSig: { ...cfg.hostSig, webgpu: chosen.kind === "webgpu" } }, chosen, restore)
    } catch (e) {
      if (!isRecoverable(chosen)(e) || !(e instanceof Error)) throw e
      gpuFailures++
      const at = await recordRestore(deps.idb, cfg.learnerId, deps.lastCU())
      // Let the failed device go before asking for another (a lost device's memory is not otherwise freed).
      const failed = chosen.backend as Partial<Pick<WebGpuBackend, "destroy">>
      failed.destroy?.()
      // After any GPU failure a CPU run keeps the failed life's widths and says why it is on the CPU.
      const next = gpuFailures === 1 ? await deps.choose() : null
      chosen = next?.kind === "webgpu" ? next : cpu(chosen.widths)
      restore = { reason: e.message, again: gpuFailures > 1, on: chosen.kind }
      const from = at === null ? "from birth (there was no checkpoint yet)" : `from its checkpoint at step ${at}`
      deps.onStatus?.(`A GPU read failed${restore.again ? " again" : ""} (${e.message}); restoring ${from} ${chosen.kind === "webgpu" ? "on WebGPU" : "on the CPU at the same widths"}`)
    }
  }
}
