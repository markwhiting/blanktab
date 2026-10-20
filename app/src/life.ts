// One learner's life on the best backend this machine offers, surviving a lost GPU (Review Focus 3).
import { loadCheckpoint, saveCheckpoint } from "@blanktab/harness/checkpoint"
import type { LogFileV2 } from "@blanktab/harness/log"
import type { SessionConfig, SessionResult } from "@blanktab/harness/session"
import { ResourceError } from "@blanktab/tensor"
import { cpuFallback, type Chosen } from "./backend"

/** Two uint32 seeds from the learner id (lrn_ + 16 hex): the world's (its crib) and a newborn learner's weights'. */
export const seedsOf = (learnerId: string) => ({ world: parseInt(learnerId.slice(4, 12), 16), learner: parseInt(learnerId.slice(12, 20), 16) })

/** A SessionConfig before the backend is known: widths and hostSig.webgpu come from the chosen backend. */
export type LifeConfig = Omit<SessionConfig, "widths" | "hostSig"> & { hostSig: Omit<LogFileV2["hostSig"], "webgpu"> }

/** Runs (or resumes) the life on one backend: runSession with a learner on `chosen.backend`. */
export type RunLife = (cfg: SessionConfig, chosen: Chosen) => Promise<SessionResult>

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
 *  1. the first one restores the life from its last checkpoint on a fresh WebGPU backend;
 *  2. a second one restores it from the checkpoint on the CPU backend at the same widths (a GPU checkpoint is valid
 *     on the CPU, only slower), for the rest of this tab's life.
 * Each restore is on the record (the checkpoint's restores and lostCU) and reported through onStatus. Any other error,
 * and any error on the CPU, ends the life (the page starts a newborn).
 */
export async function liveOnBestBackend(cfg: LifeConfig, deps: LifeDeps): Promise<SessionResult> {
  let chosen = await deps.choose()
  let gpuFailures = 0
  for (;;) {
    deps.onBackend?.(chosen)
    try {
      return await deps.run({ ...cfg, widths: chosen.widths, hostSig: { ...cfg.hostSig, webgpu: chosen.kind === "webgpu" } }, chosen)
    } catch (e) {
      if (chosen.kind !== "webgpu" || !(e instanceof ResourceError)) throw e
      gpuFailures++
      const at = await recordRestore(deps.idb, cfg.learnerId, deps.lastCU())
      const from = at === null ? "from birth (there was no checkpoint yet)" : `from its checkpoint at step ${at}`
      if (gpuFailures === 1) {
        deps.onStatus?.(`A GPU read failed (${e.message}); restoring ${from} on WebGPU.`)
        chosen = await deps.choose()
      } else {
        deps.onStatus?.(`A GPU read failed again (${e.message}); restoring ${from} on the CPU at the same widths.`)
        chosen = (deps.cpu ?? cpuFallback)(chosen.widths)
      }
    }
  }
}
