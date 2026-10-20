import type { Learner, Observation } from "@blanktab/interface"
import type { World } from "@blanktab/world"
import type { LogFile } from "./log"
import type { Queue } from "./queue"

export interface SessionConfig {
  learnerId: string
  codeSha: string
  knowledgeSha: string
  sessionsSeenAtBirth: number
  settings: Record<string, number | string>
  hostSig: LogFile["hostSig"]
  lifetimeSteps: number
  logEverySteps: number
  startSeq: number
  startSteps: number
}

const YIELD_EVERY = 50
const yieldToEventLoop = () => new Promise<void>(r => setTimeout(r, 0))

export async function runSession(
  cfg: SessionConfig,
  deps: {
    world: World; learner: Learner; queue: Queue; now: () => Date
    onTick?: (s: { steps: number; obs: Observation }) => void
    /** Awaited after every step (e.g. to hold a real-time pace). Without it the loop yields every YIELD_EVERY steps. */
    pace?: (steps: number) => Promise<void>
  },
): Promise<void> {
  const { world, learner, queue, now, onTick, pace } = deps
  if (!Number.isInteger(cfg.logEverySteps) || cfg.logEverySteps <= 0) throw new Error("logEverySteps must be a positive integer")
  let seq = cfg.startSeq
  let steps = cfg.startSteps
  while (steps < cfg.lifetimeSteps) {
    const obs = world.observe()
    world.act(await learner.step(obs))
    // The privileged trace is for milestone detectors, which a later plan adds; until then it is discarded.
    // Draining it every step keeps it from growing without bound over a lifetime.
    world.drainTrace()
    steps++
    onTick?.({ steps, obs })
    if (steps % cfg.logEverySteps === 0 || steps >= cfg.lifetimeSteps) {
      await queue.put({
        schemaVersion: 1,
        learnerId: cfg.learnerId,
        seq: seq++,
        codeSha: cfg.codeSha,
        knowledgeSha: cfg.knowledgeSha,
        writtenAt: now().toISOString(),
        stepsTotal: steps,
        settings: { ...cfg.settings },
        hostSig: { ...cfg.hostSig },
        sessionsSeenAtBirth: cfg.sessionsSeenAtBirth,
        final: steps >= cfg.lifetimeSteps,
      })
    }
    if (pace) await pace(steps)
    else if (steps % YIELD_EVERY === 0) await yieldToEventLoop()
  }
}

/** Steps after the last queued file are lost on interruption; numbering continues after it. */
export async function resumePoint(queue: Queue, learnerId: string): Promise<{ startSeq: number; startSteps: number }> {
  const last = await queue.latest(learnerId)
  return last ? { startSeq: last.seq + 1, startSteps: last.stepsTotal } : { startSeq: 0, startSteps: 0 }
}
