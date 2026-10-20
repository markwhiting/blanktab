// The learner's thread: owns the OffscreenCanvas, the world, the learner and the session loop.
// It writes log files to the shared IndexedDB queue and tells the page what happened. It never sees the token.
import { WORLD } from "@blanktab/interface"
import { createWorld } from "@blanktab/world"
import { createFlailer } from "@blanktab/learner"
import { openQueue, type Queue } from "@blanktab/harness/queue"
import { runSession } from "@blanktab/harness/session"
import type { FromWorker, ToWorker } from "./protocol"

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null
  postMessage(m: FromWorker, transfer?: Transferable[]): void
}
const post = (m: FromWorker, transfer: Transferable[] = []) => scope.postMessage(m, transfer)

const FRAME_MS = 33 // at most ~30 frames a second reach the page
const MAX_LAG_MS = 1000 // after a long stall (sleep, throttling) resume the pace instead of racing to catch up

/** Holds the loop to `stepsPerSecond` of wall time. */
function pacer(stepsPerSecond: number): () => Promise<void> {
  const interval = 1000 / stepsPerSecond
  let due = performance.now()
  return () => {
    due += interval
    const now = performance.now()
    if (due < now - MAX_LAG_MS) due = now
    return new Promise(r => setTimeout(r, Math.max(0, due - now)))
  }
}

/** Two 32-bit seeds from the learner id, so a resumed learner lives in the same crib. */
const seedsOf = (learnerId: string) => ({ world: parseInt(learnerId.slice(4, 12), 16), learner: parseInt(learnerId.slice(12, 20), 16) })

async function start({ cfg, stepsPerSecond }: ToWorker): Promise<void> {
  const moveProbability = cfg.settings.moveProbability
  if (typeof moveProbability !== "number") throw new Error("settings.moveProbability must be a number")
  const seeds = seedsOf(cfg.learnerId)
  const world = createWorld({ seed: seeds.world, canvas: new OffscreenCanvas(WORLD.width, WORLD.height) })
  const learner = createFlailer({ seed: seeds.learner, moveProbability })
  const outbox = await openQueue(indexedDB)
  const queue: Queue = {
    ...outbox,
    async put(f) {
      await outbox.put(f)
      post({ type: "queued", seq: f.seq, steps: f.stepsTotal, final: f.final })
    },
  }
  let lastFrame = -Infinity
  await runSession(cfg, {
    world, learner, queue, now: () => new Date(), pace: pacer(stepsPerSecond),
    onTick: ({ steps, obs }) => {
      const t = performance.now()
      if (t - lastFrame < FRAME_MS && steps < cfg.lifetimeSteps) return
      lastFrame = t
      // Copies: the learner may keep its observation, so its buffers are never transferred away.
      const periphery = obs.periphery.slice(), fovea = obs.fovea.slice()
      post({ type: "frame", steps, periphery, fovea }, [periphery.buffer, fovea.buffer])
    },
  })
  post({ type: "final", steps: cfg.lifetimeSteps })
}

let started = false
scope.onmessage = e => {
  if (e.data.type !== "start" || started) return
  started = true
  start(e.data).catch((err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }))
}
