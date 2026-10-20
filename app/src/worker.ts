// The learner's thread: owns the OffscreenCanvas, the world, the tensor backend, the learner and its session.
// The session writes log files and checkpoints to the shared IndexedDB; this thread tells the page what happened.
// It never sees the token.
import { WORLD } from "@blanktab/interface"
import { createWorld } from "@blanktab/world"
import { createLearner } from "@blanktab/learner"
import { createWebGpuBackend } from "@blanktab/tensor"
import type { CurvePoint } from "@blanktab/harness/log"
import { openQueue } from "@blanktab/harness/queue"
import { DEFAULT_MAX_STEPS_PER_SEC, runSession, type SessionLearner } from "@blanktab/harness/session"
import { chooseBackend } from "./backend"
import { liveOnBestBackend, resumeStatus, seedsOf, type RunLife } from "./life"
import type { FromWorker, StartMessage, ToWorker } from "./protocol"

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null
  postMessage(m: FromWorker, transfer?: Transferable[]): void
}
const post = (m: FromWorker, transfer: Transferable[] = []) => scope.postMessage(m, transfer)

const FRAME_MS = 33 // at most ~30 frames a second reach the page
const RATE_WINDOW_MS = 3000

/** Steps per second over the last RATE_WINDOW_MS. */
function rateMeter(): (steps: number) => number {
  const samples: [number, number][] = []
  return steps => {
    const t = performance.now()
    samples.push([t, steps])
    while (samples.length > 2 && samples[0]![0] < t - RATE_WINDOW_MS) samples.shift()
    const [t0, s0] = samples[0]!
    return t > t0 ? ((steps - s0) * 1000) / (t - t0) : 0
  }
}

async function start({ cfg, backend: force, widths }: StartMessage): Promise<void> {
  const id = cfg.learnerId
  const world = createWorld({ seed: seedsOf(id).world, canvas: new OffscreenCanvas(WORLD.width, WORLD.height) })
  const queue = await openQueue(indexedDB)
  const rate = rateMeter()
  let steps = 0
  let cu = 0
  let lastFrame = Number.NEGATIVE_INFINITY

  const run: RunLife = async (c, chosen, restore) => {
    // Every log file this learner has written comes before its checkpoint, so its curve is the start of the resumed one.
    const prior: CurvePoint[] = (await queue.files(id)).flatMap(f => (f.schemaVersion === 2 ? f.curve : []))
    return runSession(c, {
      world, idb: indexedDB, now: () => new Date(),
      createLearner: async (init): Promise<SessionLearner> => {
        const l = await createLearner({ backend: chosen.backend, ...init })
        const birth = "state" in init ? init.state : init
        post({ type: "learner", settings: { ...birth.settings, maxStepsPerSec: c.maxStepsPerSec ?? DEFAULT_MAX_STEPS_PER_SEC, seed: birth.seed }, widths: { ...birth.widths } })
        if (!("state" in init)) post({ type: "curve", points: [], reset: true })
        return {
          step: o => l.step(o), stats: () => l.stats(), save: () => l.save(), dispose: () => l.dispose(),
          async sleep() {
            await l.sleep()
            const s = l.stats()
            if (Number.isFinite(s.predLoss) && Number.isFinite(s.baselineLoss)) {
              post({ type: "curve", points: [{ cu: s.cu, steps, predLoss: s.predLoss, baselineLoss: s.baselineLoss }], reset: false })
            }
          },
        }
      },
      onTick: ({ steps: n, cu: now, obs }) => {
        steps = n
        cu = now
        const perSec = rate(n)
        const t = performance.now()
        if (t - lastFrame < FRAME_MS) return
        lastFrame = t
        // Copies: the learner may keep its observation, so its buffers are never transferred away.
        const periphery = obs.periphery.slice(), fovea = obs.fovea.slice()
        post({ type: "frame", steps: n, cu: now, stepsPerSec: perSec, periphery, fovea }, [periphery.buffer, fovea.buffer])
      },
      onEvent: e => {
        if (e.type === "resumed") {
          steps = e.steps
          cu = e.cu
          post({ type: "curve", points: [...prior, ...e.curveTail], reset: true })
          post({ type: "status", message: resumeStatus(e.steps, restore) })
        } else if (e.type === "restored") {
          post({ type: "status", message: `its updates kept failing at step ${e.fromSteps}: restored from its checkpoint at step ${e.toSteps} (${e.restores} restore${e.restores === 1 ? "" : "s"} so far)` })
        } else if (e.type === "logged") {
          post({ type: "queued", seq: e.seq, steps: e.steps, cu: e.cu, final: e.final })
        }
      },
    })
  }

  const r = await liveOnBestBackend(cfg, {
    choose: () => chooseBackend({ createGpu: () => createWebGpuBackend({ powerPreference: "high-performance" }), ...(force ? { force } : {}), ...(widths ? { widths } : {}) }),
    run, idb: indexedDB, lastCU: () => cu,
    onBackend: c => post({ type: "backend", kind: c.kind, label: c.label }),
    onStatus: message => post({ type: "status", message }),
  })
  post({ type: "final", steps: r.steps, cu: r.cu })
}

let started = false
scope.onmessage = e => {
  if (e.data.type !== "start" || started) return
  started = true
  start(e.data).catch((err: unknown) => post({ type: "error", message: err instanceof Error ? err.message : String(err) }))
}
