// The learner's thread: owns the OffscreenCanvas, the world, the tensor backend, the learner and its session.
// The session writes log files and checkpoints to the shared IndexedDB; this thread tells the page what happened.
// It never sees the token.
import { WORLD } from "@blanktab/interface"
import { createWorld } from "@blanktab/world"
import { buildHeldOut } from "@blanktab/harness/heldout"
import { LearnerStateVersionError, createLearner } from "@blanktab/learner"
import { createWebGpuBackend } from "@blanktab/tensor"
import type { Manifest } from "@blanktab/harness/knowledge"
import type { CurvePoint } from "@blanktab/harness/log"
import { openQueue } from "@blanktab/harness/queue"
import { DEFAULT_MAX_STEPS_PER_SEC, SessionInbox, curvePoint, runSession, type SessionLearner } from "@blanktab/harness/session"
import { chooseBackend } from "./backend"
import { frameGate } from "./frames"
import { isRecoverable, liveOnBestBackend, resumeStatus, seedsOf, type RunLife } from "./life"
import type { FromWorker, StartMessage, ToWorker } from "./protocol"

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null
  postMessage(m: FromWorker, transfer?: Transferable[]): void
}
const post = (m: FromWorker, transfer: Transferable[] = []) => scope.postMessage(m, transfer)

const FRAME_MS = 33 // at most ~30 frames a second reach the page
/** No frames while the page is hidden (the page says when it is). */
const frames = frameGate(FRAME_MS)
const RATE_WINDOW_MS = 3000
/** The page's refreshed knowledge and forced checkpoints, for whichever run of the life is current (it outlives GPU restores). */
const inbox = new SessionInbox()

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

async function start({ cfg, backend: force, widths, hidden, checkpointRecord }: StartMessage): Promise<void> {
  /** Used by the first run only: a restore after a GPU failure reads the (updated) checkpoint from the database. */
  let record = checkpointRecord
  frames.setHidden(hidden ?? false)
  const id = cfg.learnerId
  const world = createWorld({ seed: seedsOf(id).world, canvas: new OffscreenCanvas(WORLD.width, WORLD.height) })
  const queue = await openQueue(indexedDB)
  const rate = rateMeter()
  let steps = 0
  let cu = 0

  const run: RunLife = async (c, chosen, restore) => {
    // Every log file this learner has written comes before its checkpoint, so its curve is the start of the resumed one.
    const prior: CurvePoint[] = (await queue.files(id)).flatMap(f => (f.schemaVersion === 2 ? f.curve : []))
    const adopted = record
    record = undefined
    return runSession(c, {
      ...(adopted !== undefined ? { checkpointRecord: adopted } : {}),
      world, idb: indexedDB, now: () => new Date(), recoverable: isRecoverable(chosen), inbox,
      createLearner: async (init): Promise<SessionLearner> => {
        // A newborn's held-out set (decision 0015 §3) is recorded here, at birth, from a world of its own; a resumed
        // learner carries its own in its state. The learner sees it only as data.
        const l = await createLearner({
          backend: chosen.backend, ...init,
          ...("state" in init ? {} : { heldOut: buildHeldOut(init.seed, new OffscreenCanvas(WORLD.width, WORLD.height)) }),
        })
        const birth = "state" in init ? init.state : init
        post({ type: "learner", settings: { ...birth.settings, maxStepsPerSec: c.maxStepsPerSec ?? DEFAULT_MAX_STEPS_PER_SEC, seed: birth.seed }, widths: { ...birth.widths } })
        if (!("state" in init)) post({ type: "curve", points: [], reset: true })
        // Refreshed knowledge goes to the learner if it takes it (the 04a learner does not; its logs still record the sha in force).
        const takesKnowledge = l as { onKnowledge?(m: Manifest): void }
        return {
          step: o => l.step(o), stats: () => l.stats(), save: () => l.save(), dispose: () => l.dispose(),
          onKnowledge: k => takesKnowledge.onKnowledge?.(k),
          async sleep() {
            await l.sleep()
            const p = curvePoint(l.stats(), steps)
            if (p) post({ type: "curve", points: [p], reset: false })
          },
        }
      },
      onTick: ({ steps: n, cu: now, obs }) => {
        steps = n
        cu = now
        const perSec = rate(n)
        if (!frames.due(performance.now())) return
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
  // Halted after a forced save: the life is not over, and the page reloads onto that save (even if it had stopped
  // waiting for the answer; any request it makes from now on is answered true at once by the inbox).
  post(r.halted ? { type: "halted" } : { type: "final", steps: r.steps, cu: r.cu })
}

let started = false
scope.onmessage = e => {
  const m = e.data
  if (m.type === "visibility") { frames.setHidden(m.hidden); return }
  if (m.type === "knowledge") { inbox.setKnowledge(m.manifest); return }
  if (m.type === "checkpointNow") { void inbox.checkpointNow({ halt: m.halt }).then(saved => post({ type: "checkpointed", id: m.id, saved })); return }
  if (m.type !== "start" || started) return
  started = true
  start(m).catch((err: unknown) => post({
    type: "error", message: err instanceof Error ? err.message : String(err),
    ...(err instanceof LearnerStateVersionError && err.older ? { olderVersion: true } : {}),
  }))
}
