// The page's main thread: knowledge, identity, the GitHub contributor and the token live here.
// The learner itself runs in ./worker.ts; this thread only starts it and shows what it reports.
import { TICKS_PER_WORLD_SECOND } from "@blanktab/interface"
import {
  REJECTED_MESSAGE, checkToken, chooseSettings, createContributor, getOrCreateLearnerId, getToken, loadKnowledge,
  newLearnerId, openQueue, resumePoint, setToken,
  type Knowledge, type LogFile, type Queue, type SessionConfig,
} from "@blanktab/harness"
import type { FromWorker, ToWorker } from "./protocol"
import { readTestParams, type TestParams } from "./params"
import { createUi, type Ui } from "./ui"

const LIFETIME_STEPS = 3000
const LOG_EVERY_STEPS = 500
const STEPS_PER_SECOND = TICKS_PER_WORLD_SECOND // real time: a 3000-step life takes five minutes
const FLUSH_EVERY_MS = 60_000
const NEWBORN_DELAY_MS = 2_000
const PRESENCE_CHANNEL = "blanktab:presence"
const PRESENCE_WAIT_MS = 250
/** Used for any setting the knowledge doesn't provide (e.g. offline), so the log records what the learner actually used. */
const LEARNER_DEFAULTS = { moveProbability: 0.5 }

// Test-only overrides. In the production build MODE is the literal "production" and this is dropped.
const test: TestParams = import.meta.env.MODE !== "production" ? readTestParams(location.search) : {}
const lifetimeSteps = test.lifetimeSteps ?? LIFETIME_STEPS
const stepsPerSecond = test.stepsPerSecond ?? STEPS_PER_SECOND

const fetchFn: typeof fetch = (input, init) => fetch(input, init)

/** Every blanktab tab answers pings, including this one, so the count includes self. */
function answerPresence(): void {
  const ch = new BroadcastChannel(PRESENCE_CHANNEL)
  ch.onmessage = (e: MessageEvent<{ type?: string; from?: string }>) => {
    if (e.data?.type === "ping") ch.postMessage({ type: "pong", to: e.data.from })
  }
}
async function countTabs(): Promise<number> {
  const me = crypto.randomUUID()
  const ch = new BroadcastChannel(PRESENCE_CHANNEL)
  let n = 0
  ch.onmessage = (e: MessageEvent<{ type?: string; to?: string }>) => { if (e.data?.type === "pong" && e.data.to === me) n++ }
  ch.postMessage({ type: "ping", from: me })
  await new Promise(r => setTimeout(r, PRESENCE_WAIT_MS))
  ch.close()
  return Math.max(1, n)
}
async function hostSig(): Promise<LogFile["hostSig"]> {
  const nav = navigator as Navigator & { deviceMemory?: number }
  return {
    cores: Number.isFinite(nav.hardwareConcurrency) ? nav.hardwareConcurrency : 1,
    deviceMemoryGb: typeof nav.deviceMemory === "number" && Number.isFinite(nav.deviceMemory) ? nav.deviceMemory : null,
    webgpu: "gpu" in navigator,
    tabsOnMachine: await countTabs(),
  }
}

/** Birth facts are fixed for a learner's whole life: a resumed learner keeps the ones its first log recorded. */
function birth(latest: LogFile | undefined, knowledge: Knowledge): Pick<SessionConfig, "settings" | "knowledgeSha" | "sessionsSeenAtBirth"> {
  if (latest) return { settings: latest.settings, knowledgeSha: latest.knowledgeSha, sessionsSeenAtBirth: latest.sessionsSeenAtBirth }
  return {
    settings: { ...LEARNER_DEFAULTS, ...chooseSettings(knowledge, Math.random) },
    knowledgeSha: knowledge.manifest.knowledgeSha,
    sessionsSeenAtBirth: knowledge.summary.sessions.length,
  }
}

async function live(id: string, ctx: { ui: Ui; queue: Queue; knowledge: Knowledge; flush: () => void }): Promise<void> {
  const { ui, queue, knowledge, flush } = ctx
  const latest = await queue.latest(id)
  if (latest?.final) return live(newLearnerId(sessionStorage), ctx) // this tab's learner already finished its life
  const cfg: SessionConfig = {
    learnerId: id, codeSha: __CODE_SHA__, ...birth(latest, knowledge), hostSig: await hostSig(),
    lifetimeSteps, logEverySteps: LOG_EVERY_STEPS, ...(await resumePoint(queue, id)),
  }
  ui.learner({ id, settings: cfg.settings, sessionsBefore: cfg.sessionsSeenAtBirth, lifetimeSteps })
  ui.steps(cfg.startSteps)

  const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" })
  worker.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data
    if (m.type === "frame") ui.frame(m.steps, m.periphery, m.fovea)
    else if (m.type === "queued") { ui.steps(m.steps); flush() }
    else if (m.type === "error") { ui.error(m.message); worker.terminate() }
    else if (m.type === "final") {
      worker.terminate()
      ui.lifeComplete(id)
      flush()
      setTimeout(() => void live(newLearnerId(sessionStorage), ctx), NEWBORN_DELAY_MS)
    }
  }
  worker.onerror = e => { ui.error(e.message || "worker failed"); worker.terminate() }
  worker.postMessage({ type: "start", cfg, stepsPerSecond } satisfies ToWorker)
}

async function main(): Promise<void> {
  const ui = createUi()
  answerPresence()
  ui.tokenPanel(getToken(localStorage) === null)
  const [knowledge, queue] = await Promise.all([loadKnowledge(import.meta.env.BASE_URL, fetchFn), openQueue(indexedDB)])
  ui.knowledge(knowledge)

  const contributor = createContributor({
    queue, ls: localStorage, fetchFn, now: () => new Date(),
    onStatus: s => {
      ui.contribution(s)
      if (s.state === "no-token" || s.message === REJECTED_MESSAGE) ui.tokenPanel(true)
    },
  })
  const flush = () => void contributor.flush().catch((e: unknown) => {
    ui.contribution({ state: "error", message: `Couldn't read the saved logs (${e instanceof Error ? e.message : String(e)}); will try again.`, pending: 0 })
  })
  ui.onTokenSubmit(async token => {
    const r = await checkToken(token, fetchFn) // check first: a token is stored only once GitHub accepts it
    if (!r.ok) return r.reason
    setToken(localStorage, token)
    ui.tokenPanel(false)
    flush()
    return null
  })
  flush() // logs left by earlier tabs or lives
  setInterval(flush, FLUSH_EVERY_MS)

  await live(getOrCreateLearnerId(sessionStorage), { ui, queue, knowledge, flush })
}

void main().catch((e: unknown) => createUi().error(e instanceof Error ? e.message : String(e)))
