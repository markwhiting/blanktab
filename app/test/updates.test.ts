import { describe, it, expect } from "vitest"
import type { Knowledge, Manifest } from "@blanktab/harness"
import {
  CHECKPOINT_RETRY_MS, INCOMPATIBLE_MESSAGE, KNOWLEDGE_POLL, MANIFEST_POLL, RELOAD_JITTER_MS, RELOAD_TARGET_KEY, createPoller, jittered,
  startUpdates, type UpdatesDeps,
} from "../src/updates"
import { stubStorage } from "../../packages/harness/test/helpers"

/** Timers on a fake clock: advance(ms) runs, in time order, every timer due by then (and those they set). */
function fakeTimers() {
  let t = 0, next = 1
  const pending = new Map<number, { at: number; fn: () => void }>()
  const timers = {
    setTimeout: (fn: () => void, ms: number) => { const h = next++; pending.set(h, { at: t + Math.max(0, ms), fn }); return h },
    clearTimeout: (h: number) => void pending.delete(h),
  }
  const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve() }
  return {
    timers,
    now: () => t,
    pending: () => [...pending.values()].map(p => p.at - t),
    async advance(ms: number) {
      const end = t + ms
      for (;;) {
        const due = [...pending.entries()].filter(([, p]) => p.at <= end).sort((a, b) => a[1].at - b[1].at)[0]
        if (!due) break
        pending.delete(due[0])
        t = due[1].at
        due[1].fn()
        await flush()
      }
      t = end
      await flush()
    },
    flush,
  }
}

function fakeVisibility(hidden = false) {
  const fns = new Set<() => void>()
  return {
    hidden: () => hidden,
    onChange: (fn: () => void) => { fns.add(fn); return () => void fns.delete(fn) },
    set(h: boolean) { hidden = h; for (const fn of [...fns]) fn() },
    listeners: () => fns.size,
  }
}

const MIN = 60_000

describe("jittered", () => {
  it("spans exactly every ± jitter", () => {
    expect(jittered(KNOWLEDGE_POLL, () => 0)).toBe(12 * MIN)
    expect(jittered(KNOWLEDGE_POLL, () => 0.5)).toBe(15 * MIN)
    expect(jittered(KNOWLEDGE_POLL, () => 1)).toBe(18 * MIN)
    expect(jittered(MANIFEST_POLL, () => 0)).toBe(8 * MIN)
    expect(jittered(MANIFEST_POLL, () => 1)).toBe(12 * MIN)
  })
  it("the constants: knowledge every 15 ± 3 min, the manifest every 10 ± 2 min, a reload within 0-5 min", () => {
    expect(KNOWLEDGE_POLL).toEqual({ everyMs: 15 * MIN, jitterMs: 3 * MIN })
    expect(MANIFEST_POLL).toEqual({ everyMs: 10 * MIN, jitterMs: 2 * MIN })
    expect(RELOAD_JITTER_MS).toBe(5 * MIN)
  })
})

describe("createPoller", () => {
  const setup = (o: { hidden?: boolean; random?: () => number } = {}) => {
    const clock = fakeTimers()
    const vis = fakeVisibility(o.hidden)
    const at: number[] = []
    const poller = createPoller(() => { at.push(clock.now()) }, MANIFEST_POLL, { timers: clock.timers, visibility: vis, random: o.random ?? (() => 0.5) })
    return { clock, vis, at, poller }
  }

  it("checks once per jittered period, never at start, each delay within the bounds", async () => {
    let i = 0
    const draws = [0, 1, 0.25, 0.75]
    const { clock, at } = setup({ random: () => draws[i++ % draws.length]! })
    expect(at).toEqual([])
    await clock.advance(60 * MIN)
    const gaps = at.map((t, k) => t - (at[k - 1] ?? 0))
    expect(gaps.slice(0, 4)).toEqual([8 * MIN, 12 * MIN, 9 * MIN, 11 * MIN])
    for (const g of gaps) { expect(g).toBeGreaterThanOrEqual(8 * MIN); expect(g).toBeLessThanOrEqual(12 * MIN) }
  })

  it("random draws over many periods stay within every ± jitter", async () => {
    const { clock, at } = setup({ random: Math.random })
    await clock.advance(24 * 60 * MIN)
    const gaps = at.map((t, k) => t - (at[k - 1] ?? 0))
    expect(gaps.length).toBeGreaterThan(100)
    for (const g of gaps) { expect(g).toBeGreaterThanOrEqual(8 * MIN); expect(g).toBeLessThanOrEqual(12 * MIN) }
  })

  it("pauses while the tab is hidden and checks at once when it becomes visible, then resumes the period", async () => {
    const { clock, vis, at } = setup()
    await clock.advance(5 * MIN)
    vis.set(true)
    expect(clock.pending()).toEqual([]) // nothing scheduled while hidden
    await clock.advance(60 * MIN)
    expect(at).toEqual([]) // no check in an hour hidden
    vis.set(false)
    await clock.flush()
    expect(at).toEqual([65 * MIN]) // straight away
    await clock.advance(10 * MIN)
    expect(at).toEqual([65 * MIN, 75 * MIN])
  })

  it("a tab hidden from the start checks nothing until shown", async () => {
    const { clock, vis, at } = setup({ hidden: true })
    await clock.advance(60 * MIN)
    expect(at).toEqual([])
    vis.set(false)
    await clock.flush()
    expect(at).toEqual([60 * MIN])
  })

  it("checks never overlap; stop() ends it and its visibility listener", async () => {
    const clock = fakeTimers()
    const vis = fakeVisibility()
    let running = 0, most = 0, calls = 0
    let release: () => void = () => {}
    const p = createPoller(async () => {
      calls++; running++; most = Math.max(most, running)
      await new Promise<void>(r => { release = r })
      running--
    }, MANIFEST_POLL, { timers: clock.timers, visibility: vis, random: () => 0.5 })
    await clock.advance(10 * MIN)
    vis.set(true); vis.set(false) // shown again while a check is still running
    await clock.flush()
    expect(calls).toBe(1)
    release()
    await clock.flush()
    expect(most).toBe(1)
    p.stop()
    expect(vis.listeners()).toBe(0)
    await clock.advance(60 * MIN)
    vis.set(false)
    expect(calls).toBe(1)
  })

  it("a check that throws does not stop the polling", async () => {
    const clock = fakeTimers()
    let calls = 0
    createPoller(() => { calls++; throw new Error("network") }, MANIFEST_POLL, { timers: clock.timers, visibility: fakeVisibility(), random: () => 0.5 })
    await clock.advance(30 * MIN)
    expect(calls).toBe(3)
  })
})

const SHA = "a".repeat(40), NEW = "b".repeat(40)
const knowledge = (knowledgeSha: string, offline = false): Knowledge => ({
  offline, summary: { schemaVersion: 1, sessions: [] }, manifest: { knowledgeSha, codeSha: SHA, learnerStateVersion: 1 }, settingsSpace: { settings: [] },
})

describe("startUpdates", () => {
  const setup = (o: { manifest?: () => Manifest | null; knowledge?: () => Knowledge; checkpoint?: () => Promise<boolean>; hidden?: boolean; random?: () => number; storage?: Storage } = {}) => {
    const clock = fakeTimers()
    const vis = fakeVisibility(o.hidden)
    const log: string[] = []
    const given: string[] = []
    let manifestCalls = 0
    const deps: UpdatesDeps = {
      env: { timers: clock.timers, visibility: vis, random: o.random ?? (() => 0.5) },
      initial: knowledge("c".repeat(64)),
      codeSha: SHA, learnerStateVersion: 1, storage: o.storage ?? stubStorage(),
      loadKnowledge: async () => (o.knowledge ?? (() => knowledge("c".repeat(64))))(),
      loadManifest: async () => { manifestCalls++; return (o.manifest ?? (() => ({ knowledgeSha: "c".repeat(64), codeSha: SHA, learnerStateVersion: 1 })))() },
      onKnowledge: k => given.push(k.manifest.knowledgeSha[0]!),
      checkpoint: async () => { log.push(`checkpoint@${clock.now()}`); return o.checkpoint ? o.checkpoint() : true },
      reload: () => log.push(`reload@${clock.now()}`),
      notice: m => log.push(`notice:${m}`),
    }
    const u = startUpdates(deps)
    return { clock, vis, log, given, u, manifestCalls: () => manifestCalls, deps }
  }

  it("new knowledge (knowledgeSha or content) is handed on; the same, or offline, is not", async () => {
    let k = knowledge("c".repeat(64))
    const s = setup({ knowledge: () => k })
    await s.clock.advance(15 * MIN)
    expect(s.given).toEqual([]) // the same as at startup
    k = knowledge("e".repeat(64))
    await s.clock.advance(15 * MIN)
    expect(s.given).toEqual(["e"])
    k = knowledge("0".repeat(64), true)
    await s.clock.advance(15 * MIN)
    k = { ...knowledge("e".repeat(64)), summary: { schemaVersion: 1, sessions: [] }, settingsSpace: { settings: [{ name: "lr", min: 0, max: 1 }] } }
    await s.clock.advance(15 * MIN)
    expect(s.given).toEqual(["e", "e"]) // offline skipped; a new settings space is new knowledge
  })

  it("the same codeSha: nothing happens", async () => {
    const s = setup()
    await s.clock.advance(60 * MIN)
    expect(s.manifestCalls()).toBe(6)
    expect(s.log).toEqual([])
  })

  it("new compatible code: waits a random 0-5 min, checkpoints, then reloads; and polls the manifest no more", async () => {
    let m: Manifest = { knowledgeSha: "c".repeat(64), codeSha: SHA, learnerStateVersion: 1 }
    const storage = stubStorage()
    const s = setup({ manifest: () => m, random: () => 0.5, storage })
    await s.clock.advance(10 * MIN)
    m = { ...m, codeSha: NEW }
    await s.clock.advance(10 * MIN) // seen at 20 min; the reload is at 20 + 0.5 * 5 = 22.5 min
    expect(s.log).toEqual([])
    await s.clock.advance(2.5 * MIN)
    expect(s.log).toEqual([`checkpoint@${22.5 * MIN}`, `reload@${22.5 * MIN}`])
    expect(storage.getItem(RELOAD_TARGET_KEY)).toBe(NEW)
    const calls = s.manifestCalls()
    await s.clock.advance(60 * MIN)
    expect(s.manifestCalls()).toBe(calls)
  })

  it("the reload wait is within 0-5 min for any draw", async () => {
    for (const r of [0, 0.999999]) {
      let draws = 0 // the two pollers' first delays (at 0.5: the manifest's first poll at 10 min), then the reload wait
      const s = setup({ random: () => (draws++ < 2 ? 0.5 : r), manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 1 }) })
      await s.clock.advance(10 * MIN) // the first poll (0.5: at 10 min) sees new code
      await s.clock.advance(r * RELOAD_JITTER_MS)
      expect(s.log.at(-1)).toBe(`reload@${10 * MIN + r * RELOAD_JITTER_MS}`)
    }
  })

  it("a checkpoint not saved (skips outstanding) is retried later; the page reloads only once one is saved", async () => {
    const answers = [false, false, true]
    const s = setup({ random: () => 0, manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 1 }), checkpoint: async () => answers.shift()! })
    await s.clock.advance(8 * MIN) // random 0: the poll at 8 min, the reload wait 0
    await s.clock.advance(2 * CHECKPOINT_RETRY_MS)
    const t = 8 * MIN
    expect(s.log).toEqual([`checkpoint@${t}`, `checkpoint@${t + CHECKPOINT_RETRY_MS}`, `checkpoint@${t + 2 * CHECKPOINT_RETRY_MS}`, `reload@${t + 2 * CHECKPOINT_RETRY_MS}`])
  })

  it("new incompatible code (another learnerStateVersion, or none): says so verbatim, never reloads, stops polling the manifest", async () => {
    for (const learnerStateVersion of [2, undefined]) {
      const s = setup({ manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, ...(learnerStateVersion !== undefined ? { learnerStateVersion } : {}) }) })
      await s.clock.advance(10 * MIN)
      expect(s.log).toEqual([`notice:${INCOMPATIBLE_MESSAGE}`])
      expect(INCOMPATIBLE_MESSAGE).toBe("A newer version is available; this learner continues on the version it was born with.")
      await s.clock.advance(120 * MIN)
      expect(s.manifestCalls()).toBe(1)
      expect(s.log).toHaveLength(1)
    }
  })

  it("knowledge keeps refreshing after new code is found, incompatible or not", async () => {
    let k = knowledge("c".repeat(64))
    const s = setup({ knowledge: () => k, manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 9 }) })
    await s.clock.advance(10 * MIN)
    k = knowledge("e".repeat(64))
    await s.clock.advance(30 * MIN)
    expect(s.given).toEqual(["e"])
  })

  it("a reload that brought no new code (a stale cache) is not repeated for the same codeSha", async () => {
    const storage = stubStorage({ [RELOAD_TARGET_KEY]: NEW })
    const s = setup({ storage, manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 1 }) })
    await s.clock.advance(60 * MIN)
    expect(s.log).toEqual([])
  })

  it("an unreachable manifest is skipped; hidden, nothing polls; shown, both check at once", async () => {
    let m: Manifest | null = null
    const s = setup({ manifest: () => m, hidden: true })
    await s.clock.advance(60 * MIN)
    expect(s.manifestCalls()).toBe(0)
    s.vis.set(false)
    await s.clock.flush()
    expect(s.manifestCalls()).toBe(1)
    m = { knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 3 }
    s.vis.set(true); s.vis.set(false)
    await s.clock.flush()
    expect(s.log).toEqual([`notice:${INCOMPATIBLE_MESSAGE}`])
  })

  it("stop() ends every poller and a pending reload", async () => {
    const s = setup({ manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 1 }) })
    await s.clock.advance(10 * MIN)
    s.u.stop()
    await s.clock.advance(60 * MIN)
    expect(s.log).toEqual([])
  })
})
