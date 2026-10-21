import { describe, it, expect } from "vitest"
import type { Knowledge, Manifest } from "@blanktab/harness"
import {
  CHECKPOINT_RETRY_MS, INCOMPATIBLE_MESSAGE, STALE_MESSAGE, KNOWLEDGE_POLL, MANIFEST_POLL, RELOAD_JITTER_MS, RELOAD_TARGET_KEY, createPoller, jittered,
  onWorkerLoadFailure, startUpdates, type UpdatesDeps,
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
    const poller = createPoller(() => { at.push(clock.now()) }, MANIFEST_POLL, { timers: clock.timers, visibility: vis, random: o.random ?? (() => 0.5), now: clock.now })
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

  it("shown again within half a period of the last check: no check at once, the period goes on from there", async () => {
    const { clock, vis, at } = setup()
    await clock.advance(10 * MIN) // the first check
    expect(at).toEqual([10 * MIN])
    vis.set(true)
    await clock.advance(4 * MIN)
    vis.set(false) // 4 min after the check: under half of 10 min
    await clock.flush()
    expect(at).toEqual([10 * MIN])
    await clock.advance(10 * MIN)
    expect(at).toEqual([10 * MIN, 24 * MIN]) // a period after being shown
    vis.set(true)
    await clock.advance(5 * MIN)
    vis.set(false) // exactly half a period since the last check: checks at once
    await clock.flush()
    expect(at).toEqual([10 * MIN, 24 * MIN, 29 * MIN])
  })

  it("the page's own load counts as a check: shown within half a period of start, nothing runs at once", async () => {
    const { clock, vis, at } = setup()
    vis.set(true)
    await clock.advance(1 * MIN)
    vis.set(false)
    await clock.flush()
    expect(at).toEqual([])
    await clock.advance(10 * MIN)
    expect(at).toEqual([11 * MIN])
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
    }, MANIFEST_POLL, { timers: clock.timers, visibility: vis, random: () => 0.5, now: clock.now })
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
    createPoller(() => { calls++; throw new Error("network") }, MANIFEST_POLL, { timers: clock.timers, visibility: fakeVisibility(), random: () => 0.5, now: clock.now })
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
      env: { timers: clock.timers, visibility: vis, random: o.random ?? (() => 0.5), now: clock.now },
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
    expect(JSON.parse(storage.getItem(RELOAD_TARGET_KEY)!)).toEqual({ codeSha: NEW, tries: 1 })
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

  it("new incompatible code (an explicit other learnerStateVersion): says so verbatim, never reloads, stops polling the manifest", async () => {
    const s = setup({ manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 2 }) })
    await s.clock.advance(10 * MIN)
    expect(s.log).toEqual([`notice:${INCOMPATIBLE_MESSAGE}`])
    expect(INCOMPATIBLE_MESSAGE).toBe("A newer version is available; this learner continues on the version it was born with.")
    await s.clock.advance(120 * MIN)
    expect(s.manifestCalls()).toBe(1)
    expect(s.log).toHaveLength(1)
  })

  it("a manifest without learnerStateVersion is skipped (not incompatible): polling goes on, and a later complete one is acted on", async () => {
    let m: Manifest = { knowledgeSha: "c".repeat(64), codeSha: NEW }
    const s = setup({ manifest: () => m, random: () => 0.5 })
    await s.clock.advance(30 * MIN)
    expect(s.manifestCalls()).toBe(3)
    expect(s.log).toEqual([])
    m = { ...m, learnerStateVersion: 1 }
    await s.clock.advance(10 * MIN + 2.5 * MIN)
    expect(s.log).toEqual([`checkpoint@${42.5 * MIN}`, `reload@${42.5 * MIN}`])
  })

  it("knowledge keeps refreshing after new code is found, incompatible or not", async () => {
    let k = knowledge("c".repeat(64))
    const s = setup({ knowledge: () => k, manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 9 }) })
    await s.clock.advance(10 * MIN)
    k = knowledge("e".repeat(64))
    await s.clock.advance(30 * MIN)
    expect(s.given).toEqual(["e"])
  })

  it("a reload that brought the old code again (a stale cache) is retried once, after 2 manifest polls, then given up with a message", async () => {
    const storage = stubStorage()
    const load = () => setup({ storage, random: () => 0.5, manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 1 }) })
    const first = load() // still on SHA after each "reload": the browser keeps serving it
    await first.clock.advance(12.5 * MIN)
    expect(first.log).toEqual([`checkpoint@${12.5 * MIN}`, `reload@${12.5 * MIN}`])
    first.u.stop()
    const second = load()
    await second.clock.advance(20 * MIN) // polls at 10 and 20 min: the backoff
    expect(second.log).toEqual([])
    await second.clock.advance(12.5 * MIN) // the third sighting, at 30 min: one retry
    expect(second.log).toEqual([`checkpoint@${32.5 * MIN}`, `reload@${32.5 * MIN}`])
    expect(JSON.parse(storage.getItem(RELOAD_TARGET_KEY)!)).toEqual({ codeSha: NEW, tries: 2 })
    second.u.stop()
    const third = load()
    await third.clock.advance(60 * MIN)
    expect(third.log).toEqual([`notice:${STALE_MESSAGE}`])
    expect(STALE_MESSAGE).toBe("A newer version is published but this browser keeps loading the old one; reload later.")
    expect(third.manifestCalls()).toBe(1)
  })

  it("the stale-cache guard is per codeSha: a still newer one is reloaded for at once", async () => {
    const storage = stubStorage({ [RELOAD_TARGET_KEY]: JSON.stringify({ codeSha: NEW, tries: 2 }) })
    const s = setup({ storage, random: () => 0.5, manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: "d".repeat(40), learnerStateVersion: 1 }) })
    await s.clock.advance(12.5 * MIN)
    expect(s.log).toEqual([`checkpoint@${12.5 * MIN}`, `reload@${12.5 * MIN}`])
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
    s.vis.set(true)
    await s.clock.advance(6 * MIN) // more than half a period since the check on showing
    s.vis.set(false)
    await s.clock.flush()
    expect(s.log).toEqual([`notice:${INCOMPATIBLE_MESSAGE}`])
  })

  it("a late halt: the page gave up waiting, then the learner saved and halted; the page reloads on it, with no more retries", async () => {
    const storage = stubStorage()
    let answer: (saved: boolean) => void = () => {}
    let asked = 0
    // The first request is never answered in time (the page's own wait gives up: false); the learner halts later.
    const s = setup({ storage, random: () => 0, manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 1 }), checkpoint: () => { asked++; return new Promise<boolean>(r => { answer = r }) } })
    await s.clock.advance(8 * MIN) // random 0: the poll at 8 min sees new code; the reload wait is 0
    expect(asked).toBe(1)
    answer(false) // timed out
    await s.clock.flush()
    expect(s.log).toEqual([`checkpoint@${8 * MIN}`])
    await s.clock.advance(10_000)
    s.u.halted() // the worker: {type: "halted"}
    expect(s.log).toEqual([`checkpoint@${8 * MIN}`, `reload@${8 * MIN + 10_000}`])
    expect(JSON.parse(storage.getItem(RELOAD_TARGET_KEY)!)).toEqual({ codeSha: NEW, tries: 1 })
    await s.clock.advance(10 * CHECKPOINT_RETRY_MS)
    expect(asked).toBe(1) // the retry was cancelled
    expect(s.log).toHaveLength(2)
    s.u.halted() // a second report changes nothing
    expect(s.log).toHaveLength(2)
  })

  it("a halt with no reload pending still reloads (the learner has stopped; the page resumes it)", async () => {
    const s = setup()
    s.u.halted()
    expect(s.log).toEqual([`reload@0`])
  })

  it("stop() ends every poller and a pending reload", async () => {
    const s = setup({ manifest: () => ({ knowledgeSha: "c".repeat(64), codeSha: NEW, learnerStateVersion: 1 }) })
    await s.clock.advance(10 * MIN)
    s.u.stop()
    await s.clock.advance(60 * MIN)
    expect(s.log).toEqual([])
  })
})

describe("onWorkerLoadFailure (the worker's asset is gone: replaced by a deploy)", () => {
  const OLD = "a".repeat(40), NEW = "b".repeat(40)
  const deps = (manifest: Manifest | null, storage = stubStorage()) => {
    const reloads: number[] = []
    return {
      reloads, storage,
      d: { codeSha: OLD, storage, loadManifest: async () => manifest, reload: () => { reloads.push(1) } },
    }
  }
  const m = (codeSha: string): Manifest => ({ knowledgeSha: "c".repeat(64), codeSha, learnerStateVersion: 2 })
  it("newer code published: reloads onto it, recording the try (the same guard as code updates)", async () => {
    const { d, reloads, storage } = deps(m(NEW))
    expect(await onWorkerLoadFailure(d)).toBe("reloaded")
    expect(reloads).toHaveLength(1)
    expect(JSON.parse(storage.getItem(RELOAD_TARGET_KEY)!)).toEqual({ codeSha: NEW, tries: 1 })
    expect(await onWorkerLoadFailure(d)).toBe("reloaded") // the reload brought the old code back once more
    expect(JSON.parse(storage.getItem(RELOAD_TARGET_KEY)!)).toEqual({ codeSha: NEW, tries: 2 })
  })
  it("after two reloads for that code it gives up (no reload loop)", async () => {
    const { d, reloads } = deps(m(NEW), stubStorage({ [RELOAD_TARGET_KEY]: JSON.stringify({ codeSha: NEW, tries: 2 }) }))
    expect(await onWorkerLoadFailure(d)).toBe("gave-up")
    expect(reloads).toHaveLength(0)
  })
  it("the same code (or no manifest): not a deploy, nothing to reload onto", async () => {
    for (const manifest of [m(OLD), null]) {
      const { d, reloads } = deps(manifest)
      expect(await onWorkerLoadFailure(d)).toBe("same-code")
      expect(reloads).toHaveLength(0)
    }
    const { d } = deps(null)
    expect(await onWorkerLoadFailure({ ...d, loadManifest: async () => { throw new Error("offline") } })).toBe("same-code")
  })
})
