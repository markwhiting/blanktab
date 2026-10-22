import { describe, it, expect } from "vitest"
import { createCanvas } from "@napi-rs/canvas"
import { MOVE_MAGNITUDES, NOOP_ACTION, WORLD, type Action } from "@blanktab/interface"
import type { FrozenResult } from "@blanktab/learner"
import { createRng } from "@blanktab/tensor"
import {
  ROOM_KINDS, ROOM_VIEW, TASKBAR_HEIGHT, createWorld, cribRoom, flowHideseek, hashSeed, hideseekRoom, layoutHideseek,
  placesRoom, toysRoom, type PlacesRoom, type PrivilegedState, type Room, type RoomFactory, type RoomKind, type World,
} from "@blanktab/world"
import { MILESTONE_IDS, MILESTONE_SUITE_VERSION } from "../src/log"
import { uniformAction } from "../src/heldout"
import { M2_MIN_IN_VIEW, M5_MIN_PAGES, M5_REFERENCE_CLICKERS, MILESTONES_V1, YIELD_EVERY, recordRun, type CheckupTrace, type Milestone, type MilestoneId, type Policy } from "../src/milestones"

const canvas = () => createCanvas(WORLD.width, WORLD.height)
const same = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((v, i) => v === b[i])
const byId = (id: MilestoneId): Milestone => MILESTONES_V1.find(m => m.id === id)!
const NO_STATS: FrozenResult = { predLoss: NaN, baselineLoss: NaN, transitions: 0 }
const SEEDS_1_20 = Array.from({ length: 20 }, (_, i) => i + 1)

// ---- scripted (fake-learner) policies: test-only closures that read the world's privileged state ----

/** The 8-direction move code toward (tx, ty): diagonal while both axes differ, never overshooting. */
function moveToward(cx: number, cy: number, tx: number, ty: number): number {
  const dx = tx - cx, dy = ty - cy
  if (dx === 0 && dy === 0) return 0
  const sx = Math.sign(dx), sy = Math.sign(dy)
  const d = sx !== 0 && sy !== 0 ? Math.min(Math.abs(dx), Math.abs(dy)) : Math.abs(dx) + Math.abs(dy)
  const dir = sx > 0 ? (sy < 0 ? 1 : sy > 0 ? 7 : 0) : sx < 0 ? (sy < 0 ? 3 : sy > 0 ? 5 : 4) : sy < 0 ? 2 : 6
  const mag = [...MOVE_MAGNITUDES].reverse().find(m => m <= d)!
  return 1 + dir * 3 + MOVE_MAGNITUDES.indexOf(mag)
}

type Goal = { x: number; y: number; click?: boolean; scroll?: 1 | 2 } | null
/** Goes to the goal's point, then presses (and releases next step) or scrolls there; `presses` = presses made so far. */
function scripted(world: World, goal: (s: PrivilegedState, presses: number) => Goal): Policy {
  let held = false, presses = 0
  return () => {
    if (held) { held = false; return { ...NOOP_ACTION, button: 2 } }
    const s = world.state()
    const g = goal(s, presses)
    if (!g) return NOOP_ACTION
    if (s.cursor.x !== g.x || s.cursor.y !== g.y) return { ...NOOP_ACTION, move: moveToward(s.cursor.x, s.cursor.y, g.x, g.y) }
    if (g.click) { held = true; presses++; return { ...NOOP_ACTION, button: 1 } }
    if (g.scroll) return { ...NOOP_ACTION, scroll: g.scroll }
    return NOOP_ACTION
  }
}

const FACTORIES: Record<RoomKind, RoomFactory> = { crib: cribRoom, toys: toysRoom, hideseek: hideseekRoom, places: placesRoom }
/** The same room a variant world builds for `kind` (a world seeded `seed` builds its rooms from hashSeed(seed, kind)). */
function twinRoom(seed: number, kind: RoomKind): Room {
  let room: Room | undefined
  createWorld({ seed, canvas: canvas(), rooms: [kind], roomFactories: { [kind]: (s: number, w: readonly string[]) => (room = FACTORIES[kind](s, w)) } })
  return room!
}
/** The first in-view point (scanning rows) whose inspect() satisfies `ok`. */
function findPoint(room: Room, ok: (info: ReturnType<Room["inspect"]>) => boolean): { x: number; y: number } {
  for (let y = 2; y < ROOM_VIEW.height - 2; y += 2) for (let x = 2; x < ROOM_VIEW.width - 2; x += 2) if (ok(room.inspect(x, y))) return { x, y }
  throw new Error("no such point")
}
const ICON_Y = WORLD.height - TASKBAR_HEIGHT + 2 + 6
const iconAt = (i: number) => ({ x: 4 + i * 16 + 6, y: ICON_Y })

const distinct = (t: CheckupTrace) => new Set([t.initial.page, ...t.states.map(s => s.page)]).size
/** Depth-first over the site: an unvisited link on this page, else Back. */
function explorer(seed: number): (w: World) => Policy {
  const links = (twinRoom(seed, "places") as PlacesRoom).links()
  return w => {
    const seen = new Set<string>(), path: string[] = []
    return scripted(w, s => {
      const p = s.page!
      if (path.at(-1) !== p) { if (path.at(-2) === p) path.pop(); else path.push(p) }
      seen.add(p)
      const next = links[p]!.find(l => !seen.has(l.target))
      if (next) return { x: next.x + 1, y: next.y + 2, click: true }
      return path.length > 1 ? { x: 9, y: 9, click: true } : null
    })
  }
}

async function run(m: Milestone, seed: number, policy: (w: World) => Policy, steps = m.budgetSteps) {
  const world = m.makeVariant(seed, canvas())
  const trace = await recordRun(world, policy(world), steps)
  return { world, trace }
}
const judge = (m: Milestone, trace: CheckupTrace, ref?: number) => ({ measure: m.measure(trace, NO_STATS, ref), passed: m.passes(trace, NO_STATS, ref) })

const flailer = (seed: number): Policy => {
  const rnd = createRng(hashSeed(seed, "flailer"))
  return () => uniformAction(rnd)
}

describe("milestones v1 (decision 0016)", { timeout: 300_000 }, () => {
  it("suite v1: ids, order, budgets", () => {
    expect(MILESTONES_V1.map(m => m.id)).toEqual(MILESTONE_IDS[MILESTONE_SUITE_VERSION])
    expect(MILESTONES_V1.map(m => m.budgetSteps)).toEqual([300, 400, 400, 400, 400, 500])
    expect(MILESTONES_V1.filter(m => m.measuresLoss).map(m => m.id)).toEqual(["M1"])
    expect(MILESTONES_V1.every(m => m.version === 1 && m.description.length > 0)).toBe(true)
    expect(Object.fromEntries(MILESTONES_V1.map(m => [m.id, m.prerequisites]))).toEqual({
      M1: [], M2: [], M3: ["M2"], M4: ["M2"], M5: ["M3"], M6: ["M3"],
    })
    expect(MILESTONES_V1.filter(m => m.reference).map(m => m.id)).toEqual(["M5"])
  })

  it("variants are deterministic in seed and differ across seeds", () => {
    for (const m of MILESTONES_V1) {
      const a = m.makeVariant(5, canvas()).observe(), b = m.makeVariant(5, canvas()).observe(), c = m.makeVariant(6, canvas()).observe()
      expect(same(a.periphery, b.periphery) && same(a.fovea, b.fovea), m.id).toBe(true)
      expect(same(a.periphery, c.periphery), m.id).toBe(false)
    }
  })

  it("variant worlds: the rooms of the table", () => {
    const rooms = (id: MilestoneId, seed: number) => {
      const w = byId(id).makeVariant(seed, canvas())
      return { rooms: w.rooms(), start: w.state().room }
    }
    expect(rooms("M1", 3)).toEqual({ rooms: ["crib", "toys"], start: "crib" })
    expect(rooms("M2", 3)).toEqual({ rooms: ["crib"], start: "crib" })
    expect(rooms("M3", 3)).toEqual({ rooms: ["toys"], start: "toys" })
    expect(rooms("M4", 3)).toEqual({ rooms: ["hideseek"], start: "hideseek" })
    expect(rooms("M5", 3)).toEqual({ rooms: ["places"], start: "places" })
    for (const seed of [4, 5, 6, 7]) expect(rooms("M6", seed)).toEqual({ rooms: ROOM_KINDS, start: ROOM_KINDS[seed % 4] })
  })

  it("variants can be passed (seeds 1–20)", async () => {
    for (const seed of SEEDS_1_20) {
      const s2 = byId("M2").makeVariant(seed, canvas()).state()
      expect(s2.areas.interactive, `M2 seed ${seed}`).toBeGreaterThan(0)
      expect(s2.areas.interactive, `M2 seed ${seed}`).toBeLessThan(0.5)
      const s3 = byId("M3").makeVariant(seed, canvas()).state()
      expect(s3.areas.responsive, `M3 seed ${seed}`).toBeGreaterThan(0)
      expect(s3.areas.responsive, `M3 seed ${seed}`).toBeLessThan(0.5)
      expect(byId("M4").makeVariant(seed, canvas()).state().scroll!.revealedFraction, `M4 seed ${seed}`).toBeLessThan(1)
      expect(byId("M6").makeVariant(seed, canvas()).rooms()).toHaveLength(4)
      // M5: at least M5_MIN_PAGES pages are reachable from the start page (the explorer's reach is checked in the M5 test).
      const { links } = (twinRoom(seed, "places") as PlacesRoom).graph()
      const reach = new Set(["p0"])
      for (const q = ["p0"]; q.length;) for (const t of links[q.shift()!]!) if (!reach.has(t)) { reach.add(t); q.push(t) }
      expect(reach.size, `M5 seed ${seed}`).toBeGreaterThanOrEqual(M5_MIN_PAGES)
    }
  })

  it("M1 passes iff predLoss < 0.8 × baselineLoss; NaN, or a zero baseline, fails", () => {
    const m = byId("M1")
    const t: CheckupTrace = { initial: m.makeVariant(1, canvas()).state(), states: [], events: [] }
    const f = (predLoss: number, baselineLoss: number) => ({ predLoss, baselineLoss, transitions: 300 })
    expect(m.measure(t, f(0.5, 1))).toBe(0.5)
    expect(m.passes(t, f(0.79, 1))).toBe(true)
    expect(m.passes(t, f(0.8, 1))).toBe(false)
    expect(m.passes(t, f(1, 1))).toBe(false)
    expect(m.passes(t, f(NaN, 1))).toBe(false)
    expect(m.passes(t, f(0.1, NaN))).toBe(false)
    expect(m.passes(t, f(0, 0))).toBe(false)
    expect(m.passes(t, f(0.1, 0))).toBe(false)
    expect(Number.isFinite(m.measure(t, f(0.1, 0)))).toBe(false)
  })

  it("M2: parked on a crib shape passes; a still cursor on background fails; taskbar hovers count for neither side", async () => {
    const m = byId("M2")
    for (const seed of [1, 2, 3]) {
      const room = twinRoom(seed, "crib")
      const shape = findPoint(room, i => i?.interactive === true)
      const bg = findPoint(room, i => i === null)
      const parked = await run(m, seed, w => scripted(w, () => shape))
      expect(judge(m, parked.trace).passed, `seed ${seed}`).toBe(true)
      const still = await run(m, seed, w => scripted(w, () => bg))
      expect(judge(m, still.trace).passed).toBe(false)
      expect(judge(m, still.trace).measure).toBeLessThan(0.2) // only the trip there can cross a shape
      expect(still.trace.states.slice(-300).every(s => s.hovered === null)).toBe(true)
      // Parked on the taskbar: its steps are not in view, so on their own there is nothing to measure.
      const onBar = await run(m, seed, w => scripted(w, () => iconAt(0)))
      const barOnly = { ...onBar.trace, states: onBar.trace.states.filter(s => s.cursor.y >= ROOM_VIEW.height) }
      expect(barOnly.states.length).toBeGreaterThan(350)
      expect(barOnly.states.slice(-300).every(s => s.hovered?.id === "taskbar:crib")).toBe(true)
      expect(Number.isNaN(m.measure(barOnly, NO_STATS))).toBe(true) // taskbar hovers are not "over interactive"
      expect(m.passes(barOnly, NO_STATS)).toBe(false)
      // Half the budget on the shape, then on the taskbar: the taskbar steps are in neither the numerator nor the
      // denominator, so the fraction over in-view steps is the parked one.
      const mixed = await run(m, seed, w => scripted(w, s => (s.tick < 200 ? shape : iconAt(0))))
      const inView = mixed.trace.states.filter(s => s.cursor.y < ROOM_VIEW.height)
      const over = inView.filter(s => s.hovered?.interactive).length
      expect(judge(m, mixed.trace).measure).toBeCloseTo(over / inView.length / inView[0]!.areas.interactive, 10)
      expect(mixed.trace.states.some(s => s.hovered?.id.startsWith("taskbar:"))).toBe(true)
    }
  })

  it("M2: walking onto the taskbar and parking fails, however good its few in-view steps (in view ≥ 50 % of the budget)", async () => {
    const m = byId("M2")
    expect(M2_MIN_IN_VIEW).toBe(0.5)
    const down = 1 + 6 * 3 + 2 // S, 16 px: from (160, 100) onto the strip in 6 steps, 5 of them in view
    const ratios: number[] = []
    for (const seed of [1, 2, 3]) {
      let i = 0
      const { trace } = await run(m, seed, () => () => (i++ < 6 ? { ...NOOP_ACTION, move: down } : NOOP_ACTION))
      expect(trace.states.filter(s => s.cursor.y < ROOM_VIEW.height)).toHaveLength(5)
      ratios.push(judge(m, trace).measure)
      expect(judge(m, trace).passed, `seed ${seed}`).toBe(false)
    }
    expect(ratios.filter(r => r >= 2).length).toBeGreaterThan(0) // the ratio alone would have passed
  })

  it("M3: clicking only responsive toys passes; only background fails; 2 responsive clicks fail; taskbar clicks are ignored", async () => {
    const m = byId("M3")
    for (const seed of [1, 2, 3]) {
      const room = twinRoom(seed, "toys")
      const button = findPoint(room, i => !!i && (i.id.startsWith("button-") || i.id.startsWith("toggle-")))
      const bg = findPoint(room, i => i === null)
      const resp = await run(m, seed, w => scripted(w, () => ({ ...button, click: true })))
      expect(judge(m, resp.trace).passed, `seed ${seed}`).toBe(true)
      const back = await run(m, seed, w => scripted(w, () => ({ ...bg, click: true })))
      expect(judge(m, back.trace)).toEqual({ measure: 0, passed: false })
      const two = await run(m, seed, w => scripted(w, (_, n) => (n < 2 ? { ...button, click: true } : null)))
      expect(judge(m, two.trace).measure).toBeGreaterThanOrEqual(2) // the ratio alone would pass
      expect(judge(m, two.trace).passed).toBe(false)
      // Two responsive clicks, then many on the toys taskbar icon (which is interactive and responsive): still 2.
      const bar = await run(m, seed, w => scripted(w, (_, n) => (n < 2 ? { ...button, click: true } : { ...iconAt(0), click: true })))
      expect(bar.trace.events.filter(e => e.kind === "click" && e.elementId === "taskbar:toys").length).toBeGreaterThan(10)
      expect(judge(m, bar.trace).passed).toBe(false)
      expect(judge(m, bar.trace).measure).toBeCloseTo(judge(m, two.trace).measure, 10)
    }
  })

  it("M4: scrolling to the end and opening every section passes; content visible at start does not count", async () => {
    const m = byId("M4")
    for (const seed of [1, 2, 3, 4, 5]) {
      const h = layoutHideseek(hashSeed(seed, "hideseek"))
      const flow0 = flowHideseek(h, new Set(), 0)
      const closeAt = { x: h.close.x + 4, y: h.close.y + 4 }
      const tabs = h.tabs.slice(1).map(t => ({ x: t.x + Math.floor(t.w / 2), y: flow0.tabY + 7 }))
      const panelEnd = flow0.panelY + flow0.panelH
      let phase: "popup" | "tabs" | "end" = "popup", tab = 0, down = true
      const solver = (s: PrivilegedState, presses: number): Goal => {
        const sc = s.scroll!, here = { x: s.cursor.x, y: s.cursor.y }
        if (phase === "popup") {
          if (presses === 0) return s.tick < h.popup.tick ? closeAt : { ...closeAt, click: true } // wait for it, close it
          phase = "tabs"
        }
        if (phase === "tabs") {
          while (tab < tabs.length) {
            if (presses < 2 + tab) return { ...tabs[tab]!, click: true }
            if (down && sc.offset < Math.min(sc.max, panelEnd - ROOM_VIEW.height)) return { ...here, scroll: 2 }
            down = false
            if (sc.offset > 0) return { ...here, scroll: 1 }
            down = true
            tab++
          }
          phase = "end"
        }
        return sc.offset < sc.max ? { ...here, scroll: 2 } : null
      }
      const full = await run(m, seed, w => scripted(w, solver))
      expect(full.trace.states.at(-1)!.scroll!.revealedFraction, `seed ${seed}`).toBe(1)
      expect(judge(m, full.trace)).toEqual({ measure: 1, passed: true })
      // A still cursor: what was on screen at the start is in r0 and gains nothing.
      const still = await run(m, seed, () => () => NOOP_ACTION)
      expect(still.trace.initial.scroll!.revealedFraction).toBeGreaterThan(0)
      expect(judge(m, still.trace)).toEqual({ measure: 0, passed: false })
    }
  })

  it("M5: reference = mean of 16 random clickers, deterministic; a policy visiting every page passes (seeds 1–20); their clickers average to it", async () => {
    const m = byId("M5")
    expect([M5_REFERENCE_CLICKERS, M5_MIN_PAGES]).toEqual([16, 4])
    for (const seed of SEEDS_1_20) {
      const ref = m.reference!(seed, canvas())
      if (seed <= 3) expect(m.reference!(seed, canvas())).toBe(ref)
      expect(ref).toBeGreaterThanOrEqual(1)
      const dfs = await run(m, seed, explorer(seed))
      const pages = distinct(dfs.trace)
      expect(pages, `seed ${seed}`).toBeGreaterThanOrEqual(M5_MIN_PAGES)
      expect(judge(m, dfs.trace, ref).passed, `seed ${seed}: ${pages} pages, ref ${ref}`).toBe(true)
    }
    // The reference's clickers, recorded with observations (as a checkup would): their mean is exactly the reference,
    // and they rarely pass against it (a lucky clicker reaching 4+ pages can).
    let lucky = 0
    for (const seed of [1, 2]) {
      const ref = m.reference!(seed, canvas())
      let sum = 0
      for (let j = 0; j < M5_REFERENCE_CLICKERS; j++) {
        const rnd = createRng(hashSeed(seed, `random-clicker/${j}`))
        const { trace } = await run(m, seed, () => () => uniformAction(rnd))
        sum += distinct(trace)
        if (judge(m, trace, ref).passed) lucky++
      }
      expect(sum / M5_REFERENCE_CLICKERS).toBe(ref)
    }
    console.info(`M5: reference clickers passing against their own reference (seeds 1–2): ${lucky} of ${2 * M5_REFERENCE_CLICKERS}`)
  })

  it("M5: the page floor: 1.5 × the reference is not enough below 4 pages", async () => {
    const m = byId("M5")
    const seed = 1, link = (twinRoom(seed, "places") as PlacesRoom).links()["p0"]![0]!
    const { trace } = await run(m, seed, w => scripted(w, (_, n) => (n < 1 ? { x: link.x + 1, y: link.y + 2, click: true } : null)), 60)
    expect(distinct(trace)).toBe(2)
    expect(m.measure(trace, NO_STATS, 1)).toBe(2)
    expect(m.passes(trace, NO_STATS, 1)).toBe(false)
  })

  it("M6: three taskbar clicks onto other rooms pass; caregiver focus events and clicks on the active room's icon do not count", async () => {
    const m = byId("M6")
    for (const seed of [1, 2, 3, 4]) {
      const start = seed % 4
      const three = await run(m, seed, w => scripted(w, (_, n) => (n < 3 ? { ...iconAt((start + 1 + n) % 4), click: true } : null)))
      expect(judge(m, three.trace)).toEqual({ measure: 3, passed: true })
      // The caregiver carries the learner three times; each time the learner clicks the now-active room's icon.
      const carried = await run(m, seed, w => {
        let carries = 0
        return scripted(w, (s, n) => {
          if (n >= 3) return null
          if (carries === n) { w.bringToFront(ROOM_KINDS[(start + 1 + n) % 4]!); carries++ }
          return { ...iconAt(ROOM_KINDS.indexOf(w.state().room)), click: true }
        })
      })
      const focus = carried.trace.events.filter(e => e.kind === "focus")
      expect(focus.map(e => e.source)).toEqual(["caregiver", "caregiver", "caregiver"])
      expect(carried.trace.events.filter(e => e.kind === "click" && e.elementId?.startsWith("taskbar:"))).toHaveLength(3)
      expect(judge(m, carried.trace)).toEqual({ measure: 0, passed: false })
    }
  })

  it("Review Focus 5: a uniform-random policy fails each of M2–M5 on at least 9 of seeds 1–10", async () => {
    const passes: Record<string, number> = {}, measures: Record<string, string[]> = {}
    for (const m of MILESTONES_V1.filter(x => x.id !== "M1")) {
      passes[m.id] = 0
      measures[m.id] = []
      for (let seed = 1; seed <= 10; seed++) {
        const ref = m.reference?.(seed, canvas())
        const { trace } = await run(m, seed, () => flailer(seed))
        const j = judge(m, trace, ref)
        if (j.passed) passes[m.id]!++
        measures[m.id]!.push(j.measure.toFixed(3))
      }
    }
    console.info(`uniform-random policy, seeds 1–10: passes ${JSON.stringify(passes)}; measures ${JSON.stringify(measures)}`)
    for (const id of ["M2", "M3", "M4", "M5"]) expect(passes[id], id).toBeLessThanOrEqual(1)
  })

  it("recordRun: initial state, one state per step, every trace event; yields every step when given yieldNow, else every YIELD_EVERY", async () => {
    const m = byId("M3")
    let yields = 0
    const actions: Action[] = []
    const world = m.makeVariant(2, canvas())
    const initial = world.state()
    const rnd = createRng(9)
    const trace = await recordRun(world, () => { const a = uniformAction(rnd); actions.push(a); return Promise.resolve(a) }, 120, async () => { yields++ })
    expect(yields).toBe(120)
    expect(trace.initial).toEqual(initial)
    expect(trace.states).toHaveLength(120)
    expect(trace.states.map(s => s.tick)).toEqual(actions.map((_, i) => i + 1))
    // The same actions replayed on a fresh variant give the same trace.
    const twin = m.makeVariant(2, canvas())
    const events = []
    for (const a of actions) { twin.act(a); events.push(...twin.drainTrace()) }
    expect(trace.events).toEqual(events)
    expect(trace.states.at(-1)).toEqual(twin.state())
    expect(world.drainTrace()).toEqual([]) // recordRun drained everything
    // Without yieldNow: still yields to the event loop (a pending timer fires during a long run).
    expect(YIELD_EVERY).toBe(50)
    let fired = false
    setTimeout(() => { fired = true }, 0)
    await recordRun(m.makeVariant(2, canvas()), () => NOOP_ACTION, YIELD_EVERY)
    expect(fired).toBe(true)
  })
})
