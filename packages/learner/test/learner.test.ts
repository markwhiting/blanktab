import { describe, it, expect } from "vitest"
import { BUTTON_OPTIONS, KEYS, MOVE_OPTIONS, SCROLL_OPTIONS, emptyObservation, isValidAction } from "@blanktab/interface"
import type { Action, Observation } from "@blanktab/interface"
import { createCpuBackend, createRng } from "@blanktab/tensor"
import type { Backend, Storage } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, EXPLORE_EPSILON_RANGE, HEAD_LR_SCALE_RANGE, RESIDUAL_DECAY_RANGE, LEARNER_STATE_VERSION, clampSettings, createLearner, curiosityReward, LR_RANGE, sampleAction } from "../src/learner"
import type { HeldOutSet, LearnerSettings, LearnerState, UpdateInfo } from "../src/learner"
import { copyLastBaseline, targetsFromObs } from "../src/encode"
import type { Widths } from "../src/model"
import { createSyntheticStream } from "./synthetic"

const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }

async function run(opts: {
  seed: number
  steps: number
  settings?: Partial<LearnerSettings>
  stream?: { observe(): Observation; act(a: Action): void }
  at?: (step: number, l: Awaited<ReturnType<typeof createLearner>>, be: ReturnType<typeof createCpuBackend>) => void | Promise<void>
  /** Sleep after every `sleepEvery`-th step (counting the learner's own steps, so a restored learner keeps the schedule). */
  sleepEvery?: number
  /** Resume from this checkpoint instead of starting fresh (seed and settings are then the state's). */
  state?: LearnerState
}) {
  const be = createCpuBackend()
  const updates: UpdateInfo[] = []
  const onUpdate = (u: UpdateInfo) => updates.push(u)
  const learner = opts.state
    ? await createLearner({ backend: be, state: opts.state, onUpdate })
    : await createLearner({ backend: be, seed: opts.seed, widths: TINY, settings: { ...DEFAULT_SETTINGS, ...opts.settings }, onUpdate })
  const world = opts.stream ?? createSyntheticStream()
  const actions: Action[] = []
  for (let i = 1; i <= opts.steps; i++) {
    const a = await learner.step(world.observe())
    actions.push(a)
    world.act(a)
    if (opts.sleepEvery && learner.stats().steps % opts.sleepEvery === 0) await learner.sleep()
    await opts.at?.(i, learner, be)
  }
  return { learner, be, actions, updates }
}

/**
 * Bitwise deep equality (typed arrays by bytes, numbers by Object.is), returning the first differing path or null.
 * vitest's toEqual walks typed arrays element by element and takes seconds on a full LearnerState.
 */
function firstDifference(a: unknown, b: unknown, path = "state"): string | null {
  if (ArrayBuffer.isView(a) || ArrayBuffer.isView(b)) {
    if (!ArrayBuffer.isView(a) || !ArrayBuffer.isView(b) || a.constructor !== b.constructor) return path
    const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength)
    const y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength)
    if (x.length !== y.length) return path
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return path
    return null
  }
  if (typeof a !== "object" || a === null || typeof b !== "object" || b === null) return Object.is(a, b) ? null : path
  if (Array.isArray(a) !== Array.isArray(b)) return path
  const ka = Object.keys(a)
  const kb = Object.keys(b)
  if (ka.length !== kb.length || ka.some((k) => !(k in b))) return `${path} (keys)`
  for (const k of ka) {
    const d = firstDifference((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k], `${path}.${k}`)
    if (d) return d
  }
  return null
}

const finiteParams = async (l: { readParams(): Promise<Float32Array[]> }) =>
  (await l.readParams()).every((p) => p.every(Number.isFinite))

describe("learner", { timeout: 30_000 }, () => { // every test builds a real learner: >= 30 s (controller ruling; ~1-8 s unloaded)
  // Several seeds must pass (task brief + controller rulings): after 3,000 steps the action-conditioned predictor
  // beats copy-last (predLoss < baselineLoss, both EMAs with the same alpha). Measured predLoss/baselineLoss at
  // 3,000 steps (the then-default lr 1e-3, headLrScale 0.1, widths 4,4,16,16): seed 1 0.883, seed 2 0.874, seed 3 0.786.
  // Still passing at the Plan 04a.2 defaults (lr 5e-4, exploreEpsilon 0.1, residualDecay 1e-3).
  for (const seed of [1, 2, 3]) {
    it(`learns to predict the next frame better than copy-last after 3000 steps (seed ${seed})`, async () => {
      const { learner } = await run({ seed, steps: 3000 })
      const s = learner.stats()
      expect(Number.isFinite(s.predLoss) && Number.isFinite(s.baselineLoss)).toBe(true)
      expect(s.predLoss).toBeLessThan(s.baselineLoss)
      expect(s.steps).toBe(3000)
      expect(s.skippedUpdates).toBe(0)
      learner.dispose()
    }, 90_000) // ~18-20 s each on an M-series CPU; generous for slower CI machines
  }

  it("is deterministic: same seed and stream give identical actions and losses", async () => {
    const a = await run({ seed: 9, steps: 120 })
    const b = await run({ seed: 9, steps: 120 })
    const c = await run({ seed: 10, steps: 120 })
    expect(b.actions).toEqual(a.actions)
    expect(b.updates).toEqual(a.updates)
    expect(b.learner.stats()).toEqual(a.learner.stats())
    expect(c.updates).not.toEqual(a.updates)
    for (const r of [a, b, c]) r.learner.dispose()
  }, 30_000) // ~2.7 s unloaded; timed out at the 5 s default under load (controller ruling: >= 2x, min 30 s)

  describe("ε-uniform exploration (exploreEpsilon)", () => {
    // Recorded from the learner before exploreEpsilon existed (commit 863adae): seed 5, TINY, DEFAULT_SETTINGS, the
    // synthetic stream, 40 steps with a sleep every 10. Actions as move.button.scroll.key; losses as an FNV-1a hash of
    // the 39 UpdateInfo.lossBefore values' float64 bytes.
    const GOLDEN_ACTIONS = "5.2.2.33 4.2.2.41 1.2.2.34 12.1.2.12 10.0.0.38 12.1.0.16 1.2.1.26 7.2.2.33 7.0.1.11 14.0.0.30 20.0.2.10 22.1.2.20 3.2.1.31 3.2.0.29 10.0.0.2 3.1.1.40 3.2.0.15 15.1.2.25 3.2.0.2 9.0.1.8 2.2.0.6 18.2.1.27 22.1.0.44 8.2.2.32 8.2.1.16 17.2.0.22 18.1.2.12 16.1.1.44 16.1.0.43 2.1.2.38 12.0.2.32 13.0.2.7 0.2.2.2 17.0.1.26 15.1.2.0 8.2.2.21 8.2.1.20 11.2.0.7 18.1.1.39 5.1.2.33"
    const GOLDEN_LOSS_HASH = "e40e135c"
    /** The settings of that recording, on top of today's DEFAULT_SETTINGS: no exploration, no residual decay. */
    const BEFORE = { lr: 1e-3, exploreEpsilon: 0, residualDecay: 0 }
    const fnv = (xs: number[]) => {
      let h = 0x811c9dc5
      for (const b of new Uint8Array(new Float64Array(xs).buffer)) { h ^= b; h = Math.imul(h, 0x01000193) >>> 0 }
      return h.toString(16)
    }
    const fmt = (as: Action[]) => as.map((a) => `${a.move}.${a.button}.${a.scroll}.${a.key}`).join(" ")

    it("defaults to 0.1 and is clamped into [0, 0.5]; a non-finite value is rejected", () => {
      expect(DEFAULT_SETTINGS.exploreEpsilon).toBe(0.1)
      expect(EXPLORE_EPSILON_RANGE).toEqual([0, 0.5])
      expect(clampSettings({ ...DEFAULT_SETTINGS, exploreEpsilon: 0.9 }).exploreEpsilon).toBe(0.5)
      expect(clampSettings({ ...DEFAULT_SETTINGS, exploreEpsilon: -1 }).exploreEpsilon).toBe(0)
      for (const v of [Number.NaN, Number.POSITIVE_INFINITY, undefined as unknown as number]) {
        expect(() => clampSettings({ ...DEFAULT_SETTINGS, exploreEpsilon: v })).toThrow(RangeError)
      }
    })

    it("ε = 0 and residualDecay = 0 (at the then-default lr 1e-3) reproduce the learner from before either existed, bit for bit", async () => {
      const r = await run({ seed: 5, steps: 40, sleepEvery: 10, settings: BEFORE })
      expect(fmt(r.actions)).toBe(GOLDEN_ACTIONS)
      expect(r.updates.length).toBe(39)
      expect(fnv(r.updates.map((u) => u.lossBefore))).toBe(GOLDEN_LOSS_HASH)
      r.learner.dispose()
    })

    it("ε > 0 stays deterministic per seed, keeps actions valid, and changes what is done", async () => {
      const a = await run({ seed: 5, steps: 80, sleepEvery: 10, settings: { exploreEpsilon: 0.3 } })
      const b = await run({ seed: 5, steps: 80, sleepEvery: 10, settings: { exploreEpsilon: 0.3 } })
      expect(b.actions).toEqual(a.actions)
      expect(b.updates).toEqual(a.updates)
      expect(a.actions.every(isValidAction)).toBe(true)
      expect(fmt(a.actions.slice(0, 40))).not.toBe(GOLDEN_ACTIONS)
      for (const r of [a, b]) r.learner.dispose()
    })

    it("ε > 0 survives save and load: the resumed learner explores exactly as one that never stopped", async () => {
      const whole = await run({ seed: 6, steps: 60, settings: { exploreEpsilon: 0.2 } })
      const first = await run({ seed: 6, steps: 30, settings: { exploreEpsilon: 0.2 } })
      const state = await first.learner.save()
      first.learner.dispose()
      expect(state.settings.exploreEpsilon).toBe(0.2)
      const be = createCpuBackend()
      const resumed = await createLearner({ backend: be, state })
      const world = createSyntheticStream()
      for (const a of first.actions) { world.observe(); world.act(a) }
      const rest: Action[] = []
      for (let i = 0; i < 30; i++) { const a = await resumed.step(world.observe()); rest.push(a); world.act(a) }
      expect(rest).toEqual(whole.actions.slice(30))
      resumed.dispose()
      whole.learner.dispose()
    })

    it("residualDecay: defaults to 1e-3 (with lr 5e-4), clamped into [0, 1e-2]; a non-finite value is rejected", () => {
      expect(DEFAULT_SETTINGS.residualDecay).toBe(1e-3)
      expect(DEFAULT_SETTINGS.lr).toBe(5e-4)
      expect(RESIDUAL_DECAY_RANGE).toEqual([0, 1e-2])
      expect(clampSettings({ ...DEFAULT_SETTINGS, residualDecay: 0.5 }).residualDecay).toBe(1e-2)
      expect(clampSettings({ ...DEFAULT_SETTINGS, residualDecay: -1 }).residualDecay).toBe(0)
      for (const v of [Number.NaN, undefined as unknown as number]) expect(() => clampSettings({ ...DEFAULT_SETTINGS, residualDecay: v })).toThrow(RangeError)
    })

    it("residualDecay > 0 pulls the residual layers (shift gate, output layers) toward zero, deterministically", async () => {
      const resid = async (l: { readParams(): Promise<Float32Array[]> }) => {
        const ps = await l.readParams()
        const n = ps.length
        return [ps[n - 9]!, ...ps.slice(-4)].reduce((a, p) => a + p.reduce((b, v) => b + v * v, 0), 0)
      }
      const off = await run({ seed: 5, steps: 60, sleepEvery: 10, settings: { exploreEpsilon: 0, residualDecay: 0 } })
      const on = await run({ seed: 5, steps: 60, sleepEvery: 10, settings: { exploreEpsilon: 0, residualDecay: 1e-2 } })
      const on2 = await run({ seed: 5, steps: 60, sleepEvery: 10, settings: { exploreEpsilon: 0, residualDecay: 1e-2 } })
      expect(await resid(on.learner)).toBeLessThan(await resid(off.learner))
      expect(on2.updates).toEqual(on.updates)
      for (const r of [off, on, on2]) r.learner.dispose()
    })

    it("sampleAction: with probability ε a factor is uniform, otherwise it follows the policy", () => {
      const sizes = [MOVE_OPTIONS, BUTTON_OPTIONS, SCROLL_OPTIONS, KEYS.length]
      const logits = new Float32Array(sizes.reduce((a, n) => a + n, 0)).fill(-100)
      let off = 0
      for (const n of sizes) { logits[off] = 100; off += n } // the policy always picks index 0 of every factor
      const draws = 20_000
      for (const eps of [0, 0.2]) {
        const rng = createRng(11)
        const off0 = [0, 0, 0, 0]
        for (let i = 0; i < draws; i++) {
          const a = sampleAction(logits, eps, rng)
          expect(isValidAction(a)).toBe(true)
          ;[a.move, a.button, a.scroll, a.key].forEach((v, f) => { if (v !== 0) off0[f]!++ })
        }
        sizes.forEach((n, f) => {
          const want = eps * (n - 1) / n // explored, and the uniform draw missed index 0
          expect(Math.abs(off0[f]! / draws - want)).toBeLessThan(eps === 0 ? 1e-12 : 0.012)
        })
      }
    })
  })

  it("returns only valid actions, counts steps and compute", async () => {
    const { learner, actions, updates } = await run({ seed: 4, steps: 200 })
    expect(actions.every(isValidAction)).toBe(true)
    // Not degenerate: the policy samples more than one move.
    expect(new Set(actions.map((a) => a.move)).size).toBeGreaterThan(1)
    expect(updates.length).toBe(199) // no transition to learn from on the first step
    const s = learner.stats()
    expect(s.steps).toBe(200)
    expect(s.cu).toBeGreaterThan(0)
    expect(Number.isFinite(s.predLoss) && Number.isFinite(s.baselineLoss)).toBe(true)
    expect(s.sleeps).toBe(0)
    expect(s.consecutiveSkips).toBe(0)
    learner.dispose()
  })

  it("keeps liveBuffers flat between step 100 and step 600", async () => {
    let at100 = -1
    const { learner, be } = await run({ seed: 5, steps: 600, at: (i, _l, b) => { if (i === 100) at100 = b.stats.liveBuffers } })
    expect(at100).toBeGreaterThan(0)
    expect(be.stats.liveBuffers).toBe(at100)
    learner.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  }, 30_000)

  it("Review Focus 4: an all-black stream with the cursor in a corner keeps losses finite and weights NaN-free", async () => {
    for (const corner of [{ x: 0, y: 0 }, { x: 1, y: 1 }]) {
      const black = { observe: () => ({ ...emptyObservation(), cursor: { ...corner } }), act: () => {} }
      const { learner, updates } = await run({ seed: 6, steps: 150, stream: black })
      expect(updates.every((u) => Number.isFinite(u.lossBefore) && Number.isFinite(u.lossAfter) && Number.isFinite(u.reward))).toBe(true)
      expect(Number.isFinite(learner.stats().predLoss)).toBe(true)
      expect(learner.stats().skippedUpdates).toBe(0)
      expect(await finiteParams(learner)).toBe(true)
      learner.dispose()
    }
  })

  it("Review Focus 4: a corrupt cursor (NaN / Infinity) never reaches the weights", async () => {
    const s = createSyntheticStream()
    const bad = { observe: () => ({ ...s.observe(), cursor: { x: Number.NaN, y: Number.POSITIVE_INFINITY } }), act: s.act }
    const { learner } = await run({ seed: 6, steps: 50, stream: bad })
    expect(Number.isFinite(learner.stats().predLoss)).toBe(true)
    expect(await finiteParams(learner)).toBe(true)
    learner.dispose()
  })

  it("Review Focus 5: lr = 1 is clamped and the run stays finite", async () => {
    expect(clampSettings({ ...DEFAULT_SETTINGS, lr: 1 }).lr).toBe(LR_RANGE[1])
    expect(clampSettings({ ...DEFAULT_SETTINGS, lr: 1e-9 }).lr).toBe(LR_RANGE[0])
    expect(LR_RANGE).toEqual([1e-5, 1e-2])
    expect(DEFAULT_SETTINGS.headLrScale).toBe(0.1)
    expect(HEAD_LR_SCALE_RANGE).toEqual([0.01, 1])
    expect(clampSettings({ ...DEFAULT_SETTINGS, headLrScale: 5 }).headLrScale).toBe(1)
    expect(clampSettings({ ...DEFAULT_SETTINGS, headLrScale: 0 }).headLrScale).toBe(0.01)
    const { learner, updates } = await run({ seed: 7, steps: 200, settings: { lr: 1 } })
    expect(updates.every((u) => Number.isFinite(u.lossBefore) && Number.isFinite(u.lossAfter))).toBe(true)
    expect(await finiteParams(learner)).toBe(true)
    learner.dispose()
  })

  it("Review Focus 5: the extremes of entropyWeight and curiosityWeight run without NaN", async () => {
    // Ranges from the plan (Task 7 writes them into knowledge/settings-space.json): entropyWeight [0.001, 0.05],
    // curiosityWeight [0.5, 2]. Also beyond them (0 and 10x), since a setting outside the space must not NaN either.
    for (const entropyWeight of [0, 0.001, 0.05, 0.5]) {
      for (const curiosityWeight of [0, 0.5, 2, 20]) {
        const { learner, updates } = await run({ seed: 8, steps: 60, settings: { entropyWeight, curiosityWeight } })
        expect(updates.every((u) => Number.isFinite(u.lossBefore) && Number.isFinite(u.advantage))).toBe(true)
        expect(learner.stats().skippedUpdates).toBe(0)
        expect(await finiteParams(learner)).toBe(true)
        learner.dispose()
      }
    }
  }, 30_000)

  it("rejects settings that cannot be clamped", () => {
    for (const bad of [{ lr: Number.NaN }, { entropyWeight: -1 }, { curiosityWeight: Number.POSITIVE_INFINITY }, { bufferSize: 0 }, { bufferSize: 2.5 }, { replayPerSleep: -3 }, { headLrScale: Number.NaN }]) {
      expect(() => clampSettings({ ...DEFAULT_SETTINGS, ...bad })).toThrow(RangeError)
    }
  })

  it("the curiosity reward is relative learning progress, finite and clamped to [-1, 1]", async () => {
    expect(curiosityReward(0.5, 0.25, 1)).toBeCloseTo(0.25)
    expect(curiosityReward(0.011, 0.010, 0.01)).toBeCloseTo(0.1) // relative to the running loss scale
    expect(curiosityReward(5, 0, 1)).toBe(1)
    expect(curiosityReward(0, 5, 1)).toBe(-1)
    expect(curiosityReward(2e-6, 1e-6, 0)).toBeCloseTo(1) // scale floored at 1e-6
    for (const [b, a, sc] of [[Number.NaN, 0, 1], [0, Number.NaN, 1], [Number.POSITIVE_INFINITY, 0, 1], [1, 0, Number.NaN]] as const) {
      expect(curiosityReward(b, a, sc)).toBe(0)
    }
    const { learner, updates } = await run({ seed: 3, steps: 200 })
    expect(updates.length).toBeGreaterThan(0)
    for (const u of updates) {
      expect(Number.isFinite(u.reward)).toBe(true)
      expect(Math.abs(u.reward)).toBeLessThanOrEqual(1)
    }
    learner.dispose()
  })

  it("scales curiosity by the predLoss EMA from before this step's loss (no self-inclusion)", async () => {
    const { learner, updates } = await run({ seed: 5, steps: 60 })
    let ema = Number.NaN
    for (const u of updates) {
      expect(u.reward).toBe(curiosityReward(u.lossBefore, u.lossAfter, ema)) // the first update has no scale yet: 0
      ema = Number.isFinite(ema) ? ema + 0.01 * (u.lossBefore - ema) : u.lossBefore
    }
    expect(updates[0]!.reward).toBe(0)
    learner.dispose()
  })

  it("trains the residual output layers at lr x headLrScale and every other parameter at lr", async () => {
    // Adam's first step moves each parameter with a non-zero gradient by ~lr (m/sqrt(v) = sign(g)).
    const learner = await createLearner({ backend: createCpuBackend(), seed: 2, widths: TINY, settings: { ...DEFAULT_SETTINGS, lr: 1e-3, headLrScale: 0.1 } })
    const world = createSyntheticStream()
    world.act(await learner.step(world.observe()))
    const p0 = await learner.readParams()
    world.act(await learner.step(world.observe())) // first update
    const p1 = await learner.readParams()
    const maxStep = (i: number) => p1[i]!.reduce((m, v, k) => Math.max(m, Math.abs(v - p0[i]![k]!)), 0)
    const n = p0.length
    for (const i of [n - 4, n - 3, n - 2, n - 1]) expect(maxStep(i)).toBeLessThan(1.01e-4) // headP w/b, headF w/b
    // headP sees the block move; headF's target is unchanged on this step (block centred, plain fovea): zero gradient.
    for (const i of [n - 4, n - 3]) expect(maxStep(i)).toBeGreaterThan(0.9e-4)
    // On this first update the zero output layers block every gradient upstream of them (core, predictor h/action
    // layers get exactly 0), so compare with the policy and value heads, which are on the main optimizer.
    for (const i of [n - 15, n - 14, n - 13, n - 12]) { // policy w/b, value w/b (then shiftX, shiftY, shiftGate, predH, predA)
      expect(maxStep(i)).toBeGreaterThan(0.9e-3)
      expect(maxStep(i)).toBeLessThan(1.01e-3)
    }
    learner.dispose()
  })

  it("on a fixed stream, curiosity is on average positive early", async () => {
    // Fixed = independent of the learner's actions: the block ping-pongs 4 px per step (E for 40 steps, then W).
    const sweep = createSyntheticStream({ start: { x: 20, y: 100 } })
    let k = 0
    const fixed = { observe: () => { sweep.act({ move: Math.floor(k++ / 40) % 2 === 0 ? 2 : 14, button: 0, scroll: 0, key: 0 }); return sweep.observe() }, act: () => {} }
    const { learner, updates } = await run({ seed: 4, steps: 101, stream: fixed })
    const mean = updates.reduce((a, u) => a + u.reward, 0) / updates.length
    expect(updates.length).toBe(100)
    expect(mean).toBeGreaterThan(0)
    learner.dispose()
  })

  it("a non-finite loss skips the update: no throw, skippedUpdates + 1, every other parameter bitwise unchanged", async () => {
    // White-box: record CPU storages as they are allocated; the model's parameters come first, in Model.params order.
    const inner = createCpuBackend()
    const allocated: Storage[] = []
    const be: Backend = {
      ...inner,
      upload: (d) => { const st = inner.upload(d); allocated.push(st); return st },
      zeros: (n) => { const st = inner.zeros(n); allocated.push(st); return st },
    }
    const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
    const world = createSyntheticStream()
    for (let i = 0; i < 5; i++) world.act(await learner.step(world.observe()))
    const before = await learner.readParams()
    // Poison the periphery output-layer bias (Model.params index 35: ..., predA w/b, headP w (34), headP b (35), headF w/b).
    const POISONED = 35
    expect(allocated[POISONED]!.length).toBe(3000)
    ;(allocated[POISONED] as unknown as { data: Float32Array }).data[7] = Number.NaN
    const skipped = learner.stats().skippedUpdates
    const a = await learner.step(world.observe())
    expect(isValidAction(a)).toBe(true)
    expect(learner.stats().skippedUpdates).toBe(skipped + 1)
    expect(learner.stats().consecutiveSkips).toBe(1)
    const after = await learner.readParams()
    after.forEach((p, i) => {
      if (i === POISONED) return
      expect(p).toEqual(before[i])
    })
    learner.dispose()
  })

  it("replay buffer holds the last bufferSize transitions as raw frames", async () => {
    const { learner } = await run({ seed: 2, steps: 30, settings: { bufferSize: 10 } })
    const buf = learner.replay
    expect(buf.size).toBe(10)
    const t = buf.at(buf.size - 1)
    expect(t.obs.periphery).toBeInstanceOf(Uint8Array)
    expect(t.next.fovea).toBeInstanceOf(Uint8Array)
    expect(t.hIn.length).toBe(TINY.hidden)
    // Consecutive transitions share the frame between them: next of one is obs of the following.
    expect(buf.at(buf.size - 2).next).toBe(t.obs)
    learner.dispose()
  })

  it("rejects a malformed frame without counting a step or leaking", async () => {
    const be = createCpuBackend()
    const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
    await learner.step(emptyObservation())
    const live = be.stats.liveBuffers
    await expect(learner.step({ ...emptyObservation(), fovea: new Uint8Array(5) })).rejects.toThrow(RangeError)
    expect(learner.stats().steps).toBe(1)
    expect(be.stats.liveBuffers).toBe(live)
    await learner.step(emptyObservation())
    expect(learner.stats().steps).toBe(2)
    learner.dispose()
  })

  it("rejects overlapping step calls", async () => {
    const learner = await createLearner({ backend: createCpuBackend(), seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
    const first = learner.step(emptyObservation())
    await expect(learner.step(emptyObservation())).rejects.toThrow(/in progress/)
    await first
    learner.dispose()
  })

  describe("sleep, save and load", { timeout: 30_000 }, () => {
    /** The stream is part of the run: a resumed learner continues against the same stream object. */
    it("save at step 300 (sleeps every 100), load into a fresh learner, 200 more steps = an uninterrupted 500-step run", async () => {
      const full = await run({ seed: 11, steps: 500, sleepEvery: 100 })
      expect(full.learner.stats().sleeps).toBe(5)

      const world = createSyntheticStream()
      const first = await run({ seed: 11, steps: 300, sleepEvery: 100, stream: world })
      const state = await first.learner.save()
      first.learner.dispose()
      expect(first.be.stats.liveBuffers).toBe(0)
      const second = await run({ seed: -1, steps: 200, sleepEvery: 100, stream: world, state })

      expect([...first.actions, ...second.actions]).toEqual(full.actions)
      expect([...first.updates, ...second.updates]).toEqual(full.updates)
      expect(second.learner.stats()).toEqual(full.learner.stats())
      expect(await second.learner.readParams()).toEqual(await full.learner.readParams())
      expect(firstDifference(await second.learner.save(), await full.learner.save())).toBeNull()
      full.learner.dispose()
      second.learner.dispose()
    }, 30_000) // ~7.6 s unloaded (controller ruling: >= 2x, min 30 s)

    it("restores a learner saved before its first step and one saved after a single step", async () => {
      for (const n of [0, 1]) {
        const full = await run({ seed: 12, steps: 20 })
        const world = createSyntheticStream()
        const first = await run({ seed: 12, steps: n, stream: world })
        const second = await run({ seed: -1, steps: 20 - n, stream: world, state: await first.learner.save() })
        expect([...first.actions, ...second.actions]).toEqual(full.actions)
        expect([...first.updates, ...second.updates]).toEqual(full.updates)
        for (const r of [full, first, second]) r.learner.dispose()
      }
    })

    it("LearnerState survives structuredClone, is a snapshot, and a cloned state resumes identically", async () => {
      const world = createSyntheticStream()
      const { learner } = await run({ seed: 13, steps: 40, sleepEvery: 20, stream: world, settings: { bufferSize: 16 } })
      const state = await learner.save()
      const clone = structuredClone(state)
      expect(firstDifference(clone, state)).toBeNull()
      const tweaked = structuredClone(state) // the comparison itself sees a one-float change
      tweaked.adam[1].slots[3]!.v[0]! += 1e-30
      expect(firstDifference(tweaked, state)).toBe("state.adam.1.slots.3.v")
      expect(state.params[0]).toBeInstanceOf(Float32Array)
      expect(state.replay.length).toBe(16)
      expect(state.version).toBe(LEARNER_STATE_VERSION)
      expect(state.seed).toBe(13)
      expect(state.widths).toEqual(TINY)
      // A snapshot: stepping on does not change it, and loading from it does not alias it.
      const frozen = structuredClone(state)
      world.act(await learner.step(world.observe()))
      await learner.sleep()
      expect(firstDifference(state, frozen)).toBeNull()
      const w2 = createSyntheticStream()
      const w3 = createSyntheticStream()
      const a = await run({ seed: -1, steps: 30, sleepEvery: 10, stream: w2, state })
      const b = await run({ seed: -1, steps: 30, sleepEvery: 10, stream: w3, state: clone })
      expect(firstDifference(state, frozen)).toBeNull()
      expect(b.actions).toEqual(a.actions)
      expect(b.updates).toEqual(a.updates)
      for (const r of [a, b]) r.learner.dispose()
      learner.dispose()
    }, 30_000) // ~2.8 s unloaded; timed out at the 5 s default under load (controller ruling)

    it("rejects a state that does not fit, before allocating anything", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      const be = createCpuBackend()
      const bad: LearnerState[] = [
        { ...state, version: (LEARNER_STATE_VERSION + 1) as typeof LEARNER_STATE_VERSION },
        { ...state, params: state.params.slice(1) },
        { ...state, widths: { ...TINY, hidden: 8 } },
        { ...state, h: new Float32Array(3) },
        { ...state, adam: state.adam.slice(1) as unknown as LearnerState["adam"] },
        { ...state, settings: { ...state.settings, bufferSize: 2 } }, // fewer slots than the 4 stored transitions
      ]
      for (const s of bad) await expect(createLearner({ backend: be, state: s })).rejects.toThrow(RangeError)
      expect(be.stats.liveBuffers).toBe(0)
    })

    it("rejects non-finite parameters, Adam moments and recurrent state with RangeError, allocating nothing", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      const be = createCpuBackend()
      const poison = (mutate: (s: LearnerState) => void, v: number) => {
        const s = structuredClone(state)
        mutate(s)
        return { s, v }
      }
      const cases = [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY].flatMap(v => [
        poison(s => { s.params[3]![1] = v }, v),
        poison(s => { s.adam[0].slots[2]!.m[0] = v }, v),
        poison(s => { s.adam[1].slots[1]!.v[4] = v }, v),
        poison(s => { s.h[5] = v }, v),
        poison(s => { s.last!.hIn[0] = v }, v),
        poison(s => { s.replay[1]!.hIn[2] = v }, v),
      ])
      for (const { s } of cases) await expect(createLearner({ backend: be, state: s })).rejects.toThrow(RangeError)
      expect(be.stats.liveBuffers).toBe(0)
    })

    it("turns missing or malformed top-level fields into RangeError (not TypeError)", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      const be = createCpuBackend()
      const keys = ["version", "seed", "widths", "settings", "params", "adam", "rng", "h", "last", "replay", "stats", "heldOut"] as const
      for (const k of keys) {
        const s = { ...state } as Partial<LearnerState>
        delete s[k]
        await expect(createLearner({ backend: be, state: s as LearnerState }), `without ${k}`).rejects.toThrow(RangeError)
      }
      const odd: unknown[] = [null, 7, "x", [], { ...state, adam: [null, null] }, { ...state, stats: null }, { ...state, widths: null }, { ...state, params: [null] }, { ...state, last: { obs: null } }, { ...state, replay: [null] }]
      for (const s of odd) await expect(createLearner({ backend: be, state: s as LearnerState })).rejects.toThrow(RangeError)
      expect(be.stats.liveBuffers).toBe(0)
    })

    it("a restore that fails part-way (the device refuses an allocation) frees everything it had allocated", async () => {
      const { learner } = await run({ seed: 14, steps: 5 })
      const state = await learner.save()
      learner.dispose()
      // Count the allocations of a successful restore, then fail at each one in turn. The backend is patched in
      // place: a tensor's backend is its storage's, so a wrapper object would miss the model's and Adam's allocations.
      const refusing = (failAt: number) => {
        const be = createCpuBackend()
        const { upload, zeros } = be
        let k = 0
        const take = () => { if (++k === failAt) throw new Error("device lost") }
        Object.assign(be, { upload: (d: Float32Array) => { take(); return upload(d) }, zeros: (m: number) => { take(); return zeros(m) } })
        return { be, count: () => k }
      }
      const counting = refusing(0)
      ;(await createLearner({ backend: counting.be, state })).dispose()
      const n = counting.count()
      expect(n).toBeGreaterThan(50)
      for (let failAt = 1; failAt <= n; failAt++) {
        const { be: inner } = refusing(failAt)
        await expect(createLearner({ backend: inner, state }), `failing at allocation ${failAt}`).rejects.toThrow("device lost")
        expect(inner.stats.liveBuffers, `failing at allocation ${failAt}`).toBe(0)
      }
    })

    it("sleep() replays replayPerSleep uniformly sampled transitions and lowers the mean loss on the buffer", async () => {
      // Fixed stream, independent of the actions: the block sweeps E then W, 4 px per step.
      const sweep = createSyntheticStream({ start: { x: 20, y: 100 } })
      let k = 0
      const fixed = { observe: () => { sweep.act({ move: Math.floor(k++ / 40) % 2 === 0 ? 2 : 14, button: 0, scroll: 0, key: 0 }); return sweep.observe() }, act: () => {} }
      const { learner, updates } = await run({ seed: 15, steps: 120, stream: fixed, settings: { bufferSize: 100, replayPerSleep: 200 } })
      const s0 = learner.stats()
      const before = await learner.replayLoss()
      expect(learner.stats()).toEqual(s0) // measuring is not learning and not billed
      const p0 = await learner.readParams()
      await learner.sleep()
      const after = await learner.replayLoss()
      const s1 = learner.stats()
      expect(Number.isFinite(before) && Number.isFinite(after)).toBe(true)
      expect(after).toBeLessThan(before)
      expect(s1.sleeps).toBe(1)
      expect(s1.steps).toBe(s0.steps)
      expect(s1.cu).toBeGreaterThan(s0.cu)
      expect(s1.skippedUpdates).toBe(0)
      expect(updates.length).toBe(119) // sleep does not report online updates
      // Prediction only: the policy and value heads are untouched by replay.
      const p1 = await learner.readParams()
      const n = p0.length
      for (const i of [n - 12, n - 11, n - 10, n - 9]) expect(p1[i]).toEqual(p0[i])
      expect(p1[0]).not.toEqual(p0[0])
      learner.dispose()
    })

    it("sleep() on an empty buffer is a counted no-op, and sleep() cannot overlap a step", async () => {
      const be = createCpuBackend()
      const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
      const live = be.stats.liveBuffers
      await learner.sleep()
      expect(learner.stats().sleeps).toBe(1)
      expect(be.stats.liveBuffers).toBe(live)
      const step = learner.step(emptyObservation())
      const [sl, sv, rl] = [learner.sleep(), learner.save(), learner.replayLoss()] // all started while the step runs
      for (const p of [sl, sv, rl]) await expect(p).rejects.toThrow(/in progress/)
      await step
      expect(learner.stats().sleeps).toBe(1)
      learner.dispose()
      await expect(learner.sleep()).rejects.toThrow(/dispose/)
    })

    it("Review Focus 1: after 2000 steps with a sleep every 250, liveBuffers equals its value after the first sleep", async () => {
      let afterFirstSleep = -1
      const { learner, be } = await run({
        seed: 16, steps: 2000, sleepEvery: 250,
        at: (i, _l, b) => { if (i === 250) afterFirstSleep = b.stats.liveBuffers },
      })
      expect(learner.stats().sleeps).toBe(8)
      expect(afterFirstSleep).toBeGreaterThan(0)
      expect(be.stats.liveBuffers).toBe(afterFirstSleep)
      learner.dispose()
      expect(be.stats.liveBuffers).toBe(0)
    }, 60_000) // ~26-28 s on a loaded M-series CPU (controller ruling: 60 s)

    it("consecutiveSkips counts skipped updates in a row, and restoring the last checkpoint clears it", async () => {
      // Same white-box poisoning as the skip test above: a NaN in headP's bias makes every prediction loss non-finite.
      const inner = createCpuBackend()
      const allocated: Storage[] = []
      const be: Backend = {
        ...inner,
        upload: (d) => { const st = inner.upload(d); allocated.push(st); return st },
        zeros: (n) => { const st = inner.zeros(n); allocated.push(st); return st },
      }
      const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS })
      const world = createSyntheticStream()
      for (let i = 0; i < 5; i++) world.act(await learner.step(world.observe()))
      const checkpoint = await learner.save()
      ;(allocated[32] as unknown as { data: Float32Array }).data[7] = Number.NaN
      for (let i = 1; i <= 4; i++) {
        world.act(await learner.step(world.observe()))
        expect(learner.stats().consecutiveSkips).toBe(i)
      }
      await learner.sleep() // every replayed update is skipped too
      expect(learner.stats().consecutiveSkips).toBe(4 + DEFAULT_SETTINGS.replayPerSleep)
      expect(learner.stats().skippedUpdates).toBe(4 + DEFAULT_SETTINGS.replayPerSleep)
      learner.dispose()
      // The harness's response: restore from the last checkpoint and carry on.
      const restored = await createLearner({ backend: createCpuBackend(), state: checkpoint })
      expect(restored.stats().consecutiveSkips).toBe(0)
      world.act(await restored.step(world.observe()))
      expect(restored.stats().consecutiveSkips).toBe(0)
      expect(restored.stats().skippedUpdates).toBe(0)
      restored.dispose()
    })
  })
})

/** A held-out trajectory from the synthetic stream under uniformly random actions (the worker builds the real one from the world). */
function syntheticHeldOut(seed: number, n: number): HeldOutSet {
  const w = createSyntheticStream({ start: { x: 60, y: 60 } })
  const rnd = createRng(seed)
  const pick = (k: number) => Math.min(k - 1, Math.floor(rnd() * k))
  const obs = [w.observe()]
  const actions: Action[] = []
  for (let i = 0; i < n; i++) {
    const a = { move: pick(MOVE_OPTIONS), button: pick(BUTTON_OPTIONS), scroll: pick(SCROLL_OPTIONS), key: pick(KEYS.length) }
    w.act(a)
    actions.push(a)
    obs.push(w.observe())
  }
  return { obs, actions }
}
const copyLastOn = (set: HeldOutSet) =>
  set.actions.reduce((s, _, i) => s + copyLastBaseline(targetsFromObs(set.obs[i]!), targetsFromObs(set.obs[i + 1]!)), 0) / set.actions.length

describe("the held-out check (decision 0015 §3)", { timeout: 60_000 }, () => {
  const SET = syntheticHeldOut(7, 16)

  it("a newborn scores exactly copy-last on it; nothing is measured before the first sleep", async () => {
    const be = createCpuBackend()
    const learner = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET })
    const s0 = learner.stats()
    expect(Number.isNaN(s0.heldOutLoss) && Number.isNaN(s0.heldOutBaseline)).toBe(true)
    expect(s0.evalCU).toBe(0)
    await learner.sleep() // empty buffer: no training, but the held-out set is scored
    const s1 = learner.stats()
    expect(s1.heldOutBaseline).toBeCloseTo(copyLastOn(SET), 9)
    expect(s1.heldOutBaseline).toBeGreaterThan(0)
    expect(Math.abs(s1.heldOutLoss / s1.heldOutBaseline - 1)).toBeLessThan(1e-5) // newborn = copy-last
    expect(s1.evalCU).toBeGreaterThan(0)
    expect(s1.cu).toBe(0) // the scoring is not billed to training
    learner.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("never trains: actions, updates, weights, RNG and cu are bit-identical to a learner without it; evalCU grows per sleep", async () => {
    const plain = await run({ seed: 21, steps: 60, sleepEvery: 20 })
    const be = createCpuBackend()
    const updates: UpdateInfo[] = []
    const learner = await createLearner({ backend: be, seed: 21, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET, onUpdate: u => updates.push(u) })
    const world = createSyntheticStream()
    const actions: Action[] = []
    const evals: number[] = []
    const losses: number[] = []
    for (let i = 1; i <= 60; i++) {
      const a = await learner.step(world.observe())
      actions.push(a)
      world.act(a)
      if (i % 20 === 0) {
        await learner.sleep()
        evals.push(learner.stats().evalCU)
        losses.push(learner.stats().heldOutLoss)
      }
    }
    expect(actions).toEqual(plain.actions)
    expect(updates).toEqual(plain.updates)
    expect(await learner.readParams()).toEqual(await plain.learner.readParams())
    const { evalCU, heldOutLoss, heldOutBaseline, ...rest } = learner.stats()
    const { evalCU: e0, heldOutLoss: l0, heldOutBaseline: b0, ...plainRest } = plain.learner.stats()
    expect(rest).toEqual(plainRest) // incl. cu: bit-identical
    expect(e0).toBe(0)
    expect(Number.isNaN(l0) && Number.isNaN(b0)).toBe(true)
    // Each sleep scores the same set the same way: the same FLOPs each time.
    expect(evals[0]).toBeGreaterThan(0)
    expect(evals[1]! - evals[0]!).toBeCloseTo(evals[0]!, 12)
    expect(evals[2]! - evals[1]!).toBeCloseTo(evals[0]!, 12)
    expect(losses.every(Number.isFinite)).toBe(true)
    expect(heldOutLoss).toBe(losses[2])
    expect(heldOutBaseline).toBeCloseTo(copyLastOn(SET), 9)
    expect(evalCU).toBe(evals[2])
    learner.dispose()
    plain.learner.dispose()
  })

  it("is fixed for the learner's life: saved, restored bit-exactly, and an own copy (the caller's arrays can change)", async () => {
    const mine = structuredClone(SET)
    const full = await createLearner({ backend: createCpuBackend(), seed: 22, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: mine })
    mine.obs[3]!.fovea.fill(0) // the learner holds its own copy
    mine.actions[0]!.move = 0
    const w1 = createSyntheticStream()
    for (let i = 1; i <= 30; i++) { w1.act(await full.step(w1.observe())); if (i % 10 === 0) await full.sleep() }

    const w2 = createSyntheticStream()
    const first = await createLearner({ backend: createCpuBackend(), seed: 22, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET })
    for (let i = 1; i <= 20; i++) { w2.act(await first.step(w2.observe())); if (i % 10 === 0) await first.sleep() }
    const state = await first.save()
    first.dispose()
    expect(firstDifference(state.heldOut, SET)).toBeNull()
    const second = await createLearner({ backend: createCpuBackend(), state: structuredClone(state) })
    for (let i = 21; i <= 30; i++) { w2.act(await second.step(w2.observe())); if (i % 10 === 0) await second.sleep() }
    expect(second.stats()).toEqual(full.stats())
    expect(firstDifference(await second.save(), await full.save())).toBeNull()
    full.dispose()
    second.dispose()
  })

  it("a learner born without one saves heldOut null and measures nothing", async () => {
    const { learner } = await run({ seed: 23, steps: 10, sleepEvery: 5 })
    const s = learner.stats()
    expect(s.evalCU).toBe(0)
    expect(Number.isNaN(s.heldOutLoss)).toBe(true)
    expect((await learner.save()).heldOut).toBeNull()
    learner.dispose()
  })

  it("rejects a malformed held-out set (newborn or in a state) with RangeError, allocating nothing", async () => {
    const be = createCpuBackend()
    const badSets: unknown[] = [
      { obs: SET.obs, actions: SET.actions.slice(1) }, // obs must be one more than actions
      { obs: [SET.obs[0]], actions: [] }, // at least one transition
      { obs: SET.obs.map((o, i) => (i === 2 ? { ...o, fovea: new Uint8Array(5) } : o)), actions: SET.actions },
      { obs: SET.obs, actions: SET.actions.map((a, i) => (i === 1 ? { ...a, move: 999 } : a)) },
      { obs: SET.obs },
      null, 7,
    ]
    for (const h of badSets) {
      await expect(createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: h as HeldOutSet })).rejects.toThrow(RangeError)
    }
    const l = await createLearner({ backend: be, seed: 1, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET })
    const state = await l.save()
    l.dispose()
    for (const h of badSets.filter(x => x !== null)) {
      await expect(createLearner({ backend: be, state: { ...state, heldOut: h as HeldOutSet } })).rejects.toThrow(RangeError)
    }
    const { heldOut: _drop, ...without } = state
    void _drop
    await expect(createLearner({ backend: be, state: without as LearnerState })).rejects.toThrow(RangeError)
    for (const k of ["evalCU", "heldOutLoss", "heldOutBaseline"] as const) {
      const { [k]: _gone, ...st } = state.stats
      void _gone
      await expect(createLearner({ backend: be, state: { ...state, stats: st as LearnerState["stats"] } }), `stats without ${k}`).rejects.toThrow(RangeError)
    }
    for (const bad of [{ evalCU: -1 }, { evalCU: Number.NaN }, { heldOutLoss: Number.POSITIVE_INFINITY }, { heldOutBaseline: -0.1 }, { heldOutLoss: "0.1" }]) {
      await expect(createLearner({ backend: be, state: { ...state, stats: { ...state.stats, ...bad } as LearnerState["stats"] } }), JSON.stringify(bad)).rejects.toThrow(RangeError)
    }
    expect(be.stats.liveBuffers).toBe(0)
  })

  it("keeps liveBuffers flat across sleeps that score it", async () => {
    const be = createCpuBackend()
    const learner = await createLearner({ backend: be, seed: 24, widths: TINY, settings: DEFAULT_SETTINGS, heldOut: SET })
    const w = createSyntheticStream()
    let after = -1
    for (let i = 1; i <= 40; i++) {
      w.act(await learner.step(w.observe()))
      if (i % 10 === 0) await learner.sleep()
      if (i === 10) after = be.stats.liveBuffers
    }
    expect(be.stats.liveBuffers).toBe(after)
    learner.dispose()
    expect(be.stats.liveBuffers).toBe(0)
  })
})
