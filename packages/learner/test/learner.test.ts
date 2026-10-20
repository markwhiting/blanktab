import { describe, it, expect } from "vitest"
import { emptyObservation, isValidAction } from "@blanktab/interface"
import type { Action, Observation } from "@blanktab/interface"
import { createCpuBackend } from "@blanktab/tensor"
import type { Backend, Storage } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, HEAD_LR_SCALE_RANGE, clampSettings, createLearner, curiosityReward, LR_RANGE } from "../src/learner"
import type { LearnerSettings, UpdateInfo } from "../src/learner"
import type { Widths } from "../src/model"
import { createSyntheticStream } from "./synthetic"

const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }

async function run(opts: {
  seed: number
  steps: number
  settings?: Partial<LearnerSettings>
  stream?: { observe(): Observation; act(a: Action): void }
  at?: (step: number, l: Awaited<ReturnType<typeof createLearner>>, be: ReturnType<typeof createCpuBackend>) => void | Promise<void>
}) {
  const be = createCpuBackend()
  const updates: UpdateInfo[] = []
  const learner = await createLearner({
    backend: be, seed: opts.seed, widths: TINY, settings: { ...DEFAULT_SETTINGS, ...opts.settings },
    onUpdate: (u) => updates.push(u),
  })
  const world = opts.stream ?? createSyntheticStream()
  const actions: Action[] = []
  for (let i = 1; i <= opts.steps; i++) {
    const a = await learner.step(world.observe())
    actions.push(a)
    world.act(a)
    await opts.at?.(i, learner, be)
  }
  return { learner, be, actions, updates }
}

const finiteParams = async (l: { readParams(): Promise<Float32Array[]> }) =>
  (await l.readParams()).every((p) => p.every(Number.isFinite))

describe("learner", () => {
  // Several seeds must pass (task brief + controller ruling): after 1,500 steps the action-conditioned predictor
  // beats copy-last (predLoss < baselineLoss, both EMAs with the same alpha).
  for (const seed of [1, 2, 3]) {
    it(`learns to predict the next frame better than copy-last (seed ${seed})`, async () => {
      const { learner } = await run({ seed, steps: 1500 })
      const s = learner.stats()
      expect(Number.isFinite(s.predLoss) && Number.isFinite(s.baselineLoss)).toBe(true)
      expect(s.predLoss).toBeLessThan(s.baselineLoss)
      expect(s.steps).toBe(1500)
      expect(s.skippedUpdates).toBe(0)
      learner.dispose()
    }, 60_000)
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
    for (const i of [n - 12, n - 11, n - 10, n - 9]) { // policy w/b, value w/b
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
    // Poison the periphery output-layer bias (Model.params index 32: ..., predA w/b, headP w (31), headP b (32), headF w/b).
    const POISONED = 32
    expect(allocated[POISONED]!.length).toBe(3000)
    ;(allocated[POISONED] as unknown as { data: Float32Array }).data[7] = Number.NaN
    const skipped = learner.stats().skippedUpdates
    const a = await learner.step(world.observe())
    expect(isValidAction(a)).toBe(true)
    expect(learner.stats().skippedUpdates).toBe(skipped + 1)
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
})
