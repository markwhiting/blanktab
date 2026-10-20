import { describe, it, expect } from "vitest"
import { emptyObservation, isValidAction } from "@blanktab/interface"
import type { Action, Observation } from "@blanktab/interface"
import { createCpuBackend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, clampSettings, createLearner, curiosityReward, LR_RANGE } from "../src/learner"
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
  // Several seeds must pass (task brief). Each run is ~1,500 steps at tiny width on the CPU backend.
  for (const seed of [1, 2, 3]) {
    it(`learns to predict the next frame (seed ${seed}): predLoss at 1500 < 0.5 x predLoss at 100`, async () => {
      let at100 = Number.NaN
      const { learner } = await run({ seed, steps: 1500, at: (i, l) => { if (i === 100) at100 = l.stats().predLoss } })
      const at1500 = learner.stats().predLoss
      console.log(`seed ${seed}: predLoss@100=${at100.toExponential(3)} @1500=${at1500.toExponential(3)} ratio=${(at1500 / at100).toFixed(3)} baseline=${learner.stats().baselineLoss.toExponential(3)}`)
      expect(Number.isFinite(at100) && Number.isFinite(at1500)).toBe(true)
      expect(at1500).toBeLessThan(0.5 * at100)
      expect(learner.stats().steps).toBe(1500)
      expect(learner.stats().skippedUpdates).toBe(0)
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
    for (const bad of [{ lr: Number.NaN }, { entropyWeight: -1 }, { curiosityWeight: Number.POSITIVE_INFINITY }, { bufferSize: 0 }, { bufferSize: 2.5 }, { replayPerSleep: -3 }]) {
      expect(() => clampSettings({ ...DEFAULT_SETTINGS, ...bad })).toThrow(RangeError)
    }
  })

  it("the curiosity reward is finite and clamped to [-1, 1]", async () => {
    expect(curiosityReward(0.5, 0.25)).toBeCloseTo(0.25)
    expect(curiosityReward(5, 0)).toBe(1)
    expect(curiosityReward(0, 5)).toBe(-1)
    for (const [b, a] of [[Number.NaN, 0], [0, Number.NaN], [Number.POSITIVE_INFINITY, 0]] as const) expect(curiosityReward(b, a)).toBe(0)
    const { learner, updates } = await run({ seed: 3, steps: 200 })
    expect(updates.length).toBeGreaterThan(0)
    for (const u of updates) {
      expect(Number.isFinite(u.reward)).toBe(true)
      expect(Math.abs(u.reward)).toBeLessThanOrEqual(1)
      expect(u.reward).toBeCloseTo(Math.max(-1, Math.min(1, u.lossBefore - u.lossAfter)), 12)
    }
    // Learning progress is mostly positive: an update on a transition lowers its loss.
    expect(updates.filter((u) => u.reward > 0).length).toBeGreaterThan(updates.length / 2)
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
