// Review Focus 3: WebGPU missing or lost. A rejected GPU read restores the life from its checkpoint once (on a fresh
// GPU backend); a second one moves the life to the CPU at the same widths. Real runSession and a real tiny learner.
import { describe, it, expect } from "vitest"
import { IDBFactory } from "fake-indexeddb"
import { emptyObservation, PERIPHERY, decodeMove, type Action, type Observation, type WorldView } from "@blanktab/interface"
import { createCpuBackend, ResourceError, type Backend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, createLearner, type Widths } from "@blanktab/learner"
import { openQueue, runSession, type LogFileV2, type SessionConfig } from "@blanktab/harness"
import { liveOnBestBackend, type LifeConfig, type RunLife } from "../src/life"
import type { Chosen } from "../src/backend"

const ID = "lrn_00000000deadbeef"
const TINY: Widths = { conv1: 2, conv2: 2, enc: 4, hidden: 4 }
const LIFETIME_CU = 0.4

function stubWorld(): WorldView {
  let x = 40, y = 25
  return {
    observe(): Observation {
      const o = emptyObservation()
      o.periphery.fill(30)
      o.periphery.fill(220, (y * PERIPHERY.width + x) * 3, (y * PERIPHERY.width + x) * 3 + 3)
      o.cursor = { x: x / (PERIPHERY.width - 1), y: y / (PERIPHERY.height - 1) }
      return o
    },
    act(a: Action) {
      const { dx, dy } = decodeMove(a.move)
      x = Math.min(PERIPHERY.width - 1, Math.max(0, x + Math.sign(dx)))
      y = Math.min(PERIPHERY.height - 1, Math.max(0, y + Math.sign(dy)))
    },
  }
}

/** A CPU backend posing as a GPU: once `fail` is set, every read rejects the way a lost WebGPU device does. */
function flakyGpu(): Backend & { fail: boolean } {
  // Patched in place: tensors reach their backend through their storages, which point at this very object.
  const b = Object.assign(createCpuBackend(), { fail: false })
  const read = b.read.bind(b)
  b.read = s => (b.fail ? Promise.reject(new ResourceError("WebGPU read failed: device lost")) : read(s))
  return b
}

const cfg: LifeConfig = {
  learnerId: ID, codeSha: "dev", knowledgeSha: "f".repeat(64), sessionsSeenAtBirth: 0,
  hostSig: { cores: 4, deviceMemoryGb: null, tabsOnMachine: 1 }, seed: 5, learnerSettings: { ...DEFAULT_SETTINGS, replayPerSleep: 4 },
  lifetimeCU: LIFETIME_CU, sleepInterval: 4, maxStepsPerSec: 1e6, checkpointEveryMs: 0,
}

/** `failAt[i]`: the step at which the i-th GPU backend handed out starts failing (undefined: never). */
function setup(failAt: (number | undefined)[]) {
  const idb = new IDBFactory()
  const gpus: (Backend & { fail: boolean })[] = []
  const statuses: string[] = []
  const backends: string[] = []
  const choose = async (): Promise<Chosen> => {
    const backend = flakyGpu()
    gpus.push(backend)
    return { kind: "webgpu", backend, widths: TINY, label: "WebGPU (fake)" }
  }
  let cu = 0
  const run: RunLife = (c: SessionConfig, chosen: Chosen) => {
    const i = gpus.indexOf(chosen.backend as never)
    const world = stubWorld()
    return runSession(c, {
      world, idb, now: () => new Date(),
      createLearner: init => createLearner({ backend: chosen.backend, ...init }),
      onTick: ({ steps, cu: now }) => {
        cu = now
        if (i >= 0 && failAt[i] !== undefined && steps >= failAt[i]!) gpus[i]!.fail = true
      },
    })
  }
  const life = () => liveOnBestBackend(cfg, {
    idb, choose, cpu: () => ({ kind: "cpu", backend: createCpuBackend(), widths: TINY, label: "CPU" }), run, lastCU: () => cu,
    onBackend: c => backends.push(c.kind), onStatus: s => statuses.push(s),
  })
  const logs = async () => (await (await openQueue(idb)).pending()) as LogFileV2[]
  return { life, logs, gpus, statuses, backends }
}

describe("liveOnBestBackend", { timeout: 60_000 }, () => {
  it("a rejected GPU read restores the life from its checkpoint once, on a fresh GPU backend; the life completes", async () => {
    const t = setup([10, undefined])
    const r = await t.life()
    expect(r.cu).toBeGreaterThanOrEqual(LIFETIME_CU)
    expect(t.gpus).toHaveLength(2)
    expect(t.backends).toEqual(["webgpu", "webgpu"])
    expect(t.statuses.some(s => /GPU read failed/i.test(s) && /checkpoint/i.test(s))).toBe(true)
    const final = (await t.logs()).at(-1)!
    expect(final).toMatchObject({ final: true, widths: TINY, hostSig: { webgpu: true }, restores: 1 })
    expect(final.lostCU).toBeGreaterThan(0) // the work since the checkpoint is on the record
  })

  it("a second rejected GPU read moves the life to the CPU at the same widths; hostSig.webgpu says so", async () => {
    const t = setup([10, 6])
    const r = await t.life()
    expect(r.cu).toBeGreaterThanOrEqual(LIFETIME_CU)
    expect(t.gpus).toHaveLength(2) // restored on the GPU once, then no more GPU
    expect(t.backends).toEqual(["webgpu", "webgpu", "cpu"])
    expect(t.statuses.some(s => /CPU/.test(s))).toBe(true)
    const files = await t.logs()
    const final = files.at(-1)!
    expect(final).toMatchObject({ final: true, widths: TINY, hostSig: { webgpu: false }, restores: 2 })
    // Nothing went backwards: steps and CU are monotonic across the files.
    for (let i = 1; i < files.length; i++) {
      expect(files[i]!.cu).toBeGreaterThanOrEqual(files[i - 1]!.cu)
      expect(files[i]!.stepsTotal).toBeGreaterThanOrEqual(files[i - 1]!.stepsTotal)
    }
  })

  it("errors that are not a GPU failure end the life as before (no retry)", async () => {
    const idb = new IDBFactory()
    let runs = 0
    const choose = async (): Promise<Chosen> => ({ kind: "webgpu", backend: createCpuBackend(), widths: TINY, label: "WebGPU (fake)" })
    const run: RunLife = async () => { runs++; throw new RangeError("bad checkpoint") }
    await expect(liveOnBestBackend(cfg, { idb, choose, cpu: () => { throw new Error("no") }, run, lastCU: () => 0 })).rejects.toThrow("bad checkpoint")
    expect(runs).toBe(1)
  })

  it("a ResourceError on the CPU is fatal", async () => {
    const idb = new IDBFactory()
    let runs = 0
    const choose = async (): Promise<Chosen> => ({ kind: "cpu", backend: createCpuBackend(), widths: TINY, label: "CPU" })
    const run: RunLife = async () => { runs++; throw new ResourceError("out of memory") }
    await expect(liveOnBestBackend(cfg, { idb, choose, cpu: () => { throw new Error("no") }, run, lastCU: () => 0 })).rejects.toThrow("out of memory")
    expect(runs).toBe(1)
  })
})
