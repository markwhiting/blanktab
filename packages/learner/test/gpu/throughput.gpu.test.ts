import { describe, expect, it } from "vitest"
import { createWebGpuBackend, isSoftwareAdapter } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, GPU_WIDTHS, createLearner } from "../../src/index"
import { createSyntheticStream } from "../synthetic"

// The real learner's step rate on WebGPU at GPU_WIDTHS, uncapped (Plan 04c Task 2): a measurement, not a bar.
// Runs only in real Chromium via `npm run test:gpu -- throughput`.

const WARMUP = 20
const TIMED = 200
const allowSoftware = (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env.GPU_ALLOW_SOFTWARE === "1"

describe("learner throughput", () => {
  it("real learner throughput on WebGPU at GPU_WIDTHS, uncapped", { timeout: 300_000 }, async () => {
    const gpu = await createWebGpuBackend({ powerPreference: "high-performance" })
    if (!gpu) throw new Error("WebGPU backend unavailable")
    if (!allowSoftware && isSoftwareAdapter(gpu.adapterInfo)) throw new Error(`software adapter ${JSON.stringify(gpu.adapterInfo)}`)
    const learner = await createLearner({ backend: gpu, seed: 1, widths: GPU_WIDTHS, settings: { ...DEFAULT_SETTINGS } })
    const world = createSyntheticStream()
    for (let i = 0; i < WARMUP; i++) world.act(await learner.step(world.observe()))
    const t0 = performance.now()
    for (let i = 0; i < TIMED; i++) world.act(await learner.step(world.observe()))
    const seconds = (performance.now() - t0) / 1000
    learner.dispose()
    const stepsPerSec = TIMED / seconds
    const msPerStep = (seconds * 1000) / TIMED
    console.info(JSON.stringify({ backend: "webgpu", stepsPerSec, msPerStep, widths: GPU_WIDTHS, adapter: gpu.adapterInfo }))
    expect(stepsPerSec).toBeGreaterThan(0)
  })
})
