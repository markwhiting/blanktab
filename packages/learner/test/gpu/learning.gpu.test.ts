import { describe, expect, it } from "vitest"
import { createCpuBackend, createWebGpuBackend, isSoftwareAdapter } from "@blanktab/tensor"
import type { Backend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, createLearner } from "../../src/learner"
import type { Widths } from "../../src/model"
import { createSyntheticStream } from "../synthetic"

// Does the learner learn on WebGPU as it does on the CPU? Same seed, same visible-motion stream, tiny widths.
// Runs only in real Chromium via `npm run test:gpu`.

const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const STEPS = 300
const SEED = 1
const allowSoftware = (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env.GPU_ALLOW_SOFTWARE === "1"

async function live(backend: Backend) {
  const learner = await createLearner({ backend, seed: SEED, widths: TINY, settings: { ...DEFAULT_SETTINGS } })
  const world = createSyntheticStream()
  const moves: number[] = []
  for (let i = 0; i < STEPS; i++) {
    const a = await learner.step(world.observe())
    moves.push(a.move)
    world.act(a)
  }
  const stats = learner.stats()
  const params = await learner.readParams()
  learner.dispose()
  return { stats, moves, outputs: params.slice(-4) }
}

const maxAbs = (a: Float32Array) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0)

describe("the learner on WebGPU", () => {
  it(`learns like the CPU learner: same seed and stream, ${STEPS} steps, predLoss within 2%; the residual output layers move`, async () => {
    const gpu = await createWebGpuBackend({ powerPreference: "high-performance" })
    if (!gpu) throw new Error("WebGPU backend unavailable")
    if (!allowSoftware && isSoftwareAdapter(gpu.adapterInfo)) throw new Error(`software adapter ${JSON.stringify(gpu.adapterInfo)}`)
    const cpu = await live(createCpuBackend())
    const onGpu = await live(gpu)
    const rel = Math.abs(onGpu.stats.predLoss - cpu.stats.predLoss) / cpu.stats.predLoss
    const sameMoves = onGpu.moves.filter((m, i) => m === cpu.moves[i]).length
    const outMax = onGpu.outputs.map(maxAbs)
    console.log(JSON.stringify({
      adapter: gpu.adapterInfo, cpu: { predLoss: cpu.stats.predLoss, baselineLoss: cpu.stats.baselineLoss, cu: cpu.stats.cu },
      gpu: { predLoss: onGpu.stats.predLoss, baselineLoss: onGpu.stats.baselineLoss, cu: onGpu.stats.cu, skipped: onGpu.stats.skippedUpdates },
      relPredLoss: rel, sameMoves: `${sameMoves}/${STEPS}`, gpuOutputMaxAbs: outMax, cpuOutputMaxAbs: cpu.outputs.map(maxAbs),
    }))
    expect(Number.isFinite(onGpu.stats.predLoss)).toBe(true)
    expect(onGpu.stats.skippedUpdates).toBe(0)
    expect(rel).toBeLessThanOrEqual(0.02)
    for (const m of outMax) expect(m).toBeGreaterThan(0)
  })
})
