import { describe, it } from "vitest"
import { createCpuBackend } from "../src/index"
import type { Backend } from "../src/index"
import { createWebGpuBackend, isSoftwareAdapter } from "../src/webgpu"
import type { WebGpuBackend } from "../src/webgpu"
import { DEFAULT_WIDTHS, createBenchModel } from "./learner-model"
import type { BenchWidths } from "./learner-model"

// Spike 2: learner-sized training throughput. Run with `npm run bench:gpu` (real Chromium + WebGPU).
// Each backend: 20 warm-up steps, then 200 timed steps with no readback except the loss after the
// last step (which also waits for the GPU to finish all queued work). One JSON line per backend.
// BENCH_WIDTHS='{"hidden":128}' overrides default widths (spec §5.5: shrink widths before resolution).

const WARMUP = 20
const TIMED = 200
const env = (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env
const allowSoftware = env.GPU_ALLOW_SOFTWARE === "1"
const widths: BenchWidths = { ...DEFAULT_WIDTHS, ...(JSON.parse(env.BENCH_WIDTHS || "{}") as Partial<BenchWidths>) }

async function hardwareGpu(): Promise<WebGpuBackend> {
  const gpu = await createWebGpuBackend({ powerPreference: "high-performance" })
  if (!gpu) throw new Error("WebGPU backend unavailable: no navigator.gpu or no adapter")
  if (!allowSoftware && isSoftwareAdapter(gpu.adapterInfo)) {
    throw new Error(`software WebGPU adapter ${JSON.stringify(gpu.adapterInfo)}; set GPU_ALLOW_SOFTWARE=1 to allow`)
  }
  return gpu
}

async function measure(backend: Backend, adapter: WebGpuBackend["adapterInfo"] | null): Promise<void> {
  const model = createBenchModel(backend, widths)
  try {
    for (let i = 1; i <= WARMUP; i++) await model.step(i === WARMUP) // last warm-up step syncs
    const f0 = backend.stats.flops
    const live0 = backend.stats.liveBuffers
    const t0 = performance.now()
    let loss: number | null = null
    for (let i = 1; i <= TIMED; i++) loss = await model.step(i === TIMED)
    const seconds = (performance.now() - t0) / 1000
    const flopsPerStep = (backend.stats.flops - f0) / TIMED
    const record = {
      backend: backend.name,
      stepsPerSec: Math.round((TIMED / seconds) * 10) / 10,
      paramCount: model.paramCount,
      flopsPerStep,
      cuPerStep: flopsPerStep / 1e9,
      msPerStep: Math.round((seconds * 1000 * 100) / TIMED) / 100,
      steps: TIMED,
      warmup: WARMUP,
      finalLoss: loss,
      liveBufferGrowth: backend.stats.liveBuffers - live0,
      widths,
      adapter,
      userAgent: navigator.userAgent,
    }
    console.log(`BENCH ${JSON.stringify(record)}`)
    if (!Number.isFinite(loss)) throw new Error(`non-finite loss ${loss}`)
  } finally {
    model.dispose()
  }
}

// Vitest 5's `vitest bench` runs bench files as tests; the timing is measure()'s own 200-step window.
describe("learner-sized training step (spike 2)", () => {
  it("webgpu", async () => {
    const gpu = await hardwareGpu()
    await measure(gpu, gpu.adapterInfo)
  }, 300_000)

  it("cpu", async () => {
    await measure(createCpuBackend(), null)
  }, 600_000)
})
