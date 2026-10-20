import { describe, expect, it } from "vitest"
import { ResourceError } from "../../src/index"
import { createWebGpuBackend, isSoftwareAdapter } from "../../src/webgpu"
import type { WebGpuBackend } from "../../src/webgpu"
import { createBenchModel } from "../../bench/learner-model"

// GPU memory hygiene (Review Focus 2 and 4). Runs only in real Chromium via `npm run test:gpu`.

const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" })
const gpuOrNull = await createWebGpuBackend({ powerPreference: "high-performance" })
const allowSoftware = (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env.GPU_ALLOW_SOFTWARE === "1"

const requireGpu = (): WebGpuBackend => {
  if (!gpuOrNull) throw new Error("WebGPU backend unavailable: no navigator.gpu or no adapter")
  if (!allowSoftware && isSoftwareAdapter(gpuOrNull.adapterInfo)) {
    throw new Error(`software WebGPU adapter ${JSON.stringify(gpuOrNull.adapterInfo)}; set GPU_ALLOW_SOFTWARE=1 to allow`)
  }
  return gpuOrNull
}

/** Runs `fn` with `navigator.gpu` shadowed by `value` (an own property over the prototype getter), then restores it. */
async function withNavigatorGpu<T>(value: unknown, fn: () => Promise<T>): Promise<T> {
  Object.defineProperty(navigator, "gpu", { value, configurable: true })
  try {
    return await fn()
  } finally {
    delete (navigator as { gpu?: unknown }).gpu
  }
}

describe("webgpu resource hygiene", () => {
  it("50 learner-sized training steps do not grow the live GPU buffer count", async () => {
    const gpu = requireGpu()
    const before = gpu.stats.liveBuffers
    const model = createBenchModel(gpu)
    await model.step(true)
    const afterFirst = gpu.stats.liveBuffers
    let loss: number | null = null
    for (let i = 2; i <= 50; i++) loss = await model.step(i === 50)
    expect(gpu.stats.liveBuffers).toBe(afterFirst)
    expect(Number.isFinite(loss)).toBe(true)
    model.dispose()
    expect(gpu.stats.liveBuffers).toBe(before)
  })

  it("allocating beyond maxStorageBufferBindingSize throws ResourceError naming the size and the limit", () => {
    const gpu = requireGpu()
    // The backend requests the adapter's full limit, so the adapter limit is the device limit.
    const limit = adapter!.limits.maxStorageBufferBindingSize
    const len = Math.floor(limit / 4) + 1
    const live = gpu.stats.liveBuffers
    expect(() => gpu.zeros(len)).toThrow(ResourceError)
    expect(() => gpu.zeros(len)).toThrow(new RegExp(`${len * 4}.*maxStorageBufferBindingSize ${limit}`))
    expect(gpu.stats.liveBuffers).toBe(live)
  })

  it("createWebGpuBackend() resolves null when navigator.gpu is undefined", async () => {
    await withNavigatorGpu(undefined, async () => {
      expect(navigator.gpu).toBeUndefined()
      await expect(createWebGpuBackend()).resolves.toBeNull()
    })
    expect(navigator.gpu).toBeDefined()
  })

  it("createWebGpuBackend() resolves null when no adapter is available", async () => {
    await withNavigatorGpu({ requestAdapter: () => Promise.resolve(null) }, async () => {
      await expect(createWebGpuBackend()).resolves.toBeNull()
    })
  })
})
