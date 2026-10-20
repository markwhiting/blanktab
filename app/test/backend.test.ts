import { describe, it, expect } from "vitest"
import { CPU_WIDTHS, GPU_WIDTHS } from "@blanktab/learner"
import { createCpuBackend, type WebGpuAdapterInfo, type WebGpuBackend } from "@blanktab/tensor"
import { chooseBackend } from "../src/backend"

const adapter = (over: Partial<WebGpuAdapterInfo> = {}): WebGpuAdapterInfo => ({
  vendor: "apple", architecture: "metal-3", device: "", description: "Apple M4", isFallbackAdapter: false, ...over,
})
/** A stand-in for a WebGPU backend: only its adapter matters to the choice. */
const fakeGpu = (info: WebGpuAdapterInfo) => async () => ({ ...createCpuBackend(), name: "webgpu", adapterInfo: info }) as unknown as WebGpuBackend

describe("chooseBackend", () => {
  it("a hardware adapter gives WebGPU at GPU_WIDTHS", async () => {
    const c = await chooseBackend({ createGpu: fakeGpu(adapter()) })
    expect(c).toMatchObject({ kind: "webgpu", widths: GPU_WIDTHS })
    expect(c.backend.name).toBe("webgpu")
    expect(c.label).toContain("WebGPU")
    expect(c.label).toContain("Apple M4")
  })
  it("a software adapter (fallback flag, or SwiftShader by name) gives the CPU at half width", async () => {
    for (const info of [adapter({ isFallbackAdapter: true }), adapter({ vendor: "google", description: "SwiftShader Device (Subzero)" })]) {
      const c = await chooseBackend({ createGpu: fakeGpu(info) })
      expect(c).toMatchObject({ kind: "cpu", widths: CPU_WIDTHS })
      expect(c.backend.name).toBe("cpu")
      expect(c.label).toMatch(/^CPU/)
      expect(c.label).toMatch(/software/i)
    }
  })
  it("no WebGPU (no adapter, or creating it throws) gives the CPU at half width", async () => {
    for (const createGpu of [async () => null, async () => { throw new Error("requestDevice failed") }]) {
      const c = await chooseBackend({ createGpu })
      expect(c).toMatchObject({ kind: "cpu", widths: CPU_WIDTHS })
      expect(c.label).toMatch(/^CPU/)
    }
  })
  it("forced to the CPU it never asks for WebGPU; widths can be overridden (tests)", async () => {
    let asked = false
    const tiny = { conv1: 1, conv2: 1, enc: 2, hidden: 2 }
    const c = await chooseBackend({ force: "cpu", widths: tiny, createGpu: async () => { asked = true; return null } })
    expect(asked).toBe(false)
    expect(c).toMatchObject({ kind: "cpu", widths: tiny })
    expect((await chooseBackend({ widths: tiny, createGpu: fakeGpu(adapter()) })).widths).toEqual(tiny)
  })
})
