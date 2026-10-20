import { describe, expect, it } from "vitest"
import { createWebGpuBackend } from "../src/index"

describe("createWebGpuBackend outside a WebGPU browser", () => {
  it("resolves null when navigator.gpu is missing (node)", async () => {
    expect((globalThis.navigator as Navigator | undefined)?.gpu).toBeUndefined()
    await expect(createWebGpuBackend()).resolves.toBeNull()
  })
})
