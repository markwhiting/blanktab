import { defineConfig } from "vitest/config"
import { playwright } from "@vitest/browser-playwright"

// GPU suites only: real Chromium with WebGPU. The node suite (vitest.config.ts) excludes these files.
// --use-angle=metal: without it headless Chromium on macOS falls back to the SwiftShader (CPU) adapter.
// GPU_HEADED=1 runs headed if a machine's headless Chromium exposes no WebGPU adapter.
export default defineConfig({
  test: {
    include: ["**/*.gpu.test.ts"],
    benchmark: { include: ["**/*.gpu.bench.ts"] },
    testTimeout: 60_000,
    // GPU_ALLOW_SOFTWARE=1 lets the suite pass on a software adapter (e.g. SwiftShader in CI); by default it must be hardware.
    // BENCH_WIDTHS (JSON, e.g. '{"hidden":128}') overrides the spike-2 benchmark's default widths.
    env: { GPU_ALLOW_SOFTWARE: process.env.GPU_ALLOW_SOFTWARE ?? "", BENCH_WIDTHS: process.env.BENCH_WIDTHS ?? "" },
    // Show console output (adapter info, per-primitive max errors) for passing tests too.
    silent: false,
    // Away from vite's default ports and the e2e preview (4173), which other streams may hold.
    api: { port: 63417, strictPort: false },
    browser: {
      enabled: true,
      headless: process.env.GPU_HEADED !== "1",
      provider: playwright({
        launchOptions: { args: ["--enable-unsafe-webgpu", "--use-angle=metal"] },
      }),
      instances: [{ browser: "chromium" }],
    },
  },
})
