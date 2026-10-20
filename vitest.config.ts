import { configDefaults, defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    include: ["tools/**/*.test.ts", "packages/*/src/**/*.test.ts", "packages/*/test/**/*.test.ts", "app/test/**/*.test.ts"],
    // GPU suites need a real browser with WebGPU; they run only via `npm run test:gpu` (vitest.gpu.config.ts).
    exclude: [...configDefaults.exclude, "**/*.gpu.test.ts", "**/*.gpu.bench.ts"],
  },
})
