import { defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    include: ["tools/**/*.test.ts", "packages/*/src/**/*.test.ts", "packages/*/test/**/*.test.ts", "app/test/**/*.test.ts"],
  },
})
