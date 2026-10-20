import { defineConfig, devices } from "@playwright/test"

const PORT = 4173

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 90_000,
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? "github" : "list",
  use: { baseURL: `http://localhost:${PORT}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  // The e2e build honours ?testLifetime; the production (deploy) build does not.
  webServer: {
    command: `npm run build:e2e -w app && npm run preview:e2e -w app -- --port ${PORT} --strictPort`,
    url: `http://localhost:${PORT}/blanktab/`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
