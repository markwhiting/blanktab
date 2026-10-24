// Final review (Plan 05a): detmath in the page computes what Node computes. The e2e build's probe page computes the
// golden fixture's values in Chromium, on the main thread and in a module worker (where births are drawn), and both
// must equal packages/detmath/test/fixtures/golden.json (written by Node) bit for bit.
import { test, expect } from "@playwright/test"
import { readFileSync } from "node:fs"

const GOLDEN = JSON.parse(readFileSync(new URL("../../packages/detmath/test/fixtures/golden.json", import.meta.url), "utf8")) as unknown

test("detmath in Chromium (main thread and worker) equals the Node golden file bit for bit", async ({ page }) => {
  await page.goto("/learner/detmath-probe.html")
  await page.waitForFunction(() => (window as { __detmath?: unknown }).__detmath !== undefined)
  const got = await page.evaluate(() => (window as { __detmath?: unknown }).__detmath) as { main: unknown; worker: unknown } | { error: string }
  expect("error" in got ? got.error : null).toBeNull()
  const { main, worker } = got as { main: unknown; worker: unknown }
  expect(main).toEqual(GOLDEN)
  expect(worker).toEqual(GOLDEN)
})
