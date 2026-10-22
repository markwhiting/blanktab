// Guards against code or config that only works on one developer's machine
// (e.g. a test writing to /Users/<name>/...), which passes locally and fails in CI.
import { describe, it, expect } from "vitest"
import { readTracked, trackedFiles } from "./tracked"

const MACHINE_PATH = /(^|[^\w.])\/(Users|home)\/[A-Za-z0-9._-]+\//

describe("no machine-specific paths", () => {
  it("no tracked source or config file hard-codes a home-directory path", () => {
    const files = trackedFiles().filter(f => !f.startsWith("docs/") && f !== "package-lock.json" && !f.endsWith(".md"))
    const offenders = files.filter(f => MACHINE_PATH.test(readTracked(f) ?? ""))
    expect(offenders).toEqual([])
  })
})
