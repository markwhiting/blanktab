// Guards against code or config that only works on one developer's machine
// (e.g. a test writing to /Users/<name>/...), which passes locally and fails in CI.
import { execFileSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { describe, it, expect } from "vitest"

const MACHINE_PATH = /(^|[^\w.])\/(Users|home)\/[A-Za-z0-9._-]+\//

describe("no machine-specific paths", () => {
  it("no tracked source or config file hard-codes a home-directory path", () => {
    const files = execFileSync("git", ["ls-files"], { encoding: "utf8" })
      .split("\n")
      .filter(f => f && !f.startsWith("docs/") && f !== "package-lock.json" && !f.endsWith(".md"))
    const offenders = files.filter(f => {
      try { return MACHINE_PATH.test(readFileSync(f, "utf8")) } catch { return false }
    })
    expect(offenders).toEqual([])
  })
})
