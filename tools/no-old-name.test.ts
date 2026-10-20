// Guards the rename (decision 0012): the repo is markwhiting/learner, served at /learner/.
// Internal names (npm scope @blanktab/*, storage keys blanktab:*, IndexedDB "blanktab") deliberately stay.
// docs/ keeps its historical text and is excluded.
import { execFileSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { describe, it, expect } from "vitest"

// Built by concatenation so this file does not match itself.
const OLD = ["markwhiting/" + "blanktab", "/" + "blanktab/", "github.io/" + "blanktab"]

describe("no old public name", () => {
  it("no tracked non-docs file names the old repo or the old page path", () => {
    const files = execFileSync("git", ["ls-files"], { encoding: "utf8" })
      .split("\n")
      .filter(f => f && !f.startsWith("docs/") && f !== "package-lock.json")
    const offenders = files.filter(f => {
      try {
        const text = readFileSync(f, "utf8")
        return OLD.some(s => text.includes(s))
      } catch { return false }
    })
    expect(offenders).toEqual([])
  })
})
