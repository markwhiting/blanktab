// Guards the rename (decision 0012): the repo is markwhiting/learner, served at /learner/.
// Internal names (npm scope @blanktab/*, storage keys blanktab:*, IndexedDB "blanktab") deliberately stay.
// docs/ keeps its historical text and is excluded (and data/runs/, contributed logs: tools/tracked.ts).
import { describe, it, expect } from "vitest"
import { readTracked, trackedFiles } from "./tracked"

// Built by concatenation so this file does not match itself.
const OLD = ["markwhiting/" + "blanktab", "/" + "blanktab/", "github.io/" + "blanktab"]

describe("no old public name", () => {
  it("no tracked non-docs file names the old repo or the old page path", () => {
    const files = trackedFiles().filter(f => !f.startsWith("docs/") && f !== "package-lock.json")
    const offenders = files.filter(f => {
      const text = readTracked(f) ?? ""
      return OLD.some(s => text.includes(s))
    })
    expect(offenders).toEqual([])
  })
})
