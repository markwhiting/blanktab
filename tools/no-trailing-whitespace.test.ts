// Trailing whitespace is noise in diffs and hides real changes; this keeps tracked text files clean.
import { describe, it, expect } from "vitest"
import { readTracked, trackedFiles } from "./tracked"

describe("no trailing whitespace", () => {
  it("no tracked .ts/.md/.json/.yml file has trailing whitespace", () => {
    const files = trackedFiles().filter(f => /\.(ts|md|json|ya?ml)$/.test(f) && f !== "package-lock.json")
    const offenders: string[] = []
    for (const f of files) {
      const text = readTracked(f)
      if (text === undefined) continue
      text.split("\n").forEach((line, i) => { if (/[ \t]+\r?$/.test(line)) offenders.push(`${f}:${i + 1}`) })
    }
    expect(offenders).toEqual([])
  })
})
