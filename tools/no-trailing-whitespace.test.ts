// Trailing whitespace is noise in diffs and hides real changes; this keeps tracked text files clean.
import { execFileSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { describe, it, expect } from "vitest"

describe("no trailing whitespace", () => {
  it("no tracked .ts/.md/.json/.yml file has trailing whitespace", () => {
    const files = execFileSync("git", ["ls-files"], { encoding: "utf8" })
      .split("\n")
      .filter(f => /\.(ts|md|json|ya?ml)$/.test(f) && f !== "package-lock.json")
    const offenders: string[] = []
    for (const f of files) {
      let text: string
      try { text = readFileSync(f, "utf8") } catch { continue }
      text.split("\n").forEach((line, i) => { if (/[ \t]+\r?$/.test(line)) offenders.push(`${f}:${i + 1}`) })
    }
    expect(offenders).toEqual([])
  })
})
