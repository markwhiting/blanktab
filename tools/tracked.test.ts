import { describe, it, expect } from "vitest"
import { execFileSync } from "node:child_process"
import { REPO_ROOT, hygieneScans, readTracked, trackedFiles } from "./tracked"

describe("the files the hygiene tests scan", () => {
  it("a contributed log under data/runs/ is not scanned; other tracked files are", () => {
    expect(hygieneScans("data/runs/lrn_0123456789abcdef/0.json")).toBe(false)
    expect(hygieneScans("data/runs/.gitkeep")).toBe(false)
    expect(hygieneScans("data/newborns/lrn_0123456789abcdef/0.json")).toBe(false)
    expect(hygieneScans("data/other.json")).toBe(true)
    expect(hygieneScans("tools/tracked.ts")).toBe(true)
    expect(hygieneScans("")).toBe(false)
    const all = execFileSync("git", ["ls-files"], { encoding: "utf8", cwd: REPO_ROOT }).split("\n").filter(Boolean)
    expect(all.some(f => f.startsWith("data/runs/"))).toBe(true) // there are some to leave out
    const scanned = trackedFiles()
    expect(scanned.some(f => f.startsWith("data/runs/") || f.startsWith("data/newborns/"))).toBe(false)
    expect(scanned).toEqual(all.filter(hygieneScans))
  })

  it("paths are read from the repo root; a missing file reads as undefined, other errors throw", () => {
    expect(readTracked("tools/tracked.ts")).toContain("hygieneScans")
    expect(readTracked("tools/no-such-file.ts")).toBeUndefined()
    expect(() => readTracked("tools")).toThrow() // EISDIR
  })
})
