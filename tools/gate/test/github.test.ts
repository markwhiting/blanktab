import { describe, it, expect } from "vitest"
import { bytesToContent, statusErrors, loadFile, loadPr, mergePr, MAX_PR_FILES } from "../github.ts"
import { validateSessionPr } from "../validate.ts"

describe("statusErrors", () => {
  it("allows added/modified, rejects removed/renamed/copied/changed", () => {
    const e = statusErrors([
      { filename: "a", status: "added" }, { filename: "b", status: "modified" },
      { filename: "c", status: "removed" }, { filename: "d", status: "renamed" }, { filename: "e", status: "copied" }, { filename: "f", status: "changed" },
    ])
    expect(e).toHaveLength(4)
    expect(e.join("\n")).toContain("c:")
  })
})

describe("bytesToContent", () => {
  it("decodes valid UTF-8 and rejects invalid UTF-8 instead of decoding lossily", () => {
    expect(bytesToContent(Buffer.from("é"))).toEqual({ content: "é" })
    expect(bytesToContent(Buffer.from([0xff, 0xfe]))).toHaveProperty("error")
  })
  it("oversize raw bytes are reported as oversize by the validator even when they would decode shorter", () => {
    const raw = Buffer.alloc(1_048_577, 0xff) // invalid UTF-8; a lossy decode would not shrink below the limit reliably
    const c = bytesToContent(raw) as { content: string }
    const r = validateSessionPr([{ path: "data/runs/lrn_0123456789abcdef/0.json", content: c.content }], [])
    expect(r.errors.join("\n")).toContain("exceeds 1 MB")
  })
})

describe("loadFile", () => {
  const big = JSON.stringify({ type: "file", size: 5_000_000 })
  it("never downloads a file whose metadata size exceeds 1 MB", () => {
    const calls: string[][] = []
    const r = loadFile(a => { calls.push(a); if (a.join(" ").includes("raw")) throw new Error("must not download"); return Buffer.from(big) }, "o/r", "data/runs/x/0.json", "sha")
    expect(r).toEqual({ error: "file exceeds 1 MB" })
    expect(calls).toHaveLength(1)
  })
  it("downloads raw when within the limit and rejects non-files", () => {
    const g = (a: string[]) => a.join(" ").includes("raw") ? Buffer.from("{}") : Buffer.from(JSON.stringify({ type: "file", size: 2 }))
    expect(loadFile(g, "o/r", "a b.json", "sha")).toEqual({ content: "{}" })
    expect(loadFile(() => Buffer.from(JSON.stringify({ type: "symlink", size: 1 })), "o/r", "l", "sha")).toHaveProperty("error")
  })
})

describe("loadPr budget", () => {
  const prJson = (changed_files: number) => JSON.stringify({ base: { ref: "main" }, head: { sha: "s", ref: "run/x", repo: { full_name: "o/r" } }, changed_files })
  it("caps a session PR at 16 files", () => expect(MAX_PR_FILES).toBe(16))
  it("more than 16 changed files: a load error from the PR object alone; files are never listed or fetched", () => {
    const calls: string[] = []
    const r = loadPr("o/r", 7, a => { calls.push(a.join(" ")); if (a.join(" ").includes("/files") || a.join(" ").includes("contents")) throw new Error("must not list"); return Buffer.from(prJson(17)) })
    expect(r.loadErrors).toEqual(["PR changes 17 files (limit 16)"])
    expect(r.files).toEqual([])
    expect(calls).toEqual(["api repos/o/r/pulls/7"])
  })
  it("16 changed files are listed and fetched", () => {
    const changed = Array.from({ length: 16 }, (_, i) => ({ filename: `data/runs/lrn_0123456789abcdef/${i}.json`, status: "added" }))
    const r = loadPr("o/r", 7, a => {
      const j = a.join(" ")
      if (j.endsWith("pulls/7")) return Buffer.from(prJson(16))
      if (j.includes("/files")) return Buffer.from(JSON.stringify([changed]))
      return j.includes("raw") ? Buffer.from("{}") : Buffer.from(JSON.stringify({ type: "file", size: 2 }))
    })
    expect(r.loadErrors).toEqual([])
    expect(r.files).toHaveLength(16)
  })
})

describe("mergePr", () => {
  it("marks ready, then squash-merges pinned to the validated head commit", () => {
    const calls: string[][] = []
    mergePr("o/r", 7, "abc123", args => { calls.push(args); return Buffer.from("") })
    expect(calls).toEqual([
      ["pr", "ready", "7", "--repo", "o/r"],
      ["pr", "merge", "7", "--repo", "o/r", "--squash", "--delete-branch", "--match-head-commit", "abc123"],
    ])
  })
  it("a head that moved since validation: the merge error propagates (the caller logs it and retries next run)", () => {
    const ghFn = (args: string[]): Buffer => {
      if (args[1] === "merge") throw new Error("Head branch was modified")
      return Buffer.from("")
    }
    expect(() => mergePr("o/r", 7, "abc123", ghFn)).toThrow("Head branch was modified")
  })
})
