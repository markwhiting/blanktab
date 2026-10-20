import { describe, it, expect } from "vitest"
import { bytesToContent, statusErrors, loadFile } from "../github.ts"
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
