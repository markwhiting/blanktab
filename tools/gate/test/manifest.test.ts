import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { createHash } from "node:crypto"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { computeKnowledgeSha, resolveCodeSha } from "../manifest.ts"

let dir: string
beforeEach(() => { dir = mkdtempSync(join(tmpdir(), "manifest-")) })
afterEach(() => { rmSync(dir, { recursive: true, force: true }) })
const sha = (s: string) => createHash("sha256").update(s).digest("hex")

describe("computeKnowledgeSha", () => {
  it("hashes name + newline + content + newline for each present prior, sorted by name", () => {
    writeFileSync(join(dir, "settings-space.json"), "S")
    writeFileSync(join(dir, "consolidation-prior.json"), "C")
    expect(computeKnowledgeSha(dir)).toBe(sha("consolidation-prior.json\nC\nsettings-space.json\nS\n"))
  })
  it("matches the 64-lowercase-hex rule and ignores other files (summary.json, manifest.json)", () => {
    writeFileSync(join(dir, "settings-space.json"), "S")
    const a = computeKnowledgeSha(dir)
    writeFileSync(join(dir, "summary.json"), "x"); writeFileSync(join(dir, "manifest.json"), "y")
    expect(computeKnowledgeSha(dir)).toBe(a)
    expect(a).toMatch(/^[0-9a-f]{64}$/)
  })
  it("changes when a prior's content changes or a prior appears", () => {
    writeFileSync(join(dir, "settings-space.json"), "S")
    const a = computeKnowledgeSha(dir)
    writeFileSync(join(dir, "caregiver-prior.json"), "G")
    const b = computeKnowledgeSha(dir)
    expect(b).not.toBe(a)
    writeFileSync(join(dir, "settings-space.json"), "S2")
    expect(computeKnowledgeSha(dir)).not.toBe(b)
  })
})

describe("resolveCodeSha", () => {
  it("uses a 40-hex GITHUB_SHA, otherwise dev", () => {
    expect(resolveCodeSha("b".repeat(40))).toBe("b".repeat(40))
    expect(resolveCodeSha(undefined)).toBe("dev")
    expect(resolveCodeSha("nonsense")).toBe("dev")
  })
})
