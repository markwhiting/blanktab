import { describe, it, expect, beforeEach, afterEach } from "vitest"
import { execFileSync } from "node:child_process"
import { mkdirSync, mkdtempSync, writeFileSync, rmSync } from "node:fs"
import { createHash } from "node:crypto"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { LEARNER_STATE_VERSION } from "@blanktab/learner/version"
import { buildManifest, codeShaOf, computeKnowledgeSha, resolveCodeSha } from "../manifest.ts"

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

describe("codeShaOf: the last commit that changed code (anything but data/runs, docs and knowledge)", () => {
  const git = (...args: string[]) => execFileSync("git", args, { cwd: dir, encoding: "utf8", env: { ...process.env, GIT_AUTHOR_NAME: "t", GIT_AUTHOR_EMAIL: "t@t", GIT_COMMITTER_NAME: "t", GIT_COMMITTER_EMAIL: "t@t" } }).trim()
  const commit = (path: string, text: string) => {
    mkdirSync(join(dir, path, ".."), { recursive: true })
    writeFileSync(join(dir, path), text)
    git("add", "-A")
    git("commit", "-q", "-m", path)
    return git("rev-parse", "HEAD")
  }
  beforeEach(() => { git("init", "-q") })

  it("a commit touching only data/runs, or only docs, leaves it unchanged; one touching app/ changes it", () => {
    const code = commit("app/src/main.ts", "1")
    expect(codeShaOf(dir)).toBe(code)
    commit("data/runs/lrn_0123456789abcdef/0.json", "{}")
    expect(codeShaOf(dir)).toBe(code)
    commit("docs/decisions/0099-x.md", "x")
    expect(codeShaOf(dir)).toBe(code)
    const next = commit("app/src/main.ts", "2")
    expect(next).not.toBe(code)
    expect(codeShaOf(dir)).toBe(next)
    expect(next).toMatch(/^[0-9a-f]{40}$/)
  })
  it("a knowledge/-only commit leaves it unchanged (knowledge reaches learners by refresh, tracked by knowledgeSha)", () => {
    const code = commit("app/a.ts", "1")
    commit("knowledge/settings-space.json", "{}")
    expect(codeShaOf(dir)).toBe(code)
  })
  it("is dev outside a git repository or with no commits", () => {
    expect(codeShaOf(dir)).toBe("dev") // git init, nothing committed
    const bare = mkdtempSync(join(tmpdir(), "nogit-"))
    try { expect(codeShaOf(bare)).toBe("dev") } finally { rmSync(bare, { recursive: true, force: true }) }
  })
})

describe("resolveCodeSha", () => {
  it("uses a 40-hex CODE_SHA (the deploy computes it with codeShaOf), otherwise codeShaOf the repo", () => {
    expect(resolveCodeSha("b".repeat(40), dir)).toBe("b".repeat(40))
    expect(resolveCodeSha(undefined, dir)).toBe("dev") // not a repo
    expect(resolveCodeSha("nonsense", dir)).toBe("dev")
  })
})

describe("buildManifest", () => {
  it("carries knowledgeSha, codeSha and the learnerStateVersion this code reads and writes", () => {
    writeFileSync(join(dir, "settings-space.json"), "S")
    expect(buildManifest(dir, "b".repeat(40))).toEqual({ knowledgeSha: computeKnowledgeSha(dir), codeSha: "b".repeat(40), learnerStateVersion: LEARNER_STATE_VERSION })
    expect(Number.isInteger(LEARNER_STATE_VERSION) && LEARNER_STATE_VERSION >= 1).toBe(true)
  })
})
