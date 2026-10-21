import { describe, it, expect } from "vitest"
import { execFileSync } from "node:child_process"
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { boundAuthor, bytesToContent, statusErrors, listOpenPrs, loadFile, loadPr, mergePr, MAX_PR_FILES } from "../github.ts"
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

describe("listOpenPrs", () => {
  it("asks only for the fields the gate uses", () => {
    const calls: string[][] = []
    const listed = listOpenPrs("o/r", a => { calls.push(a); return Buffer.from(JSON.stringify([{ number: 3, title: "run: x", headRefName: "run/x", baseRefName: "main" }])) })
    expect(calls).toEqual([["pr", "list", "--repo", "o/r", "--state", "open", "--limit", "200", "--json", "number,title,headRefName,baseRefName"]])
    expect(listed).toEqual([{ number: 3, title: "run: x", headRefName: "run/x", baseRefName: "main" }])
  })
})

describe("loadPr head commit", () => {
  const files = (a: string[]) => {
    const j = a.join(" ")
    if (j.includes("/files")) return Buffer.from(JSON.stringify([[{ filename: "data/runs/lrn_0123456789abcdef/0.json", status: "added" }]]))
    return j.includes("raw") ? Buffer.from("{}") : Buffer.from(JSON.stringify({ type: "file", size: 2 }))
  }
  it("a PR object without a head sha loads with headSha undefined (the gate then never merges it)", () => {
    for (const head of [{ ref: "run/x", repo: { full_name: "o/r" } }, { sha: 42, ref: "run/x", repo: { full_name: "o/r" } }, { sha: "", ref: "run/x", repo: { full_name: "o/r" } }]) {
      const r = loadPr("o/r", 7, a => (a.join(" ").endsWith("pulls/7") ? Buffer.from(JSON.stringify({ base: { ref: "main" }, head, changed_files: 1 })) : files(a)))
      expect(r.headSha).toBeUndefined()
    }
  })
})

describe("mergePr", () => {
  it("refuses a missing head commit rather than merge unpinned", () => {
    const calls: string[][] = []
    for (const bad of [undefined, ""]) expect(() => mergePr("o/r", 7, bad, a => { calls.push(a); return Buffer.from("") })).toThrow("no validated head commit")
    expect(calls).toEqual([])
  })
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

describe("loadPr author", () => {
  const prWith = (user: unknown) => (a: string[]) => {
    const j = a.join(" ")
    if (j.endsWith("pulls/7")) return Buffer.from(JSON.stringify({ base: { ref: "main" }, head: { sha: "s", ref: "run/x", repo: { full_name: "o/r" } }, changed_files: 0, user }))
    return Buffer.from(JSON.stringify([[]]))
  }
  it("is the PR's user.login; anything else is undefined", () => {
    expect(loadPr("o/r", 7, prWith({ login: "ann" })).author).toBe("ann")
    for (const u of [undefined, null, {}, { login: 5 }, { login: "" }]) expect(loadPr("o/r", 7, prWith(u)).author).toBeUndefined()
  })
})

const LRN = "lrn_0123456789abcdef"
describe("boundAuthor (fake gh)", () => {
  const git = (out: string) => () => out
  it("the first commit touching the learner's dir, its merged PR, that PR's author", () => {
    const calls: string[] = []
    const gh = (a: string[]) => {
      calls.push(a.join(" "))
      return Buffer.from(JSON.stringify([{ number: 3, merged_at: null, user: { login: "eve" } }, { number: 4, merged_at: "2026-10-21T00:00:00Z", user: { login: "ann" } }]))
    }
    expect(boundAuthor("o/r", LRN, git(`${"a".repeat(40)}\n${"b".repeat(40)}\n`), gh)).toBe("ann")
    expect(calls).toEqual([`api repos/o/r/commits/${"a".repeat(40)}/pulls`])
  })
  it("nothing of the learner on main: null, and GitHub is not asked", () => {
    expect(boundAuthor("o/r", LRN, git(""), () => { throw new Error("must not ask") })).toBeNull()
  })
  it("throws when the lookup fails: no merged PR, no login, a gh error, a malformed answer", () => {
    for (const body of [[], [{ merged_at: null, user: { login: "x" } }], [{ merged_at: "t", user: {} }], { not: "a list" }]) {
      expect(() => boundAuthor("o/r", LRN, git(`${"a".repeat(40)}\n`), () => Buffer.from(JSON.stringify(body)))).toThrow()
    }
    expect(() => boundAuthor("o/r", LRN, git(`${"a".repeat(40)}\n`), () => { throw new Error("HTTP 502") })).toThrow()
  })
  it("git output that is not a commit sha is a failed lookup", () => {
    expect(() => boundAuthor("o/r", LRN, git("fatal: bad\n"), () => Buffer.from("[]"))).toThrow("no commit")
  })
  it("refuses a malformed learnerId (never put into a pathspec or URL)", () => {
    expect(() => boundAuthor("o/r", "lrn_../../x", git(""), () => Buffer.from("[]"))).toThrow()
  })
})

describe("boundAuthor (git, in a temporary repository)", { timeout: 60_000 }, () => {
  let dir: string
  const run = (...args: string[]) => execFileSync("git", args, { cwd: dir, encoding: "utf8", env: { ...process.env, GIT_AUTHOR_NAME: "t", GIT_AUTHOR_EMAIL: "t@t", GIT_COMMITTER_NAME: "t", GIT_COMMITTER_EMAIL: "t@t" } }).trim()
  const commit = (path: string) => {
    mkdirSync(join(dir, path, ".."), { recursive: true })
    writeFileSync(join(dir, path), String(Math.random()))
    run("add", "-A"); run("commit", "-q", "-m", path)
    return run("rev-parse", "HEAD")
  }
  it("the oldest commit to data/runs/<id>/ wins; another learner's and look-alike paths are ignored", () => {
    dir = mkdtempSync(join(tmpdir(), "bound-"))
    try {
      run("init", "-q")
      commit(`data/runs/lrn_fedcba9876543210/0.json`) // another learner
      commit(`data/runs/${LRN}.bak/0.json`) // look-alike: not the learner's dir
      commit(`other/data/runs/${LRN}/0.json`) // look-alike: not at the top
      const first = commit(`data/runs/${LRN}/0.json`)
      commit(`data/runs/${LRN}/1.json`)
      const asked: string[] = []
      const gh = (a: string[]) => { asked.push(a[1]!); return Buffer.from(JSON.stringify([{ merged_at: "t", user: { login: "ann" } }])) }
      const git = (args: string[]) => execFileSync("git", args, { cwd: dir, encoding: "utf8" })
      expect(boundAuthor("o/r", LRN, git, gh)).toBe("ann")
      expect(asked).toEqual([`repos/o/r/commits/${first}/pulls`])
      expect(boundAuthor("o/r", "lrn_1111111111111111", git, gh)).toBeNull()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
