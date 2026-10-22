import { afterEach, beforeEach, describe, it, expect } from "vitest"
import { execFileSync } from "node:child_process"
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { boundAuthor, bytesToContent, FETCH_MAX_BYTES, FetchBoundError, GIT_TIMEOUT_MS, gitRunner, isShallow, statusErrors, listOpenPrs, loadPrViaGit, mergePr, MAX_OWNER_WALK, MAX_PR_COMMITS, MAX_PR_FILES } from "../github.ts"
import { FIXTURE_ENV, makeFixture, prObject, type Fixture } from "./git-fixture.ts"
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

describe("listOpenPrs", () => {
  it("asks only for the fields the gate uses", () => {
    const calls: string[][] = []
    const listed = listOpenPrs("o/r", a => { calls.push(a); return Buffer.from(JSON.stringify([{ number: 3, title: "run: x", headRefName: "run/x", baseRefName: "main" }])) })
    expect(calls).toEqual([["pr", "list", "--repo", "o/r", "--state", "open", "--limit", "200", "--json", "number,title,headRefName,baseRefName,labels"]])
    expect(listed).toEqual([{ number: 3, title: "run: x", headRefName: "run/x", baseRefName: "main" }])
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
  const sha = (c: string) => c.repeat(40)
  it("boundAuthor skips direct-push commits and returns the first merged PR's author", () => {
    const calls: string[] = []
    const answers: Record<string, unknown> = { [sha("a")]: [], [sha("b")]: [{ merged_at: null, user: { login: "eve" } }], [sha("c")]: [{ merged_at: "t", user: { login: "bob" } }], [sha("d")]: [{ merged_at: "t", user: { login: "ann" } }] }
    const gh = (a: string[]) => { calls.push(a[1]!); return Buffer.from(JSON.stringify(answers[a[1]!.split("/")[4]!])) }
    expect(boundAuthor("o/r", LRN, git(["a", "b", "c", "d"].map(sha).join("\n") + "\n"), gh)).toBe("bob")
    expect(calls).toEqual(["a", "b", "c"].map(c => `repos/o/r/commits/${sha(c)}/pulls`))
  })
  it("boundAuthor: only direct pushes → null (unbound)", () => {
    for (const body of [[], [{ merged_at: null, user: { login: "x" } }], [{ merged_at: "", user: { login: "x" } }]]) {
      expect(boundAuthor("o/r", LRN, git(`${sha("a")}\n${sha("b")}\n`), () => Buffer.from(JSON.stringify(body)))).toBeNull()
    }
  })
  it("boundAuthor: a gh error or a malformed answer still throws", () => {
    for (const body of [[{ merged_at: "t", user: {} }], [{ merged_at: "t", user: null }], { not: "a list" }]) {
      expect(() => boundAuthor("o/r", LRN, git(`${sha("a")}\n`), () => Buffer.from(JSON.stringify(body)))).toThrow()
    }
    expect(() => boundAuthor("o/r", LRN, git(`${sha("a")}\n`), () => Buffer.from("not json"))).toThrow()
    expect(() => boundAuthor("o/r", LRN, git(`${sha("a")}\n`), () => { throw new Error("HTTP 502") })).toThrow("HTTP 502")
    // A failure after a direct push still throws: a skipped commit never turns an error into "unbound".
    let n = 0
    expect(() => boundAuthor("o/r", LRN, git(`${sha("a")}\n${sha("b")}\n`), () => { if (n++ === 0) return Buffer.from("[]"); throw new Error("HTTP 502") })).toThrow("HTTP 502")
  })
  it("boundAuthor walks at most MAX_OWNER_WALK commits", () => {
    expect(MAX_OWNER_WALK).toBe(20)
    const shas = Array.from({ length: MAX_OWNER_WALK + 5 }, (_, i) => i.toString(16).padStart(40, "0"))
    const asked: string[] = []
    const gh = (a: string[]) => { asked.push(a[1]!); return Buffer.from(JSON.stringify(asked.length > MAX_OWNER_WALK ? [{ merged_at: "t", user: { login: "late" } }] : [])) }
    expect(boundAuthor("o/r", LRN, git(shas.join("\n") + "\n"), gh)).toBeNull()
    expect(asked).toEqual(shas.slice(0, MAX_OWNER_WALK).map(s => `repos/o/r/commits/${s}/pulls`)) // oldest first
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

describe("loadPrViaGit (a local origin over file://)", { timeout: 60_000 }, () => {
  let fx: Fixture
  beforeEach(() => { fx = makeFixture() })
  afterEach(() => fx.cleanup())
  const D = `data/runs/${LRN}`
  /** A fake gh that answers pulls/<n> only, and records every call. */
  const ghFor = (n: number, head: string, over: Record<string, unknown> = {}) => {
    const calls: string[] = []
    const fn = (a: string[]) => {
      calls.push(a.join(" "))
      if (a.join(" ") === `api repos/o/r/pulls/${n}`) return prObject(n, head, over)
      throw new Error(`unexpected gh call: ${a.join(" ")}`)
    }
    return { fn, calls }
  }
  const missing = (oid: string) => { try { fx.git(["--no-lazy-fetch", "cat-file", "-e", oid]); return false } catch { return true } }

  it("loadPrViaGit: added and modified files with exact bytes; one gh call", () => {
    fx.commitMain({ [`${D}/0.json`]: "{\"seq\":0}\n" })
    const head = fx.pr(3, { [`${D}/0.json`]: "{\"seq\":0,\"é\":1}\n", [`${D}/1.json`]: "{\"seq\":1}\n" })
    const gh = ghFor(3, head, { changed_files: 2 })
    const r = loadPrViaGit("o/r", 3, gh.fn, fx.git)
    expect(r.loadErrors).toEqual([])
    expect(r.moved).toBeUndefined()
    expect(r.headSha).toBe(head)
    expect(r.author).toBe("ann")
    expect(r.baseRef).toBe("main")
    expect(r.files).toEqual([{ path: `${D}/0.json`, content: "{\"seq\":0,\"é\":1}\n" }, { path: `${D}/1.json`, content: "{\"seq\":1}\n" }])
    expect(gh.calls).toEqual(["api repos/o/r/pulls/3"])
  })

  it("files are the PR's own changes: commits main gained after the PR branched are not in it", () => {
    const head = fx.pr(3, { [`${D}/0.json`]: "a" })
    fx.commitMain({ "other/file.txt": "main moved on" })
    const r = loadPrViaGit("o/r", 3, ghFor(3, head).fn, fx.git)
    expect(r.loadErrors).toEqual([])
    expect(r.files.map(f => f.path)).toEqual([`${D}/0.json`])
  })

  it("a deleted, renamed or type-changed file, a symlink or a submodule: the same load errors as before", () => {
    const sub = fx.commitMain({ [`${D}/0.json`]: "a", [`${D}/1.json`]: "b", [`${D}/2.json`]: "c" })
    const head = fx.pr(4, {
      [`${D}/3.json`]: "b", // with 1.json removed: a rename to git with renames on, here an add plus a delete
      [`${D}/2.json`]: { symlink: "0.json" }, // type change
      [`${D}/link.json`]: { symlink: "../../../README.md" },
      [`${D}/sub`]: { submodule: sub },
    }, [`${D}/1.json`])
    const r = loadPrViaGit("o/r", 4, ghFor(4, head, { changed_files: 5 }).fn, fx.git)
    expect(r.loadErrors.sort()).toEqual([
      `${D}/1.json: file status "removed" is not allowed (only added or modified)`,
      `${D}/2.json: file status "changed" is not allowed (only added or modified)`,
      `${D}/link.json: not a regular file`,
      `${D}/sub: not a regular file`,
    ].sort())
    expect(r.files).toEqual([{ path: `${D}/3.json`, content: "b" }])
    expect(statusErrors([{ filename: `${D}/1.json`, status: "removed" }])).toEqual([`${D}/1.json: file status "removed" is not allowed (only added or modified)`])
  })

  it("a 2 MB blob is never downloaded (git cat-file -e fails without lazy fetch) and reads as 'file exceeds 1 MB'", () => {
    const head = fx.pr(5, { [`${D}/0.json`]: Buffer.alloc(2 * 1_048_576, 0x61), [`${D}/1.json`]: Buffer.alloc(1_048_576, 0x62) })
    const r = loadPrViaGit("o/r", 5, ghFor(5, head, { changed_files: 2 }).fn, fx.git)
    expect(r.loadErrors).toEqual([`${D}/0.json: file exceeds 1 MB`])
    // Exactly 1 MiB is within the limit (the validator's rule is "> 1 MB"), so it is fetched and read.
    expect(r.files.map(f => [f.path, f.content.length])).toEqual([[`${D}/1.json`, 1_048_576]])
    const big = fx.git(["--no-lazy-fetch", "rev-parse", `${head}:${D}/0.json`]).toString().trim()
    expect(missing(big)).toBe(true)
    // The flag is what keeps it out: git without it fetches the blob on demand from the promisor remote.
    expect(fx.git(["cat-file", "-s", big]).toString().trim()).toBe(String(2 * 1_048_576))
    expect(missing(big)).toBe(false)
  })

  it("every git call runs with --no-lazy-fetch; the fetch recurses into no submodules; the diff ignores no submodule", () => {
    const head = fx.pr(5, { [`${D}/0.json`]: "a" })
    const calls: { a: string[]; limits?: { maxFileBytes?: number } }[] = []
    const r = loadPrViaGit("o/r", 5, ghFor(5, head).fn, (a, env, limits) => { calls.push({ a, limits }); return fx.git(a, env, limits) })
    expect(r.loadErrors).toEqual([])
    expect(calls.length).toBeGreaterThan(0)
    for (const c of calls) expect(c.a[0]).toBe("--no-lazy-fetch")
    const fetch = calls.find(c => c.a.includes("fetch"))!
    expect(fetch.a).toContain("--no-recurse-submodules")
    expect(fetch.a).toContain(`+${head}:refs/gate/pr-5`) // the pinned commit, not refs/pull/5/head
    expect(fetch.limits).toEqual({ maxFileBytes: FETCH_MAX_BYTES })
    expect(calls.find(c => c.a.includes("diff"))!.a).toContain("--ignore-submodules=none")
  })

  it("more than MAX_PR_COMMITS commits: rejected from the PR object; nothing is fetched", () => {
    expect(MAX_PR_COMMITS).toBe(64)
    const never = () => { throw new Error("must not run git") }
    expect(loadPrViaGit("o/r", 8, ghFor(8, "a".repeat(40), { commits: MAX_PR_COMMITS + 1 }).fn, never).loadErrors).toEqual([`PR has ${MAX_PR_COMMITS + 1} commits (limit ${MAX_PR_COMMITS})`])
    for (const commits of [undefined, -1, 1.5, "3"]) expect(loadPrViaGit("o/r", 8, ghFor(8, "a".repeat(40), { commits }).fn, never).loadErrors).toEqual(["PR commit count is unavailable"])
  })

  it("a PR based on a main newer than the checkout: stale, no files (main's newer commits are never taken for the PR's)", () => {
    fx.commitMain({ "other/file.txt": "main moved on" }, [], false) // origin's main moves; the checkout does not
    const head = fx.pr(10, { [`${D}/0.json`]: "a" })
    const r = loadPrViaGit("o/r", 10, ghFor(10, head).fn, fx.git)
    expect(r.stale).toBe(true)
    expect(r.files).toEqual([])
    expect(r.loadErrors).toEqual([])
  })

  it("a fetch that runs past its timeout throws FetchBoundError", () => {
    const head = fx.pr(11, { [`${D}/0.json`]: "a" })
    expect(() => loadPrViaGit("o/r", 11, ghFor(11, head).fn, gitRunner({ cwd: fx.work, env: FIXTURE_ENV, timeoutMs: 1 }))).toThrow(FetchBoundError)
    expect(GIT_TIMEOUT_MS).toBe(120_000)
  })

  it("a fetch over the byte bound throws FetchBoundError; the cut-off pack is removed and nothing is kept", () => {
    const rnd = (seed: number) => { const b = Buffer.alloc(900_000); let x = seed; for (let i = 0; i < b.length; i++) { x = (Math.imul(x, 1103515245) + 12345) >>> 0; b[i] = 0x21 + ((x >>> 24) % 94) } return b }
    fx.pr(12, { [`${D}/0.json`]: rnd(1) }) // three commits of hard-to-compress printable sub-1 MB blobs: over 2 MB to fetch
    fx.pr(12, { [`${D}/1.json`]: rnd(2) }, [], true)
    const head = fx.pr(12, { [`${D}/2.json`]: rnd(3) }, [], true)
    expect(FETCH_MAX_BYTES).toBe(64 * 1_048_576)
    const err = (() => { try { loadPrViaGit("o/r", 12, ghFor(12, head, { changed_files: 3 }).fn, fx.git, 1_048_576); return undefined } catch (e) { return e } })()
    expect(err).toBeInstanceOf(FetchBoundError)
    expect((err as Error).message).toContain("over 1 MiB")
    const packDir = fx.git(["rev-parse", "--path-format=absolute", "--git-path", "objects/pack"]).toString().trim()
    expect(readdirSync(packDir).filter(f => f.startsWith("tmp_pack_"))).toEqual([])
    expect(() => fx.git(["rev-parse", "--verify", "refs/gate/pr-12"])).toThrow()
    // Under the bound the same PR loads (the bound, not the fetch, failed it).
    expect(loadPrViaGit("o/r", 12, ghFor(12, head, { changed_files: 3, commits: 3 }).fn, fx.git, 8 * 1_048_576).loadErrors).toEqual([])
  })

  it("the byte-bounded runner passes arguments to git as data, never as shell text", () => {
    const marker = join(fx.work, "pwned")
    expect(() => fx.git(["rev-parse", "--verify", "--end-of-options", `$(touch '${marker}')`, ";", "touch", marker], undefined, { maxFileBytes: 1_048_576 })).toThrow()
    expect(existsSync(marker)).toBe(false)
    expect(fx.git(["rev-parse", "HEAD"], undefined, { maxFileBytes: 1_048_576 }).toString().trim()).toMatch(/^[0-9a-f]{40}$/)
  })

  it("more than 16 changed files: rejected from the PR object; nothing is fetched", () => {
    const gitCalls: string[][] = []
    const r = loadPrViaGit("o/r", 8, ghFor(8, "a".repeat(40), { changed_files: MAX_PR_FILES + 1 }).fn, a => { gitCalls.push(a); throw new Error("must not run git") })
    expect(r.loadErrors).toEqual([`PR changes ${MAX_PR_FILES + 1} files (limit ${MAX_PR_FILES})`])
    expect(r.files).toEqual([])
    expect(gitCalls).toEqual([])
    expect(MAX_PR_FILES).toBe(16)
  })

  it("more than 16 files in the diff itself (the PR object undercounted): a load error, no file is read", () => {
    const files = Object.fromEntries(Array.from({ length: MAX_PR_FILES + 1 }, (_, i) => [`${D}/${i}.json`, "x"]))
    const head = fx.pr(9, files)
    const r = loadPrViaGit("o/r", 9, ghFor(9, head, { changed_files: 1 }).fn, fx.git)
    expect(r.loadErrors).toEqual([`PR changes ${MAX_PR_FILES + 1} files (limit ${MAX_PR_FILES})`])
    expect(r.files).toEqual([])
  })

  it("a PR object without a usable head sha or file count: nothing is fetched", () => {
    const never = () => { throw new Error("must not run git") }
    for (const sha of [undefined, 42, "", "s", "A".repeat(40)]) {
      const r = loadPrViaGit("o/r", 7, ghFor(7, sha as string).fn, never)
      expect(r.headSha).toBeUndefined()
      expect(r.loadErrors).toEqual(["head commit is unavailable"])
    }
    expect(loadPrViaGit("o/r", 7, ghFor(7, "a".repeat(40), { changed_files: "2" }).fn, never).loadErrors).toEqual(["PR file count is unavailable"])
  })

  it("the author is the PR's user.login; anything else is undefined", () => {
    const never = () => { throw new Error("must not run git") }
    expect(loadPrViaGit("o/r", 7, ghFor(7, "", { user: { login: "ann" } }).fn, never).author).toBe("ann")
    for (const u of [undefined, null, {}, { login: 5 }, { login: "" }]) expect(loadPrViaGit("o/r", 7, ghFor(7, "", { user: u }).fn, never).author).toBeUndefined()
  })
})

describe("isShallow", () => {
  it("reads git rev-parse --is-shallow-repository; anything else throws", () => {
    expect(isShallow(a => { expect(a).toEqual(["rev-parse", "--is-shallow-repository"]); return "true\n" })).toBe(true)
    expect(isShallow(() => "false\n")).toBe(false)
    expect(() => isShallow(() => "")).toThrow()
  })
})
