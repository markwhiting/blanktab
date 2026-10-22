// Thin `gh` and `git` wrappers. Everything fetched from a PR is treated as data; nothing from a PR is ever executed.
// Arguments go through execFileSync (no shell), so untrusted file names cannot inject commands.
import { execFileSync } from "node:child_process"
import { readdirSync, rmSync, statSync } from "node:fs"
import { join } from "node:path"
import { LEARNER_ID_RE } from "@blanktab/harness/log"
import type { PrFile } from "./validate.ts"

const MAX_BYTES = 1_048_576
/** A whole 3000-step life is 6 logs; anything far beyond that is not a session PR and is not even listed. */
export const MAX_PR_FILES = 16
/** How many of the commits touching a learner's dir (oldest first) boundAuthor asks GitHub about. */
export const MAX_OWNER_WALK = 20
/**
 * A PR with more commits is rejected from the PR object, before anything is fetched (decision 0017). A tab makes one
 * commit per send, and a send with nothing new makes none (contributor.ts commitFiles), so every commit adds or changes
 * a file; with at most 16 files that is about 16 commits, and 64 leaves room for re-sends. The cap bounds what the
 * fetch of the PR's head commit can bring: the PR's own commits, whose blobs over 1 MB the filter leaves out.
 */
export const MAX_PR_COMMITS = 64
/** Every git the gate runs is killed after this long. A PR fetch that runs past it is over its bound (FetchBoundError). */
export const GIT_TIMEOUT_MS = 120_000
/**
 * The most a PR fetch may write: the pack it downloads is kept as one file (unpackLimit 1) and git runs under
 * `ulimit -f`, so a larger pack is cut off and the fetch fails (FetchBoundError). A valid PR's pack holds at most
 * 16 files of at most 1 MB, plus the trees and commits of at most 64 commits: well under 32 MiB; 64 MiB is twice that.
 */
export const FETCH_MAX_BYTES = 64 * 1_048_576

/** A PR whose fetch went over its time or byte bound. The gate labels it needs-human; it is not fetched again. */
export class FetchBoundError extends Error {}

let ghCallCount = 0
export function gh(args: string[]): Buffer {
  ghCallCount++
  return execFileSync("gh", args, { maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "inherit"] })
}
/** Calls made through gh() so far in this process. */
export function ghCalls(): number { return ghCallCount }

export interface ListedPr { number: number; title: string; headRefName: string; baseRefName: string; labels?: { name?: unknown }[] }

export function listOpenPrs(repo: string, ghFn: (args: string[]) => Buffer = gh): ListedPr[] {
  return JSON.parse(ghFn(["pr", "list", "--repo", repo, "--state", "open", "--limit", "200", "--json", "number,title,headRefName,baseRefName,labels"]).toString("utf8")) as ListedPr[]
}

/**
 * `headSha` is undefined when the PR object gave none: such a PR is never merged (there is no commit to pin the merge
 * to). `author`: the PR's user.login, undefined if it gave none (such a PR is not decided).
 */
export interface LoadedPr { headRef: string; baseRef: string; headSha: string | undefined; author: string | undefined; files: PrFile[]; loadErrors: string[] }

/** Only added/modified files are acceptable; anything else (removed, renamed, copied, changed...) is an error. */
export function statusErrors(changed: { filename: string; status: string }[]): string[] {
  return changed.filter(f => f.status !== "added" && f.status !== "modified").map(f => `${f.filename}: file status "${f.status}" is not allowed (only added or modified)`)
}

/** Raw bytes -> PrFile content. Size is judged on the raw bytes before any decoding. */
export function bytesToContent(raw: Buffer): { content: string } | { error: string } {
  if (raw.length > MAX_BYTES) return { content: "\0".repeat(raw.length) } // oversize placeholder: validator's size rule rejects it on length
  try { return { content: new TextDecoder("utf-8", { fatal: true }).decode(raw) } } catch { return { error: "file is not valid UTF-8" } }
}

/**
 * git, returning raw bytes; `env` is added to the process env. With `limits.maxFileBytes`, no file git writes may grow
 * past that many bytes (rounded down to whole KiB).
 */
export interface Git { (args: string[], env?: Record<string, string>, limits?: { maxFileBytes?: number }): Buffer }
/**
 * A Git running in `cwd` (default: the current directory, the checkout of main), killed after `timeoutMs`, with `env`
 * added to every call. A file-size limit runs git under bash's `ulimit -f` (KiB in bash; sh's unit differs: 512 bytes
 * in dash, 1024 in macOS sh). The arguments reach git as bash's positional parameters, never parsed as shell text.
 */
export function gitRunner(opts: { cwd?: string; timeoutMs?: number; env?: Record<string, string> } = {}): Git {
  return (args, env, limits) => {
    const o = {
      cwd: opts.cwd, timeout: opts.timeoutMs ?? GIT_TIMEOUT_MS, maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "pipe"] as ["ignore", "pipe", "pipe"],
      env: opts.env || env ? { ...process.env, ...opts.env, ...env } : process.env,
    }
    const max = limits?.maxFileBytes
    if (max === undefined) return execFileSync("git", args, o)
    if (!Number.isSafeInteger(max) || max < 1024) throw new Error(`not a file-size limit: ${max}`)
    return execFileSync("bash", ["-c", 'ulimit -f "$1" && shift && exec git "$@"', "bash", String(Math.floor(max / 1024)), ...args], o)
  }
}
export const gitBuf: Git = gitRunner()

const SHA_RE = /^[0-9a-f]{40}$|^[0-9a-f]{64}$/
/** git's raw diff status letters, named as the REST API names them (so load errors read as they always have). */
const STATUS: Record<string, string> = { A: "added", M: "modified", D: "removed", R: "renamed", C: "copied", T: "changed" }
const REGULAR = new Set(["100644", "100755"])

/**
 * Reads PR <n>'s files with git (decision 0017). One gh call (pulls/N); a PR over MAX_PR_FILES files or MAX_PR_COMMITS
 * commits is rejected from it, before any fetch. Then a partial fetch of the head commit pulls/N named into
 * refs/gate/pr-N leaves out blobs over 1 MB, and every git call runs with `--no-lazy-fetch`, so such a blob is never
 * downloaded: one that is missing reads as "file exceeds 1 MB". The fetch is bounded by GIT_TIMEOUT_MS and
 * FETCH_MAX_BYTES; past either it throws FetchBoundError. Files are the diff from the merge base with the checkout of
 * main (HEAD) to that commit. If GitHub no longer serves the commit, the PR is `moved`. If HEAD..head is not the PR's
 * commit count, the checkout of main is not the main GitHub compares the PR with (it is older than the PR's base, or
 * GitHub has not caught up): `stale`. Either way it has no files and is left for the next run. Nothing from the PR is
 * run: fetch runs no hooks from it and recurses into no submodules; the diff runs no external diff or textconv.
 *
 * HEAD is the one main used throughout a gate run: the merge base here, the earlier logs (read from its working tree)
 * and the owner (boundAuthor's git log) all come from it.
 */
export function loadPrViaGit(repo: string, n: number, ghFn: (args: string[]) => Buffer = gh, gitFn: Git = gitBuf, fetchMaxBytes = FETCH_MAX_BYTES): LoadedPr & { moved?: true; stale?: true } {
  if (!Number.isSafeInteger(n) || n <= 0) throw new Error(`not a PR number: ${n}`)
  const pr = JSON.parse(ghFn(["api", `repos/${repo}/pulls/${n}`]).toString("utf8")) as { base: { ref: string }; head: { sha?: unknown; ref: string }; changed_files?: unknown; commits?: unknown; user?: { login?: unknown } | null }
  const headSha = typeof pr.head.sha === "string" && SHA_RE.test(pr.head.sha) ? pr.head.sha : undefined
  const login = pr.user?.login
  const base = { headRef: pr.head.ref, baseRef: pr.base.ref, headSha, author: typeof login === "string" && login !== "" ? login : undefined }
  // Files are read at the head commit; without one there is nothing to read them at (the gate leaves such a PR alone).
  if (headSha === undefined) return { ...base, files: [], loadErrors: ["head commit is unavailable"] }
  // Budget first, from the PR object alone: an oversized PR is never fetched.
  const isCount = (c: unknown): c is number => typeof c === "number" && Number.isInteger(c) && c >= 0
  const count = pr.changed_files, commits = pr.commits
  if (!isCount(count)) return { ...base, files: [], loadErrors: ["PR file count is unavailable"] }
  if (count > MAX_PR_FILES) return { ...base, files: [], loadErrors: [`PR changes ${count} files (limit ${MAX_PR_FILES})`] }
  if (!isCount(commits)) return { ...base, files: [], loadErrors: ["PR commit count is unavailable"] }
  if (commits > MAX_PR_COMMITS) return { ...base, files: [], loadErrors: [`PR has ${commits} commits (limit ${MAX_PR_COMMITS})`] }

  // --no-lazy-fetch on every call (git >= 2.44 refuses an unknown option, so an older git fails loudly): a blob the
  // filter left out is never fetched on demand.
  const g = (args: string[]) => gitFn(["--no-lazy-fetch", ...args])
  const text = (args: string[]) => g(args).toString("utf8").trim()
  const ref = `refs/gate/pr-${n}`
  // The head commit pulls/N named, not the moving refs/pull/N/head: the commit cap binds exactly what is fetched.
  // Limit MAX_BYTES + 1: the filter omits blobs of *at least* the limit, and a file of exactly 1 MB is allowed.
  const fetchArgs = ["-c", "fetch.unpackLimit=1", "-c", "transfer.unpackLimit=1", "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", `--filter=blob:limit=${MAX_BYTES + 1}`, "origin", `+${headSha}:${ref}`]
  try {
    gitFn(["--no-lazy-fetch", ...fetchArgs], undefined, { maxFileBytes: fetchMaxBytes })
  } catch (e) {
    const timedOut = (e as { code?: unknown }).code === "ETIMEDOUT"
    // A pack cut off at the limit is left as a tmp_pack_* file: that is how "over the byte bound" is told, and it is removed.
    let over = false
    try {
      const packDir = text(["rev-parse", "--path-format=absolute", "--git-path", "objects/pack"])
      for (const f of readdirSync(packDir)) {
        if (!f.startsWith("tmp_pack_")) continue
        if (statSync(join(packDir, f)).size >= Math.floor(fetchMaxBytes / 1024) * 1024) over = true
        rmSync(join(packDir, f), { force: true })
      }
    } catch (ce) {
      if (!timedOut) throw ce
    }
    if (timedOut) throw new FetchBoundError(`fetching it took over ${GIT_TIMEOUT_MS / 1000} s`)
    if (over) throw new FetchBoundError(`fetching it took over ${fetchMaxBytes / 1_048_576} MiB`)
    // GitHub no longer serves that commit (the branch was force-pushed past it): the head moved.
    if (/not our ref/.test(String((e as { stderr?: unknown }).stderr ?? ""))) return { ...base, files: [], loadErrors: [], moved: true }
    throw e
  }
  if (text(["rev-parse", "--verify", "--end-of-options", `${ref}^{commit}`]) !== headSha) return { ...base, files: [], loadErrors: [], moved: true }
  if (text(["rev-list", "--count", `HEAD..${headSha}`]) !== String(commits)) return { ...base, files: [], loadErrors: [], stale: true }
  const mergeBase = text(["merge-base", "HEAD", headSha])
  if (!SHA_RE.test(mergeBase)) throw new Error(`no merge base of ${headSha} with main`)
  // --raw gives each file's status, head-side mode and blob id in one call, pinned to headSha (not to a ref).
  const raw = g(["diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--ignore-submodules=none", "--raw", "-z", "--no-abbrev", mergeBase, headSha])
  const changed = parseRawDiff(raw)
  const loadErrors: string[] = []
  loadErrors.push(...statusErrors(changed.map(c => ({ filename: c.path, status: STATUS[c.status] ?? c.status }))))
  if (changed.length > MAX_PR_FILES) return { ...base, files: [], loadErrors: [...loadErrors, `PR changes ${changed.length} files (limit ${MAX_PR_FILES})`] }
  const files: PrFile[] = []
  for (const c of changed) {
    if (c.status !== "A" && c.status !== "M") continue
    if (!REGULAR.has(c.mode)) { loadErrors.push(`${c.path}: not a regular file`); continue }
    let size: number
    try { size = Number(text(["cat-file", "-s", c.oid])) } catch { size = Infinity } // filtered out: over 1 MB
    if (!(size <= MAX_BYTES)) { loadErrors.push(`${c.path}: file exceeds 1 MB`); continue }
    const r = bytesToContent(g(["cat-file", "blob", c.oid]))
    if ("error" in r) loadErrors.push(`${c.path}: ${r.error}`)
    else files.push({ path: c.path, content: r.content })
  }
  return { ...base, files, loadErrors }
}

/** `git diff --raw -z` output: ":<srcmode> <dstmode> <srcoid> <dstoid> <status>\0<path>\0" (two paths for R and C). */
function parseRawDiff(raw: Buffer): { status: string; mode: string; oid: string; path: string }[] {
  const parts = raw.toString("latin1").split("\0")
  const out: { status: string; mode: string; oid: string; path: string }[] = []
  const path = (p: string | undefined) => {
    if (p === undefined) throw new Error("malformed git diff output")
    try { return new TextDecoder("utf-8", { fatal: true }).decode(Buffer.from(p, "latin1")) } catch { throw new Error("a changed path is not valid UTF-8") }
  }
  for (let i = 0; i < parts.length && parts[i] !== ""; ) {
    const m = /^:(\d{6}) (\d{6}) ([0-9a-f]+) ([0-9a-f]+) ([A-Z])\d*$/.exec(parts[i]!)
    if (!m) throw new Error("malformed git diff output")
    const status = m[5]!
    let p = path(parts[i + 1])
    i += 2
    if (status === "R" || status === "C") { p = path(parts[i]); i += 1 } // the destination path
    out.push({ status, mode: m[2]!, oid: m[4]!, path: p })
  }
  return out
}

/** Runs git in the gate's checkout of main (no shell). */
export function git(args: string[]): string {
  return execFileSync("git", args, { encoding: "utf8", timeout: GIT_TIMEOUT_MS, maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "inherit"] })
}

/** True if the checkout has only part of main's history (then who owns a learner cannot be told: the gate stops). */
export function isShallow(gitFn: (args: string[]) => string = git): boolean {
  const out = gitFn(["rev-parse", "--is-shallow-repository"]).trim()
  if (out !== "true" && out !== "false") throw new Error(`git rev-parse --is-shallow-repository said ${JSON.stringify(out.slice(0, 40))}`)
  return out === "true"
}

/**
 * Who a learner belongs to (decisions 0014, 0017): the GitHub author of the first merged PR among the commits on main
 * (HEAD, the checkout of main) touching data/runs/<learnerId>/, oldest first, at most MAX_OWNER_WALK of them (needs main's full history: the caller
 * checks isShallow). Commits with no merged PR (direct pushes) are skipped. Null if none of them has one: the learner
 * is unbound, and the first author to get a PR merged claims it. Throws if a lookup fails (a gh error, a malformed
 * answer, a merged PR without a login): the caller must not decide that learner's PRs then.
 */
export function boundAuthor(repo: string, learnerId: string, gitFn: (args: string[]) => string = git, ghFn: (args: string[]) => Buffer = gh): string | null {
  if (!LEARNER_ID_RE.test(learnerId)) throw new Error(`not a learnerId: ${JSON.stringify(learnerId.slice(0, 40))}`)
  // --reverse applies after any limit, so the oldest are taken here rather than with -n.
  const commits = gitFn(["--no-lazy-fetch", "log", "--reverse", "--format=%H", "--", `:(top,literal)data/runs/${learnerId}/`]).split("\n").map(l => l.trim()).filter(l => l !== "")
  for (const sha of commits.slice(0, MAX_OWNER_WALK)) {
    if (!/^[0-9a-f]{40}$/.test(sha)) throw new Error(`git gave no commit for ${learnerId}`)
    const pulls = JSON.parse(ghFn(["api", `repos/${repo}/commits/${sha}/pulls`]).toString("utf8")) as unknown
    if (!Array.isArray(pulls)) throw new Error(`unexpected answer for the pulls of commit ${sha}`)
    const merged = (pulls as { merged_at?: unknown; user?: { login?: unknown } | null }[]).find(p => typeof p?.merged_at === "string" && p.merged_at !== "")
    if (merged === undefined) continue // a direct push: it binds no one
    const login = merged.user?.login
    if (typeof login !== "string" || login === "") throw new Error(`merged PR without an author for commit ${sha} (learner ${learnerId})`)
    return login
  }
  return null
}

const fence = (s: string) => "```\n" + s.replaceAll("`", "'").slice(0, 3000) + "\n```"

export function closePr(repo: string, n: number, reason: string, ghFn: (args: string[]) => Buffer = gh): void {
  ghFn(["pr", "close", String(n), "--repo", repo, "--comment", `Closed by the session gate:\n\n${fence(reason)}`])
}
export function labelNeedsHuman(repo: string, n: number, ghFn: (args: string[]) => Buffer = gh): void {
  ghFn(["label", "create", "needs-human", "--repo", repo, "--force", "--description", "Session PR needs a person: it touches non-log paths, or was too large or slow to fetch", "--color", "d93f0b"])
  ghFn(["pr", "edit", String(n), "--repo", repo, "--add-label", "needs-human"])
}
/** Drafts cannot be merged, so mark ready first. --match-head-commit refuses to merge if the head moved since validation. */
export function mergePr(repo: string, n: number, headSha: string | undefined, ghFn: (args: string[]) => Buffer = gh): void {
  // Never an unpinned merge: without the validated head, a commit pushed after validation could be merged unchecked.
  if (typeof headSha !== "string" || headSha === "") throw new Error(`PR #${n}: no validated head commit to pin the merge to`)
  ghFn(["pr", "ready", String(n), "--repo", repo])
  ghFn(["pr", "merge", String(n), "--repo", repo, "--squash", "--delete-branch", "--match-head-commit", headSha])
}
