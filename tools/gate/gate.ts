// Gate: decide what to do with open `run: ...` PRs, then act. Runs from main's copy; never executes PR code.
import { appendFileSync } from "node:fs"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { parseLogFile, type LogFile } from "@blanktab/harness/log"
import { validateSessionPr, type PrFile } from "./validate.ts"
import { readLearnerLogs } from "./summary.ts"
import { boundAuthor, closePr, gh, ghCalls, git, gitBuf, isShallow, labelNeedsHuman, listOpenPrs, loadPrViaGit, mergePr, type Git } from "./github.ts"

export interface OpenPr {
  number: number
  /** Must be `run: <learnerId>` for the learner whose files it carries. */
  title: string
  headRef: string
  /** branch the PR targets; only main is acceptable */
  baseRef: string
  files: PrFile[]
  previous: LogFile[]
  /** problems found while loading the PR (e.g. removed/renamed files); any entry closes the PR */
  loadErrors?: string[]
  /** head commit that was validated; merging is pinned to it */
  headSha?: string
  /** The PR's GitHub author (user.login). */
  author?: string
  /** Who the PR's learner belongs to: the author of its first merged PR (github.ts boundAuthor); null if none yet. */
  boundAuthor?: string | null
}
export type GateDecision = { pr: number; action: "merge" | "close" | "label-needs-human"; reason: string }

export const DEV_BUILD_REASON = "Development builds don't contribute."

const LOG_PATH_RE = /^data\/runs\/[^/]+\/\d+\.json$/
const LEARNER_DIR_RE = /^data\/runs\/(lrn_[0-9a-f]{16})\//
/** The learnerIds a PR's file paths name. */
export const learnersIn = (files: PrFile[]): string[] => [...new Set(files.map(f => LEARNER_DIR_RE.exec(f.path)?.[1]).filter((x): x is string => !!x))]
function logsIn(files: PrFile[]): LogFile[] {
  return files.flatMap(f => {
    if (!LOG_PATH_RE.test(f.path)) return []
    try { const p = parseLogFile(JSON.parse(f.content)); return p.ok ? [p.value] : [] } catch { return [] }
  })
}

/**
 * Learners live as long as their tab (decision 0014), so there is no "final" or "gone quiet" signal to wait for:
 * every valid session PR is merged each run. A tab that pushes after validation makes the head-pinned merge fail;
 * the PR stays open and is validated again next run.
 */
export function decide(prs: OpenPr[]): GateDecision[] {
  // Oldest PR first, whatever order they come in: an unbound learner's first merge goes to the earliest PR, never a later copy.
  return [...prs].sort((a, b) => a.number - b.number).map((pr): GateDecision => {
    if (pr.baseRef !== "main") return { pr: pr.number, action: "close", reason: "Session PRs must target main." }
    if (pr.loadErrors && pr.loadErrors.length > 0) return { pr: pr.number, action: "close", reason: pr.loadErrors.join("\n") }
    const ids = learnersIn(pr.files)
    if (ids.length === 1) {
      const id = ids[0]!
      if (pr.title !== `run: ${id}`) return { pr: pr.number, action: "close", reason: `A session PR's title must be "run: ${id}", the learner whose files it carries.` }
      // A learner belongs to the author of its first merged PR: nobody else may add to its record.
      const owner = pr.boundAuthor
      if (owner && pr.author?.toLowerCase() !== owner.toLowerCase()) return { pr: pr.number, action: "close", reason: `Learner ${id} belongs to @${owner}.` }
    }
    const v = validateSessionPr(pr.files, pr.previous)
    if (!v.ok) {
      return v.kind === "needs-human"
        ? { pr: pr.number, action: "label-needs-human", reason: v.errors.join("\n") }
        : { pr: pr.number, action: "close", reason: v.errors.join("\n") }
    }
    const logs = logsIn(pr.files)
    if (logs.some(l => l.codeSha === "dev")) return { pr: pr.number, action: "close", reason: DEV_BUILD_REASON }
    return { pr: pr.number, action: "merge", reason: "valid" }
  })
}

/** What act() does to GitHub (gh in production; injected in tests). */
export interface GateOps {
  close(n: number, reason: string): void
  label(n: number): void
  merge(n: number, headSha: string): void
  log(message: string): void
  error(message: string): void
}

/**
 * Carries out the decisions; resolves to the number merged. A merge is pinned to the head commit that was validated,
 * so a PR without one (headSha missing) is never merged: it is logged and left for the next run. `previous` is main as
 * of the start of this run, so at most one PR per learner is merged per run: the oldest (decisions are carried out by
 * PR number, ascending).
 */
export function act(decisions: GateDecision[], prs: (OpenPr & { learnerId: string | null })[], ops: GateOps): number {
  let merged = 0
  const mergedLearners = new Set<string>()
  for (const d of [...decisions].sort((a, b) => a.pr - b.pr)) {
    const pr = prs.find(p => p.number === d.pr)!
    ops.log(`PR #${d.pr}: ${d.action} (${d.reason.split("\n")[0]})`)
    try {
      if (d.action === "close") ops.close(d.pr, d.reason)
      else if (d.action === "label-needs-human") ops.label(d.pr)
      else if (d.action === "merge") {
        if (pr.learnerId && mergedLearners.has(pr.learnerId)) { ops.log(`PR #${d.pr}: deferred, another PR for ${pr.learnerId} merged this run`); continue }
        if (!pr.headSha) { ops.error(`PR #${d.pr}: no head commit was validated, so it is not merged; leaving it for the next run`); continue }
        ops.merge(d.pr, pr.headSha)
        merged++
        if (pr.learnerId) mergedLearners.add(pr.learnerId)
      }
    } catch (e) {
      ops.error(`PR #${d.pr}: action ${d.action} failed: ${e instanceof Error ? e.message : e}`)
    }
  }
  return merged
}

/** `lookup` memoised per learnerId for one gate run, failures included (one owner lookup per learner, not per PR). */
export function memoBound(lookup: (learnerId: string) => string | null): (learnerId: string) => string | null {
  const seen = new Map<string, { owner: string | null } | { error: unknown }>()
  return id => {
    let r = seen.get(id)
    if (r === undefined) {
      try { r = { owner: lookup(id) } } catch (error) { r = { error } }
      seen.set(id, r)
    }
    if ("error" in r) throw r.error
    return r.owner
  }
}

/** What runGate talks to: gh and git (in the checkout of main), and where it reports. Injected in tests. */
export interface GateDeps {
  gh: (args: string[]) => Buffer
  git: (args: string[]) => string
  gitBuf: Git
  log(message: string): void
  error(message: string): void
}

/**
 * One gate run over the open `run: ` PRs; resolves to the number merged. Aborts before deciding anything if the
 * checkout of main is shallow (who owns a learner needs main's full history). A PR that cannot be decided safely
 * (no head commit or author, a head that moved since it was read, an owner lookup or load that failed) is logged and
 * left open for the next run.
 */
export function runGate(repo: string, runsDir: string, deps: GateDeps): number {
  if (isShallow(deps.git)) throw new Error("the checkout of main is shallow: who owns a learner cannot be told (gate.yml needs fetch-depth 0)")
  let calls = 0
  const ghFn = (args: string[]) => { calls++; return deps.gh(args) }
  const prs: (OpenPr & { learnerId: string | null })[] = []
  const ownerOf = memoBound(id => boundAuthor(repo, id, deps.git, ghFn))
  // Oldest first (decide and act sort again): the earliest PR for an unbound learner is the one that can claim it.
  for (const listed of [...listOpenPrs(repo, ghFn)].sort((a, b) => a.number - b.number)) {
    if (!listed.title.startsWith("run: ")) continue // not a session PR: leave alone
    const before = calls
    try {
      const loaded = loadPrViaGit(repo, listed.number, ghFn, deps.gitBuf)
      if (loaded.headSha === undefined) { deps.error(`PR #${listed.number}: GitHub gave no head commit; leaving it for the next run`); continue }
      if (loaded.moved) { deps.log(`PR #${listed.number}: its head moved since it was read; leaving it for the next run`); continue }
      if (loaded.author === undefined) { deps.error(`PR #${listed.number}: GitHub gave no author; leaving it for the next run`); continue }
      const ids = learnersIn(loaded.files)
      const previous = ids.flatMap(id => { try { return readLearnerLogs(join(runsDir, id)) } catch { return [] } })
      let bound: string | null = null
      if (ids.length === 1) {
        try {
          bound = ownerOf(ids[0]!)
        } catch (e) {
          deps.error(`PR #${listed.number}: who learner ${ids[0]} belongs to could not be found (${e instanceof Error ? e.message : e}); leaving it for the next run`)
          continue
        }
      }
      prs.push({
        number: listed.number, title: listed.title, headRef: loaded.headRef, baseRef: loaded.baseRef, files: loaded.files, previous,
        loadErrors: loaded.loadErrors, headSha: loaded.headSha, author: loaded.author, boundAuthor: bound, learnerId: ids.length === 1 ? ids[0]! : null,
      })
    } catch (e) {
      deps.error(`PR #${listed.number}: could not load (${e instanceof Error ? e.message : e}); leaving it for the next run`)
    } finally {
      deps.log(`PR #${listed.number}: ${calls - before} gh calls`)
    }
  }
  return act(decide(prs), prs, {
    close: (n, reason) => closePr(repo, n, reason, ghFn),
    label: n => labelNeedsHuman(repo, n, ghFn),
    merge: (n, sha) => mergePr(repo, n, sha, ghFn),
    log: deps.log,
    error: deps.error,
  })
}

function main(): void {
  const repo = process.env.REPO
  if (!repo) throw new Error("REPO env var is required (owner/name)")
  const merged = runGate(repo, process.argv[2] ?? "data/runs", { gh, git, gitBuf, log: m => console.log(m), error: m => console.error(m) })
  console.log(`merged ${merged} (${ghCalls()} gh calls in all)`)
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `merged=${merged}\n`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) main()
