// Gate: decide what to do with open `run: ...` PRs, then act. Runs from main's copy; never executes PR code.
import { appendFileSync } from "node:fs"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { parseLogFile, type LogFile } from "@blanktab/harness/log"
import { validateSessionPr, type PrFile } from "./validate.ts"
import { readLearnerLogs } from "./summary.ts"
import { closePr, labelNeedsHuman, listOpenPrs, loadPr, mergePr } from "./github.ts"

export interface OpenPr {
  number: number
  headRef: string
  /** branch the PR targets; only main is acceptable */
  baseRef: string
  files: PrFile[]
  previous: LogFile[]
  /** problems found while loading the PR (e.g. removed/renamed files); any entry closes the PR */
  loadErrors?: string[]
  /** head commit that was validated; merging is pinned to it */
  headSha?: string
}
export type GateDecision = { pr: number; action: "merge" | "close" | "label-needs-human"; reason: string }

export const DEV_BUILD_REASON = "Development builds don't contribute."

const LOG_PATH_RE = /^data\/runs\/[^/]+\/\d+\.json$/
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
  return prs.map((pr): GateDecision => {
    if (pr.baseRef !== "main") return { pr: pr.number, action: "close", reason: "Session PRs must target main." }
    if (pr.loadErrors && pr.loadErrors.length > 0) return { pr: pr.number, action: "close", reason: pr.loadErrors.join("\n") }
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
 * of the start of this run, so at most one PR per learner is merged per run.
 */
export function act(decisions: GateDecision[], prs: (OpenPr & { learnerId: string | null })[], ops: GateOps): number {
  let merged = 0
  const mergedLearners = new Set<string>()
  for (const d of decisions) {
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

function main(): void {
  const repo = process.env.REPO
  if (!repo) throw new Error("REPO env var is required (owner/name)")
  const runsDir = process.argv[2] ?? "data/runs"
  const prs: (OpenPr & { learnerId: string | null })[] = []
  for (const listed of listOpenPrs(repo)) {
    if (!listed.title.startsWith("run: ")) continue // not a session PR: leave alone
    try {
      const loaded = loadPr(repo, listed.number)
      if (loaded.headSha === undefined) { console.error(`PR #${listed.number}: GitHub gave no head commit; leaving it for the next run`); continue }
      const ids = [...new Set(loaded.files.map(f => /^data\/runs\/(lrn_[0-9a-f]{16})\//.exec(f.path)?.[1]).filter((x): x is string => !!x))]
      const previous = ids.flatMap(id => { try { return readLearnerLogs(join(runsDir, id)) } catch { return [] } })
      prs.push({ number: listed.number, headRef: loaded.headRef, baseRef: loaded.baseRef, files: loaded.files, previous, loadErrors: loaded.loadErrors, headSha: loaded.headSha, learnerId: ids.length === 1 ? ids[0]! : null })
    } catch (e) {
      console.error(`PR #${listed.number}: could not load (${e instanceof Error ? e.message : e}); leaving it for the next run`)
    }
  }
  const merged = act(decide(prs), prs, {
    close: (n, reason) => closePr(repo, n, reason),
    label: n => labelNeedsHuman(repo, n),
    merge: (n, sha) => mergePr(repo, n, sha),
    log: m => console.log(m),
    error: m => console.error(m),
  })
  console.log(`merged ${merged}`)
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `merged=${merged}\n`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) main()
