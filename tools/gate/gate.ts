// Gate: decide what to do with open `run: ...` PRs, then act. Runs from main's copy; never executes PR code.
import { appendFileSync } from "node:fs"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { parseHeartbeat, parseLogFile, type LogFile } from "@blanktab/harness/log"
import { validateSessionPr, type PrFile } from "./validate.ts"
import { readLearnerLogs } from "./summary.ts"
import { closePr, labelNeedsHuman, listOpenPrs, loadPr, mergePr } from "./github.ts"

export interface OpenPr {
  number: number
  headRef: string
  /** branch the PR targets; only main is acceptable */
  baseRef: string
  /** PR updatedAt, the fallback activity time when there is no heartbeat.json */
  updatedAt: Date | null
  files: PrFile[]
  previous: LogFile[]
  heartbeatAt: Date | null
  /** problems found while loading the PR (e.g. removed/renamed files); any entry closes the PR */
  loadErrors?: string[]
  /** head commit that was validated; merging is pinned to it */
  headSha?: string
}
export type GateDecision = { pr: number; action: "merge" | "wait" | "close" | "label-needs-human"; reason: string }

const STALE_MS = 48 * 3_600_000

/** Time in the PR's own heartbeat.json, or null when absent, unparseable or future-dated. */
export function heartbeatFromFiles(files: PrFile[], now: Date): Date | null {
  for (const f of files) {
    if (!/^data\/runs\/[^/]+\/heartbeat\.json$/.test(f.path)) continue
    try {
      const hb = parseHeartbeat(JSON.parse(f.content))
      if (hb.ok) { const t = new Date(hb.value.writtenAt); return t > now ? null : t } // future-dated: ignored (caller falls back to updatedAt)
    } catch { /* malformed: treated as no heartbeat */ }
  }
  return null
}

function hasFinal(files: PrFile[]): boolean {
  return files.some(f => {
    if (!/^data\/runs\/[^/]+\/\d+\.json$/.test(f.path)) return false
    try { const p = parseLogFile(JSON.parse(f.content)); return p.ok && p.value.final } catch { return false }
  })
}

export function decide(prs: OpenPr[], now: Date): GateDecision[] {
  return prs.map((pr): GateDecision => {
    if (pr.baseRef !== "main") return { pr: pr.number, action: "close", reason: "Session PRs must target main." }
    if (pr.loadErrors && pr.loadErrors.length > 0) return { pr: pr.number, action: "close", reason: pr.loadErrors.join("\n") }
    const v = validateSessionPr(pr.files, pr.previous)
    if (!v.ok) {
      return v.kind === "needs-human"
        ? { pr: pr.number, action: "label-needs-human", reason: v.errors.join("\n") }
        : { pr: pr.number, action: "close", reason: v.errors.join("\n") }
    }
    if (hasFinal(pr.files)) return { pr: pr.number, action: "merge", reason: "valid, session is final" }
    const hb = heartbeatFromFiles(pr.files, now) ?? pr.heartbeatAt ?? pr.updatedAt
    if (hb && now.getTime() - hb.getTime() > STALE_MS) return { pr: pr.number, action: "merge", reason: "valid, heartbeat older than 48 h" }
    return { pr: pr.number, action: "wait", reason: hb ? "valid, session still active" : "valid, no heartbeat and not final" }
  })
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
      const ids = [...new Set(loaded.files.map(f => /^data\/runs\/(lrn_[0-9a-f]{16})\//.exec(f.path)?.[1]).filter((x): x is string => !!x))]
      const previous = ids.flatMap(id => { try { return readLearnerLogs(join(runsDir, id)) } catch { return [] } })
      prs.push({ number: listed.number, headRef: loaded.headRef, baseRef: loaded.baseRef, updatedAt: new Date(listed.updatedAt), files: loaded.files, previous, heartbeatAt: null, loadErrors: loaded.loadErrors, headSha: loaded.headSha, learnerId: ids.length === 1 ? ids[0]! : null })
    } catch (e) {
      console.error(`PR #${listed.number}: could not load (${e instanceof Error ? e.message : e}); leaving it for the next run`)
    }
  }
  const decisions = decide(prs, new Date())
  let merged = 0
  const mergedLearners = new Set<string>()
  for (const d of decisions) {
    const pr = prs.find(p => p.number === d.pr)!
    console.log(`PR #${d.pr}: ${d.action} (${d.reason.split("\n")[0]})`)
    try {
      if (d.action === "close") closePr(repo, d.pr, d.reason)
      else if (d.action === "label-needs-human") labelNeedsHuman(repo, d.pr)
      else if (d.action === "merge") {
        // `previous` is main as of the start of this run: merge at most one PR per learner per run
        if (pr.learnerId && mergedLearners.has(pr.learnerId)) { console.log(`PR #${d.pr}: deferred, another PR for ${pr.learnerId} merged this run`); continue }
        mergePr(repo, d.pr, pr.headSha!)
        merged++
        if (pr.learnerId) mergedLearners.add(pr.learnerId)
      }
    } catch (e) {
      console.error(`PR #${d.pr}: action ${d.action} failed: ${e instanceof Error ? e.message : e}`)
    }
  }
  console.log(`merged ${merged}`)
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `merged=${merged}\n`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) main()
