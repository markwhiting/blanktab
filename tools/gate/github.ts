// Thin `gh` wrappers. Everything fetched from a PR is treated as data; nothing from a PR is ever executed.
// Arguments go through execFileSync (no shell), so untrusted file names cannot inject commands.
import { execFileSync } from "node:child_process"
import type { PrFile } from "./validate.ts"

const MAX_BYTES = 1_048_576
const MAX_PR_FILES = 500

export function gh(args: string[]): Buffer {
  return execFileSync("gh", args, { maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "inherit"] })
}
const ghJson = <T>(args: string[]): T => JSON.parse(gh(args).toString("utf8")) as T

export interface ListedPr { number: number; title: string; headRefName: string }

export function listOpenPrs(repo: string): ListedPr[] {
  return ghJson<ListedPr[]>(["pr", "list", "--repo", repo, "--state", "open", "--limit", "200", "--json", "number,title,headRefName"])
}

export interface LoadedPr { headRef: string; headSha: string; files: PrFile[]; loadErrors: string[] }

/** Only added/modified files are acceptable; anything else (removed, renamed, copied, changed...) is an error. */
export function statusErrors(changed: { filename: string; status: string }[]): string[] {
  return changed.filter(f => f.status !== "added" && f.status !== "modified").map(f => `${f.filename}: file status "${f.status}" is not allowed (only added or modified)`)
}

/** Raw bytes -> PrFile content. Size is judged on the raw bytes before any decoding. */
export function bytesToContent(raw: Buffer): { content: string } | { error: string } {
  if (raw.length > MAX_BYTES) return { content: "\0".repeat(raw.length) } // oversize placeholder: validator's size rule rejects it on length
  try { return { content: new TextDecoder("utf-8", { fatal: true }).decode(raw) } } catch { return { error: "file is not valid UTF-8" } }
}

export function loadPr(repo: string, number: number): LoadedPr {
  const pr = ghJson<{ head: { sha: string; ref: string; repo: { full_name: string } | null } }>(["api", `repos/${repo}/pulls/${number}`])
  const headSha = pr.head.sha
  const loadErrors: string[] = []
  const changed = ghJson<{ filename: string; status: string }[]>(["api", `repos/${repo}/pulls/${number}/files`, "--paginate", "--slurp"]).flat()
  loadErrors.push(...statusErrors(changed))
  if (changed.length > MAX_PR_FILES) loadErrors.push(`PR changes ${changed.length} files (limit ${MAX_PR_FILES})`)
  const files: PrFile[] = []
  const headRepo = pr.head.repo?.full_name
  if (!headRepo) loadErrors.push("head repository is unavailable")
  else if (changed.length <= MAX_PR_FILES) {
    for (const f of changed) {
      if (f.status !== "added" && f.status !== "modified") continue
      const apiPath = f.filename.split("/").map(encodeURIComponent).join("/")
      const raw = gh(["api", "-H", "Accept: application/vnd.github.raw", `repos/${headRepo}/contents/${apiPath}?ref=${headSha}`])
      const c = bytesToContent(raw)
      if ("error" in c) loadErrors.push(`${f.filename}: ${c.error}`)
      else files.push({ path: f.filename, content: c.content })
    }
  }
  return { headRef: pr.head.ref, headSha, files, loadErrors }
}

const fence = (s: string) => "```\n" + s.replaceAll("`", "'").slice(0, 3000) + "\n```"

export function closePr(repo: string, n: number, reason: string): void {
  gh(["pr", "close", String(n), "--repo", repo, "--comment", `Closed by the session gate:\n\n${fence(reason)}`])
}
export function labelNeedsHuman(repo: string, n: number): void {
  gh(["label", "create", "needs-human", "--repo", repo, "--force", "--description", "Session PR touches non-log paths; needs a person", "--color", "d93f0b"])
  gh(["pr", "edit", String(n), "--repo", repo, "--add-label", "needs-human"])
}
/** Drafts cannot be merged, so mark ready first. --match-head-commit refuses to merge if the head moved since validation. */
export function mergePr(repo: string, n: number, headSha: string): void {
  gh(["pr", "ready", String(n), "--repo", repo])
  gh(["pr", "merge", String(n), "--repo", repo, "--squash", "--delete-branch", "--match-head-commit", headSha])
}
