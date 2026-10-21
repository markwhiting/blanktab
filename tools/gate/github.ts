// Thin `gh` wrappers. Everything fetched from a PR is treated as data; nothing from a PR is ever executed.
// Arguments go through execFileSync (no shell), so untrusted file names cannot inject commands.
import { execFileSync } from "node:child_process"
import type { PrFile } from "./validate.ts"

const MAX_BYTES = 1_048_576
/** A whole 3000-step life is 6 logs; anything far beyond that is not a session PR and is not even listed. */
export const MAX_PR_FILES = 16

export function gh(args: string[]): Buffer {
  return execFileSync("gh", args, { maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "inherit"] })
}
const ghJson = <T>(args: string[]): T => JSON.parse(gh(args).toString("utf8")) as T

export interface ListedPr { number: number; title: string; headRefName: string; baseRefName: string; updatedAt: string; createdAt: string }

export function listOpenPrs(repo: string): ListedPr[] {
  return ghJson<ListedPr[]>(["pr", "list", "--repo", repo, "--state", "open", "--limit", "200", "--json", "number,title,headRefName,baseRefName,updatedAt,createdAt"])
}

export interface LoadedPr { headRef: string; baseRef: string; headSha: string; files: PrFile[]; loadErrors: string[] }

/** Only added/modified files are acceptable; anything else (removed, renamed, copied, changed...) is an error. */
export function statusErrors(changed: { filename: string; status: string }[]): string[] {
  return changed.filter(f => f.status !== "added" && f.status !== "modified").map(f => `${f.filename}: file status "${f.status}" is not allowed (only added or modified)`)
}

/** Raw bytes -> PrFile content. Size is judged on the raw bytes before any decoding. */
export function bytesToContent(raw: Buffer): { content: string } | { error: string } {
  if (raw.length > MAX_BYTES) return { content: "\0".repeat(raw.length) } // oversize placeholder: validator's size rule rejects it on length
  try { return { content: new TextDecoder("utf-8", { fatal: true }).decode(raw) } } catch { return { error: "file is not valid UTF-8" } }
}

/** Checks size from contents metadata first, so an oversize file is never downloaded. */
export function loadFile(ghFn: (args: string[]) => Buffer, headRepo: string, path: string, sha: string): { content: string } | { error: string } {
  const url = `repos/${headRepo}/contents/${path.split("/").map(encodeURIComponent).join("/")}?ref=${sha}`
  let meta: { type?: string; size?: number }
  try { meta = JSON.parse(ghFn(["api", url]).toString("utf8")) as { type?: string; size?: number } } catch { return { error: "could not read file metadata" } }
  if (meta.type !== "file" || typeof meta.size !== "number") return { error: "not a regular file" }
  if (meta.size > MAX_BYTES) return { error: "file exceeds 1 MB" }
  const c = bytesToContent(ghFn(["api", "-H", "Accept: application/vnd.github.raw", url]))
  return c
}

export function loadPr(repo: string, number: number, ghFn: (args: string[]) => Buffer = gh): LoadedPr {
  const get = <T>(args: string[]): T => JSON.parse(ghFn(args).toString("utf8")) as T
  const pr = get<{ base: { ref: string }; head: { sha: string; ref: string; repo: { full_name: string } | null }; changed_files?: unknown }>(["api", `repos/${repo}/pulls/${number}`])
  const headSha = pr.head.sha
  const base = { headRef: pr.head.ref, baseRef: pr.base.ref, headSha }
  // Budget first, from the PR object alone: an oversized PR is never listed, let alone downloaded.
  const count = pr.changed_files
  if (typeof count !== "number" || !Number.isInteger(count) || count < 0) return { ...base, files: [], loadErrors: ["PR file count is unavailable"] }
  if (count > MAX_PR_FILES) return { ...base, files: [], loadErrors: [`PR changes ${count} files (limit ${MAX_PR_FILES})`] }
  const loadErrors: string[] = []
  const changed = get<{ filename: string; status: string }[][]>(["api", `repos/${repo}/pulls/${number}/files`, "--paginate", "--slurp"]).flat()
  loadErrors.push(...statusErrors(changed))
  if (changed.length > MAX_PR_FILES) loadErrors.push(`PR changes ${changed.length} files (limit ${MAX_PR_FILES})`)
  const files: PrFile[] = []
  const headRepo = pr.head.repo?.full_name
  if (!headRepo) loadErrors.push("head repository is unavailable")
  else if (changed.length <= MAX_PR_FILES) {
    for (const f of changed) {
      if (f.status !== "added" && f.status !== "modified") continue
      const c = loadFile(ghFn, headRepo, f.filename, headSha)
      if ("error" in c) loadErrors.push(`${f.filename}: ${c.error}`)
      else files.push({ path: f.filename, content: c.content })
    }
  }
  return { ...base, files, loadErrors }
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
export function mergePr(repo: string, n: number, headSha: string, ghFn: (args: string[]) => Buffer = gh): void {
  ghFn(["pr", "ready", String(n), "--repo", repo])
  ghFn(["pr", "merge", String(n), "--repo", repo, "--squash", "--delete-branch", "--match-head-commit", headSha])
}
