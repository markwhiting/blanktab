// The tracked files the repo-hygiene tests (no-trailing-whitespace, no-machine-paths, no-old-name) scan.
import { execFileSync } from "node:child_process"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"

export const REPO_ROOT = fileURLToPath(new URL("..", import.meta.url))

/**
 * Whether a hygiene test scans `path` (repo-relative). data/runs/ holds contributed log files, and data/newborns/ the
 * collected newborn reference files (the same format, checked by tools/eval/newborn-reference.ts): log files written by
 * tabs, checked as session logs only, so their text (formatting, a short error reason) must not be able to turn main's CI red.
 */
export function hygieneScans(path: string): boolean {
  return path !== "" && !path.startsWith("data/runs/") && !path.startsWith("data/newborns/")
}

/** Tracked files (repo-relative) the hygiene tests scan, in `git ls-files` order. */
export function trackedFiles(): string[] {
  return execFileSync("git", ["ls-files"], { encoding: "utf8", cwd: REPO_ROOT }).split("\n").filter(hygieneScans)
}

/** A tracked file's text, or undefined when it is gone from the working tree (deleted, not yet staged). */
export function readTracked(path: string): string | undefined {
  try {
    return readFileSync(join(REPO_ROOT, path), "utf8")
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return undefined
    throw e
  }
}
