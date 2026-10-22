// A throwaway GitHub stand-in for gate tests: a bare "origin" served over file:// (with filters allowed), a "work"
// clone that plays the gate's checkout of main, and an author clone that pushes main commits and refs/pull/N/head.
import { execFileSync } from "node:child_process"
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { gitRunner } from "../github.ts"

/** A file's content, or a symlink to a target, or a submodule (gitlink) at a commit. */
export type Entry = string | Buffer | { symlink: string } | { submodule: string }

const ENV = {
  GIT_AUTHOR_NAME: "t", GIT_AUTHOR_EMAIL: "t@t", GIT_COMMITTER_NAME: "t", GIT_COMMITTER_EMAIL: "t@t",
  GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1",
}

export interface Fixture {
  /** The gate's checkout of main. */
  work: string
  /** git in `work`, as loadPrViaGit's gitFn. */
  git(args: string[], env?: Record<string, string>): Buffer
  /** Commits to main on origin and (unless `pull` is false) pulls it into `work`; resolves to the new main commit. */
  commitMain(files: Record<string, Entry>, remove?: string[], pull?: boolean): string
  /** Sets refs/pull/<n>/head on origin to a commit on top of main; resolves to that commit. */
  pr(n: number, files: Record<string, Entry>, remove?: string[]): string
  cleanup(): void
}

export function makeFixture(): Fixture {
  const dir = mkdtempSync(join(tmpdir(), "gate-git-"))
  const bare = join(dir, "origin.git"), src = join(dir, "src"), work = join(dir, "work")
  const run = (cwd: string, ...args: string[]) => execFileSync("git", args, { cwd, encoding: "utf8", env: { ...process.env, ...ENV }, stdio: ["ignore", "pipe", "pipe"] }).trim()
  run(dir, "init", "-q", "--bare", "-b", "main", bare)
  run(bare, "config", "uploadpack.allowFilter", "true")
  run(dir, "init", "-q", "-b", "main", src)
  run(src, "remote", "add", "origin", `file://${bare}`)
  const apply = (files: Record<string, Entry>, remove: string[]) => {
    for (const p of remove) run(src, "rm", "-q", "--", p)
    for (const [p, e] of Object.entries(files)) {
      const abs = join(src, p)
      rmSync(abs, { force: true, recursive: true })
      if (typeof e === "object" && "submodule" in e) { run(src, "update-index", "--add", "--cacheinfo", `160000,${e.submodule},${p}`); continue }
      mkdirSync(dirname(abs), { recursive: true })
      if (typeof e === "object" && "symlink" in e) symlinkSync(e.symlink, abs)
      else writeFileSync(abs, e)
      run(src, "add", "--", p)
    }
    run(src, "commit", "-q", "--allow-empty", "-m", "c")
    return run(src, "rev-parse", "HEAD")
  }
  apply({ "README.md": "x\n" }, [])
  run(src, "push", "-q", "origin", "main")
  run(dir, "clone", "-q", `file://${bare}`, work)
  return {
    work,
    git: gitRunner({ cwd: work }),
    commitMain(files, remove = [], pull = true) {
      run(src, "checkout", "-q", "main")
      const sha = apply(files, remove)
      run(src, "push", "-q", "origin", "main")
      if (pull) run(work, "pull", "-q", "--ff-only")
      return sha
    },
    pr(n, files, remove = []) {
      run(src, "checkout", "-q", "-B", `pr-${n}`, "main")
      const sha = apply(files, remove)
      run(src, "push", "-q", "-f", "origin", `HEAD:refs/pull/${n}/head`)
      run(src, "checkout", "-q", "main")
      return sha
    },
    cleanup: () => rmSync(dir, { recursive: true, force: true }),
  }
}

/** A fake `gh` answering `api repos/o/r/pulls/<n>` with the given head commit; anything else goes to `other`. */
export function prObject(n: number, head: string, over: Record<string, unknown> = {}): Buffer {
  return Buffer.from(JSON.stringify({ number: n, base: { ref: "main" }, head: { sha: head, ref: `run/x${n}`, repo: { full_name: "o/r" } }, changed_files: 1, commits: 1, user: { login: "ann" }, ...over }))
}
