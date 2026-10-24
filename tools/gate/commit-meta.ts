// Usage: tsx tools/gate/commit-meta.ts   (in the gate's checkout of main, after meta-cli.ts; prints meta=committed|unchanged)
// The gate's one direct write to main (decision 0019; invariant 7): a commit of exactly the data/meta files meta-cli.ts
// computed from main's merged logs, never PR content, never forced. On a push that main moved past, it resets to the
// new main and rebuilds from scratch (meta-cli.ts again), so it never pushes bytes built on an old main.
import { execFileSync } from "node:child_process"
import { createHash } from "node:crypto"
import { appendFileSync } from "node:fs"
import { pathToFileURL } from "node:url"

/** The files meta-cli.ts writes (meta.ts POSTERIOR_PATH, CALIBRATION_PATH, DIAGNOSTICS_PATH, POSTERIOR_INDEX_PATH), and nothing else. */
export const META_PATH_RE = /^data\/meta\/(posterior\.json|calibration\.json|diagnostics\.json|posterior-index\.jsonl)$/

const BOT = "github-actions[bot]"
const BOT_EMAIL = "41898282+github-actions[bot]@users.noreply.github.com"
const POSTERIOR = "data/meta/posterior.json"

/** One message per staged path the gate may not commit; empty means every path is one of its files (or none is staged). */
export function metaPathErrors(staged: string[]): string[] {
  return staged.filter(p => !META_PATH_RE.test(p)).map(p => `${JSON.stringify(p)} is not a gate-computed data/meta file`)
}

const lines = (s: string, sep = "\n") => s.split(sep).filter(l => l !== "")
/** The lockfile npm ci installed from: main's must still be it, or meta-cli ran main's new code on an old install. */
const LOCKFILE = "package-lock.json"
/** A push git refused because origin's main is ahead of it (not a protection rule, an auth failure or a network error). */
const isBehind = (e: unknown) => /\[rejected\][^\n]*\((fetch first|non-fast-forward)\)/.test(e instanceof Error ? e.message : String(e))

/**
 * Stages data/meta, refuses anything but the gate's own files, commits them as the Actions bot and pushes HEAD to main.
 * A push rejected because main moved: fetch, hard reset to origin/main, drop untracked data/meta files, rebuild()
 * (meta-cli.ts on the new main) and try again, at most `maxTries` pushes in all. `git` must throw with git's stderr
 * in the message when git fails. Throws, committing nothing, if the tree holds any other change, or if HEAD's
 * package-lock.json is not `installedLock` (the blob sha npm ci installed from) at any try.
 */
export async function commitMeta(
  deps: { git: (args: string[]) => string; installedLock: string; rebuild(): Promise<boolean>; log(m: string): void },
  maxTries = 3,
): Promise<"committed" | "unchanged"> {
  const { git, log } = deps
  if (!Number.isSafeInteger(maxTries) || maxTries < 1) throw new Error(`not a number of tries: ${maxTries}`)
  if (!/^[0-9a-f]{40}([0-9a-f]{24})?$/.test(deps.installedLock)) throw new Error(`not a lockfile blob sha: ${JSON.stringify(deps.installedLock.slice(0, 80))}`)
  // node_modules is ignored, so no status check sees an install older than the code: compare the lockfile itself.
  const checkLock = () => {
    const lock = git(["rev-parse", `HEAD:${LOCKFILE}`]).trim()
    if (lock !== deps.installedLock) {
      throw new Error(`main's ${LOCKFILE} (${lock.slice(0, 12)}) is not the one npm ci installed (${deps.installedLock.slice(0, 12)}): refusing to build or commit data/meta on an old install; the next run reinstalls`)
    }
  }
  for (let attempt = 1; ; attempt++) {
    // The commit goes on top of main as fetched, never on a local commit no one has checked.
    const head = git(["rev-parse", "HEAD"]).trim(), main = git(["rev-parse", "refs/remotes/origin/main"]).trim()
    if (head !== main) throw new Error(`HEAD ${head.slice(0, 12)} is not origin/main ${main.slice(0, 12)}: refusing to commit on top of it`)
    checkLock()

    // 1. Stage only data/meta; every staged path must be one of the gate's files.
    git(["add", "--all", "--", ":(top,literal)data/meta"])
    const staged = lines(git(["diff", "--cached", "--name-only", "--no-renames", "-z"]), "\0")
    const errors = metaPathErrors(staged)
    if (errors.length > 0) throw new Error(`refusing to commit: ${errors.join("; ")}`)
    // Each staged change adds or modifies a regular, non-executable file: no deletion, no symlink, no gitlink, no mode change.
    const raw = lines(git(["diff", "--cached", "--raw", "--no-renames", "--no-abbrev", "-z"]), "\0")
    for (let i = 0; i < raw.length; i += 2) {
      const [oldMode, newMode, , , status] = raw[i]!.slice(1).split(" ")
      const ok = newMode === "100644" && ((status === "A" && oldMode === "000000") || (status === "M" && oldMode === "100644"))
      if (!ok) throw new Error(`refusing to commit: ${JSON.stringify(raw[i + 1])} is ${status} ${oldMode} → ${newMode}, not a regular file added or modified`)
    }

    // 2. Nothing else may differ from HEAD: no unstaged edit, no untracked file (npm ci must touch nothing).
    const other = lines(git(["status", "--porcelain=v1", "-z", "--untracked-files=all", "--no-renames"]), "\0")
      .filter(e => !(/^[AM] $/.test(e.slice(0, 2)) && META_PATH_RE.test(e.slice(3))))
    if (other.length > 0) throw new Error(`refusing to commit: the tree has other changes: ${other.map(e => JSON.stringify(e)).join(", ")}`)

    if (staged.length === 0) { log("data/meta unchanged: nothing to commit"); return "unchanged" }

    // 3. Commit as the Actions bot (author and committer).
    const posteriorText = git(["show", `:${POSTERIOR}`])
    const posterior = JSON.parse(posteriorText) as { mode?: unknown; page?: { eligibleLives?: unknown } }
    const n = posterior.page?.eligibleLives
    if (typeof posterior.mode !== "string" || !Number.isSafeInteger(n)) throw new Error(`${POSTERIOR} has no mode or page.eligibleLives`)
    const sha8 = createHash("sha256").update(posteriorText, "utf8").digest("hex").slice(0, 8)
    const message = `meta: ${posterior.mode}, ${n} eligible lives, posterior ${sha8}`
    git(["-c", `user.name=${BOT}`, "-c", `user.email=${BOT_EMAIL}`, "-c", "commit.gpgsign=false",
      "commit", "-q", "--no-verify", `--author=${BOT} <${BOT_EMAIL}>`, "-m", message])
    log(`committed ${message} (${staged.join(", ")})`)

    // 4. Push, never forced.
    try {
      git(["push", "origin", "HEAD:main"])
      log(`pushed to main (try ${attempt})`)
      return "committed"
    } catch (e) {
      if (!isBehind(e)) throw e
      if (attempt >= maxTries) throw new Error(`push rejected ${attempt} times because main kept moving (${maxTries} tries): giving up, the next run rebuilds`, { cause: e })
      log(`push rejected (main moved, try ${attempt}): resetting to the new main and rebuilding`)
    }
    git(["fetch", "-q", "origin", "main"])
    git(["reset", "-q", "--hard", "refs/remotes/origin/main"])
    git(["clean", "-q", "-f", "-d", "--", ":(top,literal)data/meta"]) // an untracked file of the old build is stale too
    checkLock() // before running the new main's meta-cli at all
    if (!(await deps.rebuild())) { log("the rebuild on the new main changed nothing"); return "unchanged" }
  }
}

/** git in the current directory; a failure's message carries git's stderr (commitMeta reads it to tell a rejected push). */
function runGit(args: string[]): string {
  try {
    return execFileSync("git", args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024, stdio: ["ignore", "pipe", "pipe"] })
  } catch (e) {
    const err = e as { stderr?: string | Buffer; message: string }
    throw new Error(`git ${args.join(" ")} failed: ${String(err.stderr ?? "").trim() || err.message}`, { cause: e })
  }
}

/** meta-cli.ts, run afresh in this checkout (the new main's code, config and logs); its `changed=` line is the answer. */
async function rebuildMeta(): Promise<boolean> {
  const out = execFileSync("npx", ["tsx", "tools/gate/meta-cli.ts"], { encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] })
  const m = /^changed=(true|false)$/m.exec(out)
  if (!m) throw new Error(`meta-cli.ts printed no changed= line: ${JSON.stringify(out.slice(-200))}`)
  return m[1] === "true"
}

async function main(): Promise<void> {
  // gate.yml records it right after npm ci (git hash-object package-lock.json).
  const installedLock = process.env.LOCK_BLOB
  if (!installedLock) throw new Error("LOCK_BLOB env var is required: the blob sha of the package-lock.json npm ci installed from")
  const result = await commitMeta({ git: runGit, installedLock, rebuild: rebuildMeta, log: m => console.log(m) })
  console.log(`meta=${result}`)
  if (process.env.GITHUB_OUTPUT) appendFileSync(process.env.GITHUB_OUTPUT, `meta=${result}\n`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main()
