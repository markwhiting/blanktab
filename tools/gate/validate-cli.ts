// validate.yml entry point. Runs main's copy of this script; PR files are read with git (a partial fetch of
// refs/pull/N/head; github.ts loadPrViaGit) as data, never executed.
// Env: REPO (owner/name), PR_NUMBER, GH_TOKEN. cwd: a full-history checkout of main (for data/runs/<id>/ previous state).
import { join } from "node:path"
import { validateSessionPr } from "./validate.ts"
import { readLearnerLogs } from "./summary.ts"
import { isShallow, loadPrViaGit } from "./github.ts"

const repo = process.env.REPO
const n = Number(process.env.PR_NUMBER)
if (!repo || !Number.isInteger(n) || n <= 0) { console.error("REPO and PR_NUMBER env vars are required"); process.exit(2) }
if (isShallow()) { console.error("::error::the checkout of main is shallow (validate.yml needs fetch-depth 0)"); process.exit(2) }

const loaded = loadPrViaGit(repo, n)
// A newer push started its own run (which cancels this one); this one validated nothing, so it does not pass.
if (loaded.moved) { console.error(`::error::PR #${n}'s head moved while this run read it; the run for the new head validates it`); process.exit(1) }
const ids = [...new Set(loaded.files.map(f => /^data\/runs\/(lrn_[0-9a-f]{16})\//.exec(f.path)?.[1]).filter((x): x is string => !!x))]
const previous = ids.flatMap(id => { try { return readLearnerLogs(join("data/runs", id)) } catch { return [] } })
const v = validateSessionPr(loaded.files, previous)
const errors = [...(loaded.baseRef === "main" ? [] : ["Session PRs must target main."]), ...loaded.loadErrors, ...v.errors]
if (errors.length > 0) {
  for (const e of errors) console.error(`::error::${e.replace(/[\r\n]+/g, " ").slice(0, 500)}`)
  process.exit(1)
}
console.log(`PR #${n}: ${loaded.files.length} file(s) valid`)
