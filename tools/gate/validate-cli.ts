// validate.yml entry point. Runs main's copy of this script; PR files are fetched through the API as data.
// Env: REPO (owner/name), PR_NUMBER, GH_TOKEN. cwd: a checkout of main (for data/runs/<id>/ previous state).
import { join } from "node:path"
import { validateSessionPr } from "./validate.ts"
import { readLearnerLogs } from "./summary.ts"
import { loadPr } from "./github.ts"

const repo = process.env.REPO
const n = Number(process.env.PR_NUMBER)
if (!repo || !Number.isInteger(n) || n <= 0) { console.error("REPO and PR_NUMBER env vars are required"); process.exit(2) }

const loaded = loadPr(repo, n)
const ids = [...new Set(loaded.files.map(f => /^data\/runs\/(lrn_[0-9a-f]{16})\//.exec(f.path)?.[1]).filter((x): x is string => !!x))]
const previous = ids.flatMap(id => { try { return readLearnerLogs(join("data/runs", id)) } catch { return [] } })
const v = validateSessionPr(loaded.files, previous)
const errors = [...loaded.loadErrors, ...v.errors]
if (errors.length > 0) {
  for (const e of errors) console.error(`::error::${e.replace(/[\r\n]+/g, " ").slice(0, 500)}`)
  process.exit(1)
}
console.log(`PR #${n}: ${loaded.files.length} file(s) valid`)
