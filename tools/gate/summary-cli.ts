// Usage: tsx tools/gate/summary-cli.ts <runsDir> <outFile>   (run from the repo root: it reads config/, never writes it)
// Each row is scored by the pinned search evaluator (config/evaluation.json) against the pinned references; an absent
// reference leaves every row not eligible (decision 0020).
import { mkdirSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"
import { buildSummary, pinnedScoring } from "./summary.ts"

const [runsDir, out] = process.argv.slice(2)
if (!runsDir || !out) { console.error("usage: summary-cli.ts <runsDir> <outFile>"); process.exit(2) }
const scoring = pinnedScoring(process.cwd())
const summary = buildSummary(runsDir, scoring)
mkdirSync(dirname(out), { recursive: true })
writeFileSync(out, JSON.stringify(summary) + "\n")
const key = `${scoring.evaluator.id}@${scoring.evaluator.version}`
const eligible = summary.sessions.filter(r => r.evals?.[key]?.eligible).length
console.log(`wrote ${out}: ${summary.sessions.length} rows, ${eligible} eligible under ${key}${scoring.refs.newborn ? "" : " (no newborn reference pinned yet)"}`)
