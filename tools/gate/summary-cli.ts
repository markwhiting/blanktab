// Usage: tsx tools/gate/summary-cli.ts <runsDir> <outFile>   (run from the repo root: it reads config/, never writes it)
// Each row is scored by the pinned search evaluator (config/evaluation.json) against the pinned references; an absent
// reference leaves every row not eligible (decision 0020).
import { mkdirSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"
import { getEvaluator } from "@blanktab/evaluators"
import { loadEvalConfig, loadReferences } from "../eval/config.ts"
import { buildSummary } from "./summary.ts"

const [runsDir, out] = process.argv.slice(2)
if (!runsDir || !out) { console.error("usage: summary-cli.ts <runsDir> <outFile>"); process.exit(2) }
const root = process.cwd()
const config = loadEvalConfig(root)
const refs = loadReferences(root, config)
const evaluator = getEvaluator(config.search)
const summary = buildSummary(runsDir, { evaluator, refs })
mkdirSync(dirname(out), { recursive: true })
writeFileSync(out, JSON.stringify(summary) + "\n")
const eligible = summary.sessions.filter(r => r.evals?.[config.search]?.eligible).length
console.log(`wrote ${out}: ${summary.sessions.length} rows, ${eligible} eligible under ${config.search}${refs.newborn ? "" : " (no newborn reference pinned yet)"}`)
