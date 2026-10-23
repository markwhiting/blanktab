// Usage: tsx tools/eval/rescore.ts <id@version> [runsDir]   (run from the repo root; default runsDir data/runs)
// Scores every merged life under any registered life evaluator, beside the pinned search evaluator, so a candidate can
// be compared with the pinned one before a decision (spec §2.2). Writes .local/rescore-<id@version>.tsv; reads config/.
import { mkdirSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { pathToFileURL } from "node:url"
import { getEvaluator, type EvalOutput, type LifeInput } from "@blanktab/evaluators"
import { loadEvalConfig, loadReferences } from "./config.ts"
import { collectLives } from "./lives.ts"

/** The table written, and the pinned search evaluator it was compared with. */
export interface Rescored { path: string; header: string[]; rows: string[][]; pinned: string }

/** A TSV cell: a number as JS prints it, null/undefined empty, text with tabs and newlines made spaces. */
const cell = (v: unknown): string => (v === null || v === undefined ? "" : String(v).replace(/[\t\r\n]+/g, " "))

/** The life's meta block fields the table shows, from its first file that has one ("" without). */
function metaOf(life: LifeInput): { role: string; method: string; genomeSha: string } {
  for (const f of life.files) if (f.schemaVersion === 2 && f.meta) return { role: f.meta.role, method: f.meta.method, genomeSha: f.meta.genomeSha }
  return { role: "", method: "", genomeSha: "" }
}

/**
 * Every life under `runsDir` scored by `idAtVersion` (a registered life evaluator) and by the pinned search evaluator,
 * both against the pinned references of `root`'s config/evaluation.json (an evaluator needing a reference the config
 * does not pin finds it absent). One row per learner, by learnerId: learnerId, role, method, genomeSha, then the
 * evaluator's eligible, reason, score and each of its components (columns named `<id@version>:<field>`), then
 * `pinned:<id@version>:score`. Written to `root`/.local/rescore-<id@version>.tsv.
 */
export function rescore(idAtVersion: string, runsDir: string, root: string): Rescored {
  const e = getEvaluator(idAtVersion)
  if (e.kind !== "life" || !e.evaluateLife) throw new Error(`${idAtVersion} is not a life evaluator: it scores no single life`)
  const config = loadEvalConfig(root)
  const refs = loadReferences(root, config)
  const pinned = getEvaluator(config.search)
  const lives = collectLives(runsDir)
  const outs: EvalOutput[] = lives.map(l => e.evaluateLife!(l, refs))
  const pinnedOuts: EvalOutput[] = lives.map(l => pinned.evaluateLife!(l, refs))
  const comps = [...new Set(outs.flatMap(o => Object.keys(o.components)))].sort()
  const header = [
    "learnerId", "role", "method", "genomeSha",
    ...["eligible", "reason", "score", ...comps].map(k => `${idAtVersion}:${k}`),
    `pinned:${config.search}:score`,
  ]
  const rows = lives.map((l, i) => {
    const o = outs[i]!, m = metaOf(l)
    return [l.learnerId, m.role, m.method, m.genomeSha, String(o.eligible), o.reason ?? "", o.score, ...comps.map(k => o.components[k]), pinnedOuts[i]!.score].map(cell)
  })
  const dir = join(root, ".local")
  mkdirSync(dir, { recursive: true })
  const path = join(dir, `rescore-${idAtVersion}.tsv`)
  writeFileSync(path, [header, ...rows].map(r => r.join("\t")).join("\n") + "\n")
  return { path, header, rows, pinned: config.search }
}

async function main(): Promise<void> {
  const [id, runsDir = "data/runs"] = process.argv.slice(2)
  if (!id) { console.error("usage: rescore.ts <id@version> [runsDir]"); process.exit(2) }
  const r = rescore(id, runsDir, process.cwd())
  const at = (name: string) => r.header.indexOf(name)
  const pinnedCol = r.header.length - 1
  for (const row of r.rows) console.log(`${row[0]}  ${id}: ${row[at(`${id}:score`)] || `- (${row[at(`${id}:reason`)]})`}  pinned ${r.pinned}: ${row[pinnedCol] || "-"}`)
  const eligible = r.rows.filter(row => row[at(`${id}:eligible`)] === "true").length
  console.log(`wrote ${r.path}: ${r.rows.length} lives, ${eligible} eligible under ${id}`)
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) await main()
