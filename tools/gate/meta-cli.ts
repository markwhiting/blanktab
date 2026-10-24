// Usage: tsx tools/gate/meta-cli.ts   (run from the root of a full-history checkout of main, after the gate's merges)
// Builds data/meta/** from main's merged logs and pinned config (buildMeta), writes the files that changed, appends the
// index line when the posterior's bytes changed, and prints `changed=true|false`. It reads config/ and knowledge/ and
// never writes them; committing is commit-meta.ts's (Task 11).
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { parseCentre } from "@blanktab/meta"
import { loadEvalConfig, loadReferences } from "../eval/config.ts"
import { loadGenomeSpace } from "./gate.ts"
import { git, isShallow } from "./github.ts"
import { buildMeta, CALIBRATION_PATH, parseCalibration, POSTERIOR_INDEX_PATH, POSTERIOR_PATH, posteriorHistory } from "./meta.ts"

const root = git(["rev-parse", "--show-toplevel"]).trim()
// Without the full history, posteriorHistory misses old posteriors and every draw made under one would be flagged.
if (isShallow()) { console.error("::error::the checkout is shallow (meta-cli needs main's full history)"); process.exit(2) }
const read = (rel: string): string | null => (existsSync(join(root, rel)) ? readFileSync(join(root, rel), "utf8") : null)

const config = loadEvalConfig(root)
const space = loadGenomeSpace(root)
const centre = parseCentre(JSON.parse(readFileSync(join(root, "knowledge/genome.json"), "utf8")), space)
const calibrationText = read(CALIBRATION_PATH)
const posteriorText = read(POSTERIOR_PATH)

const files = await buildMeta({
  runsDir: join(root, "data/runs"), config, refs: loadReferences(root, config), space, centre,
  sourceCommit: git(["rev-parse", "HEAD"]).trim(),
  builtAt: new Date().toISOString(), // the one clock read: it goes into the index line only
  previous: { calibration: calibrationText === null ? null : parseCalibration(calibrationText), posteriorText },
  posteriorsBySha: posteriorHistory(git),
})

let changed = false
for (const [rel, text] of [[POSTERIOR_PATH, files[POSTERIOR_PATH]], [CALIBRATION_PATH, files[CALIBRATION_PATH]]] as const) {
  if (read(rel) === text) continue
  mkdirSync(dirname(join(root, rel)), { recursive: true })
  writeFileSync(join(root, rel), text)
  changed = true
}
if (files.indexLine !== null) { appendFileSync(join(root, POSTERIOR_INDEX_PATH), files.indexLine + "\n"); changed = true }
console.log(`changed=${changed}`)
