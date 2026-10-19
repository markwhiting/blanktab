// Usage: tsx tools/gate/summary-cli.ts <runsDir> <outFile>
import { mkdirSync, writeFileSync } from "node:fs"
import { dirname } from "node:path"
import { buildSummary } from "./summary.ts"

const [runsDir, out] = process.argv.slice(2)
if (!runsDir || !out) { console.error("usage: summary-cli.ts <runsDir> <outFile>"); process.exit(2) }
mkdirSync(dirname(out), { recursive: true })
writeFileSync(out, JSON.stringify(buildSummary(runsDir)) + "\n")
console.log(`wrote ${out}`)
