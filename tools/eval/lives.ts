// Every learner's merged log files, as the evaluators read them (spec §2.1 LifeInput).
import { existsSync, readdirSync } from "node:fs"
import { join } from "node:path"
import { LEARNER_ID_RE } from "@blanktab/harness/log"
import type { LifeInput } from "@blanktab/evaluators"
import { readLearnerLogs } from "../gate/summary.ts"

/**
 * One LifeInput per learner dir under `runsDir` (a `lrn_…` name), sorted by learnerId: its parseable <n>.json files whose
 * learnerId is the dir's, sorted by seq (readLearnerLogs: heartbeats and junk are skipped). A dir with none is left out;
 * a missing `runsDir` gives none.
 */
export function collectLives(runsDir: string): LifeInput[] {
  if (!existsSync(runsDir)) return []
  const out: LifeInput[] = []
  for (const id of readdirSync(runsDir).sort()) {
    if (!LEARNER_ID_RE.test(id)) continue
    const files = readLearnerLogs(join(runsDir, id)).filter(f => f.learnerId === id)
    if (files.length > 0) out.push({ learnerId: id, files })
  }
  return out
}
