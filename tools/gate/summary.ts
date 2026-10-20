// Builds the shared summary the next tab reads: one row per learner, from its highest-seq log file.
import { readdirSync, readFileSync, existsSync } from "node:fs"
import { join } from "node:path"
import { parseLogFile, LEARNER_ID_RE, type LogFile, type SessionRow, type Summary } from "@blanktab/harness/log"

export type { SessionRow, Summary }

/** Reads every parseable <n>.json in a learner dir (heartbeat and junk are skipped). */
export function readLearnerLogs(dir: string): LogFile[] {
  const out: LogFile[] = []
  for (const name of readdirSync(dir)) {
    if (!/^\d+\.json$/.test(name)) continue
    try {
      const p = parseLogFile(JSON.parse(readFileSync(join(dir, name), "utf8")))
      if (p.ok) out.push(p.value)
    } catch { /* unreadable file: skip */ }
  }
  return out.sort((a, b) => a.seq - b.seq)
}

export function buildSummary(runsDir: string): Summary {
  const sessions: SessionRow[] = []
  if (existsSync(runsDir)) {
    for (const id of readdirSync(runsDir).sort()) {
      if (!LEARNER_ID_RE.test(id)) continue
      const logs = readLearnerLogs(join(runsDir, id)).filter(l => l.learnerId === id)
      const last = logs[logs.length - 1]
      if (!last) continue
      sessions.push({
        learnerId: id, settings: last.settings, hostSig: last.hostSig, stepsTotal: last.stepsTotal,
        final: last.final, codeSha: last.codeSha, knowledgeSha: last.knowledgeSha, sessionsSeenAtBirth: last.sessionsSeenAtBirth,
      })
    }
  }
  return { schemaVersion: 1, sessions }
}
