// Gate validator for session PRs. Runs in Node (GitHub Action) on PR files treated purely as data.
import { parseLogFile, parseHeartbeat, type LogFile } from "@blanktab/harness/log"

export interface PrFile { path: string; content: string }

const MAX_BYTES = 1_048_576
const PATH_RE = /^data\/runs\/(lrn_[0-9a-f]{16})\/(\d+\.json|heartbeat\.json)$/

export function validateSessionPr(files: PrFile[], previous: LogFile[]): { ok: boolean; errors: string[]; kind: "log" | "needs-human" } {
  const errors: string[] = []
  let kind: "log" | "needs-human" = "log"
  if (files.length === 0) return { ok: false, errors: ["PR contains no files"], kind }

  const dirIds = new Set<string>()
  const logs: { path: string; log: LogFile }[] = []

  for (const f of files) {
    const m = PATH_RE.exec(f.path)
    if (!m) {
      kind = "needs-human"
      errors.push(`${f.path}: path is not a session log path (data/runs/<learnerId>/<n>.json or heartbeat.json)`)
      continue
    }
    const dirId = m[1]!
    dirIds.add(dirId)
    if (new TextEncoder().encode(f.content).length > MAX_BYTES) {
      errors.push(`${f.path}: file exceeds 1 MB`)
      continue
    }
    let json: unknown
    try { json = JSON.parse(f.content) } catch { errors.push(`${f.path}: invalid JSON`); continue }

    if (m[2] === "heartbeat.json") {
      const hb = parseHeartbeat(json)
      if (!hb.ok) errors.push(`${f.path}: ${hb.error}`)
      else if (hb.value.learnerId !== dirId) errors.push(`${f.path}: path learnerId ${dirId} does not match learnerId ${hb.value.learnerId} in file`)
      continue
    }
    const p = parseLogFile(json)
    if (!p.ok) { errors.push(`${f.path}: ${p.error}`); continue }
    if (p.value.learnerId !== dirId) errors.push(`${f.path}: path learnerId ${dirId} does not match learnerId ${p.value.learnerId} in file`)
    if (`${p.value.seq}.json` !== m[2]) errors.push(`${f.path}: filename does not match seq ${p.value.seq}`)
    logs.push({ path: f.path, log: p.value })
  }

  if (dirIds.size > 1) errors.push(`all files must be for the same learnerId (found ${[...dirIds].join(", ")})`)

  // seq: unique and continuing from previous
  const seen = new Set<number>()
  for (const { path, log } of logs) {
    if (seen.has(log.seq)) errors.push(`${path}: duplicate seq ${log.seq}`)
    seen.add(log.seq)
  }
  const sorted = [...logs].sort((a, b) => a.log.seq - b.log.seq)
  const prevSorted = [...previous].sort((a, b) => a.seq - b.seq)
  const prevLast = prevSorted[prevSorted.length - 1]
  let expected = prevLast ? prevLast.seq + 1 : 0
  let lastSteps = prevLast ? prevLast.stepsTotal : -Infinity
  const reported = new Set<number>()
  for (const { path, log } of sorted) {
    if (reported.has(log.seq)) continue // duplicate already reported
    reported.add(log.seq)
    if (log.seq !== expected) {
      errors.push(`${path}: expected seq ${expected} but got ${log.seq}`)
      expected = log.seq
    }
    expected++
    if (log.stepsTotal < lastSteps) errors.push(`${path}: stepsTotal decreased (${log.stepsTotal} < ${lastSteps})`)
    lastSteps = Math.max(lastSteps, log.stepsTotal)
  }

  // final: at most one, and it has the highest seq
  const finals = sorted.filter(l => l.log.final)
  if (finals.length > 1) errors.push(`at most one final file allowed (found ${finals.length})`)
  else if (finals.length === 1 && finals[0] !== sorted[sorted.length - 1]) errors.push(`${finals[0]!.path}: final file must have the highest seq`)

  return { ok: errors.length === 0, errors, kind }
}
