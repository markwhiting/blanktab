// Session log format. Browser-safe: no Node imports. Shared by the tab (writer) and the gate (validator).

export interface LogFile {
  schemaVersion: 1
  learnerId: string // "lrn_" + 16 hex chars
  seq: number // 0,1,2,... per learner
  codeSha: string // commit the page was built from
  knowledgeSha: string // sha256 hex of the prior files the session loaded
  writtenAt: string // ISO time
  stepsTotal: number // cumulative, never decreases across seq
  settings: Record<string, number | string>
  hostSig: { cores: number; deviceMemoryGb: number | null; webgpu: boolean; tabsOnMachine: number }
  sessionsSeenAtBirth: number
  final: boolean
}

export type ParseResult = { ok: true; value: LogFile } | { ok: false; error: string }

export const LEARNER_ID_RE = /^lrn_[0-9a-f]{16}$/

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)
const isNonNegInt = (v: unknown): v is number => isNum(v) && Number.isInteger(v) && v >= 0
const isIso = (v: unknown): v is string => typeof v === "string" && v.length > 0 && !Number.isNaN(Date.parse(v))

const MAX_SETTING_STRING = 200
const LOG_KEYS = ["schemaVersion", "learnerId", "seq", "codeSha", "knowledgeSha", "writtenAt", "stepsTotal", "settings", "hostSig", "sessionsSeenAtBirth", "final"]
const HOST_KEYS = ["cores", "deviceMemoryGb", "webgpu", "tabsOnMachine"]
const HEARTBEAT_KEYS = ["learnerId", "writtenAt"]
const unknownKey = (o: Record<string, unknown>, allowed: string[], where: string): string | null => {
  const k = Object.keys(o).find(key => !allowed.includes(key))
  return k === undefined ? null : `unknown key ${JSON.stringify(k)} in ${where}`
}

export function parseLogFile(json: unknown): ParseResult {
  const fail = (error: string): ParseResult => ({ ok: false, error })
  if (!isObj(json)) return fail("log must be a JSON object")
  const uk = unknownKey(json, LOG_KEYS, "log")
  if (uk) return fail(uk)
  if (json.schemaVersion !== 1) return fail("schemaVersion must be 1")
  if (typeof json.learnerId !== "string" || !LEARNER_ID_RE.test(json.learnerId)) return fail("learnerId must be lrn_ + 16 hex chars")
  if (!isNonNegInt(json.seq)) return fail("seq must be a non-negative finite integer")
  if (typeof json.codeSha !== "string" || json.codeSha === "") return fail("codeSha must be a non-empty string")
  if (typeof json.knowledgeSha !== "string" || json.knowledgeSha === "") return fail("knowledgeSha must be a non-empty string")
  if (!isIso(json.writtenAt)) return fail("writtenAt must be an ISO time string")
  if (!isNonNegInt(json.stepsTotal)) return fail("stepsTotal must be a non-negative finite integer")
  if (!isObj(json.settings)) return fail("settings must be an object")
  for (const [k, v] of Object.entries(json.settings)) {
    if (typeof v === "string") {
      if (v.length > MAX_SETTING_STRING) return fail(`settings.${k} string exceeds ${MAX_SETTING_STRING} chars`)
      continue
    }
    if (!isNum(v)) return fail(`settings.${k} must be a string or a finite number`)
  }
  const h = json.hostSig
  if (!isObj(h)) return fail("hostSig must be an object")
  const hk = unknownKey(h, HOST_KEYS, "hostSig")
  if (hk) return { ok: false, error: hk }
  if (!isNum(h.cores)) return fail("hostSig.cores must be a finite number")
  if (h.deviceMemoryGb !== null && !isNum(h.deviceMemoryGb)) return fail("hostSig.deviceMemoryGb must be a finite number or null")
  if (typeof h.webgpu !== "boolean") return fail("hostSig.webgpu must be a boolean")
  if (!isNum(h.tabsOnMachine)) return fail("hostSig.tabsOnMachine must be a finite number")
  if (!isNonNegInt(json.sessionsSeenAtBirth)) return fail("sessionsSeenAtBirth must be a non-negative finite integer")
  if (typeof json.final !== "boolean") return fail("final must be a boolean")
  return {
    ok: true,
    value: {
      schemaVersion: 1,
      learnerId: json.learnerId,
      seq: json.seq,
      codeSha: json.codeSha,
      knowledgeSha: json.knowledgeSha,
      writtenAt: json.writtenAt,
      stepsTotal: json.stepsTotal,
      settings: { ...(json.settings as Record<string, number | string>) },
      hostSig: { cores: h.cores, deviceMemoryGb: h.deviceMemoryGb as number | null, webgpu: h.webgpu, tabsOnMachine: h.tabsOnMachine },
      sessionsSeenAtBirth: json.sessionsSeenAtBirth,
      final: json.final,
    },
  }
}

/** Heartbeat file shape: { learnerId, writtenAt }. */
export function parseHeartbeat(json: unknown): { ok: true; value: { learnerId: string; writtenAt: string } } | { ok: false; error: string } {
  if (!isObj(json)) return { ok: false, error: "heartbeat must be a JSON object" }
  const uk = unknownKey(json, HEARTBEAT_KEYS, "heartbeat")
  if (uk) return { ok: false, error: uk }
  if (typeof json.learnerId !== "string" || !LEARNER_ID_RE.test(json.learnerId)) return { ok: false, error: "heartbeat learnerId must be lrn_ + 16 hex chars" }
  if (!isIso(json.writtenAt)) return { ok: false, error: "heartbeat writtenAt must be an ISO time string" }
  return { ok: true, value: { learnerId: json.learnerId, writtenAt: json.writtenAt } }
}
