// Session log format. Browser-safe: no Node imports. Shared by the tab (writer) and the gate (validator).

/** Schema v1: the flailer's logs (a lifetime in steps). Still accepted by the parser and the gate. */
export interface LogFileV1 {
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

/**
 * One point of the loss curve, at a sleep: the learner's running prediction loss on its own stream vs the copy-last
 * baseline, and (optional: learners from Plan 04a.2 on) its score on its fixed held-out set, which it never trains on
 * (decision 0015 §3): the mean loss there and copy-last's on the same transitions.
 */
export interface CurvePoint { cu: number; steps: number; predLoss: number; baselineLoss: number; heldOut?: HeldOutScore }
export interface HeldOutScore { loss: number; baseline: number }

/** Bound on a held-out loss in a log file (a mean squared error of pixel values in [0, 1] predicted near [0, 1]). */
export const MAX_HELD_OUT_LOSS = 100

/** At most this many curve points per log file (the points since the previous file, thinned if need be). */
export const MAX_CURVE_POINTS = 64

/**
 * Schema v2: a learner that learns. Every v1 field (settings now holds the learner settings, maxStepsPerSec and the
 * seed) plus its compute lived and its loss curve.
 */
export interface LogFileV2 extends Omit<LogFileV1, "schemaVersion"> {
  schemaVersion: 2
  /** Compute units lived (1 CU = 1e9 FLOPs on the learner's backend), cumulative: never decreases across seq. */
  cu: number
  /** The consolidation condition (spec §5): "A" = one set of weights with uniform replay in sleep. */
  condition: "A"
  widths: { conv1: number; conv2: number; enc: number; hidden: number }
  /** Since the previous log file, oldest first; finite values only (a point is omitted while a loss is NaN). */
  curve: CurvePoint[]
  /** Updates skipped for a non-finite loss over the whole life, including those lost to restores. Never decreases. */
  skippedUpdates: number
  /** Times the learner was restored from its checkpoint after RESTORE_AFTER_SKIPS skips in a row. Never decreases. */
  restores: number
  /** CU spent after a restored checkpoint and thrown away by the restore (not in `cu`). Never decreases. */
  lostCU: number
  /**
   * Compute units spent scoring the held-out set (decision 0015 §3), cumulative, not in `cu`. Optional (files written
   * before it existed have none); finite, >= 0, and never decreasing across seq when present.
   */
  evalCU?: number
}

/** Bound on LogFileV2.restores. */
export const MAX_RESTORES = 1000

export type LogFile = LogFileV1 | LogFileV2

/** One row per learner in the published summary, from its highest-seq log file. Built by tools/gate/summary.ts. */
export interface SessionRow {
  learnerId: string
  settings: LogFile["settings"]
  hostSig: LogFile["hostSig"]
  stepsTotal: number
  final: boolean
  codeSha: string
  knowledgeSha: string
  sessionsSeenAtBirth: number
  /** writtenAt of that highest-seq file. Absent in summaries built before it existed. */
  lastWrittenAt?: string
  /** Compute units lived (v2 logs); null for v1 learners. Absent in summaries built before it existed. */
  cu?: number | null
  /** The last curve point's predLoss ÷ baselineLoss (v2 logs), or null without one. Absent in older summaries. */
  lossRatio?: number | null
  /**
   * The held-out ratio (decision 0015 §3): heldOut.loss ÷ heldOut.baseline at the latest curve point that has one, or
   * null without one. Absent in older summaries.
   */
  heldOutRatio?: number | null
}

/** knowledge/summary.json, what the next tab reads. */
export interface Summary { schemaVersion: 1; sessions: SessionRow[] }

export type ParseResult = { ok: true; value: LogFile } | { ok: false; error: string }

export const LEARNER_ID_RE = /^lrn_[0-9a-f]{16}$/

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)
const isNonNegInt = (v: unknown): v is number => isNum(v) && Number.isInteger(v) && v >= 0
const ISO_RE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/ // exactly what Date.prototype.toISOString emits
export const CODE_SHA_RE = /^([0-9a-f]{40}|dev)$/ // GITHUB_SHA, or "dev" for local builds
export const KNOWLEDGE_SHA_RE = /^[0-9a-f]{64}$/ // sha256 hex
const isIso = (v: unknown): v is string => typeof v === "string" && ISO_RE.test(v) && !Number.isNaN(Date.parse(v))

/**
 * A summary row as the page reads it, or null if it is unusable (the page drops it rather than fail).
 * Checks only what the page relies on; an invalid lastWrittenAt is dropped, not the row.
 */
export function parseSessionRow(v: unknown): SessionRow | null {
  if (!isObj(v) || typeof v.learnerId !== "string" || !LEARNER_ID_RE.test(v.learnerId)) return null
  if (!isObj(v.settings) || !isObj(v.hostSig) || !isNum(v.hostSig.cores) || typeof v.hostSig.webgpu !== "boolean") return null
  if (!isNum(v.stepsTotal) || !isNum(v.sessionsSeenAtBirth) || typeof v.final !== "boolean") return null
  const { lastWrittenAt, cu, lossRatio, heldOutRatio, ...rest } = v
  return {
    ...(rest as unknown as SessionRow),
    ...(isIso(lastWrittenAt) ? { lastWrittenAt } : {}),
    // Numbers or null: anything else (absent, a string, NaN as null) reads as null.
    cu: isNum(cu) && cu >= 0 ? cu : null,
    lossRatio: isNum(lossRatio) && lossRatio >= 0 ? lossRatio : null,
    heldOutRatio: isNum(heldOutRatio) && heldOutRatio >= 0 ? heldOutRatio : null,
  }
}

const MAX_SETTING_STRING = 200
const MAX_SETTING_KEYS = 50
/** Generous bound on a logged width, so a hostile file cannot claim absurd sizes. */
const MAX_WIDTH = 65_536
export const SETTING_KEY_RE = /^[A-Za-z][A-Za-z0-9_]{0,63}$/
const LOG_KEYS_V1 = ["schemaVersion", "learnerId", "seq", "codeSha", "knowledgeSha", "writtenAt", "stepsTotal", "settings", "hostSig", "sessionsSeenAtBirth", "final"]
const LOG_KEYS_V2 = [...LOG_KEYS_V1, "cu", "condition", "widths", "curve", "skippedUpdates", "restores", "lostCU", "evalCU"]
const HOST_KEYS = ["cores", "deviceMemoryGb", "webgpu", "tabsOnMachine"]
const WIDTH_KEYS = ["conv1", "conv2", "enc", "hidden"] as const
const CURVE_KEYS = ["cu", "steps", "predLoss", "baselineLoss", "heldOut"]
const HELD_OUT_KEYS = ["loss", "baseline"]
const unknownKey = (o: Record<string, unknown>, allowed: readonly string[], where: string): string | null => {
  const k = Object.keys(o).find(key => !allowed.includes(key))
  return k === undefined ? null : `unknown key ${JSON.stringify(k.slice(0, 80))} in ${where}`
}
const isNonNeg = (v: unknown): v is number => isNum(v) && v >= 0

export function parseLogFile(json: unknown): ParseResult {
  const fail = (error: string): ParseResult => ({ ok: false, error })
  if (!isObj(json)) return fail("log must be a JSON object")
  if (json.schemaVersion !== 1 && json.schemaVersion !== 2) return fail("schemaVersion must be 1 or 2")
  const v2 = json.schemaVersion === 2
  const uk = unknownKey(json, v2 ? LOG_KEYS_V2 : LOG_KEYS_V1, "log")
  if (uk) return fail(uk)
  if (typeof json.learnerId !== "string" || !LEARNER_ID_RE.test(json.learnerId)) return fail("learnerId must be lrn_ + 16 hex chars")
  if (!isNonNegInt(json.seq)) return fail("seq must be a non-negative finite integer")
  if (typeof json.codeSha !== "string" || !CODE_SHA_RE.test(json.codeSha)) return fail("codeSha must be 40 lowercase hex chars or \"dev\"")
  if (typeof json.knowledgeSha !== "string" || !KNOWLEDGE_SHA_RE.test(json.knowledgeSha)) return fail("knowledgeSha must be 64 lowercase hex chars")
  if (!isIso(json.writtenAt)) return fail("writtenAt must be an ISO time string")
  if (!isNonNegInt(json.stepsTotal)) return fail("stepsTotal must be a non-negative finite integer")
  if (!isObj(json.settings)) return fail("settings must be an object")
  const entries = Object.entries(json.settings)
  if (entries.length > MAX_SETTING_KEYS) return fail(`settings may have at most ${MAX_SETTING_KEYS} keys`)
  for (const [k, v] of entries) {
    if (!SETTING_KEY_RE.test(k)) return fail(`settings key ${JSON.stringify(k.slice(0, 80))} is not allowed (must match ${SETTING_KEY_RE})`)
    if (typeof v === "string") {
      if (v.length > MAX_SETTING_STRING) return fail(`settings.${k} string exceeds ${MAX_SETTING_STRING} chars`)
      continue
    }
    if (!isNum(v)) return fail(`settings.${k} must be a string or a finite number`)
  }
  const h = json.hostSig
  if (!isObj(h)) return fail("hostSig must be an object")
  const hk = unknownKey(h, HOST_KEYS, "hostSig")
  if (hk) return fail(hk)
  if (!isNum(h.cores)) return fail("hostSig.cores must be a finite number")
  if (h.deviceMemoryGb !== null && !isNum(h.deviceMemoryGb)) return fail("hostSig.deviceMemoryGb must be a finite number or null")
  if (typeof h.webgpu !== "boolean") return fail("hostSig.webgpu must be a boolean")
  if (!isNum(h.tabsOnMachine)) return fail("hostSig.tabsOnMachine must be a finite number")
  if (!isNonNegInt(json.sessionsSeenAtBirth)) return fail("sessionsSeenAtBirth must be a non-negative finite integer")
  if (typeof json.final !== "boolean") return fail("final must be a boolean")
  const v1: LogFileV1 = {
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
  }
  if (!v2) return { ok: true, value: v1 }

  if (!isNonNeg(json.cu)) return fail("cu must be a finite number >= 0")
  const cu = json.cu
  if (json.condition !== "A") return fail("condition must be \"A\"")
  const w = json.widths
  if (!isObj(w)) return fail("widths must be an object")
  const wk = unknownKey(w, WIDTH_KEYS, "widths")
  if (wk) return fail(wk)
  for (const k of WIDTH_KEYS) {
    const x = w[k]
    if (!(isNonNegInt(x) && x >= 1 && x <= MAX_WIDTH)) return fail(`widths.${k} must be an integer in [1, ${MAX_WIDTH}]`)
  }
  if (!Array.isArray(json.curve)) return fail("curve must be an array")
  if (json.curve.length > MAX_CURVE_POINTS) return fail(`curve may have at most ${MAX_CURVE_POINTS} points (has ${json.curve.length})`)
  const curve: CurvePoint[] = []
  for (const [i, p] of (json.curve as unknown[]).entries()) {
    if (!isObj(p)) return fail(`curve[${i}] must be an object`)
    const pk = unknownKey(p, CURVE_KEYS, `curve[${i}]`)
    if (pk) return fail(pk)
    if (!isNonNeg(p.cu) || !isNonNegInt(p.steps) || !isNonNeg(p.predLoss) || !isNonNeg(p.baselineLoss)) {
      return fail(`curve[${i}]: cu, predLoss and baselineLoss must be finite numbers >= 0 and steps a non-negative integer`)
    }
    const prev = curve[i - 1]
    if (prev && (p.cu < prev.cu || p.steps < prev.steps)) return fail(`curve[${i}] goes back in cu or steps`)
    if (p.cu > cu || p.steps > json.stepsTotal) return fail(`curve[${i}] is ahead of the file's cu or stepsTotal`)
    let heldOut: HeldOutScore | undefined
    if ("heldOut" in p) {
      const h = p.heldOut
      if (!isObj(h)) return fail(`curve[${i}].heldOut must be an object`)
      const hk = unknownKey(h, HELD_OUT_KEYS, `curve[${i}].heldOut`)
      if (hk) return fail(hk)
      if (!(isNonNeg(h.loss) && h.loss <= MAX_HELD_OUT_LOSS && isNonNeg(h.baseline) && h.baseline <= 1)) {
        return fail(`curve[${i}].heldOut: loss must be a finite number in [0, ${MAX_HELD_OUT_LOSS}] and baseline one in [0, 1]`)
      }
      heldOut = { loss: h.loss, baseline: h.baseline }
    }
    curve.push({ cu: p.cu, steps: p.steps, predLoss: p.predLoss, baselineLoss: p.baselineLoss, ...(heldOut ? { heldOut } : {}) })
  }
  if (!isNonNegInt(json.skippedUpdates)) return fail("skippedUpdates must be a non-negative finite integer")
  if (!(isNonNegInt(json.restores) && json.restores <= MAX_RESTORES)) return fail(`restores must be an integer in [0, ${MAX_RESTORES}]`)
  if (!isNonNeg(json.lostCU)) return fail("lostCU must be a finite number >= 0")
  if ("evalCU" in json && !isNonNeg(json.evalCU)) return fail("evalCU must be a finite number >= 0")
  return {
    ok: true,
    value: {
      ...v1,
      schemaVersion: 2,
      cu,
      condition: "A",
      widths: { conv1: w.conv1 as number, conv2: w.conv2 as number, enc: w.enc as number, hidden: w.hidden as number },
      curve,
      skippedUpdates: json.skippedUpdates,
      restores: json.restores,
      lostCU: json.lostCU,
      ...("evalCU" in json ? { evalCU: json.evalCU as number } : {}),
    },
  }
}
