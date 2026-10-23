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

/**
 * What a curve point's losses must be (predLoss, baselineLoss, heldOut.loss, heldOut.baseline): a finite number >= 0.
 * The one rule, shared by the writer (session curvePoint) and the parser, so a file the tab writes always parses. No
 * upper bound: a diverged learner's result is kept on the record (AGENTS.md invariant 7).
 */
export const isCurveLoss = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0

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
  /** Milestone checkups run since the previous file (strictly increasing levels, each on the log schedule and <= cu). */
  checkups?: CheckupResult[]
  /** Which suite the checkups used (MILESTONE_IDS key). Required when `checkups` is present. */
  milestoneSuiteVersion?: number
  /** The caregiver's activity since the previous file. */
  caregiver?: CaregiverSummary
  /** How this life's genome was drawn, and the genome (spec §7.3; Plan 05a). Absent in files from older code. */
  meta?: MetaBlock
}

/** Who a life is (spec §7.3): a search draw, a confirmation life, a newborn reference, or a life born before 05a. */
export type MetaRole = "search" | "confirm" | "newborn" | "legacy"
/** How its genome was chosen (spec §4.1). */
export type MetaMethod = "centre" | "calibrate" | "uniform" | "ts" | "confirm" | "newborn" | "legacy"
/** Why a final file ended its life early or on budget (set by final.ts; spec §7.3). */
export type MetaEnd = "failed" | "orphan" | "budget"
/**
 * The `meta` block (spec §7.3), schema v1. Every field but `filters`, `heldOut` and `end` is constant across a learner's
 * files (the gate checks). `genome` is what ran; `genomeSha` is genomeSha(genomeSchema, genome) (@blanktab/meta).
 */
export interface MetaBlock {
  v: 1; role: MetaRole; method: MetaMethod; genomeSchema: number; genome: Record<string, number | string>; genomeSha: string
  drawSeed: number | null; posteriorSha: string | null; evalPanel: number | null; evalSeed: number | null
  confirmation: { id: number; arm: string } | null; inheritance: { filters: "none" }
  /**
   * Role "legacy" only (decision 0022): the genes the life's settings did not record (legacyGenome's `missing`): their
   * values in `genome` are placeholders, not claims. Sorted, unique, each a key of `genome`.
   */
  missing?: string[]
  /** base64 of FILTER_FLOATS_PER_CONV1·widths.conv1 float32 LE: periphery conv1 w,b then fovea conv1 w,b (Model.params order). */
  filters?: string
  /** Role "newborn" only: its one held-out score at birth (plan gap 5). */
  heldOut?: HeldOutScore
  /** Final files only. */
  end?: MetaEnd
}
/** A search life's evaluation budget (spec §4.2): its first 2,048 CU and the checkups at 512, 1,024 and 2,048. */
export const BUDGET_CU = 2048
/** The log levels whose file carries `meta.filters` (spec §5.1). */
export const FILTER_LEVELS: readonly number[] = [512, 1024, 2048]
/** A newborn reference's checkups (spec §7.3). */
export const NEWBORN_LEVELS: readonly number[] = [512, 1024, 2048]
/** Floats in `meta.filters` per unit of widths.conv1: periphery 5·5·3 weights + 1 bias, fovea 3·3·3 + 1 (spec §5.1). */
export const FILTER_FLOATS_PER_CONV1 = 104
/** The number of shared checkup panels: @blanktab/meta PANELS (log.ts imports no package; log.test.ts pins it). */
export const MAX_EVAL_PANEL = 16
/** Confirmation seeds start here (spec §4.5). */
export const MIN_EVAL_SEED = 10_000

/** The log schedule: the first threshold (2^3 CU: a young life writes no log files in a burst). */
export const FIRST_LOG_CU = 8
/** Doubling stops here (2^9 CU); from then on a log is written at every multiple of LOG_STEP_CU, forever. */
export const LAST_DOUBLING_CU = 512
/** About 8 logs an hour at the speed cap. */
export const LOG_STEP_CU = 1024

/** The log schedule, forever: 8, 16, ..., 512, then 1024, 2048, 3072, ... (every LOG_STEP_CU). */
export function* logThresholds(): Iterable<number> {
  let t = FIRST_LOG_CU
  for (; t <= LAST_DOUBLING_CU; t *= 2) yield t
  for (; ; t += LOG_STEP_CU) yield t
}

/** The first threshold of logThresholds() above `cu` (the next log crossing). */
export function nextCrossing(cu: number): number {
  if (cu >= LAST_DOUBLING_CU) return (Math.floor(cu / LOG_STEP_CU) + 1) * LOG_STEP_CU
  let t = FIRST_LOG_CU
  while (t <= cu) t *= 2
  return t
}

/** Whether `n` is one of logThresholds(). */
export function isLogLevel(n: number): boolean {
  if (!Number.isInteger(n) || n < FIRST_LOG_CU) return false
  if (n <= LAST_DOUBLING_CU) return (n & (n - 1)) === 0 // a power of two from 8 to 512
  return n % LOG_STEP_CU === 0
}

/** The highest log level <= cu, or null below FIRST_LOG_CU. */
export function levelAtOrBelow(cu: number): number | null {
  if (!(cu >= FIRST_LOG_CU)) return null
  if (cu >= LAST_DOUBLING_CU) return Math.floor(cu / LOG_STEP_CU) * LOG_STEP_CU || LAST_DOUBLING_CU
  let t = FIRST_LOG_CU
  while (t * 2 <= cu) t *= 2
  return t
}

/** The milestone suite (decision 0016): version -> milestone ids in suite order. */
export const MILESTONE_SUITE_VERSION = 1
export const MILESTONE_IDS: Readonly<Record<number, readonly string[]>> = { 1: ["M1", "M2", "M3", "M4", "M5", "M6"] }
/** The largest suite size of any known suite (summary rows carry no suite version). */
export const MAX_SUITE_SIZE: number = Math.max(...Object.values(MILESTONE_IDS).map(ids => ids.length))
/** At most this many checkups per log file. */
export const MAX_CHECKUPS_PER_FILE = 8

/** A milestone checkup run at a log-schedule level (decision 0016). */
export interface CheckupResult {
  level: number // the log-schedule CU threshold it is for
  steps: number // the learner's steps when it ran
  passed: string[] // ids passed, in suite order; score = passed.length
  measures: Record<string, number | null> // each milestone's measured value; null if not finite
  evalCU: number // what it cost (inside the file's evalCU)
  error?: true // a milestone threw; it counts as not passed
  /** Why each milestone that threw did (decision 0017): id -> a short reason. Only with `error: true`; absent before 04c. */
  errorReasons?: Record<string, string>
}

/** A milestone's error reason: 1 to 80 printable ASCII characters, so a log stays small: no stack, at most an 80-character message fragment. */
export const ERROR_REASON_RE = /^[\x20-\x7E]{1,80}$/

/**
 * What the caregiver did since the previous file: `carries` in all, `to` by destination, whether the safety net fired.
 * `lostCarries` (decision 0017): carries made since the previous file and then thrown away by a restore or a failed
 * life (they are not in `carries` or `to`). An integer >= 1 when present; absent when none were lost.
 */
export interface CaregiverSummary { carries: number; to: Record<string, number>; safetyNet: boolean; lostCarries?: number }

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
  /** The latest checkup: its level and how many milestones it passed, or null without one. Absent in older summaries. */
  checkupScore?: { level: number; score: number } | null
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
  const { lastWrittenAt, cu, lossRatio, heldOutRatio, checkupScore, ...rest } = v
  return {
    ...(rest as unknown as SessionRow),
    ...(isIso(lastWrittenAt) ? { lastWrittenAt } : {}),
    // Numbers or null: anything else (absent, a string, NaN as null) reads as null.
    cu: isNum(cu) && cu >= 0 ? cu : null,
    lossRatio: isNum(lossRatio) && lossRatio >= 0 ? lossRatio : null,
    heldOutRatio: isNum(heldOutRatio) && heldOutRatio >= 0 ? heldOutRatio : null,
    // A log-schedule level and a score no suite can exceed (the row carries no suite version).
    checkupScore: isObj(checkupScore) && isNonNegInt(checkupScore.level) && isLogLevel(checkupScore.level)
      && isNonNegInt(checkupScore.score) && checkupScore.score <= MAX_SUITE_SIZE
      ? { level: checkupScore.level, score: checkupScore.score } : null,
  }
}

const MAX_SETTING_STRING = 200
const MAX_SETTING_KEYS = 50
/** Generous bound on a logged width, so a hostile file cannot claim absurd sizes. */
const MAX_WIDTH = 65_536
export const SETTING_KEY_RE = /^[A-Za-z][A-Za-z0-9_]{0,63}$/
const LOG_KEYS_V1 = ["schemaVersion", "learnerId", "seq", "codeSha", "knowledgeSha", "writtenAt", "stepsTotal", "settings", "hostSig", "sessionsSeenAtBirth", "final"]
const LOG_KEYS_V2 = [...LOG_KEYS_V1, "cu", "condition", "widths", "curve", "skippedUpdates", "restores", "lostCU", "evalCU", "checkups", "milestoneSuiteVersion", "caregiver", "meta"]
const HOST_KEYS = ["cores", "deviceMemoryGb", "webgpu", "tabsOnMachine"]
const WIDTH_KEYS = ["conv1", "conv2", "enc", "hidden"] as const
const CURVE_KEYS = ["cu", "steps", "predLoss", "baselineLoss", "heldOut"]
const HELD_OUT_KEYS = ["loss", "baseline"]
const CHECKUP_KEYS = ["level", "steps", "passed", "measures", "evalCU", "error", "errorReasons"]
const CAREGIVER_KEYS = ["carries", "to", "safetyNet", "lostCarries"]
const META_KEYS = ["v", "role", "method", "genomeSchema", "genome", "genomeSha", "drawSeed", "posteriorSha", "evalPanel", "evalSeed",
  "confirmation", "inheritance", "missing", "filters", "heldOut", "end"]
const CARE_DEST_RE = /^[a-z]{1,16}$/
const MAX_CARE_DESTS = 16
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
    if (!isNonNeg(p.cu) || !isNonNegInt(p.steps) || !isCurveLoss(p.predLoss) || !isCurveLoss(p.baselineLoss)) {
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
      if (!(isCurveLoss(h.loss) && isCurveLoss(h.baseline))) return fail(`curve[${i}].heldOut: loss and baseline must be finite numbers >= 0`)
      heldOut = { loss: h.loss, baseline: h.baseline }
    }
    curve.push({ cu: p.cu, steps: p.steps, predLoss: p.predLoss, baselineLoss: p.baselineLoss, ...(heldOut ? { heldOut } : {}) })
  }
  if (!isNonNegInt(json.skippedUpdates)) return fail("skippedUpdates must be a non-negative finite integer")
  if (!(isNonNegInt(json.restores) && json.restores <= MAX_RESTORES)) return fail(`restores must be an integer in [0, ${MAX_RESTORES}]`)
  if (!isNonNeg(json.lostCU)) return fail("lostCU must be a finite number >= 0")
  if ("evalCU" in json && !isNonNeg(json.evalCU)) return fail("evalCU must be a finite number >= 0")
  const widths = { conv1: w.conv1 as number, conv2: w.conv2 as number, enc: w.enc as number, hidden: w.hidden as number }
  let meta: MetaBlock | undefined
  if ("meta" in json) {
    const r = parseMeta(json.meta, { cu, stepsTotal: json.stepsTotal, final: json.final, widths })
    if (typeof r === "string") return fail(r)
    meta = r
  }
  const newborn = meta?.role === "newborn"
  if (newborn && json.seq !== 0) return fail("a newborn file (meta.role \"newborn\") must have seq 0")
  let checkups: CheckupResult[] | undefined
  let suiteVersion: number | undefined
  if ("milestoneSuiteVersion" in json) {
    const sv = json.milestoneSuiteVersion
    if (!(isNonNegInt(sv) && Object.hasOwn(MILESTONE_IDS, sv))) return fail("milestoneSuiteVersion is not a known suite")
    suiteVersion = sv
  }
  if ("checkups" in json) {
    // Only a newborn reference checks levels above its cu (0): it is checked at NEWBORN_LEVELS untrained (spec §7.3).
    const r = parseCheckups(json.checkups, suiteVersion, cu, json.stepsTotal, { allowAhead: newborn })
    if (typeof r === "string") return fail(r)
    checkups = r
    // Honesty (decision 0016: checkups are billed to evalCU): a file with checkups carries evalCU, and what they cost
    // fits inside it (evalCU is cumulative, so it holds this file's checkups and more).
    if (checkups.length > 0) {
      if (!("evalCU" in json)) return fail("checkups need the file's evalCU")
      const ev = json.evalCU as number
      const sum = checkups.reduce((a, c) => a + c.evalCU, 0)
      if (sum - ev > EVAL_CU_TOLERANCE * Math.max(ev, sum)) return fail(`checkups' evalCU (${sum}) exceeds the file's evalCU (${ev})`)
    }
  }
  if (newborn && !(checkups && checkups.length === NEWBORN_LEVELS.length && checkups.every((c, i) => c.level === NEWBORN_LEVELS[i]))) {
    return fail(`a newborn file must hold exactly ${NEWBORN_LEVELS.length} checkups, at levels ${NEWBORN_LEVELS.join(", ")}`)
  }
  let caregiver: CaregiverSummary | undefined
  if ("caregiver" in json) {
    const r = parseCaregiver(json.caregiver)
    if (typeof r === "string") return fail(r)
    caregiver = r
  }
  return {
    ok: true,
    value: {
      ...v1,
      schemaVersion: 2,
      cu,
      condition: "A",
      widths,
      curve,
      skippedUpdates: json.skippedUpdates,
      restores: json.restores,
      lostCU: json.lostCU,
      ...("evalCU" in json ? { evalCU: json.evalCU as number } : {}),
      ...(checkups ? { checkups } : {}),
      ...(suiteVersion !== undefined ? { milestoneSuiteVersion: suiteVersion } : {}),
      ...(caregiver ? { caregiver } : {}),
      ...(meta ? { meta } : {}),
    },
  }
}

/** Relative float tolerance for the checkups' evalCU sum against the file's evalCU (and, in the gate, its increase). */
export const EVAL_CU_TOLERANCE = 1e-9

/**
 * The checkups array, or the error string. Strict like the curve: unknown keys, bad values and bad order all fail.
 * Also checks a checkpoint's pending tail (checkpoint.ts), against its own cu and steps.
 */
export function parseCheckups(
  v: unknown, suiteVersion: number | undefined, cu: number, stepsTotal: number, opts: { allowAhead?: boolean } = {},
): CheckupResult[] | string {
  if (!Array.isArray(v)) return "checkups must be an array"
  if (v.length > MAX_CHECKUPS_PER_FILE) return `checkups may have at most ${MAX_CHECKUPS_PER_FILE} entries (has ${v.length})`
  if (suiteVersion === undefined) return "checkups need a milestoneSuiteVersion"
  const ids = MILESTONE_IDS[suiteVersion]!
  const out: CheckupResult[] = []
  for (const [i, c] of (v as unknown[]).entries()) {
    const at = `checkups[${i}]`
    if (!isObj(c)) return `${at} must be an object`
    const uk = unknownKey(c, CHECKUP_KEYS, at)
    if (uk) return uk
    if (!isNonNegInt(c.level) || !isLogLevel(c.level)) return `${at}.level must be a log-schedule level`
    if (c.level > cu && !opts.allowAhead) return `${at}.level is ahead of the file's cu`
    const prev = out[i - 1]
    if (prev && c.level <= prev.level) return `${at}.level must be above the previous checkup's`
    if (!isNonNegInt(c.steps) || c.steps > stepsTotal) return `${at}.steps must be a non-negative integer no more than stepsTotal`
    if (!Array.isArray(c.passed)) return `${at}.passed must be an array`
    let last = -1
    for (const id of c.passed as unknown[]) {
      const idx = typeof id === "string" ? ids.indexOf(id) : -1
      if (idx < 0) return `${at}.passed has an id outside the suite`
      if (idx <= last) return `${at}.passed must be in suite order without duplicates`
      last = idx
    }
    const meas = c.measures
    if (!isObj(meas)) return `${at}.measures must be an object`
    const mk = unknownKey(meas, ids, `${at}.measures`)
    if (mk) return mk
    for (const id of ids) {
      const m = meas[id]
      if (!(m === null || isNum(m))) return `${at}.measures.${id} must be a finite number or null`
    }
    if (!isNonNeg(c.evalCU)) return `${at}.evalCU must be a finite number >= 0`
    if ("error" in c && c.error !== true) return `${at}.error must be true when present`
    let errorReasons: Record<string, string> | undefined
    if ("errorReasons" in c) {
      if (c.error !== true) return `${at}.errorReasons needs error: true`
      const er = c.errorReasons
      if (!isObj(er)) return `${at}.errorReasons must be an object`
      const reasonIds = Object.keys(er)
      if (reasonIds.length === 0) return `${at}.errorReasons must not be empty`
      for (const id of reasonIds) {
        if (!ids.includes(id)) return `${at}.errorReasons has an id outside the suite`
        if (meas[id] !== null) return `${at}.errorReasons.${id} needs a null measure`
        const r = er[id]
        if (typeof r !== "string" || !ERROR_REASON_RE.test(r)) return `${at}.errorReasons.${id} must match ${ERROR_REASON_RE}`
      }
      errorReasons = Object.fromEntries(ids.filter(id => id in er).map(id => [id, er[id] as string]))
    }
    out.push({
      level: c.level, steps: c.steps, passed: [...(c.passed as string[])],
      measures: Object.fromEntries(ids.map(id => [id, meas[id] as number | null])),
      evalCU: c.evalCU, ...(c.error === true ? { error: true as const } : {}),
      ...(errorReasons ? { errorReasons } : {}),
    })
  }
  return out
}

const ROLES: readonly MetaRole[] = ["search", "confirm", "newborn", "legacy"]
/** The methods each role may have (agent's call, decision 0022: a role and its method never disagree). */
const ROLE_METHODS: Readonly<Record<MetaRole, readonly MetaMethod[]>> = {
  search: ["centre", "calibrate", "uniform", "ts"], confirm: ["confirm"], newborn: ["newborn"], legacy: ["legacy"],
}
const ENDS: readonly MetaEnd[] = ["failed", "orphan", "budget"]
/** At most this many genes in `meta.genome` (@blanktab/meta MAX_GENES). */
const MAX_META_GENES = 40
const MAX_GENE_STRING = 64
const CONFIRM_ARM_RE = /^[a-z]{1,16}$/
const BASE64_RE = /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/
const isUint32 = (v: unknown): v is number => isNonNegInt(v) && v <= 0xffff_ffff

/** The byte length `s` decodes to as padded base64, or null if it is not padded base64. Length first: never a long regex run. */
function base64Bytes(s: string, expectBytes: number): number | null {
  if (s.length !== 4 * Math.ceil(expectBytes / 3) || !BASE64_RE.test(s)) return null
  return (s.length / 4) * 3 - (s.endsWith("==") ? 2 : s.endsWith("=") ? 1 : 0)
}

/**
 * The `meta` block (spec §7.3), or what is wrong with it, checked against the file (or checkpoint) it sits in.
 * Structure only, browser-safe: genome bounds, the genomeSha and constancy across files are the gate's (validate.ts).
 * Strict keys; `filters` only in a file whose levelAtOrBelow(cu) is a FILTER_LEVELS level, of exactly
 * FILTER_FLOATS_PER_CONV1·widths.conv1 float32; `heldOut` only for role "newborn"; `missing` only for role "legacy";
 * `end` only in a final file. A newborn's file has cu 0, stepsTotal 0, final true, method "newborn" and a panel.
 */
export function parseMeta(v: unknown, f: { cu: number; stepsTotal: number; final: boolean; widths: LogFileV2["widths"] }): MetaBlock | string {
  if (!isObj(v)) return "meta must be an object"
  const uk = unknownKey(v, META_KEYS, "meta")
  if (uk) return uk
  if (v.v !== 1) return "meta.v must be 1"
  const role = v.role as MetaRole
  if (!ROLES.includes(role)) return `meta.role must be one of ${ROLES.join(", ")}`
  const method = v.method as MetaMethod
  if (!ROLE_METHODS[role].includes(method)) return `meta.method must be one of ${ROLE_METHODS[role].join(", ")} for role ${role}`
  if (!(isNonNegInt(v.genomeSchema) && v.genomeSchema >= 1)) return "meta.genomeSchema must be an integer >= 1"
  const g = v.genome
  if (!isObj(g)) return "meta.genome must be an object"
  const genes = Object.entries(g)
  if (genes.length > MAX_META_GENES) return `meta.genome may have at most ${MAX_META_GENES} genes`
  for (const [k, x] of genes) {
    if (!SETTING_KEY_RE.test(k)) return `meta.genome key ${JSON.stringify(k.slice(0, 80))} is not allowed (must match ${SETTING_KEY_RE})`
    if (typeof x === "string" ? x.length > MAX_GENE_STRING : !isNum(x)) return `meta.genome.${k} must be a finite number or a string of at most ${MAX_GENE_STRING} chars`
  }
  if (typeof v.genomeSha !== "string" || !KNOWLEDGE_SHA_RE.test(v.genomeSha)) return "meta.genomeSha must be 64 lowercase hex chars"
  if (!(v.drawSeed === null || isUint32(v.drawSeed))) return "meta.drawSeed must be a uint32 or null"
  if (!(v.posteriorSha === null || (typeof v.posteriorSha === "string" && KNOWLEDGE_SHA_RE.test(v.posteriorSha)))) return "meta.posteriorSha must be 64 lowercase hex chars or null"
  // A draw has both (seeded by the posterior's sha) or neither (offline, centre; newborn; legacy): birthDraw's rule.
  if ((v.drawSeed === null) !== (v.posteriorSha === null)) return "meta.drawSeed and meta.posteriorSha must be null together"
  if (!(v.evalPanel === null || (isNonNegInt(v.evalPanel) && v.evalPanel >= 1 && v.evalPanel <= MAX_EVAL_PANEL))) return `meta.evalPanel must be an integer in [1, ${MAX_EVAL_PANEL}] or null`
  if (!(v.evalSeed === null || (isNonNegInt(v.evalSeed) && v.evalSeed >= MIN_EVAL_SEED))) return `meta.evalSeed must be an integer >= ${MIN_EVAL_SEED} or null`
  const c = v.confirmation
  let confirmation: MetaBlock["confirmation"] = null
  if (c !== null) {
    if (!isObj(c)) return "meta.confirmation must be { id, arm } or null"
    const ck = unknownKey(c, ["id", "arm"], "meta.confirmation")
    if (ck) return ck
    if (!isNonNegInt(c.id)) return "meta.confirmation.id must be a non-negative integer"
    if (typeof c.arm !== "string" || !CONFIRM_ARM_RE.test(c.arm)) return `meta.confirmation.arm must match ${CONFIRM_ARM_RE}`
    confirmation = { id: c.id, arm: c.arm }
  }
  const inh = v.inheritance
  if (!isObj(inh)) return "meta.inheritance must be { filters: \"none\" }"
  const ik = unknownKey(inh, ["filters"], "meta.inheritance")
  if (ik) return ik
  if (inh.filters !== "none") return "meta.inheritance.filters must be \"none\""
  let missing: string[] | undefined
  if ("missing" in v) {
    if (role !== "legacy") return "meta.missing is only for role legacy"
    const m = v.missing
    if (!Array.isArray(m) || m.length > MAX_META_GENES) return `meta.missing must be an array of at most ${MAX_META_GENES} gene names`
    for (const [i, k] of (m as unknown[]).entries()) {
      if (typeof k !== "string" || !Object.hasOwn(g, k)) return "meta.missing must name genes of meta.genome"
      if (i > 0 && !(k > (m[i - 1] as string))) return "meta.missing must be sorted, without duplicates"
    }
    missing = [...(m as string[])]
  }
  let filters: string | undefined
  if ("filters" in v) {
    const level = levelAtOrBelow(f.cu)
    if (level === null || !FILTER_LEVELS.includes(level)) return `meta.filters is only in a file at ${FILTER_LEVELS.join(", ")} CU`
    const bytes = FILTER_FLOATS_PER_CONV1 * f.widths.conv1 * 4
    if (typeof v.filters !== "string" || base64Bytes(v.filters, bytes) !== bytes) {
      return `meta.filters must be base64 of ${FILTER_FLOATS_PER_CONV1 * f.widths.conv1} float32 (${FILTER_FLOATS_PER_CONV1}·widths.conv1)`
    }
    filters = v.filters
  }
  let heldOut: HeldOutScore | undefined
  if ("heldOut" in v) {
    if (role !== "newborn") return "meta.heldOut is only for role newborn"
    const h = v.heldOut
    if (!isObj(h)) return "meta.heldOut must be an object"
    const hk = unknownKey(h, HELD_OUT_KEYS, "meta.heldOut")
    if (hk) return hk
    if (!(isCurveLoss(h.loss) && isCurveLoss(h.baseline))) return "meta.heldOut: loss and baseline must be finite numbers >= 0"
    heldOut = { loss: h.loss, baseline: h.baseline }
  }
  let end: MetaEnd | undefined
  if ("end" in v) {
    if (!f.final) return "meta.end is only in a final file"
    if (!ENDS.includes(v.end as MetaEnd)) return `meta.end must be one of ${ENDS.join(", ")}`
    end = v.end as MetaEnd
  }
  if (role === "newborn") {
    if (!(f.cu === 0 && f.stepsTotal === 0 && f.final)) return "meta.role newborn needs a final file at cu 0 and stepsTotal 0"
    if (v.evalPanel === null) return "meta.evalPanel must be set for role newborn"
  }
  return {
    v: 1, role, method, genomeSchema: v.genomeSchema, genome: { ...(g as Record<string, number | string>) }, genomeSha: v.genomeSha,
    drawSeed: v.drawSeed as number | null, posteriorSha: v.posteriorSha as string | null,
    evalPanel: v.evalPanel as number | null, evalSeed: v.evalSeed as number | null, confirmation, inheritance: { filters: "none" },
    ...(missing ? { missing } : {}), ...(filters !== undefined ? { filters } : {}), ...(heldOut ? { heldOut } : {}), ...(end ? { end } : {}),
  }
}

/** A caregiver summary (as in a log file, and as a caregiver state's carries/to/safetyNet), or what is wrong with it. */
export function parseCaregiver(v: unknown): CaregiverSummary | string {
  if (!isObj(v)) return "caregiver must be an object"
  const uk = unknownKey(v, CAREGIVER_KEYS, "caregiver")
  if (uk) return uk
  if (!isNonNegInt(v.carries)) return "caregiver.carries must be a non-negative integer"
  if (typeof v.safetyNet !== "boolean") return "caregiver.safetyNet must be a boolean"
  if (!isObj(v.to)) return "caregiver.to must be an object"
  const entries = Object.entries(v.to)
  if (entries.length > MAX_CARE_DESTS) return `caregiver.to may have at most ${MAX_CARE_DESTS} keys`
  let sum = 0
  for (const [k, n] of entries) {
    if (!CARE_DEST_RE.test(k)) return `caregiver.to key ${JSON.stringify(k.slice(0, 80))} is not allowed (must match ${CARE_DEST_RE})`
    if (!isNonNegInt(n)) return `caregiver.to.${k} must be a non-negative integer`
    sum += n
  }
  if (sum !== v.carries) return "caregiver.to values must sum to carries"
  // Written only when some were lost (0 is never written: files without loss stay as they were).
  if ("lostCarries" in v && !(isNonNegInt(v.lostCarries) && v.lostCarries >= 1)) return "caregiver.lostCarries must be an integer >= 1 when present"
  return {
    carries: v.carries, to: { ...(v.to as Record<string, number>) }, safetyNet: v.safetyNet,
    ...("lostCarries" in v ? { lostCarries: v.lostCarries as number } : {}),
  }
}
