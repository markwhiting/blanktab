// The gate's meta-learner files, data/meta/** (spec §4.3, §4.6, §7.2, §7.4; decision 0019): computed from main's
// merged logs and pinned config only, never from PR content. 05a builds the calibration design, the noise check's
// looks and the re-verification of every draw; 05b adds the GP (models, incumbent, confirmation).
//
// Every byte is a pure function of the input: rows in learnerId order, fixed seeds, keys sorted at every level, and no
// clock. The one time, `builtAt`, is passed in and goes only into the index line, never into a file whose sha matters.
import { createHash } from "node:crypto"
import { sobolPoints } from "@blanktab/detmath"
import { lifeBudget, type LogFile, type LogFileV2, type MetaBlock } from "@blanktab/harness/log"
import {
  birthDraw, checkGenome, fromUnit, genomeSha, parsePosterior, searched, sha256Hex,
  type Genome, type GenomeSpace, type Mode, type Posterior,
} from "@blanktab/meta"
import {
  cohortContext, getEvaluator, GPU_WIDTHS_CLASS, SIGNAL_LOOKS, type EvalOutput, type LifeInput, type References,
} from "@blanktab/evaluators"
import type { EvalConfig } from "../eval/config.ts"
import { collectLives } from "../eval/lives.ts"

export const POSTERIOR_PATH = "data/meta/posterior.json"
export const CALIBRATION_PATH = "data/meta/calibration.json"
export const POSTERIOR_INDEX_PATH = "data/meta/posterior-index.jsonl"
/** Counts that move with every birth (born, checked, flags, reasons): kept out of posterior.json, so its bytes, and
 * the sha every tab seeds its draw with, change only when the rows, the mode, the looks or the design change. */
export const DIAGNOSTICS_PATH = "data/meta/diagnostics.json"

/** The calibration design's Sobol seed (spec §4.6). */
const CALIB_SEED = "calib/1"
/** The design: the centre plus this many Sobol points. */
const CALIB_POINTS = 9
/** Repeats per design genome before look 1, and after a no-signal look 1 or 2 (spec §4.6; plan conflict 2). */
const REPEATS_FIRST = 3
const REPEATS_MORE = 6
/** The methods whose eligible lives are rows in 05a (spec §4.3 also lists `ts`, which 05b adds). */
const ROW_METHODS: readonly string[] = ["calibrate", "uniform", "centre"]

type ShaGenome = { genomeSha: string; values: Genome }
export interface CalibrationLook { look: 1 | 2 | 3; lives: number; learnerIds: string[]; output: EvalOutput; sourceCommit: string }
export interface CalibrationState {
  schemaVersion: 1; genomeSchema: number; evaluator: string; signal: string; seed: "calib/1"
  genomes: ShaGenome[] // centre first, then 9 Sobol points decoded
  repeats: 3 | 6
  looks: CalibrationLook[] // append-only
  outcome: "pending" | "signal" | "flat"
}
export interface MetaInput {
  runsDir: string; config: EvalConfig; refs: References; space: GenomeSpace; centre: Genome; sourceCommit: string
  /** When this build ran (ISO 8601): written only into the index line. The caller reads the clock; buildMeta never does. */
  builtAt: string
  previous: { calibration: CalibrationState | null; posteriorText: string | null }
  /** posterior.json versions from main's history, by sha256 of their bytes: at least every one a life names. */
  posteriorsBySha: ReadonlyMap<string, string>
}
export interface MetaFiles {
  "data/meta/posterior.json": string; "data/meta/calibration.json": string; "data/meta/diagnostics.json": string
  /** One JSON line (no newline) for posterior-index.jsonl when the posterior's bytes changed, else null. */
  indexLine: string | null
}

const isV2 = (f: LogFile): f is LogFileV2 => f.schemaVersion === 2
const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const byId = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0)

/** `v` with the keys of every object sorted, so serialisation never depends on insertion order. */
function sortKeys(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(sortKeys)
  if (isObj(v)) return Object.fromEntries(Object.keys(v).sort().map(k => [k, sortKeys(v[k])]))
  return v
}
/** The gate's JSON: sorted keys, one-space indent, a final newline, no trailing spaces. */
const serialise = (v: unknown): string => JSON.stringify(sortKeys(v), null, 1) + "\n"

const fail = (why: string): never => { throw new Error(`${CALIBRATION_PATH}: ${why}`) }
const CAL_KEYS = ["schemaVersion", "genomeSchema", "evaluator", "signal", "seed", "genomes", "repeats", "looks", "outcome"]
const LOOK_KEYS = ["look", "lives", "learnerIds", "output", "sourceCommit"]
const exactKeys = (v: Record<string, unknown>, keys: readonly string[]) => {
  const ks = Object.keys(v)
  return ks.length === keys.length && keys.every(k => Object.hasOwn(v, k))
}

/** data/meta/calibration.json's text → its state; throws on any shape error (the gate then stops, writing nothing). */
export function parseCalibration(text: string): CalibrationState {
  const v: unknown = JSON.parse(text)
  if (!isObj(v) || !exactKeys(v, CAL_KEYS)) return fail(`must have exactly the keys ${CAL_KEYS.join(", ")}`)
  if (v.schemaVersion !== 1) fail("schemaVersion must be 1")
  if (!Number.isSafeInteger(v.genomeSchema) || (v.genomeSchema as number) < 1) fail("genomeSchema must be an integer >= 1")
  if (typeof v.evaluator !== "string" || typeof v.signal !== "string") fail("evaluator and signal must be id@version strings")
  if (v.seed !== CALIB_SEED) fail(`seed must be ${CALIB_SEED}`)
  const genomes = v.genomes
  if (!Array.isArray(genomes) || genomes.length !== 1 + CALIB_POINTS
    || !genomes.every(g => isObj(g) && exactKeys(g, ["genomeSha", "values"]) && typeof g.genomeSha === "string" && isObj(g.values))) {
    fail(`genomes must be ${1 + CALIB_POINTS} { genomeSha, values }`)
  }
  if (v.repeats !== REPEATS_FIRST && v.repeats !== REPEATS_MORE) fail(`repeats must be ${REPEATS_FIRST} or ${REPEATS_MORE}`)
  const looks = v.looks
  if (!Array.isArray(looks) || looks.length > SIGNAL_LOOKS.length) fail(`looks must be an array of at most ${SIGNAL_LOOKS.length}`)
  ;(looks as unknown[]).forEach((l, i) => {
    if (!isObj(l) || !exactKeys(l, LOOK_KEYS) || l.look !== i + 1 || !Number.isSafeInteger(l.lives) || !Array.isArray(l.learnerIds)
      || !l.learnerIds.every(x => typeof x === "string") || !isObj(l.output) || typeof l.sourceCommit !== "string") {
      fail(`look ${i + 1} must be { ${LOOK_KEYS.join(", ")} } with look ${i + 1}`)
    }
  })
  if (v.outcome !== "pending" && v.outcome !== "signal" && v.outcome !== "flat") fail("outcome must be pending, signal or flat")
  return v as unknown as CalibrationState
}

/**
 * The posterior.json versions main committed whose sha256 is in `wanted`, by sha: `git log --format=%H
 * --diff-filter=d -- <path>` (commits that delete the file have no bytes), newest first, then `git show <c>:<path>`
 * until every wanted sha is found or the history ends. A sha never found is absent (its lives are flagged unknown).
 */
export function posteriorHistory(git: (args: string[]) => string, wanted: ReadonlySet<string>): Map<string, string> {
  const out = new Map<string, string>()
  if (wanted.size === 0) return out
  const commits = git(["log", "--format=%H", "--diff-filter=d", "--", POSTERIOR_PATH]).split("\n").map(s => s.trim()).filter(Boolean)
  for (const c of commits) {
    const text = git(["show", `${c}:${POSTERIOR_PATH}`])
    const sha = createHash("sha256").update(text, "utf8").digest("hex")
    if (wanted.has(sha)) out.set(sha, text)
    if (out.size === wanted.size) break
  }
  return out
}

/** The posteriorShas the lives under `runsDir` were born with (their first meta block's), for posteriorHistory. */
export function wantedPosteriors(runsDir: string): Set<string> {
  const out = new Set<string>()
  for (const life of collectLives(runsDir)) {
    const sha = birthOf(life.files)?.meta.posteriorSha
    if (typeof sha === "string") out.add(sha)
  }
  return out
}

/** The design (spec §4.6): the centre, then fromUnit of 9 Sobol points seeded "calib/1", each with its genomeSha. */
async function design(space: GenomeSpace, centre: Genome): Promise<ShaGenome[]> {
  const genomes = [centre, ...sobolPoints(searched(space).length, CALIB_POINTS, CALIB_SEED).map(u => fromUnit(space, u, centre))]
  return Promise.all(genomes.map(async values => ({ genomeSha: await genomeSha(space.schemaVersion, values), values: { ...values } })))
}

/** The previous calibration state, checked against the pins and the space it must have been built for. */
async function keptCalibration(c: CalibrationState, config: EvalConfig, space: GenomeSpace): Promise<CalibrationState> {
  // Rows never mix evaluators or genome schemas (spec §2.2, §4.6): a change of pin or schema restarts calibration
  // under a recorded decision, which removes this file. The gate refuses rather than restart on its own.
  if (c.evaluator !== config.search) fail(`was built for search evaluator ${c.evaluator}, but ${config.search} is pinned (a recorded decision restarts calibration)`)
  if (c.signal !== config.signal) fail(`was built for signal check ${c.signal}, but ${config.signal} is pinned (a recorded decision restarts calibration)`)
  if (c.genomeSchema !== space.schemaVersion) fail(`was built for genome schema ${c.genomeSchema}, but the space is ${space.schemaVersion} (a recorded decision restarts calibration)`)
  for (const [i, g] of c.genomes.entries()) {
    const bad = checkGenome(space, g.values)
    if (bad !== null) fail(`genome ${i} is not in the space: ${bad}`)
    if (g.genomeSha !== await genomeSha(space.schemaVersion, g.values)) fail(`genome ${i}'s genomeSha does not match its values`)
  }
  return c
}

/** The first meta block a life logged (its birth), and the file it is in. */
function birthOf(files: readonly LogFile[]): { file: LogFileV2; meta: MetaBlock } | null {
  for (const f of files) if (isV2(f) && f.meta) return { file: f, meta: f.meta }
  return null
}
const widthsClassOf = (w: LogFileV2["widths"]): "gpu" | "cpu" =>
  w.conv1 === GPU_WIDTHS_CLASS.conv1 && w.conv2 === GPU_WIDTHS_CLASS.conv2 && w.enc === GPU_WIDTHS_CLASS.enc && w.hidden === GPU_WIDTHS_CLASS.hidden ? "gpu" : "cpu"

const sameGenome = (a: Genome, b: Genome): boolean => {
  const ka = Object.keys(a).sort(), kb = Object.keys(b).sort()
  return ka.length === kb.length && ka.every((k, i) => k === kb[i] && a[k] === b[k])
}

interface Checked { reason: string | null; genomeMismatch: boolean }

/**
 * Re-verification (spec §7.4): the life's draw recomputed from its learnerId, the posterior bytes its posteriorSha
 * names, the space and centre and its widths class. A row is lost for an unknown or unreadable posterior, a mismatched
 * method, seed or panel, a logged genome outside the space or not matching its genomeSha, or a calibrate genome
 * outside the design. A genome that differs from the recomputed one alone keeps the row, on its logged genome.
 */
async function verify(learnerId: string, meta: MetaBlock, widthsClass: "gpu" | "cpu", ctx: {
  space: GenomeSpace; centre: Genome; centreSha: string; design: ReadonlySet<string>; posteriors: ReadonlyMap<string, Posterior | null>
}): Promise<Checked> {
  const no = (reason: string): Checked => ({ reason, genomeMismatch: false })
  let posterior: Posterior | null = null
  if (meta.posteriorSha !== null) {
    if (!ctx.posteriors.has(meta.posteriorSha)) return no("unknown posteriorSha")
    posterior = ctx.posteriors.get(meta.posteriorSha)!
    if (posterior === null) return no("posterior unreadable") // a tab that cannot parse it logs no sha (draws centre)
  }
  const d = birthDraw({ learnerId, space: ctx.space, centre: ctx.centre, centreSha: ctx.centreSha, widthsClass, posterior, posteriorSha: meta.posteriorSha })
  if (meta.method !== d.method) return no(`method ${meta.method}, recomputed ${d.method}`)
  if (meta.drawSeed !== d.drawSeed) return no(`drawSeed ${meta.drawSeed}, recomputed ${d.drawSeed}`)
  if (meta.evalPanel !== d.evalPanel) return no(`evalPanel ${meta.evalPanel}, recomputed ${d.evalPanel}`)
  const bad = checkGenome(ctx.space, meta.genome)
  if (bad !== null) return no(`genome not in the space: ${bad}`)
  if (meta.genomeSha !== await genomeSha(meta.genomeSchema, meta.genome)) return no("genomeSha does not match the genome")
  if (meta.method === "calibrate" && !ctx.design.has(meta.genomeSha)) return no("calibrate genome not in the calibration design")
  return { reason: null, genomeMismatch: !sameGenome(meta.genome, d.genome) }
}

/** The lives' rows and counts under the pinned search evaluator, after re-verification. */
interface Lives {
  rows: { life: LifeInput; meta: MetaBlock; out: EvalOutput }[]
  checked: number; flagged: { learnerId: string; reason: string }[]
  /** Verified lives whose logged genome differs from the recomputed one, and those of them that are rows. */
  mismatched: string[]; mismatchedRows: string[]
  perGenome: { genomeSha: string; born: number; rows: number; failed: number; abandoned: number; failedRows: number }[]
  unscored: Record<string, number>
}

async function readLives(input: MetaInput, calibration: CalibrationState, centreSha: string): Promise<Lives> {
  const { space, centre, refs, config } = input
  const search = getEvaluator(config.search)
  if (search.kind !== "life" || !search.evaluateLife) throw new Error(`${config.search} is not a life evaluator`)
  // The posteriors by sha, each parsed once. The map's own keys are checked: a wrong key would verify wrong draws.
  const posteriors = new Map<string, Posterior | null>()
  for (const [sha, text] of input.posteriorsBySha) {
    if (await sha256Hex(new TextEncoder().encode(text)) !== sha) throw new Error(`posteriorsBySha: the key ${sha} is not the sha256 of its bytes`)
    posteriors.set(sha, parsePosterior(text))
  }
  const designShas = calibration.genomes.map(g => g.genomeSha)
  const ctx = { space, centre, centreSha, design: new Set(designShas), posteriors }
  const out: Lives = {
    rows: [], checked: 0, flagged: [], mismatched: [], mismatchedRows: [], unscored: {},
    perGenome: designShas.map(genomeSha => ({ genomeSha, born: 0, rows: 0, failed: 0, abandoned: 0, failedRows: 0 })),
  }
  for (const life of collectLives(input.runsDir)) { // sorted by learnerId
    const birth = birthOf(life.files)
    // 05a draws are role search; newborn, legacy and confirm lives are not draws and never rows. Other genome schemas
    // cannot be recomputed against this space, and are never rows of this fit.
    if (!birth || birth.meta.role !== "search" || birth.meta.genomeSchema !== space.schemaVersion) continue
    const widthsClass = widthsClassOf(birth.file.widths)
    // TODO(05b): recompute with the centre in force at this birth (the posterior's centre, or genome.json at the
    // birth's knowledge), not main's current one. Exact in 05a, where no promotion moves the centre.
    const checked = await verify(life.learnerId, birth.meta, widthsClass, ctx)
    out.checked++
    if (checked.reason !== null) { out.flagged.push({ learnerId: life.learnerId, reason: checked.reason }); continue }
    if (checked.genomeMismatch) out.mismatched.push(life.learnerId)

    const scored: EvalOutput = search.evaluateLife(life, refs)
    const isRow = scored.eligible && typeof scored.score === "number" && widthsClass === "gpu" && ROW_METHODS.includes(birth.meta.method)
      && life.files.every(f => !isV2(f) || !f.meta || f.meta.genomeSchema === space.schemaVersion)
    if (isRow) {
      out.rows.push({ life, meta: birth.meta, out: scored })
      if (checked.genomeMismatch) out.mismatchedRows.push(life.learnerId)
    } else { const why = scored.eligible ? "not a row of this fit" : scored.reason ?? "ineligible"; out.unscored[why] = (out.unscored[why] ?? 0) + 1 }

    const g = birth.meta.method === "calibrate" ? designShas.indexOf(birth.meta.genomeSha) : -1
    if (g < 0) continue
    // Failed: its final file says so. Abandoned: it ended, not failed, without its budget checkup (tab closed, orphan).
    const last = life.files[life.files.length - 1]!
    const end = isV2(last) ? last.meta?.end : undefined
    const budgetCU = lifeBudget(birth.meta).budgetCU
    const reachedBudget = life.files.some(f => isV2(f) && (f.checkups ?? []).some(c => c.level === budgetCU))
    const pg = out.perGenome[g]!
    pg.born++
    if (isRow) pg.rows++
    if (end === "failed") { pg.failed++; if (isRow) pg.failedRows++ }
    else if (last.final && !reachedBudget) pg.abandoned++
  }
  return out
}

/** Calibration rows (method calibrate) grouped by their design genome, for the signal check. */
function calibrationGroups(rows: Lives["rows"]): Record<string, LifeInput[]> {
  const groups: Record<string, LifeInput[]> = {}
  for (const r of rows) if (r.meta.method === "calibrate") (groups[r.meta.genomeSha] ??= []).push(r.life)
  return groups
}

/** The next look (spec §4.6), or none: at most one per build, once enough new calibration rows have merged. */
function nextLook(c: CalibrationState, rows: Lives["rows"], input: MetaInput): CalibrationState {
  const n = c.looks.length
  const calib = rows.filter(r => r.meta.method === "calibrate")
  if (c.outcome !== "pending" || n >= SIGNAL_LOOKS.length) return c
  // Each look needs its scheduled size, and as many new lives since the last look as the schedule spaces them, so a
  // burst of merges cannot bring two looks close together (controller ruling).
  const lastLives = n > 0 ? c.looks[n - 1]!.lives : 0
  const spacing = SIGNAL_LOOKS[n]! - (n > 0 ? SIGNAL_LOOKS[n - 1]! : 0)
  if (calib.length < Math.max(SIGNAL_LOOKS[n]!, lastLives + spacing)) return c
  const signal = getEvaluator(input.config.signal)
  if (signal.kind !== "cohort" || !signal.evaluateCohort) throw new Error(`${input.config.signal} is not a cohort evaluator`)
  const output = signal.evaluateCohort(calibrationGroups(rows), input.refs, cohortContext(input.config))
  if (!output.eligible) return c // not enough to judge yet: wait for more rows
  const look: CalibrationLook = {
    look: (n + 1) as 1 | 2 | 3, lives: calib.length, learnerIds: calib.map(r => r.life.learnerId).sort(byId), output, sourceCommit: input.sourceCommit,
  }
  const looks = [...c.looks, look]
  if (output.score === 1) return { ...c, looks, outcome: "signal" }
  if (looks.length === SIGNAL_LOOKS.length) return { ...c, looks, outcome: "flat" }
  return { ...c, looks, repeats: REPEATS_MORE } // paused, on the page; births keep calibrating (plan conflict 2)
}

const mismatch = (learnerIds: string[], of: number) => ({ count: learnerIds.length, rate: of > 0 ? learnerIds.length / of : 0, learnerIds })

const modeOf = (c: CalibrationState): Mode => (c.outcome === "signal" ? "search" : c.outcome === "flat" ? "flat" : "calibrate")

/**
 * data/meta/posterior.json, calibration.json and diagnostics.json from main's merged logs, and the index line when the
 * posterior changed. The posterior holds only what follows from the rows (and the mode, looks and design).
 */
export async function buildMeta(input: MetaInput): Promise<MetaFiles> {
  const { space, centre, config } = input
  const centreSha = await genomeSha(space.schemaVersion, centre)
  const prevCal = input.previous.calibration
  let calibration: CalibrationState = prevCal
    ? await keptCalibration(prevCal, config, space)
    : { schemaVersion: 1, genomeSchema: space.schemaVersion, evaluator: config.search, signal: config.signal, seed: CALIB_SEED, genomes: await design(space, centre), repeats: REPEATS_FIRST, looks: [], outcome: "pending" }

  const lives = await readLives(input, calibration, centreSha)
  calibration = nextLook(calibration, lives.rows, input)
  const calibRows = lives.rows.filter(r => r.meta.method === "calibrate").length
  const n = calibration.looks.length
  const posterior = {
    schemaVersion: 1, genomeSchema: space.schemaVersion, evaluator: config.search, sourceCommit: input.sourceCommit, mode: modeOf(calibration),
    centre: { genomeSha: centreSha, values: centre },
    calibration: { genomes: calibration.genomes, need: lives.perGenome.map(g => Math.max(0, calibration.repeats - g.rows)) },
    models: {}, incumbent: null, confirmation: null,
    diagnostics: {
      perGenome: lives.perGenome.map(g => ({ genomeSha: g.genomeSha, rows: g.rows, failed: g.failedRows })),
      // Rows kept on a logged genome that differs from the recomputed draw (spec §7.4).
      genomeMismatch: mismatch(lives.mismatchedRows, lives.rows.length),
    },
    page: {
      eligibleLives: lives.rows.length, lastPromotion: null, calibrationLives: calibRows,
      // The next look's size while calibrating; the deciding look's once decided.
      calibrationTarget: SIGNAL_LOOKS[calibration.outcome === "pending" ? Math.min(n, SIGNAL_LOOKS.length - 1) : n - 1]!, looks: n,
    },
  }
  let posteriorText = serialise(posterior)
  if (parsePosterior(posteriorText) === null) throw new Error("buildMeta produced a posterior that parsePosterior refuses")
  // Unchanged rows keep the previous bytes, and their sourceCommit: those rows came from that commit too.
  const prevText = input.previous.posteriorText
  let changed = true
  if (prevText !== null) {
    let prevCommit: unknown
    try { prevCommit = (JSON.parse(prevText) as { sourceCommit?: unknown }).sourceCommit } catch { prevCommit = undefined }
    if (typeof prevCommit === "string" && serialise({ ...posterior, sourceCommit: prevCommit }) === prevText) {
      posteriorText = prevText
      changed = false
    }
  }
  const indexLine = changed
    ? JSON.stringify({ posteriorSha: await sha256Hex(new TextEncoder().encode(posteriorText)), sourceCommit: input.sourceCommit, builtAt: input.builtAt })
    : null
  const diagnostics = {
    schemaVersion: 1, evaluator: config.search,
    perGenome: lives.perGenome.map(g => ({ genomeSha: g.genomeSha, born: g.born, rows: g.rows, failed: g.failed, abandoned: g.abandoned })),
    verify: { checked: lives.checked, ineligible: lives.flagged, genomeMismatch: mismatch(lives.mismatched, lives.checked) },
    unscored: lives.unscored,
  }
  return { [POSTERIOR_PATH]: posteriorText, [CALIBRATION_PATH]: serialise(calibration), [DIAGNOSTICS_PATH]: serialise(diagnostics), indexLine } as MetaFiles
}
