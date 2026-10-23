// The learning genome, schema v1 (spec §3; decision 0021): the gene space (knowledge/genome-space.json), the promoted
// genome (knowledge/genome.json), the unit-cube maps the search works in, and decoding to the values each component
// takes. Pure and deterministic: the maps use @blanktab/detmath's dexp/dlog only, so every engine decodes a unit point
// to the same bytes.
import { dexp, dlog } from "@blanktab/detmath"

export type GeneValue = number | string
export type Genome = Record<string, GeneValue>
export interface GeneSpec {
  name: string; type: "continuous" | "integer" | "categorical"; scale?: "linear" | "log" | "horizon"
  min?: number; max?: number; values?: string[]; centre: GeneValue
  owner: "gp" | "bandit" | "fixed"; implemented?: false; target: "learner" | "caregiver" | "harness"
}
export interface GenomeSpace { schemaVersion: number; genes: GeneSpec[] }
export interface Decoded { learner: Record<string, number>; caregiver: Record<string, number>; harness: { maxStepsPerSec: number; sleepInterval: number } }

/** A copy of @blanktab/harness/log's SETTING_KEY_RE (meta cannot import harness); tools/genome-centre.test.ts pins it. */
export const SETTING_KEY_RE = /^[A-Za-z][A-Za-z0-9_]{0,63}$/
/** At most this many genes in a space. */
export const MAX_GENES = 40
/** The format version of knowledge/genome.json itself (its `genomeSchema` names the space's schemaVersion). */
export const GENOME_FILE_VERSION = 1

const TYPES = ["continuous", "integer", "categorical"] as const
const SCALES = ["linear", "log", "horizon"] as const
const TARGETS = ["learner", "caregiver", "harness"] as const
const GENE_KEYS = new Set(["name", "type", "scale", "min", "max", "values", "centre", "owner", "implemented", "target"])
const HARNESS_GENES = ["maxStepsPerSec", "sleepInterval"] as const

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v)
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v)
const has = <T extends string>(list: readonly T[], v: unknown): v is T => list.includes(v as T)
const fail = (msg: string): never => { throw new RangeError(msg) }

function parseGene(v: unknown, i: number): GeneSpec {
  if (!isObj(v)) return fail(`gene ${i} is not an object`)
  const name = v.name
  if (typeof name !== "string" || !SETTING_KEY_RE.test(name)) return fail(`gene ${i}: name ${JSON.stringify(name)} must match ${SETTING_KEY_RE}`)
  const at = `gene ${name}`
  for (const k of Object.keys(v)) if (!GENE_KEYS.has(k)) fail(`${at}: unknown key ${JSON.stringify(k)}`)
  const { type, scale, min, max, values, centre, owner, implemented, target } = v
  if (!has(TYPES, type)) return fail(`${at}: type must be one of ${TYPES.join(", ")}`)
  if (!has(TARGETS, target)) return fail(`${at}: target must be one of ${TARGETS.join(", ")}`)
  if (owner === "bandit") return fail(`${at}: owner "bandit" is reserved (no bandit genes in schema v1)`)
  if (owner !== "gp" && owner !== "fixed") return fail(`${at}: owner must be "gp" or "fixed"`)
  if (implemented !== undefined && implemented !== false) return fail(`${at}: implemented may only be false`)
  if (implemented === false && owner !== "fixed") return fail(`${at}: an implemented:false gene must be owner "fixed"`)
  const tail: Pick<GeneSpec, "owner" | "implemented" | "target"> = implemented === false ? { owner, implemented: false, target } : { owner, target }

  if (type === "categorical") {
    if (scale !== undefined || min !== undefined || max !== undefined) return fail(`${at}: a categorical gene has no scale, min or max`)
    if (owner === "gp") return fail(`${at}: a categorical gene cannot be searched by the GP`)
    if (!Array.isArray(values) || values.length === 0 || !values.every(x => typeof x === "string"))
      return fail(`${at}: values must be a non-empty array of strings`)
    if (new Set(values).size !== values.length) return fail(`${at}: values must not repeat`)
    if (typeof centre !== "string" || !values.includes(centre)) return fail(`${at}: centre must be one of values`)
    return { name, type, values: [...values] as string[], centre, ...tail }
  }
  if (values !== undefined) return fail(`${at}: only a categorical gene has values`)
  if (!has(SCALES, scale)) return fail(`${at}: scale must be one of ${SCALES.join(", ")}`)
  if (!isNum(min) || !isNum(max) || !isNum(centre)) return fail(`${at}: min, max and centre must be finite numbers`)
  if (type === "integer" && !(Number.isInteger(min) && Number.isInteger(max) && Number.isInteger(centre)))
    return fail(`${at}: an integer gene's min, max and centre must be integers`)
  if (min > max) return fail(`${at}: min > max`)
  if (owner === "gp" && !(min < max)) return fail(`${at}: a searched gene needs min < max`)
  if ((scale === "log" || scale === "horizon") && !(min > 0)) return fail(`${at}: a ${scale} gene needs min > 0`)
  if (scale === "horizon" && !(max < 1)) return fail(`${at}: a horizon gene needs max < 1`)
  if (centre < min || centre > max) return fail(`${at}: centre ${centre} is outside [${min}, ${max}]`)
  return { name, type, scale, min, max, centre, ...tail }
}

/** knowledge/genome-space.json → a GenomeSpace. RangeError on any violation. */
export function parseGenomeSpace(v: unknown): GenomeSpace {
  if (!isObj(v)) return fail("genome space is not an object")
  const { schemaVersion, genes } = v
  if (!Number.isInteger(schemaVersion) || (schemaVersion as number) < 1) return fail("schemaVersion must be an integer >= 1")
  if (!Array.isArray(genes)) return fail("genes must be an array")
  if (genes.length > MAX_GENES) return fail(`at most ${MAX_GENES} genes, got ${genes.length}`)
  const out = genes.map(parseGene)
  const seen = new Set<string>()
  for (const g of out) {
    if (seen.has(g.name)) fail(`duplicate gene ${g.name}`)
    seen.add(g.name)
  }
  return { schemaVersion: schemaVersion as number, genes: out }
}

/** knowledge/genome.json → its values (the promoted genome), after checkGenome. RangeError on any violation. */
export function parseCentre(v: unknown, space: GenomeSpace): Genome {
  if (!isObj(v)) return fail("genome file is not an object")
  if (v.schemaVersion !== GENOME_FILE_VERSION) return fail(`genome file schemaVersion must be ${GENOME_FILE_VERSION}`)
  if (v.genomeSchema !== space.schemaVersion) return fail(`genomeSchema ${String(v.genomeSchema)} is not the space's ${space.schemaVersion}`)
  if (!isObj(v.provenance) || typeof v.provenance.source !== "string") return fail("provenance must be an object with a string source")
  if (!isObj(v.values)) return fail("values must be an object")
  const values = v.values as Genome
  const bad = checkGenome(space, values)
  if (bad !== null) return fail(`genome values: ${bad}`)
  return Object.fromEntries(space.genes.map(g => [g.name, values[g.name]!]))
}

/** The genes the GP searches (owner "gp"), in file order: the unit cube's axes. */
export function searched(space: GenomeSpace): GeneSpec[] {
  return space.genes.filter(g => g.owner === "gp")
}

// Horizon scale: the search runs on h = 1/(1 − γ), so equal steps in u are equal ratios of the planning horizon.
const horizon = (v: number) => 1 / (1 - v)

function encode(g: GeneSpec, v: number): number {
  const min = g.min!, max = g.max!
  if (g.scale === "log") return dlog(v / min) / dlog(max / min)
  if (g.scale === "horizon") return dlog(horizon(v) / horizon(min)) / dlog(horizon(max) / horizon(min))
  return (v - min) / (max - min)
}

function decode(g: GeneSpec, u: number): number {
  const min = g.min!, max = g.max!
  let v: number
  if (g.scale === "log") v = min * dexp(u * dlog(max / min))
  else if (g.scale === "horizon") {
    const hMin = horizon(min)
    v = 1 - 1 / (hMin * dexp(u * dlog(horizon(max) / hMin)))
  } else v = min + u * (max - min)
  if (g.type === "integer") v = Math.round(v)
  return Math.min(max, Math.max(min, v)) // rounding at the ends of the cube never leaves the range
}

/** The genome's point in the unit cube, one coordinate per searched gene. RangeError on a non-numeric searched value. */
export function toUnit(space: GenomeSpace, g: Genome): number[] {
  return searched(space).map(gene => {
    const v = g[gene.name]
    if (!isNum(v)) return fail(`gene ${gene.name}: ${JSON.stringify(v)} is not a finite number`)
    return encode(gene, v)
  })
}

/** The genome at unit point `u` (integers rounded); every non-searched gene is taken from `centre`. */
export function fromUnit(space: GenomeSpace, u: readonly number[], centre: Genome): Genome {
  const genes = searched(space)
  if (u.length !== genes.length) return fail(`u has ${u.length} coordinates, the space searches ${genes.length} genes`)
  const out: Genome = {}
  let j = 0
  for (const gene of space.genes) {
    if (gene.owner === "gp") {
      const x = u[j++]!
      if (!(x >= 0 && x <= 1)) return fail(`u for ${gene.name} must be in [0, 1], got ${x}`)
      out[gene.name] = decode(gene, x)
    } else {
      if (!(gene.name in centre)) return fail(`centre lacks gene ${gene.name}`)
      out[gene.name] = centre[gene.name]!
    }
  }
  return out
}

/**
 * Why `g` is not a valid genome of `space`, or null. Its keys must be the genes exactly; each value of the gene's type,
 * inside [min, max] (integral for integers, among values for categoricals); every fixed gene (implemented:false ones
 * included) exactly at its centre.
 */
export function checkGenome(space: GenomeSpace, g: Genome): string | null {
  const names = new Set(space.genes.map(x => x.name))
  for (const k of Object.keys(g)) if (!names.has(k)) return `unknown gene ${JSON.stringify(k)}`
  for (const gene of space.genes) {
    if (!Object.hasOwn(g, gene.name)) return `missing gene ${gene.name}`
    const v = g[gene.name]
    if (gene.type === "categorical") {
      if (typeof v !== "string" || !gene.values!.includes(v)) return `${gene.name} must be one of ${gene.values!.join(", ")}, got ${JSON.stringify(v)}`
    } else {
      if (!isNum(v)) return `${gene.name} must be a finite number, got ${JSON.stringify(v)}`
      if (gene.type === "integer" && !Number.isInteger(v)) return `${gene.name} must be an integer, got ${v}`
      if (v < gene.min! || v > gene.max!) return `${gene.name} ${v} is outside [${gene.min}, ${gene.max}]`
    }
    if (gene.owner === "fixed" && v !== gene.centre)
      return `${gene.name} is ${gene.implemented === false ? "not implemented" : "fixed"} and must stay at its centre ${JSON.stringify(gene.centre)}`
  }
  return null
}

/**
 * The values each component takes, by target: implemented genes only (a reserved slot never reaches code).
 * Not bounds-checked, so a legacy genome decodes too; call checkGenome first where bounds matter.
 */
export function decodeGenome(g: Genome, space: GenomeSpace): Decoded {
  const out = { learner: {} as Record<string, number>, caregiver: {} as Record<string, number>, harness: {} as Record<string, number> }
  for (const gene of space.genes) {
    if (gene.implemented === false) continue
    const v = g[gene.name]
    if (!isNum(v)) return fail(`gene ${gene.name}: ${JSON.stringify(v)} is not a finite number`)
    if (gene.target === "harness" && !has(HARNESS_GENES, gene.name)) return fail(`gene ${gene.name}: no harness setting of that name`)
    out[gene.target][gene.name] = v
  }
  for (const k of HARNESS_GENES) if (!(k in out.harness)) fail(`the space has no harness gene ${k}`)
  return { learner: out.learner, caregiver: out.caregiver, harness: { maxStepsPerSec: out.harness.maxStepsPerSec!, sleepInterval: out.harness.sleepInterval! } }
}

/** A pre-05a life's genome, and which genes its settings did not record (filled with centres as placeholders only). */
export interface LegacyGenome { genome: Genome; missing: string[] }

/**
 * Genes every life born before 05a ran with a known value although its settings never record them: gamma was the
 * constant 0.9 (learner GAMMA) until Plan 05a made it a setting. Not the gene's centre, which a promotion may move.
 */
export const PRE_05A_CONSTANTS: Readonly<Record<string, GeneValue>> = Object.freeze({ gamma: 0.9 })

/**
 * The genome of a life born before 05a, from its flat settings: each gene its setting's value, other settings ignored.
 * Genes known by construction are filled, not listed: PRE_05A_CONSTANTS, and every implemented:false slot at its
 * centre (which is, by the schema's rule, the code without the mechanism). Any other gene absent from the settings was
 * not recorded: we do not know what that life ran with (an older learner may have lacked the mechanism, or had no
 * caregiver), so it is listed in `missing` (sorted) and its centre stands in only as a placeholder, never as a claim
 * (controller ruling, decision 0022). NOT bounds-checked (today's picker spans exceed the v1 ranges; plan gap 4).
 * RangeError when a present value has the wrong type.
 */
export function legacyGenome(settings: Record<string, GeneValue>, space: GenomeSpace): LegacyGenome {
  const genome: Genome = {}
  const missing: string[] = []
  for (const gene of space.genes) {
    if (!Object.hasOwn(settings, gene.name)) {
      if (Object.hasOwn(PRE_05A_CONSTANTS, gene.name)) genome[gene.name] = PRE_05A_CONSTANTS[gene.name]!
      else if (gene.implemented === false) genome[gene.name] = gene.centre
      else { genome[gene.name] = gene.centre; missing.push(gene.name) }
      continue
    }
    const v = settings[gene.name]
    if (gene.type === "categorical" ? typeof v !== "string" : !isNum(v)) return fail(`setting ${gene.name}: ${JSON.stringify(v)} has the wrong type`)
    genome[gene.name] = v!
  }
  return { genome, missing: missing.sort() }
}
