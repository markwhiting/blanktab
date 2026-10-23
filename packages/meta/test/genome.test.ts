import { describe, it, expect } from "vitest"
import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"
import { createRng } from "@blanktab/detmath"
import {
  checkGenome, decodeGenome, fromUnit, genomeSha, legacyGenome, parseCentre, parseGenomeSpace, searched, sha256Hex, toUnit,
  type GeneSpec, type Genome,
} from "@blanktab/meta"

const readJson = (name: string): unknown => JSON.parse(readFileSync(new URL(`../../../knowledge/${name}`, import.meta.url), "utf8"))
const SPACE_JSON = readJson("genome-space.json") as { schemaVersion: number; genes: GeneSpec[] }
const space = parseGenomeSpace(SPACE_JSON)
const centre = parseCentre(readJson("genome.json"), space)

/** The committed space with gene `name` replaced by `patch` applied to it (or removed with null). */
function withGene(name: string, patch: Record<string, unknown> | null): unknown {
  const genes = SPACE_JSON.genes.flatMap(g => g.name !== name ? [g] : patch === null ? [] : [{ ...g, ...patch }])
  return { ...SPACE_JSON, genes }
}

describe("parseGenomeSpace", () => {
  it("parseGenomeSpace accepts the committed file and rejects each violation listed (one case each)", () => {
    expect(space.schemaVersion).toBe(1)
    expect(space.genes).toHaveLength(29)
    expect(searched(space).map(g => g.name)).toEqual([
      "lr", "headLrScale", "residualDecay", "entropyWeight", "curiosityWeight", "exploreEpsilon", "bufferSize",
      "replayPerSleep", "gamma", "caregiverPeriod", "caregiverTau", "caregiverEpsilon",
    ])
    const lr = SPACE_JSON.genes[0]!
    const filler = (i: number) => ({ name: `g${i}`, type: "continuous", scale: "linear", min: 0, max: 0, centre: 0, owner: "fixed", target: "learner" })
    const cases: [string, unknown, RegExp][] = [
      ["duplicate name", { ...SPACE_JSON, genes: [...SPACE_JSON.genes, lr] }, /duplicate gene lr/],
      ["name not matching SETTING_KEY_RE", withGene("lr", { name: "1lr" }), /must match/],
      ["name with a dot", withGene("lr", { name: "l.r" }), /must match/],
      ["more than 40 genes", { ...SPACE_JSON, genes: [...SPACE_JSON.genes, ...Array.from({ length: 12 }, (_, i) => filler(i))] }, /at most 40 genes/],
      ["centre below min", withGene("lr", { centre: 1e-4 }), /centre .* outside/],
      ["centre above max", withGene("caregiverThreshold", { centre: 0.02 }), /centre .* outside/],
      ["gp gene with min = max", withGene("lr", { min: 5e-4, max: 5e-4 }), /searched gene needs min < max/],
      ["gp gene with min > max", withGene("lr", { min: 2e-3, max: 1.25e-4 }), /min > max/],
      ["log gene with min <= 0", withGene("lr", { min: 0 }), /log gene needs min > 0/],
      ["horizon gene with min <= 0", withGene("gamma", { min: -0.5 }), /horizon gene needs min > 0/],
      ["implemented:false with owner gp", withGene("latentWeight", { owner: "gp" }), /implemented:false gene must be owner "fixed"/],
      ["categorical without values", withGene("creditMode", { values: undefined }), /values must be a non-empty array/],
      ["categorical with duplicated values", withGene("creditMode", { values: ["off", "off"] }), /values must not repeat/],
      ["owner bandit (reserved)", withGene("creditMode", { owner: "bandit", implemented: undefined }), /"bandit" is reserved/],
    ]
    for (const [what, v, why] of cases) {
      expect(() => parseGenomeSpace(v), what).toThrow(RangeError)
      expect(() => parseGenomeSpace(v), what).toThrow(why)
    }
  })
  it("rejects malformed structure: not an object, bad schemaVersion, unknown type, scale, owner or target", () => {
    for (const [what, v, why] of [
      ["null", null, /genome space is not an object/],
      ["schemaVersion 0", { ...SPACE_JSON, schemaVersion: 0 }, /schemaVersion must be an integer >= 1/],
      ["genes not an array", { ...SPACE_JSON, genes: {} }, /genes must be an array/],
      ["unknown key", withGene("lr", { colour: "red" }), /unknown key "colour"/],
      ["unknown type", withGene("lr", { type: "real" }), /type must be one of/],
      ["unknown scale", withGene("lr", { scale: "exp" }), /scale must be one of/],
      ["unknown owner", withGene("lr", { owner: "me" }), /owner must be "gp" or "fixed"/],
      ["unknown target", withGene("lr", { target: "world" }), /target must be one of/],
      ["implemented true", withGene("lr", { implemented: true }), /implemented may only be false/],
      ["non-finite min", withGene("lr", { min: null }), /min, max and centre must be finite numbers/],
      ["integer gene with a fractional bound", withGene("bufferSize", { min: 500.5 }), /must be integers/],
      ["horizon gene with max >= 1", withGene("gamma", { max: 1 }), /horizon gene needs max < 1/],
      ["categorical centre not among values", withGene("creditMode", { centre: "on" }), /centre must be one of values/],
      ["a searched categorical (bandit genes only)", withGene("creditMode", { owner: "gp", implemented: undefined }), /cannot be searched by the GP/],
    ] as [string, unknown, RegExp][]) {
      expect(() => parseGenomeSpace(v), what).toThrow(RangeError)
      expect(() => parseGenomeSpace(v), what).toThrow(why)
    }
  })
})

describe("unit-cube maps", () => {
  it("toUnit(fromUnit(u)) = u for searched genes on 1,000 seeded u; integers round-trip as the value that ran", () => {
    const genes = searched(space)
    const rng = createRng(0x9e3779b9)
    for (let i = 0; i < 1000; i++) {
      const u = genes.map(() => rng())
      const g = fromUnit(space, u, centre)
      expect(checkGenome(space, g)).toBeNull()
      const back = toUnit(space, g)
      genes.forEach((gene, j) => {
        if (gene.type === "integer") {
          expect(Number.isInteger(g[gene.name])).toBe(true)
          // The re-encoded u is the value that ran: decoding it again gives the same integer.
          expect(fromUnit(space, back, centre)[gene.name]).toBe(g[gene.name])
        } else {
          expect(Math.abs(back[j]! - u[j]!), gene.name).toBeLessThan(1e-12)
        }
      })
    }
  })
  it("the ends of the cube decode to min and max, and the maps are linear, log and horizon as specified", () => {
    const genes = searched(space)
    const lo = fromUnit(space, genes.map(() => 0), centre)
    const hi = fromUnit(space, genes.map(() => 1), centre)
    for (const g of genes) {
      expect(lo[g.name], g.name).toBeCloseTo(g.min!, 15)
      expect(hi[g.name], g.name).toBeCloseTo(g.max!, 15)
    }
    const u = toUnit(space, centre)
    const at = (n: string) => u[genes.findIndex(g => g.name === n)]!
    expect(at("lr")).toBeCloseTo(0.5, 12) // 5e-4 is the geometric middle of 1.25e-4 – 2e-3
    expect(at("bufferSize")).toBeCloseTo(Math.log(4) / Math.log(8), 12)
    // gamma: h = 1/(1 − γ) = 10 between h 2.5 and 40, the geometric middle.
    expect(at("gamma")).toBeCloseTo(0.5, 12)
  })
  it("fromUnit keeps non-searched genes at the given centre and rejects a wrong-length or out-of-cube u", () => {
    const n = searched(space).length
    const g = fromUnit(space, Array(n).fill(0.25), centre)
    for (const gene of space.genes.filter(x => x.owner !== "gp")) expect(g[gene.name]).toBe(centre[gene.name])
    expect(() => fromUnit(space, Array(n - 1).fill(0.5), centre)).toThrow(RangeError)
    expect(() => fromUnit(space, [...Array(n - 1).fill(0.5), 1.5], centre)).toThrow(RangeError)
    expect(() => fromUnit(space, [...Array(n - 1).fill(0.5), Number.NaN], centre)).toThrow(RangeError)
  })
})

describe("decodeGenome", () => {
  it("decodeGenome(centre) gives exactly DEFAULT_SETTINGS-shaped values: lr 5e-4 … gamma 0.9, caregiver CAREGIVER_DEFAULTS, harness 30/250", () => {
    expect(decodeGenome(centre, space)).toEqual({
      learner: {
        lr: 5e-4, entropyWeight: 0.01, curiosityWeight: 1, bufferSize: 2000, replayPerSleep: 64, headLrScale: 0.1,
        exploreEpsilon: 0.1, residualDecay: 1e-3, gamma: 0.9,
      },
      caregiver: { caregiverPeriod: 500, caregiverTau: 0.05, caregiverEpsilon: 0.1, caregiverThreshold: 0.01, caregiverStuckSteps: 2000 },
      harness: { maxStepsPerSec: 30, sleepInterval: 250 },
    })
  })
  it("decodeGenome never emits an implemented:false gene", () => {
    const reserved = space.genes.filter(g => g.implemented === false).map(g => g.name)
    expect(reserved).toHaveLength(13)
    const rng = createRng(7)
    for (let i = 0; i < 50; i++) {
      const d = decodeGenome(fromUnit(space, searched(space).map(() => rng()), centre), space)
      const keys = [...Object.keys(d.learner), ...Object.keys(d.caregiver), ...Object.keys(d.harness)]
      for (const r of reserved) expect(keys).not.toContain(r)
      expect(keys).toHaveLength(16)
    }
  })
})

describe("checkGenome", () => {
  it("checkGenome rejects a missing key, an extra key, an out-of-range value, a fractional integer and a reserved gene off centre", () => {
    expect(checkGenome(space, centre)).toBeNull()
    const missing = Object.fromEntries(Object.entries(centre).filter(([k]) => k !== "lr"))
    for (const [what, g, why] of [
      ["missing key", missing, /missing gene lr/],
      ["extra key", { ...centre, moveProbability: 0.5 }, /unknown gene "moveProbability"/],
      ["out of range", { ...centre, lr: 1e-2 }, /lr 0\.01 is outside/],
      ["fractional integer", { ...centre, bufferSize: 2000.5 }, /bufferSize must be an integer/],
      ["reserved gene off centre", { ...centre, latentWeight: 1 }, /latentWeight is not implemented and must stay at its centre/],
      ["reserved categorical off centre", { ...centre, creditMode: "gae16" }, /creditMode is not implemented and must stay at its centre/],
      ["string for a number", { ...centre, lr: "5e-4" }, /lr must be a finite number/],
      ["categorical not among values", { ...centre, modulator: "xyz" }, /modulator must be one of/],
      ["non-finite", { ...centre, curiosityWeight: Number.NaN }, /curiosityWeight must be a finite number/],
    ] as [string, Genome, RegExp][]) expect(checkGenome(space, g), what).toMatch(why)
  })
  it("checkGenome holds every fixed gene at its centre, implemented ones too, even inside a wider range", () => {
    const wide = parseGenomeSpace(withGene("maxStepsPerSec", { min: 10, max: 60 }))
    expect(checkGenome(wide, centre)).toBeNull()
    expect(checkGenome(wide, { ...centre, maxStepsPerSec: 40 })).toMatch(/maxStepsPerSec is fixed and must stay at its centre 30/)
    // A searched gene may take any value in its range.
    expect(checkGenome(space, { ...centre, lr: 1e-3 })).toBeNull()
  })
})

describe("legacyGenome", () => {
  it("legacyGenome maps flat settings (entropyWeight 0.001 kept, out of range) and fills absent genes with centres", () => {
    const settings = {
      lr: 4.5e-4, entropyWeight: 0.001, curiosityWeight: 1.7, bufferSize: 3000, replayPerSleep: 100, headLrScale: 0.09,
      exploreEpsilon: 0.11, residualDecay: 9e-4, maxStepsPerSec: 30, moveProbability: 0.4, caregiverPeriod: 700, seed: 12,
    }
    const g = legacyGenome(settings, space)
    expect(Object.keys(g).sort()).toEqual(space.genes.map(x => x.name).sort())
    expect(g.entropyWeight).toBe(0.001)
    expect(g.lr).toBe(4.5e-4)
    expect(g.caregiverPeriod).toBe(700)
    expect(g.gamma).toBe(0.9)
    expect(g.sleepInterval).toBe(250)
    expect(g.caregiverTau).toBe(0.05)
    expect(g.creditMode).toBe("off")
    expect("moveProbability" in g).toBe(false)
    expect(checkGenome(space, g)).toMatch(/entropyWeight/)
    expect(() => legacyGenome({ lr: "x" }, space)).toThrow(RangeError)
  })
})

describe("legacyGenome on real committed logs", () => {
  // Pre-05a lives store their caregiver settings flat inside `settings`, beside the learner's and maxStepsPerSec
  // (harness session.ts: settings = { ...learner, maxStepsPerSec, seed, ...caregiver.settings }); there is no other
  // place. Checked over every committed data/runs file on 2026-10-23.
  const settingsOf = (path: string) =>
    (JSON.parse(readFileSync(new URL(`../../../data/runs/${path}`, import.meta.url), "utf8")) as { settings: Record<string, number> }).settings
  it("a life with a caregiver: every learner, caregiver and harness value is read from settings; seed is ignored", () => {
    const settings = settingsOf("lrn_dfd5c5085147992a/0.json")
    const g = legacyGenome(settings, space)
    for (const k of ["lr", "entropyWeight", "curiosityWeight", "bufferSize", "replayPerSleep", "headLrScale", "exploreEpsilon",
      "residualDecay", "maxStepsPerSec", "caregiverPeriod", "caregiverTau", "caregiverEpsilon", "caregiverThreshold", "caregiverStuckSteps"])
      expect(g[k], k).toBe(settings[k])
    expect(g.lr).toBe(0.0005837009815593332)
    expect(g.bufferSize).toBe(3149)
    expect("seed" in g).toBe(false)
    expect(g.gamma).toBe(0.9) // not logged before 05a; every such life learned at 0.9
    expect(g.sleepInterval).toBe(250)
    expect(decodeGenome(g, space).caregiver).toEqual({
      caregiverPeriod: 500, caregiverTau: 0.05, caregiverEpsilon: 0.1, caregiverThreshold: 0.01, caregiverStuckSteps: 2000,
    })
  })
  it("an older life without caregiver keys gets the caregiver centres", () => {
    const settings = settingsOf("lrn_1787a3c93600cc87/3.json")
    expect("caregiverPeriod" in settings).toBe(false)
    const g = legacyGenome(settings, space)
    expect(g.lr).toBe(settings.lr)
    expect(g.caregiverPeriod).toBe(500)
  })
})

describe("genomeSha", () => {
  it("genomeSha is order-independent and stable (golden hex)", async () => {
    const a = await genomeSha(1, { b: 2, a: "x", c: 0.5 })
    const b = await genomeSha(1, { c: 0.5, a: "x", b: 2 })
    expect(a).toBe(b)
    const canonical = JSON.stringify({ schemaVersion: 1, values: { a: "x", b: 2, c: 0.5 } })
    expect(a).toBe(createHash("sha256").update(canonical).digest("hex"))
    expect(a).toBe("b0f5b1b6271bf4b53c93bc04384ecda05b6b352203105d97a784f10c7f37b3fb")
    expect(await genomeSha(2, { a: "x", b: 2, c: 0.5 })).not.toBe(a)
    expect(await genomeSha(1, centre)).toMatch(/^[0-9a-f]{64}$/)
  })
  it("sha256Hex is the standard sha256", async () => {
    expect(await sha256Hex(new Uint8Array())).toBe("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    expect(await sha256Hex(new TextEncoder().encode("abc"))).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
  })
})

describe("parseCentre", () => {
  it("returns knowledge/genome.json's values and rejects a wrong genomeSchema or a genome that fails checkGenome", () => {
    expect(Object.keys(centre)).toHaveLength(29)
    const file = readJson("genome.json") as Record<string, unknown>
    for (const [what, v] of [
      ["null", null],
      ["schemaVersion 2", { ...file, schemaVersion: 2 }],
      ["genomeSchema 2", { ...file, genomeSchema: 2 }],
      ["values missing", { ...file, values: undefined }],
      ["values off range", { ...file, values: { ...centre, lr: 1 } }],
      ["provenance missing", { ...file, provenance: undefined }],
    ] as [string, unknown][]) expect(() => parseCentre(v, space), what).toThrow(RangeError)
  })
})
