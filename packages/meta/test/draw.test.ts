import { describe, it, expect } from "vitest"
import { createHash } from "node:crypto"
import { readFileSync, writeFileSync } from "node:fs"
import { chiSquareSf } from "@blanktab/detmath"
import {
  birthDraw, checkGenome, evalPanelOf, fromUnit, parseCentre, parseGenomeSpace, parsePosterior, searched,
  PANELS, UNIFORM_SHARE,
  type Draw, type DrawInput, type Genome, type Mode, type Posterior,
} from "@blanktab/meta"

const readJson = (name: string): unknown => JSON.parse(readFileSync(new URL(`../../../knowledge/${name}`, import.meta.url), "utf8"))
const space = parseGenomeSpace(readJson("genome-space.json"))
const centre = parseCentre(readJson("genome.json"), space)
const FIXTURE = new URL("./fixtures/draws.json", import.meta.url)
const HEX = (s: string) => createHash("sha256").update(s).digest("hex")

/** 10 calibration genomes: the centre, then 9 fixed points of the cube (the gate's Sobol points stand in here). */
const n = searched(space).length
const CALIB: Genome[] = [centre, ...Array.from({ length: 9 }, (_, k) => fromUnit(space, Array.from({ length: n }, (_, j) => ((k + 1) * 0.1 + j * 0.037) % 1), centre))]

function posterior(mode: Mode, need: number[] = CALIB.map(() => 3), genomes: Genome[] = CALIB): Posterior {
  return {
    schemaVersion: 1, genomeSchema: 1, evaluator: "search-composite@1", sourceCommit: "a".repeat(40), mode,
    centre: { genomeSha: HEX("centre"), values: centre },
    calibration: { genomes: genomes.map((values, i) => ({ genomeSha: HEX(`g${i}`), values })), need },
    models: {}, incumbent: null, confirmation: null, diagnostics: {},
    page: { eligibleLives: 0, lastPromotion: null, calibrationLives: 0, calibrationTarget: 30, looks: 0 },
  }
}

/** The input a tab would build: the posterior is parsed from its own bytes, and posteriorSha is their sha256. */
function input(learnerId: string, p: Posterior | null, widthsClass: "gpu" | "cpu" = "gpu"): DrawInput {
  if (p === null) return { learnerId, space, centre, widthsClass, posterior: null, posteriorSha: null }
  const text = JSON.stringify(p)
  return { learnerId, space, centre, widthsClass, posterior: parsePosterior(text), posteriorSha: HEX(text) }
}

const ids = (k: number, tag: string) => Array.from({ length: k }, (_, i) => `lrn_${tag}_${i.toString(36)}`)

describe("birthDraw", () => {
  it("birthDraw is a pure function: the same input gives a deep-equal Draw; 20 draws equal fixtures/draws.json", () => {
    const cases: DrawInput[] = []
    const modes: (Mode | null)[] = [null, "calibrate", "search", "paused", "flat"]
    for (let i = 0; i < 20; i++) {
      const mode = modes[i % modes.length]!
      cases.push(input(`lrn_fixture_${i}`, mode === null ? null : posterior(mode), i % 3 === 2 ? "cpu" : "gpu"))
    }
    const draws = cases.map(c => birthDraw(c))
    for (let i = 0; i < cases.length; i++) expect(birthDraw(cases[i]!)).toEqual(draws[i])
    // The 20 cases cover every method 05a has.
    expect(new Set(draws.map(d => d.method))).toEqual(new Set(["centre", "calibrate", "uniform"]))
    if (process.env.DRAW_WRITE === "1") writeFileSync(FIXTURE, JSON.stringify(draws, null, 2) + "\n")
    const fixture = JSON.parse(readFileSync(FIXTURE, "utf8")) as Draw[]
    expect(draws).toEqual(fixture)
  })

  it("no posterior → centre, drawSeed null, genome = centre", () => {
    for (const id of ids(50, "none")) {
      for (const w of ["gpu", "cpu"] as const) {
        const d = birthDraw(input(id, null, w))
        expect(d).toEqual({ role: "search", method: "centre", drawSeed: null, posteriorSha: null, evalPanel: evalPanelOf(id), genome: centre })
      }
    }
    // A posterior without its sha (or the reverse) is no posterior either.
    const p = posterior("flat")
    expect(birthDraw({ learnerId: "lrn_x", space, centre, widthsClass: "gpu", posterior: p, posteriorSha: null }).drawSeed).toBeNull()
    expect(birthDraw({ learnerId: "lrn_x", space, centre, widthsClass: "gpu", posterior: null, posteriorSha: HEX("x") }).drawSeed).toBeNull()
  })

  it("parsePosterior: garbage, wrong schemaVersion, a need/genomes length mismatch → null", () => {
    const good = posterior("calibrate")
    expect(parsePosterior(JSON.stringify(good))).toEqual(good)
    const bad: [string, string][] = [
      ["garbage", "{not json"],
      ["empty", ""],
      ["an array", "[]"],
      ["null", "null"],
      ["wrong schemaVersion", JSON.stringify({ ...good, schemaVersion: 2 })],
      ["need/genomes length mismatch", JSON.stringify({ ...good, calibration: { ...good.calibration, need: [1, 2] } })],
      ["unknown mode", JSON.stringify({ ...good, mode: "ts" })],
      ["negative need", JSON.stringify({ ...good, calibration: { ...good.calibration, need: good.calibration.need.map(() => -1) } })],
      ["calibrate with no genomes", JSON.stringify({ ...good, calibration: { genomes: [], need: [] } })],
      ["a bad genomeSha", JSON.stringify({ ...good, centre: { ...good.centre, genomeSha: "xyz" } })],
      ["an unknown top-level key", JSON.stringify({ ...good, extra: 1 })],
      ["a missing key", JSON.stringify({ ...good, page: undefined })],
    ]
    for (const [why, text] of bad) expect(parsePosterior(text), why).toBeNull()
  })

  it("calibrate on gpu: every draw is calibrate; over 2,000 learnerIds the genome frequencies follow need [3,0,…] (only index 0) and [1,1,…] (uniform, χ² p > 0.001)", () => {
    const only0 = posterior("calibrate", CALIB.map((_, i) => i === 0 ? 3 : 0))
    for (const id of ids(2000, "c0")) {
      const d = birthDraw(input(id, only0))
      expect(d.method).toBe("calibrate")
      expect(d.role).toBe("search")
      expect(d.calibrationIndex).toBe(0)
      expect(d.genome).toEqual(CALIB[0])
    }
    const counts = CALIB.map(() => 0)
    const even = posterior("calibrate", CALIB.map(() => 1))
    for (const id of ids(2000, "c1")) {
      const d = birthDraw(input(id, even))
      expect(d.method).toBe("calibrate")
      expect(d.genome).toEqual(CALIB[d.calibrationIndex!])
      counts[d.calibrationIndex!]!++
    }
    const e = 2000 / CALIB.length
    const chi2 = counts.reduce((s, c) => s + (c - e) * (c - e) / e, 0)
    expect(chiSquareSf(chi2, CALIB.length - 1)).toBeGreaterThan(0.001)
    // Σneed = 0: uniform over the genomes.
    const zero = posterior("calibrate", CALIB.map(() => 0))
    const seen = new Set(ids(2000, "cz").map(id => birthDraw(input(id, zero)).calibrationIndex))
    expect(seen.size).toBe(CALIB.length)
  })

  it("calibrate on cpu: only uniform or centre; uniform share 0.2 ± 0.03 over 2,000 ids", () => {
    let uniform = 0
    for (const id of ids(2000, "cpu")) {
      const d = birthDraw(input(id, posterior("calibrate"), "cpu"))
      expect(["uniform", "centre"]).toContain(d.method)
      expect(d.calibrationIndex).toBeUndefined()
      if (d.method === "uniform") uniform++
      else expect(d.genome).toEqual(centre)
    }
    expect(Math.abs(uniform / 2000 - UNIFORM_SHARE)).toBeLessThanOrEqual(0.03)
  })

  it("flat/paused/search: uniform share 0.2 ± 0.03; uniform genomes pass checkGenome and keep fixed genes at centre", () => {
    const fixed = space.genes.filter(g => g.owner !== "gp").map(g => g.name)
    for (const mode of ["flat", "paused", "search"] as const) {
      for (const w of ["gpu", "cpu"] as const) {
        let uniform = 0
        const p = posterior(mode)
        for (const id of ids(2000, `${mode}${w}`)) {
          const d = birthDraw(input(id, p, w))
          expect(d.role).toBe("search")
          expect(typeof d.drawSeed).toBe("number")
          if (d.method === "centre") { expect(d.genome).toEqual(centre); continue }
          expect(d.method).toBe("uniform")
          uniform++
          expect(checkGenome(space, d.genome)).toBeNull()
          for (const name of fixed) expect(d.genome[name]).toBe(centre[name])
        }
        expect(Math.abs(uniform / 2000 - UNIFORM_SHARE), `${mode} ${w}`).toBeLessThanOrEqual(0.03)
      }
    }
  })

  it("evalPanelOf is in 1..16 and roughly uniform over 1,600 ids", () => {
    const counts = Array.from({ length: PANELS }, () => 0)
    for (const id of ids(1600, "panel")) {
      const p = evalPanelOf(id)
      expect(Number.isInteger(p) && p >= 1 && p <= PANELS).toBe(true)
      counts[p - 1]!++
    }
    const e = 1600 / PANELS
    const chi2 = counts.reduce((s, c) => s + (c - e) * (c - e) / e, 0)
    expect(chiSquareSf(chi2, PANELS - 1)).toBeGreaterThan(0.001)
    expect(Math.min(...counts)).toBeGreaterThan(0)
  })
})
