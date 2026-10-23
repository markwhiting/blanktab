import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { parseLogFile, type LogFile } from "@blanktab/harness/log"
import { createRng } from "@blanktab/detmath"
import {
  ALPHA_LOOK, REGISTRY, cohortContext, getEvaluator, parseNewbornReference, searchComposite, signalCheck,
  type Evaluator, type LifeInput, type NewbornReference,
} from "@blanktab/evaluators"
import { lid, meta, scoredLife } from "./fixtures/build"

const REF = parseNewbornReference(JSON.parse(readFileSync(new URL("./fixtures/reference.json", import.meta.url), "utf8"))) as NewbornReference
const refs = { newborn: REF }
const ctx = cohortContext({ search: "search-composite@1" })
const sha = (g: number) => (g + 1).toString(16).padStart(64, "0")

/**
 * Life r of genome g with a planted genome effect: M2 rises 0.4 per genome and the held-out ratio falls 0.01 per genome;
 * the repeat noise (±0.05 on M2, ±0.002 on the ratio) is far smaller than a genome step. `label` is the genome it is
 * filed under (its meta says so too), which a shuffle changes.
 */
function life(g: number, r: number, label = g): LifeInput {
  const learnerId = lid(0x100 * (g + 1) + r)
  const files = scoredLife(learnerId, {
    ms: { M1: 0.9, M2: 0.3 + 0.4 * g + 0.05 * (r - 1), M3: 0.2, M4: 0.3, M5: 0.8, M6: 0 },
    ratio: 0.98 - 0.01 * g + 0.002 * (r - 1),
    meta: meta({ genomeSha: sha(label) }),
  })
  return { learnerId, files }
}
const grouped = (lives: { life: LifeInput; label: number }[]): Record<string, LifeInput[]> => {
  const out: Record<string, LifeInput[]> = {}
  for (const { life: l, label } of lives) (out[sha(label)] ??= []).push(l)
  return out
}
const parses = (files: LogFile[]) => files.every(f => parseLogFile(JSON.parse(JSON.stringify(f))).ok)
const run = (groups: Record<string, LifeInput[]>) => signalCheck.evaluateCohort!(groups, refs, ctx)

describe("signal-check@1 (spec §2.6, §4.6)", () => {
  it("is registered as a cohort evaluator, and needs the pinned search evaluator in its context", () => {
    expect(REGISTRY["signal-check@1"]).toBe(signalCheck)
    expect(getEvaluator("signal-check@1")).toBe(signalCheck)
    expect(signalCheck).toMatchObject({ id: "signal-check", version: 1, kind: "cohort", needs: [] })
    expect(ctx.search).toBe(searchComposite)
    expect(() => signalCheck.evaluateCohort!({}, refs)).toThrow(/pinned search evaluator/)
    expect(() => signalCheck.evaluateCohort!({}, refs, { search: signalCheck })).toThrow(/pinned search evaluator/)
  })

  it("10 genomes × 3 lives with a planted genome effect → score 1", () => {
    const lives = Array.from({ length: 30 }, (_, i) => ({ life: life(Math.floor(i / 3), i % 3), label: Math.floor(i / 3) }))
    expect(lives.every(l => parses(l.life.files))).toBe(true)
    const out = run(grouped(lives))
    expect(out).toMatchObject({ evaluator: "signal-check", version: 1, eligible: true, score: 1 })
    const d = out.diagnostics as { H: number; df: number; p: number; icc: number; n: number; k: number; search: string; learnerIds: Record<string, string[]> }
    expect(d).toMatchObject({ n: 30, k: 10, df: 9, search: "search-composite@1" })
    expect(d.p).toBeLessThan(ALPHA_LOOK)
    expect(d.icc).toBeGreaterThan(0.9)
    expect(out.components).toEqual({ H: d.H, p: d.p, icc: d.icc })
    expect(Object.keys(d.learnerIds).length).toBe(10)
    expect(d.learnerIds[sha(0)]).toEqual([lid(0x100), lid(0x101), lid(0x102)])
    // Pure: the same input gives byte-identical output, whatever the order of groups and lives.
    const reversed = grouped([...lives].reverse())
    expect(JSON.stringify(run(reversed))).toBe(JSON.stringify(out))
  })

  it("the same lives with genome labels shuffled by a seeded permutation → score 0 on ≥ 95 of 100 permutations", () => {
    const rng = createRng(20261023)
    let zeros = 0
    for (let t = 0; t < 100; t++) {
      const labels = Array.from({ length: 30 }, (_, i) => Math.floor(i / 3))
      for (let i = labels.length - 1; i > 0; i--) { // Fisher–Yates
        const j = Math.floor(rng() * (i + 1));
        [labels[i], labels[j]] = [labels[j]!, labels[i]!]
      }
      const lives = labels.map((label, i) => ({ life: life(Math.floor(i / 3), i % 3, label), label }))
      const out = run(grouped(lives))
      expect(out.eligible).toBe(true)
      if (out.score === 0) zeros++
    }
    expect(zeros).toBeGreaterThanOrEqual(95)
  })

  it("5 genomes → ineligible", () => {
    // 5 genomes × 4 lives: 20 lives, but fewer than 6 genomes.
    const five = Array.from({ length: 20 }, (_, i) => ({ life: life(Math.floor(i / 4), i % 4), label: Math.floor(i / 4) }))
    expect(run(grouped(five))).toMatchObject({ eligible: false, reason: "fewer than 6 genomes with 2 or more lives", score: null })
    // 7 genomes × 2 lives: enough genomes, but fewer than 20 lives.
    const few = Array.from({ length: 14 }, (_, i) => ({ life: life(Math.floor(i / 2), i % 2), label: Math.floor(i / 2) }))
    expect(run(grouped(few))).toMatchObject({ eligible: false, reason: "fewer than 20 lives", score: null })
    // 6 genomes × 2 lives plus 8 singletons: 20 lives, 6 repeated genomes, so it is eligible.
    const mixed = [
      ...Array.from({ length: 12 }, (_, i) => ({ life: life(Math.floor(i / 2), i % 2), label: Math.floor(i / 2) })),
      ...Array.from({ length: 8 }, (_, i) => ({ life: life(6 + i, 0), label: 6 + i })),
    ]
    expect(run(grouped(mixed)).eligible).toBe(true)
  })

  it("only eligible, correctly filed, once-listed lives count; each is reported", () => {
    const lives = Array.from({ length: 30 }, (_, i) => ({ life: life(Math.floor(i / 3), i % 3), label: Math.floor(i / 3) }))
    const groups = grouped(lives)
    // A life filed under the wrong genome, one listed twice, and one the search evaluator rejects (no newborn reference
    // would reject all, so use a non-search role).
    groups[sha(0)]!.push(life(1, 5)) // its meta says genome 1
    groups[sha(2)]!.push(groups[sha(1)]![0]!)
    const legacy = scoredLife(lid(0xfff), { ms: {}, ratio: 0.9, meta: meta({ genomeSha: sha(3), role: "legacy", method: "legacy", evalPanel: null, drawSeed: null, posteriorSha: null }) })
    groups[sha(3)]!.push({ learnerId: lid(0xfff), files: legacy })
    const out = run(groups)
    expect(out.diagnostics).toMatchObject({ n: 30, mislabelled: 1, duplicates: 1, ineligible: 1 })
    expect(out.score).toBe(1)
  })

  it("scores lives through the pinned search evaluator it is given, not a fixed one", () => {
    // A stand-in search evaluator that ranks every life the same: no signal can be found through it.
    const flat: Evaluator = {
      ...searchComposite, id: "flat-search", version: 1,
      evaluateLife: (l, r) => ({ ...searchComposite.evaluateLife!(l, r), evaluator: "flat-search", score: 0, components: { C: 0, H: 0 } }),
      rankAverage: outs => outs.filter(o => o.eligible).map(() => 0.5),
    }
    const lives = Array.from({ length: 30 }, (_, i) => ({ life: life(Math.floor(i / 3), i % 3), label: Math.floor(i / 3) }))
    const out = signalCheck.evaluateCohort!(grouped(lives), refs, { search: flat })
    expect(out).toMatchObject({ eligible: true, score: 0, diagnostics: { search: "flat-search@1", p: 1, icc: 0 } })
  })
})
