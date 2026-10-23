import { describe, it, expect } from "vitest"
import { parseLogFile, type CheckupResult, type LogFileV2 } from "@blanktab/harness/log"
import { REGISTRY, newbornBaseline, parseNewbornReference, type LifeInput } from "@blanktab/evaluators"
import { checkup, CPU, file, lid, meta, typicalLife, type Mid } from "./fixtures/build"

const LEVELS = [512, 1024, 2048] as const

/** A newborn's one file (spec §2.3; plan gap 5): seq 0, cu 0, final, its three checkups and its held-out score. */
function newborn(k: number, cks: CheckupResult[], ratio: number, over: Partial<LogFileV2> = {}): LifeInput {
  const learnerId = lid(0x1000 + k)
  const f = {
    ...file({
      learnerId, seq: 0, cu: 0, final: true, checkups: cks.map(c => ({ ...c, steps: 0 })),
      meta: meta({ role: "newborn", method: "newborn", drawSeed: null, posteriorSha: null, evalPanel: 1 + (k % 16), heldOut: { loss: ratio * 2, baseline: 2 } }),
    }),
    ...over,
  }
  return { learnerId, files: [f] }
}
const parses = (l: LifeInput) => l.files.every(f => parseLogFile(JSON.parse(JSON.stringify(f))).ok)

/**
 * Newborn k's measures. M2 = 0.5·(k mod 8) at every level, passing when ≥ 2. M3 passes at 512 for k < 32, at 512 and
 * 1024 for 32 ≤ k < 48, never above. M6 is always null. M1 = 1 at 512, null elsewhere. M4 = M5 = 0.
 */
function cohort(): LifeInput[] {
  return Array.from({ length: 64 }, (_, k) => {
    const cks = LEVELS.map(level => {
      const m2 = 0.5 * (k % 8)
      const m3pass = (level === 512 && k < 48) || (level === 1024 && k >= 32 && k < 48)
      const passed: Mid[] = [...(m2 >= 2 ? ["M2" as const] : []), ...(m3pass ? ["M3" as const] : [])]
      return checkup(level, { M1: level === 512 ? 1 : null, M2: m2, M3: m3pass ? 2 : 0, M4: 0, M5: 0, M6: null }, passed)
    })
    return newborn(k, cks, 0.9 + 0.01 * (k % 5))
  })
}

describe("newborn-baseline@1 (spec §2.3)", () => {
  it("is registered as a cohort evaluator", () => {
    expect(REGISTRY["newborn-baseline@1"]).toBe(newbornBaseline)
    expect(newbornBaseline).toMatchObject({ id: "newborn-baseline", version: 1, kind: "cohort", needs: [] })
  })

  it("64 synthetic newborns → the medians and rates computed by hand; CPU-width and suite-2 newborns are ignored", () => {
    const lives = cohort()
    expect(lives.every(parses)).toBe(true)
    const wild = LEVELS.map(level => checkup(level, { M1: 50, M2: 50, M3: 50, M4: 50, M5: 50, M6: 50 }, ["M1", "M2", "M3", "M4", "M5", "M6"]))
    const cpu = newborn(100, wild, 5, { widths: CPU })
    expect(parses(cpu)).toBe(true)
    // Suite 2 does not exist yet, so its file is built by hand (parseLogFile would reject it); and genome schema 2.
    const suite2 = newborn(101, wild, 5, { milestoneSuiteVersion: 2 })
    const schema2 = newborn(102, wild, 5)
    ;(schema2.files[0] as LogFileV2).meta!.genomeSchema = 2
    // A search life in the group is not a newborn.
    const search: LifeInput = { learnerId: lid(0xd0), files: typicalLife(lid(0xd0)) }
    // Order must not matter: shuffle the cohort deterministically, extras in the middle.
    const shuffled = [...lives.slice(40), cpu, suite2, schema2, search, ...lives.slice(0, 40).reverse()]
    const out = newbornBaseline.evaluateCohort!({ newborn: shuffled }, {})
    expect(out).toMatchObject({ evaluator: "newborn-baseline", version: 1, eligible: true, score: null })
    const reference = (out.diagnostics as { reference: unknown }).reference
    expect(parseNewbornReference(reference)).toEqual(reference)
    // 192 checkups. M2: 0, 0.5, ..., 3.5, each 24 times; the 96th and 97th are 1.5 and 2 → 1.75. Passing (≥ 2): half.
    // M3: 0 (128 times) or 2 (64): median 0; passes 48 + 16 = 64 of 192; sustained (≥ 2 of 3 levels): k in 32..47, 16 of 64.
    // M1: 1 at 512 (64), null as 0 elsewhere (128): median 0. M6: all null → 0.
    // Held-out ratios 0.90 (k mod 5 = 0, 13 lives), 0.91 (13), 0.92 (13), 0.93 (13), 0.94 (12): the 32nd and 33rd are 0.92.
    expect(reference).toEqual({
      evaluator: "newborn-baseline@1", suiteVersion: 1, genomeSchema: 1, widthsClass: "gpu",
      learnerIds: Array.from({ length: 64 }, (_, k) => lid(0x1000 + k)),
      milestones: {
        M1: { median: 0, passRate: 0, sustainedPassRate: 0 },
        M2: { median: 1.75, passRate: 0.5, sustainedPassRate: 0.5 },
        M3: { median: 0, passRate: 64 / 192, sustainedPassRate: 0.25 },
        M4: { median: 0, passRate: 0, sustainedPassRate: 0 },
        M5: { median: 0, passRate: 0, sustainedPassRate: 0 },
        M6: { median: 0, passRate: 0, sustainedPassRate: 0 },
      },
      heldOutMedian: expect.closeTo(0.92, 12),
    })
    expect(out.components).toEqual({ lives: 64, heldOutMedian: expect.closeTo(0.92, 12) })
    expect((out.diagnostics as { ignored: number }).ignored).toBe(4)
  })

  it("no counted newborns is ineligible; the output is byte-identical for the same input", () => {
    const none = newbornBaseline.evaluateCohort!({ newborn: [] }, {})
    expect(none).toMatchObject({ eligible: false, reason: "no newborn lives", score: null })
    expect(newbornBaseline.evaluateCohort!({}, {})).toMatchObject({ eligible: false, reason: "no newborn lives" })
    const lives = cohort()
    expect(JSON.stringify(newbornBaseline.evaluateCohort!({ newborn: lives }, {}))).toBe(JSON.stringify(newbornBaseline.evaluateCohort!({ newborn: lives }, {})))
  })

  it("parseNewbornReference rejects malformed references", () => {
    const good = (newbornBaseline.evaluateCohort!({ newborn: cohort() }, {}).diagnostics as { reference: Record<string, unknown> }).reference
    expect(parseNewbornReference(good)).not.toBeNull()
    const bad: unknown[] = [
      null, [], { ...good, evaluator: "newborn-baseline@2" }, { ...good, widthsClass: "cpu" }, { ...good, extra: 1 },
      { ...good, heldOutMedian: 0 }, { ...good, heldOutMedian: Number.NaN }, { ...good, learnerIds: ["x"] },
      { ...good, learnerIds: [lid(2), lid(1)] }, { ...good, milestones: { ...(good.milestones as object), M7: { median: 0, passRate: 0, sustainedPassRate: 0 } } },
      { ...good, milestones: { ...(good.milestones as object), M2: { median: -1, passRate: 0, sustainedPassRate: 0 } } },
      { ...good, milestones: { ...(good.milestones as object), M2: { median: 0, passRate: 1.5, sustainedPassRate: 0 } } },
    ]
    for (const b of bad) expect(parseNewbornReference(b), JSON.stringify(b)).toBeNull()
  })
})
