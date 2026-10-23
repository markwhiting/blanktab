import { describe, it, expect } from "vitest"
import { readFileSync } from "node:fs"
import { parseLogFile, type LogFile } from "@blanktab/harness/log"
import { normInv } from "@blanktab/detmath"
import {
  REGISTRY, getEvaluator, parseNewbornReference, rankAverage, searchComposite, type EvalOutput, type LifeInput, type NewbornReference,
} from "@blanktab/evaluators"
import { checkup, CPU, file, lid, meta, point, typicalLife } from "./fixtures/build"

const HAND_BYTES = readFileSync(new URL("./fixtures/hand-life.json", import.meta.url), "utf8")
const REF_BYTES = readFileSync(new URL("./fixtures/reference.json", import.meta.url), "utf8")

/** Every fixture file must be a log the gate would accept the shape of. */
function parsed(files: LogFile[]): LogFile[] {
  return files.map(f => {
    const r = parseLogFile(JSON.parse(JSON.stringify(f)))
    if (!r.ok) throw new Error(`fixture does not parse: ${r.error}`)
    return r.value
  })
}
const lifeOf = (learnerId: string, files: LogFile[]): LifeInput => ({ learnerId, files: parsed(files) })
const handLife = (): LifeInput => {
  const j = JSON.parse(HAND_BYTES) as { learnerId: string; files: LogFile[] }
  return lifeOf(j.learnerId, j.files)
}
const ref = (over: Partial<NewbornReference> = {}): NewbornReference => {
  const r = parseNewbornReference({ ...JSON.parse(REF_BYTES), ...over })
  if (!r) throw new Error("reference fixture does not parse")
  return r
}
const score = (life: LifeInput, refs = { newborn: ref() }) => searchComposite.evaluateLife!(life, refs)

describe("search-composite@1 (spec §2.4)", () => {
  it("is registered under its id@version, with the newborn reference as its need and its own budget", () => {
    expect(searchComposite.budget).toEqual({ budgetCU: 2048, levels: [512, 1024, 2048] })
    expect(REGISTRY["search-composite@1"]).toBe(searchComposite)
    expect(getEvaluator("search-composite@1")).toBe(searchComposite)
    expect(searchComposite).toMatchObject({ id: "search-composite", version: 1, kind: "life", needs: ["newborn"] })
    expect(() => getEvaluator("search-composite@99")).toThrow(/unknown evaluator/)
  })

  it("a hand-built life with known measures gives the hand-computed C, H and score to 1e-12", () => {
    // Newborn medians (fixtures/reference.json): M2 0.9, M3 2, M4 0.25, M5 0.7, M6 0. κ: M2 0.2, M3 0.2, M4 0.06, M5 0.15, M6 0.3.
    const clip = (v: number) => Math.min(2, Math.max(-2, v))
    const L512 = [Math.log(1.4 / 1.1), -2 /* ln(.2/2.2) = −2.398, clipped */, Math.log(0.16 / 0.31), Math.log(0.15 / 0.85) /* M5 null → 0 */, 0]
    const L1024 = [Math.log(2.7 / 1.1), Math.log(0.8 / 2.2), Math.log(0.76 / 0.31), Math.log(1.75 / 0.85), Math.log(1.3 / 0.3)]
    const L2048 = [2 /* ln(30.2/1.1) = 3.31 */, -2 /* M3 null: ln(.2/2.2) */, Math.log(0.11 / 0.31), Math.log(0.55 / 0.85), 2 /* ln(4.3/.3) */]
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    const C = mean([mean(L512), mean(L1024), mean(L2048)])
    // Held-out points in (1024, 2048]: 0.93, 0.91, 0.88, 0.90 (1024 itself, 2100 and 3000 are outside; 1300 has none).
    const H = clip(-Math.log(0.905))
    const out = score(handLife())
    expect(out.eligible).toBe(true)
    expect(out.reason).toBeUndefined()
    expect(out.components.C).toBeCloseTo(C, 12)
    expect(out.components.H).toBeCloseTo(H, 12)
    expect(out.score).toBeCloseTo(C + H, 12)
    expect(Object.keys(out.components)).toEqual(["C", "H"])
    const d = out.diagnostics as {
      levels: number[]; q: Record<string, Record<string, number>>; qM1: Record<string, number | null>
      perLevel: Record<string, number>; perMilestone: Record<string, number>; heldOutRatio: number; heldOutPoints: number; heldOutFallback: boolean
    }
    expect(d.levels).toEqual([512, 1024, 2048])
    expect(d.q["1024"]!.M6).toBeCloseTo(Math.log(1.3 / 0.3), 12)
    expect(d.q["512"]!.M3).toBe(-2)
    expect(d.perLevel["512"]).toBeCloseTo(mean(L512), 12)
    expect(d.perMilestone.M4).toBeCloseTo(mean([L512[2]!, L1024[2]!, L2048[2]!]), 12)
    // M1 is a diagnostic only (κ 0.08 = 0.1 × its 0.8 threshold; a null M1 stays null).
    expect(d.qM1["512"]).toBeCloseTo(Math.log(0.98 / 1.03), 12)
    expect(d.qM1["1024"]).toBeNull()
    expect(d.qM1["2048"]).toBeCloseTo(Math.log(0.93 / 1.03), 12)
    expect(d.heldOutRatio).toBeCloseTo(0.905, 12)
    expect(d.heldOutPoints).toBe(4)
    expect(d.heldOutFallback).toBe(false)
    // The newborn held-out median outside 1 ± 0.01: H = −ln(ρ̃ / newborn median).
    const fb = score(handLife(), { newborn: ref({ heldOutMedian: 0.95 }) })
    expect(fb.components.H).toBeCloseTo(-Math.log(0.905 / 0.95), 12)
    expect(fb.components.C).toBeCloseTo(C, 12)
    expect((fb.diagnostics as { heldOutFallback: boolean }).heldOutFallback).toBe(true)
    // Inside 1 ± 0.01: no fallback.
    expect(score(handLife(), { newborn: ref({ heldOutMedian: 1.009 }) }).components.H).toBeCloseTo(H, 12)
  })

  it("each ineligibility reason, one fixture each", () => {
    const id = lid(0xb0)
    const typical = typicalLife(id)
    expect(score(lifeOf(id, typical)).eligible).toBe(true)
    // Born under the v1 budget explicitly is the same as written before the field.
    expect(score(lifeOf(id, typicalLife(id, { meta: meta({ budget: { budgetCU: 2048, levels: [512, 1024, 2048] } }) }))).eligible).toBe(true)
    const cases: [string, LogFile[], object?][] = [
      ["not a search life", typicalLife(id, { meta: meta({ role: "legacy", method: "legacy", evalPanel: null, drawSeed: null, posteriorSha: null }) })],
      ["no meta block before the checkup", [file({ learnerId: id, seq: 0, cu: 512, meta: null, checkups: [checkup(512, {})] }), ...typical.slice(1)]],
      ["not gpu widths", typicalLife(id, { widths: CPU })],
      ["different evaluation budget", typicalLife(id, { meta: meta({ budget: { budgetCU: 4096, levels: [1024, 2048, 4096] } }) })],
      ["no newborn reference", typical, {}],
      ["no 2048 checkup", [...typical.slice(0, 3), file({ learnerId: id, seq: 3, cu: 3072, curve: [point(3000, 0.9)] })]],
      ["fewer than 2 of 3 levels", [
        file({ learnerId: id, seq: 0, cu: 512, curve: [point(512, 0.99)] }),
        file({ learnerId: id, seq: 1, cu: 1024 }),
        ...typical.slice(2),
      ]],
      ["fewer than 3 held-out points in (1024, 2048]", [
        ...typical.slice(0, 2),
        file({ learnerId: id, seq: 2, cu: 2048, curve: [point(1200, 0.95), point(1600), point(2048, 0.93)] }),
        typical[3]!,
      ]],
      ["abandoned", [...typical.slice(0, 2), file({ learnerId: id, seq: 2, cu: 1500, final: true, meta: meta({ end: "orphan" }) })]],
    ]
    // Suite 2 does not exist yet, so its file is built by hand (parseLogFile would reject it).
    const suite2 = score({ learnerId: id, files: [...typical.slice(0, 3), { ...typical[3]!, milestoneSuiteVersion: 2 } as LogFile] })
    expect(suite2).toMatchObject({ eligible: false, reason: "wrong milestone suite", score: null })
    for (const [reason, files, refs] of cases) {
      const out = score(lifeOf(id, files), (refs ?? { newborn: ref() }) as { newborn: NewbornReference })
      expect(out, reason).toMatchObject({ evaluator: "search-composite", version: 1, eligible: false, reason, score: null, components: { C: null, H: null } })
    }
  })

  it("a failed life before 2048 scores C = H = −2; an orphan before 2048 is ineligible 'abandoned'", () => {
    const id = lid(0xc0)
    const head = typicalLife(id).slice(0, 2)
    const failed = score(lifeOf(id, [...head, file({ learnerId: id, seq: 2, cu: 1300, final: true, meta: meta({ end: "failed" }) })]))
    expect(failed).toMatchObject({ eligible: true, score: -4, components: { C: -2, H: -2 } })
    expect((failed.diagnostics as { failed: boolean }).failed).toBe(true)
    // Failed before any checkup at all is still a row.
    expect(score(lifeOf(id, [file({ learnerId: id, seq: 0, cu: 20, final: true, meta: meta({ end: "failed" }) })])).score).toBe(-4)
    // An orphan, and a final file with no end (older code), are abandoned.
    for (const end of ["orphan", undefined] as const) {
      const m = meta(end ? { end } : {})
      expect(score(lifeOf(id, [...head, file({ learnerId: id, seq: 2, cu: 1300, final: true, meta: m })])).reason).toBe("abandoned")
    }
    // A life that failed past 2048 CU before its 2048 checkup ran is a failed row too, not dropped.
    const typical = typicalLife(id)
    const failedLate = score(lifeOf(id, [...typical.slice(0, 3), file({ learnerId: id, seq: 3, cu: 2600, final: true, meta: meta({ end: "failed" }) })]))
    expect(failedLate).toMatchObject({ eligible: true, score: -4, components: { C: -2, H: -2 }, diagnostics: { failed: true, cu: 2600 } })
    // An orphan past 2048 without the checkup is not abandoned (it lived its budget) but has no checkup.
    expect(score(lifeOf(id, [...typical.slice(0, 3), file({ learnerId: id, seq: 3, cu: 2600, final: true, meta: meta({ end: "orphan" }) })])).reason)
      .toBe("no 2048 checkup")
    // A life still running below 2048 (its last file is not final) is not abandoned: its checkup may come.
    expect(score(lifeOf(id, head)).reason).toBe("no 2048 checkup")
  })

  it("only cu ≤ 2048 data is read: appending later files changes nothing", () => {
    const base = handLife()
    const id = base.learnerId
    const later = parsed([
      file({ learnerId: id, seq: 4, cu: 4096, curve: [point(4000, 0.01)], checkups: [checkup(3072, { M2: 99 }), checkup(4096, { M3: 99 })] }),
      file({ learnerId: id, seq: 5, cu: 5120, final: true, widths: CPU, meta: null }),
    ])
    const a = score(base)
    const b = score({ learnerId: id, files: [...base.files, ...later] })
    expect(JSON.stringify(b)).toBe(JSON.stringify(a))
    // A failed end after the budget does not undo a scored life either.
    const c = score({ learnerId: id, files: [...base.files, ...parsed([file({ learnerId: id, seq: 4, cu: 4096, final: true, meta: meta({ end: "failed" }) })])] })
    expect(JSON.stringify(c)).toBe(JSON.stringify(a))
  })

  it("targets: ties get mid-ranks; output equals normInv((rank−0.5)/n) of the re-ranked average", () => {
    const out = (C: number | null, H: number | null, eligible = true): EvalOutput => ({
      evaluator: "search-composite", version: 1, eligible, score: C !== null && H !== null ? C + H : null, components: { C, H }, diagnostics: {},
    })
    // Eligible rows (in order): C = 0.1, 0.5, 0.5, −2, 1; H = 0.3, 0.3, 0.1, −2, 0.2. The ineligible row is skipped.
    const outs = [out(0.1, 0.3), out(null, null, false), out(0.5, 0.3), out(0.5, 0.1), out(-2, -2), out(1, 0.2)]
    // Mid-ranks: C → 2, 3.5, 3.5, 1, 5; H → 4.5, 4.5, 2, 1, 3. u = (rank − 0.5)/5.
    const uC = [1.5, 3, 3, 0.5, 4.5].map(r => r / 5)
    const uH = [4, 4, 1.5, 0.5, 2.5].map(r => r / 5)
    const a = uC.map((u, i) => (u + uH[i]!) / 2) // 0.55, 0.7, 0.45, 0.1, 0.7
    const got = rankAverage(outs)
    expect(got.length).toBe(5)
    got.forEach((g, i) => expect(g).toBeCloseTo(a[i]!, 12))
    // Re-ranked a: 0.55 → 3, 0.7 → 4.5 (a tie: ranks are summed exactly, so floating
    // point cannot split it), 0.45 → 2, 0.1 → 1; so (rank − 0.5)/5 = 0.5, 0.8, 0.3, 0.1, 0.8.
    const p = [0.5, 0.8, 0.3, 0.1, 0.8]
    const y = searchComposite.targets!(outs)
    expect(y).toEqual(p.map(normInv))
    // And Φ⁻¹ itself (R's qnorm), to 1e-9.
    const qnorm = [0, 0.8416212335729143, -0.5244005127080407, -1.2815515655446004, 0.8416212335729143]
    y.forEach((v, i) => expect(v).toBeCloseTo(qnorm[i]!, 9))
    // A tie floating point would split: u = 0.1 + 0.7 and 0.3 + 0.5 (0.7999999999999999 and 0.8) have equal rank sums.
    const split = [out(1, 4), out(2, 3), out(3, 5), out(4, 1), out(5, 2)]
    expect(searchComposite.targets!(split)).toEqual([0.3, 0.3, 0.9, 0.3, 0.7].map(normInv))
    expect(searchComposite.targets!([out(1, 1)])).toEqual([0])
    // Rows never mix evaluator versions: an output of another evaluator or version is refused, eligible or not.
    for (const alien of [{ ...out(0, 0), version: 2 }, { ...out(0, 0, false), evaluator: "other" }]) {
      expect(() => searchComposite.targets!([out(1, 1), alien])).toThrow(/never mix evaluator versions/)
      expect(() => rankAverage([alien, out(1, 1)])).toThrow(/never mix evaluator versions/)
    }
    expect(searchComposite.targets!([])).toEqual([])
  })

  it("the same input JSON bytes give byte-identical JSON output twice", () => {
    const run = () => JSON.stringify(score(handLife(), { newborn: ref() }))
    const first = run()
    expect(run()).toBe(first)
    expect(first).toContain("\"evaluator\":\"search-composite\"")
  })
})
