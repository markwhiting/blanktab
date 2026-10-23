import { describe, it, expect } from "vitest"
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { BUDGET_CU, FILTER_LEVELS } from "@blanktab/harness/log"
import { GPU_WIDTHS } from "@blanktab/learner"
import { GPU_WIDTHS_CLASS, REFERENCE_PARSERS, REGISTRY, parseNewbornReference, searchComposite, type Evaluator } from "@blanktab/evaluators"
import { PENDING_PINS, loadEvalConfig, loadReferences, neededReferences } from "./config"

const ROOT = fileURLToPath(new URL("../..", import.meta.url))
const config = loadEvalConfig(ROOT)


describe("config/evaluation.json (spec §2.2)", () => {
  it("is spec §2.2 verbatim, with this plan's decision", () => {
    expect(config).toEqual({
      schemaVersion: 1,
      search: "search-composite@1", confirm: "confirm-fisher@1", signal: "signal-check@1", baseline: "newborn-baseline@1",
      budgetCU: 2048, levels: [512, 1024, 2048],
      references: { newborn: "config/references/newborn-baseline@1-gpu-suite1-genome1.json" },
      decision: "docs/decisions/0020-evaluators-v1.md",
    })
    expect(existsSync(join(ROOT, config.decision))).toBe(true)
  })

  it("every pinned id@version is in REGISTRY", () => {
    for (const pin of [config.search, config.confirm, config.signal, config.baseline]) {
      // PENDING_PINS (Task 8's) must not be registered yet: registering one fails here until it leaves the list.
      if (PENDING_PINS.includes(pin)) expect(REGISTRY[pin], `${pin} is registered: remove it from PENDING_PINS`).toBeUndefined()
      else expect(REGISTRY[pin], pin).toBeDefined()
    }
    // Each registry entry's key is its own id@version, and its kind fits its role.
    for (const [key, e] of Object.entries(REGISTRY)) expect(`${e.id}@${e.version}`).toBe(key)
    expect(REGISTRY[config.search]!.kind).toBe("life")
    expect(REGISTRY[config.baseline]!.kind).toBe("cohort")
  })

  it("budgetCU equals BUDGET_CU and levels equal FILTER_LEVELS", () => {
    expect(config.budgetCU).toBe(BUDGET_CU)
    expect(config.levels).toEqual([...FILTER_LEVELS])
    // The pinned search evaluator carries its own budget (a versioned evaluator never moves with harness constants), and
    // the config must agree with whichever evaluator is pinned.
    const budget = REGISTRY[config.search]!.budget
    expect(budget, `${config.search} declares no budget`).toBeDefined()
    expect(budget!.budgetCU).toBe(config.budgetCU)
    expect([...budget!.levels]).toEqual(config.levels)
  })

  it("references.newborn is under config/references/ and, if present, parses", () => {
    // What loads is what the pinned evaluators need.
    expect(neededReferences(config)).toEqual(["newborn"])
    expect(Object.keys(REFERENCE_PARSERS)).toEqual(["newborn"])
    const rel = config.references.newborn!
    expect(rel.startsWith("config/references/")).toBe(true)
    expect(rel).not.toContain("..")
    const path = join(ROOT, rel)
    const refs = loadReferences(ROOT, config)
    if (existsSync(path)) {
      expect(parseNewbornReference(JSON.parse(readFileSync(path, "utf8")))).not.toBeNull()
      expect(refs.newborn).toBeDefined()
    } else {
      expect(refs).toEqual({})
    }
  })

  it("GPU_WIDTHS_CLASS equals the learner's GPU_WIDTHS", () => {
    expect({ ...GPU_WIDTHS_CLASS }).toEqual({ ...GPU_WIDTHS })
  })

  it("loading rejects a malformed config or reference, and reads a present reference", () => {
    const dir = mkdtempSync(join(tmpdir(), "evalcfg-"))
    mkdirSync(join(dir, "config", "references"), { recursive: true })
    const write = (o: unknown) => writeFileSync(join(dir, "config", "evaluation.json"), JSON.stringify(o))
    const good = { ...config }
    for (const bad of [
      { ...good, schemaVersion: 2 }, { ...good, search: "search-composite" }, { ...good, extra: true },
      { ...good, levels: [512, "1024"] }, { ...good, references: { newborn: "data/x.json" } },
      { ...good, references: { newborn: "config/references/../../x.json" } }, { ...good, budgetCU: -1 },
      { ...good, references: {} }, { ...good, references: { ...good.references, spare: "config/references/spare.json" } },
      { ...good, search: "search-composite@7" },
    ]) {
      write(bad)
      expect(() => loadEvalConfig(dir), JSON.stringify(bad)).toThrow()
    }
    write(good)
    const c = loadEvalConfig(dir)
    expect(loadReferences(dir, c)).toEqual({})
    const refPath = join(dir, c.references.newborn!)
    writeFileSync(refPath, "{\"not\": \"a reference\"}")
    expect(() => loadReferences(dir, c)).toThrow(/newborn reference/)
    const ref = JSON.parse(readFileSync(new URL("../../packages/evaluators/test/fixtures/reference.json", import.meta.url), "utf8"))
    writeFileSync(refPath, JSON.stringify(ref))
    expect(loadReferences(dir, c)).toEqual({ newborn: ref })
  })

  it("an evaluator needing an unknown reference fails config load with a clear message", () => {
    const dir = mkdtempSync(join(tmpdir(), "evalcfg-"))
    mkdirSync(join(dir, "config"), { recursive: true })
    writeFileSync(join(dir, "config", "evaluation.json"), JSON.stringify(config))
    const needsMystery: Evaluator = { ...searchComposite, needs: ["newborn", "mystery"] }
    const evaluators = { ...REGISTRY, [config.search]: needsMystery }
    expect(() => loadEvalConfig(dir, { evaluators, parsers: REFERENCE_PARSERS }))
      .toThrow(`search-composite@1 needs reference "mystery", which no parser is registered for`)
    // With a parser but no pinned path, it names the missing pin instead.
    expect(() => loadEvalConfig(dir, { evaluators, parsers: { ...REFERENCE_PARSERS, mystery: () => ({}) } }))
      .toThrow(`search-composite@1 needs reference "mystery", which references does not pin`)
  })
})
