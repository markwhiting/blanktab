import { describe, it, expect } from "vitest"
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { BUDGET_CU, FILTER_LEVELS } from "@blanktab/harness/log"
import { GPU_WIDTHS } from "@blanktab/learner"
import { GPU_WIDTHS_CLASS, REGISTRY, SEARCH_BUDGET_CU, SEARCH_LEVELS, parseNewbornReference } from "@blanktab/evaluators"
import { loadEvalConfig, loadReferences } from "./config"

const ROOT = fileURLToPath(new URL("../..", import.meta.url))
const config = loadEvalConfig(ROOT)

/**
 * Pins whose evaluators land in Plan 05a Task 8. Each must NOT be registered yet, so Task 8 has to empty this list when it
 * registers them; after that every pin must resolve.
 */
const PENDING_TASK_8 = ["signal-check@1", "confirm-fisher@1"]

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
      if (PENDING_TASK_8.includes(pin)) expect(REGISTRY[pin], `${pin} is registered: remove it from PENDING_TASK_8`).toBeUndefined()
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
    // search-composite@1 carries its own copies (a versioned evaluator never moves with harness constants).
    expect(SEARCH_BUDGET_CU).toBe(config.budgetCU)
    expect([...SEARCH_LEVELS]).toEqual(config.levels)
  })

  it("references.newborn is under config/references/ and, if present, parses", () => {
    expect(config.references.newborn.startsWith("config/references/")).toBe(true)
    expect(config.references.newborn).not.toContain("..")
    const path = join(ROOT, config.references.newborn)
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
    ]) {
      write(bad)
      expect(() => loadEvalConfig(dir), JSON.stringify(bad)).toThrow()
    }
    write(good)
    const c = loadEvalConfig(dir)
    expect(loadReferences(dir, c)).toEqual({})
    const refPath = join(dir, c.references.newborn)
    writeFileSync(refPath, "{\"not\": \"a reference\"}")
    expect(() => loadReferences(dir, c)).toThrow(/newborn reference/)
    const ref = JSON.parse(readFileSync(new URL("../../packages/evaluators/test/fixtures/reference.json", import.meta.url), "utf8"))
    writeFileSync(refPath, JSON.stringify(ref))
    expect(loadReferences(dir, c)).toEqual({ newborn: ref })
  })
})
