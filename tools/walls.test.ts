import { ESLint } from "eslint"
import { describe, it, expect } from "vitest"
const eslint = new ESLint()
async function errors(code: string, filePath: string) {
  const [r] = await eslint.lintText(code, { filePath })
  return r!.messages.filter(m => m.ruleId === "no-restricted-imports" || m.ruleId === "no-restricted-syntax")
}
describe("package walls", () => {
  it("learner cannot import world or harness", async () => {
    expect(await errors(`import "@blanktab/world"`, "packages/learner/src/x.ts")).toHaveLength(1)
    expect(await errors(`import "@blanktab/harness"`, "packages/learner/src/x.ts")).toHaveLength(1)
    expect(await errors(`import "../../world/src/index"`, "packages/learner/src/x.ts")).toHaveLength(1)
  })
  it("world cannot import learner, tensor or harness", async () => {
    for (const p of ["learner", "tensor", "harness"])
      expect(await errors(`import "@blanktab/${p}"`, "packages/world/src/x.ts")).toHaveLength(1)
  })
  it("tensor and interface import no workspace package", async () => {
    expect(await errors(`import "@blanktab/interface"`, "packages/tensor/src/x.ts")).toHaveLength(1)
    expect(await errors(`import "@blanktab/tensor"`, "packages/interface/src/x.ts")).toHaveLength(1)
  })
  it("dynamic import and require of a forbidden package are errors", async () => {
    expect(await errors(`await import("@blanktab/world")`, "packages/learner/src/x.ts")).toHaveLength(1)
    expect(await errors(`require("@blanktab/world")`, "packages/learner/src/x.ts")).toHaveLength(1)
  })
  it("allowed imports pass", async () => {
    expect(await errors(`import "@blanktab/tensor"; import "@blanktab/interface"`, "packages/learner/src/x.ts")).toHaveLength(0)
    expect(await errors(`import "@blanktab/world"; import "@blanktab/learner"`, "packages/harness/src/x.ts")).toHaveLength(0)
  })
})

const DETERMINISM_RULES = ["no-restricted-globals", "no-restricted-properties"]
async function determinismErrors(code: string, filePath: string) {
  const [r] = await eslint.lintText(code, { filePath })
  return r!.messages.filter(m => DETERMINISM_RULES.includes(m.ruleId ?? ""))
}
describe("world determinism and text walls (packages/world/src)", () => {
  const SRC = "packages/world/src/x.ts"
  it("forbids Math.random, the Date global, performance and fillText", async () => {
    for (const code of [
      "Math.random()",
      "const { random } = Math",
      "Date.now()",
      "new Date()",
      "performance.now()",
      "globalThis.Date.now()",
      "globalThis.performance.now()",
      "declare const ctx: CanvasRenderingContext2D; ctx.fillText('a', 0, 0)",
      "declare const ctx: CanvasRenderingContext2D; ctx.strokeText('a', 0, 0)",
    ]) expect(await determinismErrors(code, SRC), code).toHaveLength(1)
  })
  it("allows ordinary Math, and the same code outside world/src (tests, bench, other packages)", async () => {
    expect(await determinismErrors("Math.floor(Math.PI)", SRC)).toHaveLength(0)
    for (const f of ["packages/world/test/x.test.ts", "packages/world/bench/x.ts", "packages/harness/src/x.ts"])
      expect(await determinismErrors("Math.random(); Date.now(); performance.now()", f), f).toHaveLength(0)
  })
  it("keeps the package wall (incl. its no-restricted-syntax dynamic-import ban) on world/src", async () => {
    expect(await errors(`await import("@blanktab/learner")`, SRC)).toHaveLength(1)
    expect(await errors(`require("@blanktab/harness")`, SRC)).toHaveLength(1)
    expect(await errors(`import "@blanktab/tensor"`, SRC)).toHaveLength(1)
  })
})
