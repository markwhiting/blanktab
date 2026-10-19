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
