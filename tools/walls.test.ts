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

describe("walls for detmath, meta and evaluators", () => {
  it("detmath imports nothing; meta only detmath; evaluators only detmath and @blanktab/harness/log", async () => {
    for (const p of ["interface", "tensor", "learner", "world", "harness", "meta", "evaluators"])
      expect(await errors(`import "@blanktab/${p}"`, "packages/detmath/src/x.ts"), p).toHaveLength(1)
    for (const p of ["interface", "tensor", "learner", "world", "harness", "evaluators"])
      expect(await errors(`import "@blanktab/${p}"`, "packages/meta/src/x.ts"), p).toHaveLength(1)
    for (const p of ["interface", "tensor", "learner", "world", "harness", "meta"])
      expect(await errors(`import "@blanktab/${p}"`, "packages/evaluators/src/x.ts"), p).toHaveLength(1)
    expect(await errors(`import "@blanktab/harness/session"`, "packages/evaluators/src/x.ts")).toHaveLength(1)
    expect(await errors(`await import("@blanktab/harness/session")`, "packages/evaluators/src/x.ts")).toHaveLength(1)
    expect(await errors(`import "../../harness/src/session"`, "packages/evaluators/src/x.ts")).toHaveLength(1)
    expect(await errors(`await import("@blanktab/world")`, "packages/meta/src/x.ts")).toHaveLength(1)
    expect(await errors(`require("@blanktab/meta")`, "packages/detmath/test/x.ts")).toHaveLength(1)
    expect(await errors(`import "@blanktab/detmath"`, "packages/meta/src/x.ts")).toHaveLength(0)
    expect(await errors(`import "@blanktab/detmath"; import "@blanktab/harness/log"`, "packages/evaluators/src/x.ts")).toHaveLength(0)
    expect(await errors(`await import("@blanktab/harness/log")`, "packages/evaluators/src/x.ts")).toHaveLength(0)
    expect(await errors(`import "@blanktab/evaluators"; import "@blanktab/meta"`, "packages/harness/src/x.ts")).toHaveLength(0)
  })
  it("learner and world cannot import meta, evaluators or detmath", async () => {
    for (const pkg of ["learner", "world"])
      for (const p of ["meta", "evaluators", "detmath"])
        expect(await errors(`import "@blanktab/${p}"`, `packages/${pkg}/src/x.ts`), `${pkg} -> ${p}`).toHaveLength(1)
  })
  it("Math.random, Date, performance and node:fs are errors in packages/{detmath,meta,evaluators}/src", async () => {
    for (const pkg of ["detmath", "meta", "evaluators"]) {
      const src = `packages/${pkg}/src/x.ts`
      for (const code of ["Math.random()", "Date.now()", "new Date()", "performance.now()", "globalThis.Date.now()", "process.env.X"])
        expect(await determinismErrors(code, src), `${pkg}: ${code}`).toHaveLength(1)
      for (const code of [`import "node:fs"`, `import "node:path"`, `import "fs"`, `import "path"`, `import "fs/promises"`, `await import("node:fs")`])
        expect(await errors(code, src), `${pkg}: ${code}`).toHaveLength(1)
      for (const g of ["crypto.getRandomValues(new Uint8Array(1))", "fetch('x')", "indexedDB.open('x')", "localStorage.x",
        "sessionStorage.x", "setTimeout(() => 0)", "setInterval(() => 0)", "navigator.userAgent", "globalThis.process.env"])
        expect(await determinismErrors(g, src), `${pkg}: ${g}`).toHaveLength(1)
      for (const b of ["os", "child_process", "crypto", "http", "https", "net", "worker_threads", "node:os"])
        expect(await errors(`import "${b}"`, src), `${pkg}: import ${b}`).toHaveLength(1)
      expect(await errors(`require("child_process")`, src)).toHaveLength(1)
      // Tests may use Node and the clock.
      expect(await errors(`import "node:fs"`, `packages/${pkg}/test/x.test.ts`)).toHaveLength(0)
      expect(await determinismErrors("Date.now()", `packages/${pkg}/test/x.test.ts`)).toHaveLength(0)
    }
  })
  it("all maths goes through detmath: Math transcendentals and ** are errors in packages/{detmath,meta,evaluators}/src", async () => {
    const MATH = ["exp", "expm1", "log", "log1p", "log2", "log10", "pow", "sin", "cos", "tan", "asin", "acos", "atan",
      "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "cbrt", "hypot"]
    for (const pkg of ["detmath", "meta", "evaluators"]) {
      const src = `packages/${pkg}/src/x.ts`
      for (const f of MATH) expect(await determinismErrors(`Math.${f}(1, 2)`, src), `${pkg}: Math.${f}`).toHaveLength(1)
      expect(await errors("const y = 2 ** 3", src), `${pkg}: **`).toHaveLength(1)
      expect(await errors("let y = 2; y **= 3", src), `${pkg}: **=`).toHaveLength(1)
      // The exactly rounded and integer operations stay allowed.
      expect(await determinismErrors("Math.sqrt(2); Math.abs(-1); Math.floor(1.5); Math.imul(3, 5)", src)).toHaveLength(0)
      // Tests use Math.exp and friends as oracles.
      const test = `packages/${pkg}/test/x.test.ts`
      expect(await determinismErrors("Math.exp(1); Math.log(2)", test)).toHaveLength(0)
      expect(await errors("const y = 2 ** 3", test)).toHaveLength(0)
    }
    // The world and learner are not under this rule.
    expect(await errors("const y = 2 ** 3", "packages/learner/src/x.ts")).toHaveLength(0)
  })
})
