import { describe, it, expect } from "vitest"
import { readTestParams } from "../src/params"

describe("readTestParams", () => {
  it("reads positive integer testLifetime and testStepsPerSecond", () => {
    expect(readTestParams("?testLifetime=600&testStepsPerSecond=100")).toEqual({ lifetimeSteps: 600, stepsPerSecond: 100 })
  })
  it("ignores missing, zero, negative, fractional and absurd values", () => {
    expect(readTestParams("")).toEqual({})
    for (const v of ["0", "-5", "1.5", "abc", "99999999"]) expect(readTestParams(`?testLifetime=${v}&testStepsPerSecond=${v}`)).toEqual({})
  })
})
