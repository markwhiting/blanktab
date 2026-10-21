import { describe, it, expect } from "vitest"
import { TEST_KNOBS, TINY_WIDTHS, readTestParams } from "../src/params"

describe("readTestParams", () => {
  it("reads testLifetimeCU, testStepsPerSecond, testSleepInterval, testBackend=cpu and testWidths=tiny", () => {
    expect(readTestParams("?testLifetimeCU=0.05&testStepsPerSecond=100&testSleepInterval=4&testBackend=cpu&testWidths=tiny")).toEqual({
      lifetimeCU: 0.05, stepsPerSecond: 100, sleepInterval: 4, backend: "cpu", widths: TINY_WIDTHS,
    })
  })
  it("ignores missing, zero, negative, malformed and absurd values, and unknown backends or widths", () => {
    expect(readTestParams("")).toEqual({})
    for (const v of ["0", "-5", "abc", "1e9", "99999999", "Infinity", ""]) {
      expect(readTestParams(`?testLifetimeCU=${v}&testStepsPerSecond=${v}&testSleepInterval=${v}`)).toEqual({})
    }
    expect(readTestParams("?testStepsPerSecond=1.5&testSleepInterval=2.5")).toEqual({})
    expect(readTestParams("?testBackend=webgpu&testWidths=huge")).toEqual({})
  })
  it("reads testPollMs and testReloadJitterMs (whole milliseconds, at most a day)", () => {
    expect(readTestParams("?testPollMs=500&testReloadJitterMs=200")).toEqual({ pollMs: 500, reloadJitterMs: 200 })
    for (const v of ["0", "-1", "2.5", "abc", "86400001"]) expect(readTestParams(`?testPollMs=${v}&testReloadJitterMs=${v}`)).toEqual({})
  })
  it("names every knob it reads (the production build test checks none of them survive)", () => {
    expect([...TEST_KNOBS].sort()).toEqual(["testBackend", "testLifetimeCU", "testPollMs", "testReloadJitterMs", "testSleepInterval", "testStepsPerSecond", "testWidths"])
  })
})
