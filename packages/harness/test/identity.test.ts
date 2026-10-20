import { describe, it, expect } from "vitest"
import { getOrCreateLearnerId, newLearnerId } from "../src/identity"
import { LEARNER_ID_RE } from "../src/log"

import { stubStorage } from "./helpers"

describe("identity", () => {
  it("creates a valid id and returns the same one from the same storage", () => {
    const s = stubStorage()
    const a = getOrCreateLearnerId(s)
    expect(a).toMatch(LEARNER_ID_RE)
    expect(getOrCreateLearnerId(s)).toBe(a)
    expect(s.getItem("blanktab:learnerId")).toBe(a)
  })
  it("gives different storages different ids", () => {
    expect(getOrCreateLearnerId(stubStorage())).not.toBe(getOrCreateLearnerId(stubStorage()))
  })
  it("replaces a malformed stored id", () => {
    const s = stubStorage({ "blanktab:learnerId": "garbage" })
    expect(getOrCreateLearnerId(s)).toMatch(LEARNER_ID_RE)
  })
  it("uses the injected rng", () => {
    expect(getOrCreateLearnerId(stubStorage(), () => 0)).toBe("lrn_0000000000000000")
  })
  it("newLearnerId replaces the tab's id (a newborn after a final session)", () => {
    const s = stubStorage()
    const a = getOrCreateLearnerId(s)
    const b = newLearnerId(s)
    expect(b).toMatch(LEARNER_ID_RE)
    expect(b).not.toBe(a)
    expect(getOrCreateLearnerId(s)).toBe(b)
  })
})
