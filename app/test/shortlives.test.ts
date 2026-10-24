import { describe, it, expect } from "vitest"
import { SessionInbox } from "@blanktab/harness/session"
import { consentKey, readConsent, shortLivesGate, writeConsent } from "../src/shortlives"
import { stubStorage } from "../../packages/harness/test/helpers"

describe("the short-lives gate (final review: ruling on a past-budget life)", () => {
  it("a life resumed past its budget: the stored switch alone never reaches the session; a confirm does", () => {
    const inbox = new SessionInbox()
    const g = shortLivesGate(inbox, { on: true, consent: false })
    g.run()
    g.resumed(2048)
    expect(g.meta(2048)).toBe(2048)
    expect(inbox.shortLives).toBe(false)
    g.switched(false)
    g.switched(true) // on again, without the page's confirm: still not
    expect(inbox.shortLives).toBe(false)
    g.switched(true, true)
    expect(inbox.shortLives).toBe(true)
    // A restore resumes it past its budget again: the consent is kept.
    g.run()
    g.resumed(3072)
    g.meta(2048)
    expect(inbox.shortLives).toBe(true)
  })

  it("with a stored consent (a reload after the confirm), the switch reaches the session at once", () => {
    const inbox = new SessionInbox()
    const g = shortLivesGate(inbox, { on: true, consent: true })
    g.run()
    g.resumed(4096)
    g.meta(2048)
    expect(inbox.shortLives).toBe(true)
  })

  it("lives below their budget keep the switch as it is: a newborn, and one resumed before its budget checkup", () => {
    for (const resumed of [null, 1024]) {
      const inbox = new SessionInbox()
      const g = shortLivesGate(inbox, { on: true, consent: false })
      expect(inbox.shortLives).toBe(true)
      g.run()
      if (resumed !== null) g.resumed(resumed)
      expect(g.meta(2048)).toBe(resumed ?? 0)
      expect(inbox.shortLives).toBe(true)
      g.switched(false)
      expect(inbox.shortLives).toBe(false)
      g.switched(true)
      expect(inbox.shortLives).toBe(true)
    }
  })

  it("consent is stored per learnerId: yes, no, or none", () => {
    const ls = stubStorage()
    const a = "lrn_000000000000000a", b = "lrn_000000000000000b"
    expect(readConsent(ls, a)).toBeNull()
    writeConsent(ls, a, "yes")
    writeConsent(ls, b, "no")
    expect([readConsent(ls, a), readConsent(ls, b)]).toEqual(["yes", "no"])
    expect(ls.getItem(consentKey(a))).toBe("1")
    writeConsent(ls, a, null)
    expect(readConsent(ls, a)).toBeNull()
  })
})
