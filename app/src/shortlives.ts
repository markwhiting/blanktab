// The short-lives switch as a life obeys it (decisions 0019, 0022; final review, controller ruling): the page's switch
// ends a life at its budget checkup, but a life that had already had that checkup when this tab resumed it (or when the
// switch is turned on) ends only after an explicit confirm for that learner. The stored switch alone never ends one.

/** What the worker hands the session: its inbox's switch, read at clean sleeps. */
export interface ShortLivesInbox { setShortLives(on: boolean): void }

/** localStorage: this learner's answer to the past-budget question ("1" yes, "0" no), so it survives a code-update reload. */
export const consentKey = (learnerId: string): string => `blanktab:shortLivesConsent:${learnerId}`
export type Consent = "yes" | "no" | null
export function readConsent(ls: Pick<Storage, "getItem">, learnerId: string): Consent {
  const v = ls.getItem(consentKey(learnerId))
  return v === "1" ? "yes" : v === "0" ? "no" : null
}
export function writeConsent(ls: Pick<Storage, "setItem" | "removeItem">, learnerId: string, c: Consent): void {
  if (c === null) ls.removeItem(consentKey(learnerId))
  else ls.setItem(consentKey(learnerId), c === "yes" ? "1" : "0")
}

/**
 * The worker's side: what it tells the session's inbox. The switch reaches the session only if the life had not had its
 * budget checkup when this run resumed it (`resumed` then `meta`, as the session emits them before any sleep), or if
 * the page says this learner's end was confirmed (`consent`).
 */
export function shortLivesGate(inbox: ShortLivesInbox, init: { on: boolean; consent: boolean }) {
  let on = init.on, consent = init.consent
  let past = false
  let resumedLevel = 0
  const apply = () => inbox.setShortLives(on && (!past || consent))
  apply()
  return {
    /** The page's switch changed; `confirmed`: the page asked about this learner and was told yes. */
    switched(o: boolean, confirmed = false): void {
      on = o
      if (confirmed) consent = true
      apply()
    },
    /** A run of the life starts (the first, or a restore): until `resumed`, a newborn's (nothing checked yet). */
    run(): void { resumedLevel = 0 },
    /** The session resumed the life: the highest checkup level it has had. */
    resumed(lastLevel: number): void { resumedLevel = lastLevel },
    /** The life's block is known: its budget. Returns the level this run resumed at (0 for a newborn), for the page. */
    meta(budgetCU: number): number {
      past = resumedLevel >= budgetCU
      apply()
      return resumedLevel
    },
  }
}
