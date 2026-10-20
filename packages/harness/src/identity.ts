import { LEARNER_ID_RE } from "./log"

const KEY = "blanktab:learnerId"

const cryptoRng = (): number => crypto.getRandomValues(new Uint32Array(1))[0]! / 2 ** 32

/** Per-tab identity: sessionStorage is per tab and survives reload, so a reload keeps the id and a second tab gets its own. */
export function getOrCreateLearnerId(session: Storage, rng: () => number = cryptoRng): string {
  return storedLearnerId(session) ?? newLearnerId(session, rng)
}

/** This tab's learnerId, or null if it has none yet (a new tab). */
export function storedLearnerId(session: Storage): string | null {
  const existing = session.getItem(KEY)
  return existing !== null && LEARNER_ID_RE.test(existing) ? existing : null
}

/** This tab takes over a learner (an orphan from a closed tab): its id becomes this tab's. */
export function adoptLearnerId(session: Storage, learnerId: string): void {
  if (!LEARNER_ID_RE.test(learnerId)) throw new RangeError(`not a learnerId: ${learnerId}`)
  session.setItem(KEY, learnerId)
}

/** A newborn: replaces this tab's id (used after the previous learner's session went final). */
export function newLearnerId(session: Storage, rng: () => number = cryptoRng): string {
  let hex = ""
  for (let i = 0; i < 16; i++) hex += Math.min(15, Math.floor(rng() * 16)).toString(16)
  const id = `lrn_${hex}`
  session.setItem(KEY, id)
  return id
}
