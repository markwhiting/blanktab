import { LEARNER_ID_RE } from "./log"

const KEY = "blanktab:learnerId"

const cryptoRng = (): number => crypto.getRandomValues(new Uint32Array(1))[0]! / 2 ** 32

/** Per-tab identity: sessionStorage is per tab and survives reload, so a reload keeps the id and a second tab gets its own. */
export function getOrCreateLearnerId(session: Storage, rng: () => number = cryptoRng): string {
  const existing = session.getItem(KEY)
  if (existing !== null && LEARNER_ID_RE.test(existing)) return existing
  let hex = ""
  for (let i = 0; i < 16; i++) hex += Math.min(15, Math.floor(rng() * 16)).toString(16)
  const id = `lrn_${hex}`
  session.setItem(KEY, id)
  return id
}
