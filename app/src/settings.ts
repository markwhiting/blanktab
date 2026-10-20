// A newborn's settings from the knowledge's picks (chooseSettings). Pure, so it is testable.
import { DEFAULT_SETTINGS, type LearnerSettings } from "@blanktab/learner"
import { DEFAULT_MAX_STEPS_PER_SEC } from "@blanktab/harness/session"

const INTEGER: readonly (keyof LearnerSettings)[] = ["bufferSize", "replayPerSleep"]

/**
 * Every LearnerSettings key the picks hold replaces the default (bufferSize and replayPerSleep rounded, at least 1);
 * `maxStepsPerSec` is the speed cap. Any other pick (moveProbability, kept in the space for the old flailer's logs) is
 * not used. The session clamps the result (clampSettings) and logs the values it used.
 */
export function birthSettings(picks: Record<string, number>): { learnerSettings: LearnerSettings; maxStepsPerSec: number } {
  const learnerSettings: LearnerSettings = { ...DEFAULT_SETTINGS }
  for (const k of Object.keys(DEFAULT_SETTINGS) as (keyof LearnerSettings)[]) {
    const v = picks[k]
    if (typeof v !== "number" || !Number.isFinite(v)) continue
    learnerSettings[k] = INTEGER.includes(k) ? Math.max(1, Math.round(v)) : v
  }
  const cap = picks.maxStepsPerSec
  return { learnerSettings, maxStepsPerSec: typeof cap === "number" && Number.isFinite(cap) && cap > 0 ? cap : DEFAULT_MAX_STEPS_PER_SEC }
}
