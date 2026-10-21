// Test-only knobs. main.ts reads them only when import.meta.env.MODE !== "production",
// so the deploy build drops this code entirely (checked by app/test/build.test.ts).
import type { Widths } from "@blanktab/learner"

export interface TestParams {
  /** Overrides the lifetime in compute units (default: none, a learner lives as long as its tab). */
  lifetimeCU?: number
  /** Overrides the speed cap (the learner's maxStepsPerSec setting). */
  stepsPerSecond?: number
  /** Overrides the steps between sleeps (default DEFAULT_SLEEP_INTERVAL). */
  sleepInterval?: number
  /** Skips WebGPU. */
  backend?: "cpu"
  /** Overrides the widths of a newborn (on either backend). */
  widths?: Widths
}

/** Every query knob readTestParams reads. */
export const TEST_KNOBS = ["testLifetimeCU", "testStepsPerSecond", "testSleepInterval", "testBackend", "testWidths"] as const

/** testWidths=tiny: a learner small enough for a few-second e2e life on any CPU. */
export const TINY_WIDTHS: Widths = Object.freeze({ conv1: 1, conv2: 1, enc: 2, hidden: 2 })

const positive = (v: string | null, max: number, integer: boolean): number | undefined => {
  if (v === null || !/^\d+(\.\d+)?$/.test(v)) return undefined
  const n = Number(v)
  if (integer && !Number.isInteger(n)) return undefined
  return n > 0 && n <= max ? n : undefined
}

export function readTestParams(search: string): TestParams {
  const q = new URLSearchParams(search)
  const out: TestParams = {}
  const lifetimeCU = positive(q.get("testLifetimeCU"), 1_000_000, false)
  const stepsPerSecond = positive(q.get("testStepsPerSecond"), 10_000, true)
  const sleepInterval = positive(q.get("testSleepInterval"), 1_000_000, true)
  if (lifetimeCU !== undefined) out.lifetimeCU = lifetimeCU
  if (stepsPerSecond !== undefined) out.stepsPerSecond = stepsPerSecond
  if (sleepInterval !== undefined) out.sleepInterval = sleepInterval
  if (q.get("testBackend") === "cpu") out.backend = "cpu"
  if (q.get("testWidths") === "tiny") out.widths = { ...TINY_WIDTHS }
  return out
}
