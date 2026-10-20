// Test-only knobs. main.ts reads them only when import.meta.env.MODE !== "production",
// so the deploy build drops this code entirely (checked by app/test/build.test.ts).
export interface TestParams { lifetimeSteps?: number; stepsPerSecond?: number }

const positiveInt = (v: string | null, max: number): number | undefined => {
  if (v === null || !/^\d+$/.test(v)) return undefined
  const n = Number(v)
  return n > 0 && n <= max ? n : undefined
}

export function readTestParams(search: string): TestParams {
  const q = new URLSearchParams(search)
  const lifetimeSteps = positiveInt(q.get("testLifetime"), 1_000_000)
  const stepsPerSecond = positiveInt(q.get("testStepsPerSecond"), 10_000)
  return { ...(lifetimeSteps ? { lifetimeSteps } : {}), ...(stepsPerSecond ? { stepsPerSecond } : {}) }
}
