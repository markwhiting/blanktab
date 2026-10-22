import { ROOM_VIEW, type PrivilegedState, type World } from "@blanktab/world"
import type { CheckupTrace, Policy } from "./types"

/** Without a `yieldNow`, recordRun yields to the event loop (a setTimeout 0) every this many steps. */
export const YIELD_EVERY = 50

const yieldToLoop = () => new Promise<void>(resolve => setTimeout(resolve, 0))

/**
 * Runs `policy` in `world` for `budgetSteps` steps (observe, act) and records the privileged trace: the state before
 * the first action, the state after each action, and every trace event (any already pending first). Awaits `yieldNow`
 * after every step when given (a checkup passes the session's pacer, so frozen steps obey its speed cap); otherwise
 * yields every YIELD_EVERY steps so a long run never blocks the page.
 */
export async function recordRun(world: World, policy: Policy, budgetSteps: number, yieldNow?: () => Promise<void>): Promise<CheckupTrace> {
  if (!(Number.isInteger(budgetSteps) && budgetSteps >= 0)) throw new RangeError(`recordRun: budgetSteps must be a non-negative integer, got ${budgetSteps}`)
  const events = world.drainTrace()
  const initial = world.state()
  const states = []
  for (let i = 1; i <= budgetSteps; i++) {
    world.act(await policy(world.observe()))
    states.push(world.state())
    for (const e of world.drainTrace()) events.push(e)
    if (yieldNow) await yieldNow()
    else if (i % YIELD_EVERY === 0) await yieldToLoop()
  }
  return { initial, states, events }
}

// Reading traces (shared by the milestones).

/** States whose cursor is in the room viewport, not on the taskbar strip below it. */
export const inViewStates = (states: readonly PrivilegedState[]) => states.filter(s => s.cursor.y < ROOM_VIEW.height)

/** The mean, NaN for no values. */
export const mean = (xs: readonly number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN)

/** Taskbar icons (ids "taskbar:<kind>") are outside the room viewport and its areas. */
export const isTaskbarId = (id: string | undefined) => id?.startsWith("taskbar:") ?? false
