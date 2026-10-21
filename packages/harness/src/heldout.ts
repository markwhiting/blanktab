// The learner's held-out set (decision 0015 §3), built by the harness side (the learner cannot see the world): a
// trajectory of a separate world under uniformly random actions, recorded once at birth and handed to the learner as data.
import { BUTTON_OPTIONS, KEYS, MOVE_OPTIONS, SCROLL_OPTIONS, type Action, type Observation } from "@blanktab/interface"
import type { HeldOutSet } from "@blanktab/learner"
import { createRng } from "@blanktab/tensor"
import { createWorld, hashSeed, type CanvasLike } from "@blanktab/world"

/** Transitions in a held-out set. */
export const HELD_OUT_TRANSITIONS = 64

/**
 * A learner's held-out set: `n` transitions (n + 1 observations) of a fresh world seeded hashSeed(learnerSeed,
 * "heldout") — so a different layout from the world it lives in — driven by actions drawn uniformly per factor from
 * an RNG of its own, seeded hashSeed(learnerSeed, "heldout-actions"). A pure function of the seed (and `n`); the
 * canvas is the world's to draw on.
 */
export function buildHeldOut(learnerSeed: number, canvas: CanvasLike, n = HELD_OUT_TRANSITIONS): HeldOutSet {
  const world = createWorld({ seed: hashSeed(learnerSeed, "heldout"), canvas })
  const rnd = createRng(hashSeed(learnerSeed, "heldout-actions"))
  const pick = (k: number) => Math.min(k - 1, Math.floor(rnd() * k))
  const obs: Observation[] = [world.observe()]
  const actions: Action[] = []
  for (let i = 0; i < n; i++) {
    const a: Action = { move: pick(MOVE_OPTIONS), button: pick(BUTTON_OPTIONS), scroll: pick(SCROLL_OPTIONS), key: pick(KEYS.length) }
    world.act(a)
    world.drainTrace()
    actions.push(a)
    obs.push(world.observe())
  }
  return { obs, actions }
}
