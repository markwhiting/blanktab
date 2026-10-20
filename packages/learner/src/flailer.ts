import { BUTTON_OPTIONS, MOVE_OPTIONS, type Action, type Learner } from "@blanktab/interface"
import { mulberry32 } from "./rng"

const BUTTON_PROBABILITY = 0.05

/** Placeholder learner: random moves and button presses; ignores its observation. */
export function createFlailer(opts: { seed: number; moveProbability: number }): Learner {
  const rnd = mulberry32(opts.seed)
  return {
    step(): Promise<Action> {
      const move = rnd() < opts.moveProbability ? 1 + Math.floor(rnd() * (MOVE_OPTIONS - 1)) : 0
      const button = rnd() < BUTTON_PROBABILITY ? 1 + Math.floor(rnd() * (BUTTON_OPTIONS - 1)) : 0
      return Promise.resolve({ move, button, scroll: 0, key: 0 })
    },
  }
}
