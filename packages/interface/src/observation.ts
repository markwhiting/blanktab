import type { Action } from "./action"
import { FOVEA, PERIPHERY } from "./constants"

// Pixels and a cursor only. No clock, no world state.
export interface Observation {
  periphery: Uint8Array
  fovea: Uint8Array
  cursor: { x: number; y: number } // normalized 0..1
  buttonHeld: boolean
}

export function emptyObservation(): Observation {
  return {
    periphery: new Uint8Array(PERIPHERY.width * PERIPHERY.height * PERIPHERY.channels),
    fovea: new Uint8Array(FOVEA.width * FOVEA.height * FOVEA.channels),
    cursor: { x: 0, y: 0 },
    buttonHeld: false,
  }
}

// The learner-facing world surface. Nothing privileged crosses it.
export interface WorldView { observe(): Observation; act(a: Action): void }

// The learner surface the harness drives. Async because GPU learners await readback of policy logits.
export interface Learner { step(o: Observation): Promise<Action> }
