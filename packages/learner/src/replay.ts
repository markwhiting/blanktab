import type { Action, Observation } from "@blanktab/interface"

/**
 * One learning transition, kept as raw data so it can be replayed and checkpointed (no tensors).
 * Replaying it means: h = core(obsToInputs(obs), hIn), then predict(h, action, targetsFromObs(obs)) against
 * targetsFromObs(next) (the residual base is recomputed from `obs`, so it is not stored).
 * `obs` and `next` are the learner's own copies of the observations; consecutive transitions share the
 * frame between them (`next` of one is `obs` of the following), so a frame is stored once.
 */
export interface Transition {
  /** Observation fed to the core (inputs_{t-1}). */
  obs: Observation
  /** Recurrent state fed to the core with `obs` (h_{t-2}, detached), [hidden] floats. */
  hIn: Float32Array
  /** Action taken after `obs` (a_{t-1}). */
  action: Action
  /** Observation that followed (its targets are y_t). */
  next: Observation
}

/** FIFO ring of the last `capacity` transitions: pushing into a full buffer evicts the oldest. */
export class ReplayBuffer {
  readonly capacity: number
  private readonly items: Transition[] = []
  /** Index in `items` of the oldest transition once the ring is full. */
  private head = 0

  constructor(capacity: number) {
    if (!(Number.isInteger(capacity) && capacity >= 1)) throw new RangeError(`ReplayBuffer: capacity must be an integer >= 1, got ${capacity}`)
    this.capacity = capacity
  }

  get size(): number {
    return this.items.length
  }

  push(t: Transition): void {
    if (this.items.length < this.capacity) {
      this.items.push(t)
      return
    }
    this.items[this.head] = t
    this.head = (this.head + 1) % this.capacity
  }

  /** The i-th transition, oldest first (0 = oldest, size-1 = newest). */
  at(i: number): Transition {
    if (!(Number.isInteger(i) && i >= 0 && i < this.items.length)) throw new RangeError(`ReplayBuffer: index ${i} out of range [0, ${this.items.length})`)
    return this.items[(this.head + i) % this.items.length]!
  }
}
