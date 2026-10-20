import { ShapeError } from "./backend"
import type { Storage } from "./backend"
import type { Tensor } from "./tensor"

export interface AdamOptions {
  lr: number
  beta1?: number
  beta2?: number
  eps?: number
}

interface Slot {
  p: Tensor
  m: Storage
  v: Storage
  /** Per-parameter step count, so a parameter skipped (null grad) keeps correct bias correction. */
  t: number
}

const checkLr = (lr: number) => {
  if (!(Number.isFinite(lr) && lr >= 0)) throw new RangeError(`Adam: lr must be a finite number >= 0, got ${lr}`)
}

/** Adam with bias correction; moments live on each parameter's backend and update in place. */
export class Adam {
  private lr: number
  private readonly beta1: number
  private readonly beta2: number
  private readonly eps: number
  private readonly slots: Slot[]

  constructor(params: Tensor[], opts: AdamOptions) {
    checkLr(opts.lr)
    this.lr = opts.lr
    this.beta1 = opts.beta1 ?? 0.9
    this.beta2 = opts.beta2 ?? 0.999
    this.eps = opts.eps ?? 1e-8
    for (const [name, b] of [["beta1", this.beta1], ["beta2", this.beta2]] as const) {
      if (!(b >= 0 && b < 1)) throw new RangeError(`Adam: ${name} must be in [0, 1), got ${b}`)
    }
    if (!(this.eps > 0)) throw new RangeError(`Adam: eps must be > 0, got ${this.eps}`)
    if (new Set(params).size !== params.length) throw new Error("Adam: the same parameter was passed more than once")
    this.slots = params.map((p) => ({ p, m: p.backend.zeros(p.size), v: p.backend.zeros(p.size), t: 0 }))
  }

  /** One update for every parameter with a gradient; parameters whose grad is null are skipped. */
  step(): void {
    for (const s of this.slots) {
      const g = s.p.grad
      if (!g) continue
      if (g.size !== s.p.size) throw new ShapeError(`Adam: grad size ${g.size} does not match parameter size ${s.p.size}`)
      s.t++
      s.p.backend.adam(s.p.storage, g.storage, s.m, s.v, this.lr, this.beta1, this.beta2, this.eps, s.t)
    }
  }

  /** Disposes every parameter gradient and sets it to null. */
  zeroGrad(): void {
    for (const { p } of this.slots) {
      p.grad?.dispose()
      p.grad = null
    }
  }

  setLr(lr: number): void {
    checkLr(lr)
    this.lr = lr
  }

  /** Frees the moment buffers. The optimizer must not be used afterwards. */
  dispose(): void {
    for (const s of this.slots) {
      s.p.backend.dispose(s.m)
      s.p.backend.dispose(s.v)
    }
  }
}
