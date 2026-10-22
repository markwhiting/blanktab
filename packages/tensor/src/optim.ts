import { ShapeError } from "./backend"
import type { Storage } from "./backend"
import type { Tensor } from "./tensor"

export interface AdamOptions {
  lr: number
  beta1?: number
  beta2?: number
  eps?: number
  /**
   * Decoupled weight decay (AdamW-style, on the parameter, not the gradient): each update of a decayed parameter is
   * p ← (1 − weightDecay)·p − Adam step. A per-update rate, not scaled by lr. In [0, 1); default 0 (plain Adam, bit-exact).
   */
  weightDecay?: number
  /** The parameters weightDecay applies to (a subset of the optimized ones); default all of them. */
  decayed?: readonly Tensor[]
}

/** One parameter's optimizer state, in the order the parameters were passed to the constructor. */
export interface AdamSlotState {
  m: Float32Array
  v: Float32Array
  /** Steps taken for this parameter (bias correction). */
  t: number
}
/** Everything Adam carries between steps (plain typed arrays, structured-clone friendly). */
export interface AdamState { slots: AdamSlotState[] }

interface Slot {
  readonly p: Tensor
  m: Storage
  v: Storage
  /** Per-parameter step count, so a parameter skipped (null grad) keeps correct bias correction. */
  t: number
  /** Decoupled weight decay for this parameter (0: none). */
  readonly decay: number
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
    const wd = opts.weightDecay ?? 0
    if (!(wd >= 0 && wd < 1)) throw new RangeError(`Adam: weightDecay must be in [0, 1), got ${wd}`)
    const decayed = new Set(opts.decayed ?? params)
    for (const d of decayed) if (!params.includes(d)) throw new RangeError("Adam: a decayed tensor is not one of the optimized parameters")
    // A refused allocation part-way frees the moments already made, so a failed constructor leaks nothing.
    const made: Storage[] = []
    const alloc = (p: Tensor) => {
      const st = p.backend.zeros(p.size)
      made.push(st)
      return st
    }
    try {
      this.slots = params.map((p) => ({ p, m: alloc(p), v: alloc(p), t: 0, decay: decayed.has(p) ? wd : 0 }))
    } catch (e) {
      made.forEach((st, j) => params[j >> 1]!.backend.dispose(st))
      throw e
    }
  }

  /** One update for every parameter with a gradient; parameters whose grad is null are skipped. */
  step(): void {
    for (const s of this.slots) {
      const g = s.p.grad
      if (!g) continue
      if (g.size !== s.p.size) throw new ShapeError(`Adam: grad size ${g.size} does not match parameter size ${s.p.size}`)
      s.t++
      s.p.backend.adam(s.p.storage, g.storage, s.m, s.v, this.lr, this.beta1, this.beta2, this.eps, s.t, s.decay)
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

  /**
   * A copy of the moments and step counts, in parameter order. lr and betas are not part of it (they are settings).
   * Every read settles before it resolves or rejects (with the first rejection), so the caller may then free the
   * optimizer: nothing is freed under a pending read.
   */
  async state(): Promise<AdamState> {
    const ts = this.slots.map((s) => s.t)
    const reads = this.slots.flatMap((s) => [s.m, s.v].map((st) => new Promise<Float32Array>((resolve) => resolve(s.p.backend.read(st)))))
    let first: { e: unknown } | null = null
    const rs = await Promise.allSettled(reads.map((r) => r.catch((e: unknown) => { first ??= { e }; throw e })))
    if (first) throw (first as { e: unknown }).e
    const values = rs.map((r) => (r as PromiseFulfilledResult<Float32Array>).value)
    return { slots: ts.map((t, i) => ({ m: values[2 * i]!, v: values[2 * i + 1]!, t })) }
  }

  /**
   * Restores moments and step counts from state() (copied, so the caller's arrays are not aliased). Validates the
   * whole state first and throws RangeError, changing nothing, unless it matches these parameters slot for slot.
   */
  setState(state: AdamState): void {
    const { slots } = state
    if (slots.length !== this.slots.length) throw new RangeError(`Adam.setState: ${slots.length} slots for ${this.slots.length} parameters`)
    slots.forEach((st, i) => {
      const n = this.slots[i]!.p.size
      if (st.m.length !== n || st.v.length !== n) throw new RangeError(`Adam.setState: slot ${i} has m/v length ${st.m.length}/${st.v.length}, parameter size ${n}`)
      if (!(Number.isInteger(st.t) && st.t >= 0)) throw new RangeError(`Adam.setState: slot ${i} step count must be an integer >= 0, got ${st.t}`)
    })
    // Upload everything first and swap only once every buffer exists: a refused allocation frees the new
    // buffers and leaves the optimizer exactly as it was.
    const fresh: Storage[] = [] // m0, v0, m1, v1, ...
    try {
      slots.forEach((st, i) => {
        const be = this.slots[i]!.p.backend
        fresh.push(be.upload(Float32Array.from(st.m)))
        fresh.push(be.upload(Float32Array.from(st.v)))
      })
    } catch (e) {
      fresh.forEach((f, j) => this.slots[j >> 1]!.p.backend.dispose(f))
      throw e
    }
    slots.forEach((st, i) => {
      const s = this.slots[i]!
      const be = s.p.backend
      be.dispose(s.m)
      be.dispose(s.v)
      s.m = fresh[2 * i]!
      s.v = fresh[2 * i + 1]!
      s.t = st.t
    })
  }

  /** Frees the moment buffers. The optimizer must not be used afterwards. */
  dispose(): void {
    for (const s of this.slots) {
      s.p.backend.dispose(s.m)
      s.p.backend.dispose(s.v)
    }
  }
}
