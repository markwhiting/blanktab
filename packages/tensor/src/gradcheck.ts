import { ResourceError } from "./backend"
import { backward, disposeGraph } from "./autograd"
import { readSyncCpu } from "./cpu"
import { mul, sum } from "./ops"
import { createRng } from "./random"
import { Tensor, getDefaultBackend, tensor } from "./tensor"

/**
 * Compares autodiff gradients of `f` at `inputs` against central finite differences.
 *
 * A non-scalar output is projected to a scalar with fixed pseudo-random weights w in ±[0.5, 1.5]
 * (L = Σ w·f), so every output element is checked — plain `sum` would hide errors in e.g. softmax,
 * whose rows always sum to 1. Differences and the projection are computed in float64; the step is
 * divided by the actual float32 perturbation (f32(x+eps) − f32(x−eps)), not by 2·eps.
 *
 * Returns max over all input elements of |analytic − numeric| / max(1, |analytic|, |numeric|)
 * (relative for large gradients, absolute below 1, so float32 noise on near-zero gradients does
 * not dominate). Runs on the default backend, which must be the CPU backend; `f` must build its
 * result from its arguments (any constants it creates internally are not freed).
 */
export function gradCheck(f: (...xs: Tensor[]) => Tensor, inputs: { data: Float32Array; shape: number[] }[], eps = 1e-3): number {
  const be = getDefaultBackend()
  if (be.name !== "cpu") throw new ResourceError(`gradCheck runs on the CPU backend only, default backend is "${be.name}"`)
  if (!(eps > 0)) throw new RangeError(`gradCheck: eps must be > 0, got ${eps}`)

  const leaves = (datas: Float32Array[]) => datas.map((d, i) => tensor(d, inputs[i]!.shape, { requiresGrad: true }))
  const base = inputs.map((x) => Float32Array.from(x.data))

  // Analytic pass.
  const xs = leaves(base)
  const out = f(...xs)
  const rng = createRng(0x9e3779b9)
  const w = Float32Array.from({ length: out.size }, () => (rng() < 0.5 ? -1 : 1) * (0.5 + rng()))
  const wt = tensor(w, [...out.shape])
  const loss = sum(mul(out, wt))
  backward(loss)
  const analytic = xs.map((x) => (x.grad ? readSyncCpu(x.grad.storage) : new Float32Array(x.size)))
  disposeGraph(loss, [])
  for (const x of xs) {
    x.grad?.dispose()
    x.dispose()
  }
  wt.dispose()

  /** L(datas) = Σ w·f in float64, freeing everything it allocates. */
  const project = (datas: Float32Array[]): number => {
    const ys = leaves(datas)
    const y = f(...ys)
    const vals = readSyncCpu(y.storage)
    disposeGraph(y, [])
    for (const t of ys) t.dispose()
    let acc = 0
    for (let k = 0; k < vals.length; k++) acc += w[k]! * vals[k]!
    return acc
  }

  let worst = 0
  base.forEach((data, i) => {
    for (let j = 0; j < data.length; j++) {
      const orig = data[j]!
      const plus = base.map((d) => Float32Array.from(d))
      const minus = base.map((d) => Float32Array.from(d))
      plus[i]![j] = orig + eps
      minus[i]![j] = orig - eps
      const h = plus[i]![j]! - minus[i]![j]!
      const numeric = (project(plus) - project(minus)) / h
      const a = analytic[i]![j]!
      const err = Math.abs(a - numeric) / Math.max(1, Math.abs(a), Math.abs(numeric))
      if (!(err <= worst)) worst = err // NaN propagates as a failure
    }
  })
  return worst
}
