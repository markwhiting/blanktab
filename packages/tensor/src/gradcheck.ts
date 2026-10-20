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
 * whose rows always sum to 1. The numeric gradient is a Richardson extrapolation of central
 * differences at steps eps and 2·eps, (4·D(eps) − D(2·eps)) / 3, error O(eps⁴), computed in float64.
 * Each difference is divided by the actual float32 perturbation, not by 2·step.
 *
 * Returns max over all input elements of |analytic − numeric| / max(|analytic|, |numeric|, 1e-2·S, 1e-12),
 * where S = max |numeric| over every checked element. The floor is scale-invariant: a function
 * scaled by 1e-4 is judged exactly like the unscaled one, while float32 noise on gradients far
 * below the function's own gradient scale does not dominate. Runs on the default backend, which must be the CPU backend; `f` must build its
 * result from its arguments (any constants it creates internally are not freed).
 */
export function gradCheck(f: (...xs: Tensor[]) => Tensor, inputs: { data: Float32Array; shape: number[] }[], eps = 2e-2): number {
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

  const numeric = base.map((d) => new Float64Array(d.length))
  base.forEach((data, i) => {
    for (let j = 0; j < data.length; j++) {
      /** Central difference with step `step`, divided by the actual float32 perturbation. */
      const central = (step: number) => {
        const plus = base.map((d) => Float32Array.from(d))
        const minus = base.map((d) => Float32Array.from(d))
        plus[i]![j] = data[j]! + step
        minus[i]![j] = data[j]! - step
        return (project(plus) - project(minus)) / (plus[i]![j]! - minus[i]![j]!)
      }
      // Richardson extrapolation: cancels the O(h²) truncation term, so h can be large enough
      // that float32 roundoff in the forward pass stays small.
      numeric[i]![j] = (4 * central(eps) - central(2 * eps)) / 3
    }
  })
  let scale = 0
  for (const n of numeric) for (const v of n) scale = Math.max(scale, Math.abs(v))
  let worst = 0
  numeric.forEach((ns, i) =>
    ns.forEach((n, j) => {
      const a = analytic[i]![j]!
      const err = Math.abs(a - n) / Math.max(Math.abs(a), Math.abs(n), 1e-2 * scale, 1e-12)
      if (!(err <= worst)) worst = err // NaN propagates as a failure
    }),
  )
  return worst
}
