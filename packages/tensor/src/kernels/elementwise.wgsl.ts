import type { BinaryOp, UnaryOp } from "../backend"
import { WG, prelude } from "./common.wgsl"

// tanh and sigmoid use exp(-|x|) forms so large |x| saturates instead of overflowing to NaN.
const unaryExpr: Record<UnaryOp, string> = {
  neg: "-x",
  relu: "select(0.0, x, x > 0.0)",
  step: "select(0.0, 1.0, x > 0.0)",
  tanh: "tanhStable(x)",
  sigmoid: "sigmoidStable(x)",
  exp: "exp(x)",
  log: "log(x)",
  recip: "1.0 / x",
  sqrt: "sqrt(x)",
  square: "x * x",
}

export const binaryExpr: Record<BinaryOp, string> = { add: "x + y", sub: "x - y", mul: "x * y", div: "x / y" }

const stable = /* wgsl */ `
fn tanhStable(x: f32) -> f32 {
  let e = exp(-2.0 * abs(x));
  let t = (1.0 - e) / (1.0 + e);
  return select(-t, t, x >= 0.0);
}
fn sigmoidStable(x: f32) -> f32 {
  let e = exp(-abs(x));
  return select(e / (1.0 + e), 1.0 / (1.0 + e), x >= 0.0);
}
`

/** out[i] = f(a[i]). Uniform: n. */
export const unaryKernel = (op: UnaryOp) => /* wgsl */ `
${prelude}
${stable}
struct U { n: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatId(gid, nwg);
  if (i >= u.n) { return; }
  let x = a[i];
  out[i] = ${unaryExpr[op]};
}
`

/** out[i] = a[i] op c. Uniform: n, c. */
export const scalarKernel = (op: BinaryOp) => /* wgsl */ `
${prelude}
struct U { n: u32, c: f32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatId(gid, nwg);
  if (i >= u.n) { return; }
  let x = a[i];
  let y = u.c;
  out[i] = ${binaryExpr[op]};
}
`
