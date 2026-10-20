import { WG, prelude } from "./common.wgsl"

/**
 * In-place bias-corrected Adam, matching cpu.ts:
 * m = b1 m + (1-b1) g; v = b2 v + (1-b2) g²; p -= lr (m/c1) / (sqrt(v/c2) + eps).
 * Uniform: n, lr, b1, oneMinusB1, b2, oneMinusB2, eps, c1, c2 (c = 1 - beta^t, computed in float64 on the host).
 */
export const adamKernel = () => /* wgsl */ `
${prelude}
struct U { n: u32, lr: f32, b1: f32, ob1: f32, b2: f32, ob2: f32, eps: f32, c1: f32, c2: f32 }
@group(0) @binding(0) var<storage, read_write> p: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatId(gid, nwg);
  if (i >= u.n) { return; }
  let gi = g[i];
  let mi = u.b1 * m[i] + u.ob1 * gi;
  let vi = u.b2 * v[i] + u.ob2 * gi * gi;
  m[i] = mi;
  v[i] = vi;
  p[i] = p[i] - (u.lr * (mi / u.c1)) / (sqrt(vi / u.c2) + u.eps);
}
`
