import type { BinaryOp } from "../backend"
import { WG, prelude } from "./common.wgsl"
import { binaryExpr } from "./elementwise.wgsl"

/** Max rank for broadcasting kernels: shapes and strides travel as vec4 uniforms. */
export const MAX_RANK = 4

/**
 * NumPy-broadcast a op b. Shapes are left-padded to rank 4; strides are 0 on broadcast dims.
 * Uniform: shape (out), sa, sb (vec4<u32>), n.
 */
export const binaryKernel = (op: BinaryOp) => /* wgsl */ `
${prelude}
struct U { shape: vec4<u32>, sa: vec4<u32>, sb: vec4<u32>, n: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatId(gid, nwg);
  if (i >= u.n) { return; }
  var rem = i;
  var ia = 0u;
  var ib = 0u;
  for (var d = 3i; d >= 0i; d--) {
    let c = rem % u.shape[d];
    rem = rem / u.shape[d];
    ia += c * u.sa[d];
    ib += c * u.sb[d];
  }
  let x = a[ia];
  let y = b[ib];
  out[i] = ${binaryExpr[op]};
}
`
