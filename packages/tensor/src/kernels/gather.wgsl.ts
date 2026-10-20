import { WG, prelude } from "./common.wgsl"

/** out[i, c] = a[idx[i], c]. Indices are validated on the host. Uniform: cols, n (= idx.length * cols). */
export const gatherRowsKernel = () => /* wgsl */ `
${prelude}
struct U { cols: u32, n: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatId(gid, nwg);
  if (i >= u.n) { return; }
  out[i] = a[idx[i / u.cols] * u.cols + i % u.cols];
}
`

/**
 * Adjoint of gatherRows without float atomics: the host builds, per destination row r, the list of
 * source positions j with idx[j] == r (ascending, CSR in offsets/list), and one thread per output
 * element sums them in that order — the CPU's order. Uniform: cols, n (= rows * cols).
 */
export const scatterAddRowsKernel = () => /* wgsl */ `
${prelude}
struct U { cols: u32, n: u32 }
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> list: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatId(gid, nwg);
  if (i >= u.n) { return; }
  let r = i / u.cols;
  let c = i % u.cols;
  var s = 0.0;
  for (var j = offsets[r]; j < offsets[r + 1u]; j++) { s += g[list[j] * u.cols + c]; }
  out[i] = s;
}
`
