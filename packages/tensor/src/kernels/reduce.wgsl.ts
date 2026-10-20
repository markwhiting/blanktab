import { WG, prelude } from "./common.wgsl"

/**
 * One workgroup per row. Each thread reduces a strided slice of the row, then thread 0 combines the
 * 256 partials. sum is Neumaier-compensated; max propagates NaN. Uniform: rows, cols (cols >= 1 for max).
 */
export const reduceLastKernel = (op: "sum" | "max") => /* wgsl */ `
${prelude}
struct U { rows: u32, cols: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
var<workgroup> part: array<vec2<f32>, ${WG}>;
fn maxNan(x: f32, y: f32) -> f32 {
  if (isNan(x)) { return x; }
  if (isNan(y)) { return y; }
  return max(x, y);
}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let row = wid.x + wid.y * nwg.x;
  if (row >= u.rows) { return; }
  let z = nwg.z - 1u;
  let base = row * u.cols;
  ${op === "sum"
    ? `var acc = vec2<f32>(0.0, 0.0);
  for (var c = lid; c < u.cols; c += ${WG}u) { acc = sumAdd(acc, a[base + c], z); }
  part[lid] = vec2<f32>(acc.x, select(0.0, acc.y, isFinite(acc.x)));`
    : `var m = a[base];
  for (var c = lid; c < u.cols; c += ${WG}u) { m = maxNan(m, a[base + c]); }
  part[lid] = vec2<f32>(m, 0.0);`}
  workgroupBarrier();
  if (lid == 0u) {
    ${op === "sum"
      ? `var total = vec2<f32>(0.0, 0.0);
    for (var t = 0u; t < ${WG}u; t++) { total = sumAdd(sumAdd(total, part[t].x, z), part[t].y, z); }
    out[row] = sumResult(total);`
      : `var m2 = part[0].x;
    for (var t = 1u; t < ${WG}u; t++) { m2 = maxNan(m2, part[t].x); }
    out[row] = m2;`}
  }
}
`

/**
 * Inverse of broadcast: one thread per output element sums (compensated) over the reduced dims.
 * Shapes are rank-4 padded. Uniform: outShape (padded target; 1 on reduced dims), redShape (extent
 * of reduced dims; 1 elsewhere), inStride (contiguous strides of the input), nOut, nRed.
 */
export const sumToShapeKernel = () => /* wgsl */ `
${prelude}
struct U { outShape: vec4<u32>, redShape: vec4<u32>, inStride: vec4<u32>, nOut: u32, nRed: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let o = flatId(gid, nwg);
  if (o >= u.nOut) { return; }
  let z = nwg.z - 1u;
  var rem = o;
  var base = 0u;
  for (var d = 3i; d >= 0i; d--) {
    base += (rem % u.outShape[d]) * u.inStride[d];
    rem = rem / u.outShape[d];
  }
  var acc = vec2<f32>(0.0, 0.0);
  for (var r = 0u; r < u.nRed; r++) {
    var rr = r;
    var off = base;
    for (var d = 3i; d >= 0i; d--) {
      off += (rr % u.redShape[d]) * u.inStride[d];
      rr = rr / u.redShape[d];
    }
    acc = sumAdd(acc, a[off], z);
  }
  out[o] = sumResult(acc);
}
`
