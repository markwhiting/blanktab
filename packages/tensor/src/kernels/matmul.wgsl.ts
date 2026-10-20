import { prelude } from "./common.wgsl"

export const TILE = 16

/**
 * C[m,n] = op(A)·op(B), 16×16 tiles through workgroup memory; transpose flags are uniforms so one
 * pipeline serves all four variants. Products are Neumaier-accumulated so long k stays near float64
 * accuracy. A(i,p) = transA ? a[p*m+i] : a[i*k+p]; B(p,j) = transB ? b[j*k+p] : b[p*n+j].
 */
export const matmulKernel = () => /* wgsl */ `
${prelude}
struct U { m: u32, k: u32, n: u32, transA: u32, transB: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
var<workgroup> As: array<array<f32, ${TILE}>, ${TILE}>;
var<workgroup> Bs: array<array<f32, ${TILE}>, ${TILE}>;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let z = nwg.z - 1u;
  let row = wid.y * ${TILE}u + lid.y;
  let col = wid.x * ${TILE}u + lid.x;
  var acc = vec2<f32>(0.0, 0.0);
  let tiles = (u.k + ${TILE - 1}u) / ${TILE}u;
  for (var t = 0u; t < tiles; t++) {
    let pa = t * ${TILE}u + lid.x;
    var av = 0.0;
    if (row < u.m && pa < u.k) {
      if (u.transA != 0u) { av = a[pa * u.m + row]; } else { av = a[row * u.k + pa]; }
    }
    As[lid.y][lid.x] = av;
    let pb = t * ${TILE}u + lid.y;
    var bv = 0.0;
    if (pb < u.k && col < u.n) {
      if (u.transB != 0u) { bv = b[col * u.k + pb]; } else { bv = b[pb * u.n + col]; }
    }
    Bs[lid.y][lid.x] = bv;
    workgroupBarrier();
    for (var p = 0u; p < ${TILE}u; p++) { acc = sumAdd(acc, As[lid.y][p] * Bs[p][lid.x], z); }
    workgroupBarrier();
  }
  if (row < u.m && col < u.n) { out[row * u.n + col] = sumResult(acc); }
}
`
