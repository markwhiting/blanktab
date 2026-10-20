import { WG, prelude } from "./common.wgsl"

const geometry = "struct U { c: u32, h: u32, w: u32, kh: u32, kw: u32, stride: u32, pad: u32, oh: u32, ow: u32, n: u32 }"

/** One thread per output element of [c*kh*kw, oh*ow]; zero padding. */
export const im2colKernel = () => /* wgsl */ `
${prelude}
${geometry}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatId(gid, nwg);
  if (i >= u.n) { return; }
  let hw = u.oh * u.ow;
  let row = i / hw;
  let pos = i % hw;
  let oy = pos / u.ow;
  let ox = pos % u.ow;
  let kx = row % u.kw;
  let ky = (row / u.kw) % u.kh;
  let ch = row / (u.kw * u.kh);
  let iy = i32(oy * u.stride + ky) - i32(u.pad);
  let ix = i32(ox * u.stride + kx) - i32(u.pad);
  var v = 0.0;
  if (iy >= 0i && iy < i32(u.h) && ix >= 0i && ix < i32(u.w)) { v = x[(ch * u.h + u32(iy)) * u.w + u32(ix)]; }
  out[i] = v;
}
`

/**
 * Adjoint of im2col as a gather: one thread per image pixel sums every column entry that read it,
 * in (ky, kx) order (the CPU's order), so no float atomics are needed.
 */
export const col2imKernel = () => /* wgsl */ `
${prelude}
${geometry}
@group(0) @binding(0) var<storage, read> cols: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = flatId(gid, nwg);
  if (i >= u.n) { return; }
  let ix = i % u.w;
  let iy = (i / u.w) % u.h;
  let ch = i / (u.w * u.h);
  let hw = u.oh * u.ow;
  var s = 0.0;
  for (var ky = 0u; ky < u.kh; ky++) {
    let ty = i32(iy + u.pad) - i32(ky);
    if (ty < 0i || u32(ty) % u.stride != 0u) { continue; }
    let oy = u32(ty) / u.stride;
    if (oy >= u.oh) { continue; }
    for (var kx = 0u; kx < u.kw; kx++) {
      let tx = i32(ix + u.pad) - i32(kx);
      if (tx < 0i || u32(tx) % u.stride != 0u) { continue; }
      let ox = u32(tx) / u.stride;
      if (ox >= u.ow) { continue; }
      s += cols[((ch * u.kh + ky) * u.kw + kx) * hw + oy * u.ow + ox];
    }
  }
  out[i] = s;
}
`
