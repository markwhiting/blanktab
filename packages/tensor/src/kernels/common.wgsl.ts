/** Threads per workgroup for 1-D (elementwise-style) kernels. */
export const WG = 256

/**
 * Shared WGSL helpers. Non-finite checks use bit patterns because WGSL lets implementations assume
 * NaN/Inf are absent in float comparisons, so `x != x` is not a reliable NaN test.
 */
export const prelude = /* wgsl */ `
fn flatId(gid: vec3<u32>, nwg: vec3<u32>) -> u32 { return gid.x + gid.y * nwg.x * ${WG}u; }
fn isNan(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn isFinite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
// Value barrier: XOR with a runtime zero the compiler cannot see through. Kernels pass
// z = num_workgroups.z - 1 (dispatches are 2-D, so z is 0).
fn opaque(x: f32, z: u32) -> f32 { return bitcast<f32>(bitcast<u32>(x) ^ z); }
// Neumaier-compensated summation: (s, c) accumulates with roughly double the float32 precision.
// Under fast math (Metal) the error term (s - t) + v is rewritten to (s + v) - t = 0, silently
// dropping the compensation; the barriers on t and on the difference make that rewrite impossible.
fn sumAdd(acc: vec2<f32>, v: f32, z: u32) -> vec2<f32> {
  let s = acc.x;
  let t = opaque(s + v, z);
  let corr = select(opaque(v - t, z) + s, opaque(s - t, z) + v, abs(s) >= abs(v));
  return vec2<f32>(t, acc.y + corr);
}
// A non-finite running sum is the answer (Inf, or NaN for Inf-Inf); the correction would only turn Inf into NaN.
fn sumResult(acc: vec2<f32>) -> f32 { return select(acc.x, acc.x + acc.y, isFinite(acc.x)); }
`
