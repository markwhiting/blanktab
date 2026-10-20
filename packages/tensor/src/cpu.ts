import { ResourceError, ShapeError, broadcastShape, im2colGeometry } from "./backend"
import type { Backend, BinaryOp, Storage, UnaryOp } from "./backend"

interface CpuStorage extends Storage {
  data: Float32Array
  disposed: boolean
}

const unaryFns: Record<UnaryOp, (x: number) => number> = {
  neg: (x) => -x,
  // NaN fails every comparison, so test x <= 0 (not x > 0) to let NaN through.
  relu: (x) => (x <= 0 ? 0 : x),
  step: (x) => (x > 0 ? 1 : x <= 0 ? 0 : NaN),
  tanh: Math.tanh,
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  exp: Math.exp,
  log: Math.log,
  recip: (x) => 1 / x,
  sqrt: Math.sqrt,
  square: (x) => x * x,
}

const binaryFns: Record<BinaryOp, (x: number, y: number) => number> = {
  add: (x, y) => x + y,
  sub: (x, y) => x - y,
  mul: (x, y) => x * y,
  div: (x, y) => x / y,
}

const prod = (s: number[]) => s.reduce((p, d) => p * d, 1)

/** Strides of `shape` right-aligned into `rank` dims; broadcast (size-1/missing) dims get stride 0. */
function broadcastStrides(shape: number[], rank: number): number[] {
  const out = new Array<number>(rank).fill(0)
  let stride = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    out[rank - shape.length + i] = shape[i] === 1 ? 0 : stride
    stride *= shape[i]!
  }
  return out
}

/** Synchronous read of a CPU-backend storage (a copy). Throws ResourceError for other backends or disposed storage. */
export function readSyncCpu(s: Storage): Float32Array {
  const c = s as CpuStorage
  if (c.backend.name !== "cpu" || !(c.data instanceof Float32Array)) throw new ResourceError("readSyncCpu: storage is not a CPU storage")
  if (c.disposed) throw new ResourceError("storage used after dispose")
  return Float32Array.from(c.data)
}

export function createCpuBackend(): Backend {
  const stats = { flops: 0, liveBuffers: 0 }
  const make = (data: Float32Array): CpuStorage => {
    stats.liveBuffers++
    return { length: data.length, backend, data, disposed: false }
  }
  const get = (s: Storage): Float32Array => {
    const c = s as CpuStorage
    if (c.backend !== backend) throw new ResourceError("storage belongs to a different backend")
    if (c.disposed) throw new ResourceError("storage used after dispose")
    return c.data
  }
  const checkLen = (what: string, s: Storage, want: number) => {
    if (s.length !== want) throw new ShapeError(`${what}: expected length ${want}, got ${s.length}`)
  }

  const backend: Backend = {
    name: "cpu",
    stats,
    upload: (data) => make(Float32Array.from(data)),
    zeros: (length) => make(new Float32Array(length)),
    // Executor form so a disposed/foreign storage rejects instead of throwing synchronously.
    read: (s) => new Promise((resolve) => resolve(Float32Array.from(get(s)))),
    dispose(s) {
      const c = s as CpuStorage
      if (c.backend !== backend || c.disposed) return
      c.disposed = true
      c.data = new Float32Array(0)
      stats.liveBuffers--
    },

    unary(op, a) {
      const x = get(a)
      const f = unaryFns[op]
      const out = new Float32Array(x.length)
      for (let i = 0; i < x.length; i++) out[i] = f(x[i]!)
      stats.flops += out.length
      return make(out)
    },

    scalar(op, a, c) {
      const x = get(a)
      const f = binaryFns[op]
      // The constant is float32, as on the GPU (a uniform f32), so both backends round identically.
      const c32 = Math.fround(c)
      const out = new Float32Array(x.length)
      for (let i = 0; i < x.length; i++) out[i] = f(x[i]!, c32)
      stats.flops += out.length
      return make(out)
    },

    binary(op, a, aShape, b, bShape, outShape) {
      const x = get(a)
      const y = get(b)
      checkLen("binary lhs", a, prod(aShape))
      checkLen("binary rhs", b, prod(bShape))
      const expected = broadcastShape(aShape, bShape)
      if (expected.length !== outShape.length || expected.some((d, i) => d !== outShape[i])) {
        throw new ShapeError(`binary: outShape [${outShape.join(",")}] does not match broadcast of [${aShape.join(",")}] and [${bShape.join(",")}]`)
      }
      const f = binaryFns[op]
      const rank = outShape.length
      const n = prod(outShape)
      const sa = broadcastStrides(aShape, rank)
      const sb = broadcastStrides(bShape, rank)
      const out = new Float32Array(n)
      const idx = new Array<number>(rank).fill(0)
      let ia = 0
      let ib = 0
      for (let i = 0; i < n; i++) {
        out[i] = f(x[ia]!, y[ib]!)
        for (let d = rank - 1; d >= 0; d--) {
          idx[d]!++
          ia += sa[d]!
          ib += sb[d]!
          if (idx[d]! < outShape[d]!) break
          ia -= sa[d]! * outShape[d]!
          ib -= sb[d]! * outShape[d]!
          idx[d] = 0
        }
      }
      stats.flops += n
      return make(out)
    },

    sumToShape(a, fromShape, toShape) {
      const x = get(a)
      checkLen("sumToShape input", a, prod(fromShape))
      const rank = fromShape.length
      if (toShape.length > rank) throw new ShapeError(`sumToShape: cannot reduce [${fromShape.join(",")}] to [${toShape.join(",")}]`)
      const padded = [...new Array<number>(rank - toShape.length).fill(1), ...toShape]
      for (let d = 0; d < rank; d++) {
        if (padded[d] !== fromShape[d] && padded[d] !== 1) {
          throw new ShapeError(`sumToShape: cannot reduce [${fromShape.join(",")}] to [${toShape.join(",")}]`)
        }
      }
      const so = broadcastStrides(padded, rank)
      const acc = new Float64Array(prod(toShape))
      const n = x.length
      const idx = new Array<number>(rank).fill(0)
      let io = 0
      for (let i = 0; i < n; i++) {
        acc[io] = acc[io]! + x[i]!
        for (let d = rank - 1; d >= 0; d--) {
          idx[d]!++
          io += so[d]!
          if (idx[d]! < fromShape[d]!) break
          io -= so[d]! * fromShape[d]!
          idx[d] = 0
        }
      }
      stats.flops += n
      return make(Float32Array.from(acc))
    },

    matmul(a, b, m, k, n, transA, transB) {
      const x = get(a)
      const y = get(b)
      checkLen("matmul lhs", a, m * k)
      checkLen("matmul rhs", b, k * n)
      const out = new Float32Array(m * n)
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0
          for (let p = 0; p < k; p++) {
            const av = transA ? x[p * m + i]! : x[i * k + p]!
            const bv = transB ? y[j * k + p]! : y[p * n + j]!
            s += av * bv
          }
          out[i * n + j] = s
        }
      }
      stats.flops += 2 * m * k * n
      return make(out)
    },

    reduceLast(op, a, rows, cols) {
      const x = get(a)
      checkLen("reduceLast", a, rows * cols)
      if (op === "max" && cols === 0) throw new ShapeError(`reduceLast max: cannot reduce ${rows} empty rows`)
      const out = new Float32Array(rows)
      for (let r = 0; r < rows; r++) {
        // max: a NaN anywhere in the row makes the row max NaN (Math.max semantics).
        let acc = op === "sum" ? 0 : -Infinity
        for (let c = 0; c < cols; c++) {
          const v = x[r * cols + c]!
          acc = op === "sum" ? acc + v : Math.max(acc, v)
        }
        out[r] = acc
      }
      stats.flops += rows * cols
      return make(out)
    },

    im2col(xs, c, h, w, kh, kw, stride, pad) {
      const x = get(xs)
      checkLen("im2col", xs, c * h * w)
      const [oh, ow] = im2colGeometry("im2col", c, h, w, kh, kw, stride, pad)
      const out = new Float32Array(c * kh * kw * oh * ow)
      for (let ch = 0; ch < c; ch++)
        for (let ky = 0; ky < kh; ky++)
          for (let kx = 0; kx < kw; kx++) {
            const row = (ch * kh + ky) * kw + kx
            for (let oy = 0; oy < oh; oy++) {
              const iy = oy * stride - pad + ky
              if (iy < 0 || iy >= h) continue
              for (let ox = 0; ox < ow; ox++) {
                const ix = ox * stride - pad + kx
                if (ix < 0 || ix >= w) continue
                out[row * oh * ow + oy * ow + ox] = x[(ch * h + iy) * w + ix]!
              }
            }
          }
      return make(out)
    },

    col2im(cs, c, h, w, kh, kw, stride, pad) {
      const cols = get(cs)
      const [oh, ow] = im2colGeometry("col2im", c, h, w, kh, kw, stride, pad)
      checkLen("col2im", cs, c * kh * kw * oh * ow)
      const out = new Float32Array(c * h * w)
      for (let ch = 0; ch < c; ch++)
        for (let ky = 0; ky < kh; ky++)
          for (let kx = 0; kx < kw; kx++) {
            const row = (ch * kh + ky) * kw + kx
            for (let oy = 0; oy < oh; oy++) {
              const iy = oy * stride - pad + ky
              if (iy < 0 || iy >= h) continue
              for (let ox = 0; ox < ow; ox++) {
                const ix = ox * stride - pad + kx
                if (ix < 0 || ix >= w) continue
                const o = (ch * h + iy) * w + ix
                out[o] = out[o]! + cols[row * oh * ow + oy * ow + ox]!
              }
            }
          }
      return make(out)
    },

    gatherRows(a, rows, cols, idx) {
      const x = get(a)
      checkLen("gatherRows", a, rows * cols)
      const out = new Float32Array(idx.length * cols)
      for (let i = 0; i < idx.length; i++) {
        const r = idx[i]!
        if (r < 0 || r >= rows) throw new ShapeError(`gatherRows: index ${r} out of range for ${rows} rows`)
        out.set(x.subarray(r * cols, (r + 1) * cols), i * cols)
      }
      return make(out)
    },

    scatterAddRows(grad, rows, cols, idx) {
      const g = get(grad)
      checkLen("scatterAddRows", grad, idx.length * cols)
      const out = new Float32Array(rows * cols)
      for (let i = 0; i < idx.length; i++) {
        const r = idx[i]!
        if (r < 0 || r >= rows) throw new ShapeError(`scatterAddRows: index ${r} out of range for ${rows} rows`)
        for (let c = 0; c < cols; c++) out[r * cols + c] = out[r * cols + c]! + g[i * cols + c]!
      }
      return make(out)
    },

    adam(p, g, m, v, lr, b1, b2, eps, t) {
      const pd = get(p)
      const gd = get(g)
      const md = get(m)
      const vd = get(v)
      if (gd.length !== pd.length || md.length !== pd.length || vd.length !== pd.length) {
        throw new ShapeError("adam: p, g, m, v must have equal length")
      }
      if (new Set([p, g, m, v]).size !== 4) throw new ShapeError("adam: p, g, m, v must be distinct storages")
      const c1 = 1 - Math.pow(b1, t)
      const c2 = 1 - Math.pow(b2, t)
      for (let i = 0; i < pd.length; i++) {
        const gi = gd[i]!
        const mi = b1 * md[i]! + (1 - b1) * gi
        const vi = b2 * vd[i]! + (1 - b2) * gi * gi
        md[i] = mi
        vd[i] = vi
        pd[i] = pd[i]! - (lr * (mi / c1)) / (Math.sqrt(vi / c2) + eps)
      }
      stats.flops += 10 * pd.length
    },
  }
  return backend
}
