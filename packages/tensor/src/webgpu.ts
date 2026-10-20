import { ResourceError, ShapeError, broadcastShape, im2colGeometry } from "./backend"
import type { Backend, Storage } from "./backend"
import { adamKernel } from "./kernels/adam.wgsl"
import { MAX_RANK, binaryKernel } from "./kernels/binary.wgsl"
import { WG } from "./kernels/common.wgsl"
import { scalarKernel, unaryKernel } from "./kernels/elementwise.wgsl"
import { gatherRowsKernel, scatterAddRowsKernel } from "./kernels/gather.wgsl"
import { col2imKernel, im2colKernel } from "./kernels/im2col.wgsl"
import { TILE, matmulKernel } from "./kernels/matmul.wgsl"
import { reduceLastKernel, sumToShapeKernel } from "./kernels/reduce.wgsl"

interface GpuStorage extends Storage {
  buffer: GPUBuffer
  disposed: boolean
}

export interface WebGpuOptions {
  powerPreference?: "high-performance" | "low-power"
}

/** Identity of the adapter a WebGPU backend runs on (for logs and benchmark records). */
export interface WebGpuAdapterInfo {
  readonly vendor: string
  readonly architecture: string
  readonly device: string
  readonly description: string
  readonly isFallbackAdapter: boolean
}

export interface WebGpuBackend extends Backend {
  readonly name: "webgpu"
  readonly adapterInfo: WebGpuAdapterInfo
  /**
   * Lets the device go (e.g. after a fatal read, before asking for a new one): pending work is dropped, released
   * buffers are destroyed, then the device. Every later read rejects with a ResourceError. Idempotent.
   */
  destroy(): void
}

/** True for fallback/software adapters (SwiftShader, llvmpipe, ...), which do not exercise a real GPU compiler. */
export function isSoftwareAdapter(info: WebGpuAdapterInfo): boolean {
  return info.isFallbackAdapter || /swiftshader|llvmpipe|lavapipe|software/i.test(`${info.vendor} ${info.architecture} ${info.device} ${info.description}`)
}

/** Resolves null when WebGPU (navigator.gpu) or an adapter is unavailable. */
export async function createWebGpuBackend(opts: WebGpuOptions = {}): Promise<WebGpuBackend | null> {
  const gpu = typeof navigator === "undefined" ? undefined : navigator.gpu
  if (!gpu) return null
  const adapter = await gpu.requestAdapter(opts.powerPreference ? { powerPreference: opts.powerPreference } : undefined)
  if (!adapter) return null
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  })
  const i = adapter.info
  // isFallbackAdapter moved from GPUAdapter to GPUAdapterInfo; older browsers have only the former.
  const legacyFallback = (adapter as { isFallbackAdapter?: boolean }).isFallbackAdapter
  const adapterInfo: WebGpuAdapterInfo = {
    vendor: i.vendor,
    architecture: i.architecture,
    device: i.device,
    description: i.description,
    isFallbackAdapter: (i as { isFallbackAdapter?: boolean }).isFallbackAdapter ?? legacyFallback ?? false,
  }
  return webGpuBackendOn(device, adapterInfo)
}

// WebGPU flag values (spec constants; TypeScript's DOM lib types the interfaces but not these namespaces).
const BUF = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 } as const
const MAP_MODE_READ = 0x1

const prod = (s: number[]) => s.reduce((p, d) => p * d, 1)
const fmt = (s: number[]) => `[${s.join(",")}]`

/** Contiguous strides of `shape` left-padded to rank 4; size-1 dims get stride 0 (broadcast). */
function paddedStrides(shape: number[]): number[] {
  const out = [0, 0, 0, 0]
  let stride = 1
  for (let i = shape.length - 1; i >= 0; i--) {
    out[MAX_RANK - shape.length + i] = shape[i] === 1 ? 0 : stride
    stride *= shape[i]!
  }
  return out
}
const padShape = (shape: number[]) => [...new Array<number>(MAX_RANK - shape.length).fill(1), ...shape]

type Field = ["u" | "f", number]
const u = (x: number): Field => ["u", x]
const fl = (x: number): Field => ["f", x]

/** Bytes of a uniform holding `fields` (4-byte scalars, vec4s as 4 consecutive u32), rounded up to 16. */
const uniformBytes = (fields: Field[]) => Math.max(16, Math.ceil((fields.length * 4) / 16) * 16)

/** Writes uniform fields at `offset` of `view` (little-endian). */
function writeUniform(view: DataView, offset: number, fields: Field[]): void {
  fields.forEach(([t, x], i) => (t === "u" ? view.setUint32(offset + i * 4, x, true) : view.setFloat32(offset + i * 4, x, true)))
}

// Submit pending work once this many passes are queued (or this many dead buffers await destruction).
const MAX_PENDING_PASSES = 64
const MAX_PENDING_TRASH = 256
/** Largest uniform any kernel declares (sumToShape: 14 u32 = 64 bytes), with headroom. */
const MAX_UNIFORM_BYTES = 128

function webGpuBackendOn(device: GPUDevice, adapterInfo: WebGpuAdapterInfo): WebGpuBackend {
  const stats = { flops: 0, liveBuffers: 0 }
  const maxBinding = device.limits.maxStorageBufferBindingSize
  const maxBufferSize = device.limits.maxBufferSize
  const maxGroups = device.limits.maxComputeWorkgroupsPerDimension
  const usage = BUF.STORAGE | BUF.COPY_SRC | BUF.COPY_DST

  // ---- uniforms: one slot per pass of the open batch in a single ring buffer, staged on the CPU
  // and uploaded with one writeBuffer just before the batch is submitted. Creating a mapped buffer
  // per dispatch cost ~38 µs each, two thirds of a learner-sized step (spike 2). Reusing the ring
  // across batches is safe: queue operations run in order, so the next batch's write lands after
  // the previous batch's passes have read their slots.
  const uniformSlot = Math.ceil(MAX_UNIFORM_BYTES / device.limits.minUniformBufferOffsetAlignment) * device.limits.minUniformBufferOffsetAlignment
  const uniformRing = device.createBuffer({ size: MAX_PENDING_PASSES * uniformSlot, usage: BUF.UNIFORM | BUF.COPY_DST })
  const uniformStaging = new ArrayBuffer(MAX_PENDING_PASSES * uniformSlot)
  const uniformView = new DataView(uniformStaging)

  // ---- error reporting: every batch of GPU work runs inside error scopes; the next read() awaits
  // the scopes closed since the previous read and rejects with the first error, then the state is
  // clear again (not sticky). Errors outside any scope (e.g. buffer creation) are queued the same way.
  const pendingErrors: Promise<GPUError | null>[] = []
  device.addEventListener("uncapturederror", (e) => {
    pendingErrors.push(Promise.resolve((e as GPUUncapturedErrorEvent).error))
  })
  const pushScopes = () => {
    device.pushErrorScope("out-of-memory")
    device.pushErrorScope("validation")
  }
  const popScopes = () => {
    const validation = device.popErrorScope()
    const oom = device.popErrorScope()
    pendingErrors.push(validation, oom)
  }
  let lost: string | null = null
  let destroyed = false
  void device.lost.then((info) => {
    lost = `WebGPU device lost (${info.reason}): ${info.message}`
  })

  // ---- command batching: passes go into one encoder, submitted on read() or when the batch is large ----
  let encoder: GPUCommandEncoder | null = null
  let pendingPasses = 0
  /** Buffers released while pending passes may still reference them; destroyed after the next submit. */
  const trash: GPUBuffer[] = []
  const flush = () => {
    if (encoder) {
      if (pendingPasses > 0) device.queue.writeBuffer(uniformRing, 0, uniformStaging, 0, pendingPasses * uniformSlot)
      device.queue.submit([encoder.finish()])
      popScopes() // opened by beginBatch, so it covers pipeline/bind-group creation, encoding and submit
      encoder = null
      pendingPasses = 0
    }
    // Destroying after submit is safe: the queue keeps buffers alive until submitted work completes.
    for (const b of trash) b.destroy()
    trash.length = 0
  }
  const release = (b: GPUBuffer) => {
    if (encoder) {
      trash.push(b)
      if (trash.length >= MAX_PENDING_TRASH) flush()
    } else b.destroy()
  }

  const pipelines = new Map<string, GPUComputePipeline>()
  const pipeline = (key: string, code: () => string): GPUComputePipeline => {
    let p = pipelines.get(key)
    if (!p) {
      p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ label: key, code: code() }), entryPoint: "main" } })
      pipelines.set(key, p)
    }
    return p
  }

  const checkBytes = (what: string, bytes: number) => {
    if (bytes > maxBinding) {
      throw new ResourceError(`${what}: allocation of ${bytes} bytes exceeds maxStorageBufferBindingSize ${maxBinding} bytes`)
    }
    if (bytes > maxBufferSize) throw new ResourceError(`${what}: allocation of ${bytes} bytes exceeds maxBufferSize ${maxBufferSize} bytes`)
  }
  /** A raw buffer of `bytes` (min 4: zero-size buffers cannot be bound), optionally filled. */
  const rawBuffer = (what: string, bytes: number, init?: ArrayBufferView): GPUBuffer => {
    checkBytes(what, bytes)
    if (lost) throw new ResourceError(lost)
    const size = Math.max(4, Math.ceil(bytes / 4) * 4)
    const b = device.createBuffer({ size, usage, mappedAtCreation: init !== undefined })
    if (init) {
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(init.buffer, init.byteOffset, init.byteLength))
      b.unmap()
    }
    return b
  }
  const make = (length: number, init?: Float32Array): GpuStorage => {
    const buffer = rawBuffer("webgpu", length * 4, init) // GPU buffers are zero-initialized
    stats.liveBuffers++
    return { length, backend, buffer, disposed: false }
  }
  /** Allocates an n-element output and runs `fill` on it; the output is freed if `fill` throws. */
  const produce = (n: number, fill: (out: GPUBuffer) => void): GpuStorage => {
    const out = make(n)
    try {
      fill(out.buffer)
    } catch (e) {
      backend.dispose(out)
      throw e
    }
    return out
  }
  const get = (s: Storage): GPUBuffer => {
    const g = s as GpuStorage
    if (g.backend !== backend) throw new ResourceError("storage belongs to a different backend")
    if (g.disposed) throw new ResourceError("storage used after dispose")
    return g.buffer
  }
  const checkLen = (what: string, s: Storage, want: number) => {
    if (s.length !== want) throw new ShapeError(`${what}: expected length ${want}, got ${s.length}`)
  }
  const checkRank = (what: string, ...shapes: number[][]) => {
    for (const s of shapes) {
      if (s.length > MAX_RANK) throw new ShapeError(`${what}: rank ${s.length} of ${fmt(s)} exceeds the WebGPU maximum of ${MAX_RANK}`)
    }
  }

  /** Workgroup grid for `groups` workgroups, folded into 2-D when it exceeds the per-dimension limit. */
  const grid = (what: string, groups: number): [number, number] => {
    const x = Math.min(groups, maxGroups)
    const y = Math.ceil(groups / x)
    if (y > maxGroups) throw new ResourceError(`${what}: ${groups} workgroups exceed the device limit`)
    return [x, y]
  }

  /** The open batch encoder; starting a batch opens its error scopes (closed in flush). */
  const beginBatch = (): GPUCommandEncoder => {
    if (!encoder) {
      pushScopes()
      encoder = device.createCommandEncoder()
    }
    return encoder
  }
  /** Records one compute pass. `buffers` bind at 0..k-1; the uniform binds at k. Owned temporaries are released after. */
  const dispatch = (key: string, code: () => string, buffers: GPUBuffer[], fields: Field[], groups: [number, number], temps: GPUBuffer[] = []) => {
    const size = uniformBytes(fields)
    if (size > MAX_UNIFORM_BYTES) throw new Error(`${key}: uniform of ${size} bytes exceeds MAX_UNIFORM_BYTES ${MAX_UNIFORM_BYTES}`)
    const enc = beginBatch()
    const p = pipeline(key, code)
    const offset = pendingPasses * uniformSlot
    writeUniform(uniformView, offset, fields)
    const bind = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [
        ...buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        { binding: buffers.length, resource: { buffer: uniformRing, offset, size } },
      ],
    })
    const pass = enc.beginComputePass()
    pass.setPipeline(p)
    pass.setBindGroup(0, bind)
    pass.dispatchWorkgroups(groups[0], groups[1])
    pass.end()
    trash.push(...temps)
    if (++pendingPasses >= MAX_PENDING_PASSES || trash.length >= MAX_PENDING_TRASH) flush()
  }
  /** One thread per element of an n-element output. */
  const dispatch1d = (key: string, code: () => string, buffers: GPUBuffer[], fields: Field[], n: number, temps: GPUBuffer[] = []) => {
    const freeTemps = () => temps.forEach(release)
    if (n === 0) return freeTemps()
    let groups: [number, number]
    try {
      groups = grid(key, Math.ceil(n / WG))
    } catch (e) {
      freeTemps()
      throw e
    }
    dispatch(key, code, buffers, fields, groups, temps)
  }

  const backend: WebGpuBackend = {
    name: "webgpu",
    adapterInfo: Object.freeze({ ...adapterInfo }),
    stats,
    destroy() {
      if (destroyed) return
      destroyed = true
      lost ??= "WebGPU backend destroyed"
      encoder = null
      pendingPasses = 0
      for (const b of trash) b.destroy()
      trash.length = 0
      device.destroy()
    },
    upload: (data) => make(data.length, data),
    zeros: (length) => make(length),

    read(s) {
      let buf: GPUBuffer
      try {
        buf = get(s)
        if (lost) throw new ResourceError(lost)
      } catch (e) {
        return Promise.reject(e instanceof Error ? e : new Error(String(e)))
      }
      flush()
      const bytes = s.length * 4
      pushScopes()
      const staging = device.createBuffer({ size: Math.max(4, bytes), usage: BUF.MAP_READ | BUF.COPY_DST })
      const enc = device.createCommandEncoder()
      if (bytes > 0) enc.copyBufferToBuffer(buf, 0, staging, 0, bytes)
      device.queue.submit([enc.finish()])
      popScopes()
      // This read owns every error raised since the previous read (including its own copy).
      const errors = Promise.all(pendingErrors.splice(0))
      return (async () => {
        try {
          const first = (await errors).find((e) => e !== null)
          if (first) throw new ResourceError(`WebGPU error: ${first.message}`)
          await staging.mapAsync(MAP_MODE_READ)
          return new Float32Array(staging.getMappedRange(0, Math.max(4, bytes)).slice(0, bytes))
        } catch (e) {
          if (e instanceof ResourceError) throw e
          throw new ResourceError(`WebGPU read failed: ${lost ?? (e instanceof Error ? e.message : String(e))}`)
        } finally {
          staging.destroy() // also unmaps
        }
      })()
    },

    dispose(s) {
      const g = s as GpuStorage
      if (g.backend !== backend || g.disposed) return
      g.disposed = true
      stats.liveBuffers--
      release(g.buffer)
    },

    copy(s) {
      const x = get(s)
      // Recorded in the open batch (no compute pass, so no uniform slot), ordered after pending writers of x.
      return produce(s.length, (o) => {
        if (s.length > 0) beginBatch().copyBufferToBuffer(x, 0, o, 0, s.length * 4)
      })
    },

    unary(op, a) {
      const x = get(a)
      const out = produce(a.length, (o) => dispatch1d(`unary:${op}`, () => unaryKernel(op), [x, o], [u(a.length)], a.length))
      stats.flops += a.length
      return out
    },

    scalar(op, a, c) {
      const x = get(a)
      const out = produce(a.length, (o) => dispatch1d(`scalar:${op}`, () => scalarKernel(op), [x, o], [u(a.length), fl(c)], a.length))
      stats.flops += a.length
      return out
    },

    binary(op, a, aShape, b, bShape, outShape) {
      const x = get(a)
      const y = get(b)
      checkLen("binary lhs", a, prod(aShape))
      checkLen("binary rhs", b, prod(bShape))
      const expected = broadcastShape(aShape, bShape)
      if (expected.length !== outShape.length || expected.some((d, i) => d !== outShape[i])) {
        throw new ShapeError(`binary: outShape ${fmt(outShape)} does not match broadcast of ${fmt(aShape)} and ${fmt(bShape)}`)
      }
      checkRank("binary", aShape, bShape, outShape)
      const n = prod(outShape)
      const fields = [...padShape(outShape), ...paddedStrides(aShape), ...paddedStrides(bShape), n].map(u)
      const out = produce(n, (o) => dispatch1d(`binary:${op}`, () => binaryKernel(op), [x, y, o], fields, n))
      stats.flops += n
      return out
    },

    sumToShape(a, fromShape, toShape) {
      const x = get(a)
      checkLen("sumToShape input", a, prod(fromShape))
      const rank = fromShape.length
      const bad = () => new ShapeError(`sumToShape: cannot reduce ${fmt(fromShape)} to ${fmt(toShape)}`)
      if (toShape.length > rank) throw bad()
      const padded = [...new Array<number>(rank - toShape.length).fill(1), ...toShape]
      for (let d = 0; d < rank; d++) if (padded[d] !== fromShape[d] && padded[d] !== 1) throw bad()
      checkRank("sumToShape", fromShape)
      const from4 = padShape(fromShape)
      const to4 = padShape(padded)
      const red4 = from4.map((d, i) => (to4[i] === 1 ? d : 1))
      const inStride = [0, 0, 0, 0]
      for (let i = MAX_RANK - 1, s = 1; i >= 0; i--) {
        inStride[i] = s
        s *= from4[i]!
      }
      const nOut = prod(toShape)
      const fields = [...to4, ...red4, ...inStride, nOut, prod(red4)].map(u)
      const out = produce(nOut, (o) => dispatch1d("sumToShape", sumToShapeKernel, [x, o], fields, nOut))
      stats.flops += a.length
      return out
    },

    matmul(a, b, m, k, n, transA, transB) {
      const x = get(a)
      const y = get(b)
      checkLen("matmul lhs", a, m * k)
      checkLen("matmul rhs", b, k * n)
      const gx = Math.ceil(n / TILE)
      const gy = Math.ceil(m / TILE)
      if (gx > maxGroups || gy > maxGroups) throw new ResourceError(`matmul: ${m}x${n} output exceeds the device workgroup limit`)
      const out = produce(m * n, (o) => {
        if (m * n > 0) dispatch("matmul", matmulKernel, [x, y, o], [m, k, n, transA ? 1 : 0, transB ? 1 : 0].map(u), [gx, gy])
      })
      stats.flops += 2 * m * k * n
      return out
    },

    reduceLast(op, a, rows, cols) {
      const x = get(a)
      checkLen("reduceLast", a, rows * cols)
      if (op === "max" && cols === 0) throw new ShapeError(`reduceLast max: cannot reduce ${rows} empty rows`)
      // Empty rows (cols 0, sum only) need no pass: GPU buffers start zeroed.
      const out = produce(rows, (o) => {
        if (rows > 0 && cols > 0) dispatch(`reduceLast:${op}`, () => reduceLastKernel(op), [x, o], [u(rows), u(cols)], grid("reduceLast", rows))
      })
      stats.flops += rows * cols
      return out
    },

    im2col(xs, c, h, w, kh, kw, stride, pad) {
      const x = get(xs)
      checkLen("im2col", xs, c * h * w)
      const [oh, ow] = im2colGeometry("im2col", c, h, w, kh, kw, stride, pad)
      const n = c * kh * kw * oh * ow
      return produce(n, (o) => dispatch1d("im2col", im2colKernel, [x, o], [c, h, w, kh, kw, stride, pad, oh, ow, n].map(u), n))
    },

    col2im(cs, c, h, w, kh, kw, stride, pad) {
      const cols = get(cs)
      const [oh, ow] = im2colGeometry("col2im", c, h, w, kh, kw, stride, pad)
      checkLen("col2im", cs, c * kh * kw * oh * ow)
      const n = c * h * w
      return produce(n, (o) => dispatch1d("col2im", col2imKernel, [cols, o], [c, h, w, kh, kw, stride, pad, oh, ow, n].map(u), n))
    },

    gatherRows(a, rows, cols, idx) {
      const x = get(a)
      checkLen("gatherRows", a, rows * cols)
      for (const r of idx) if (r < 0 || r >= rows) throw new ShapeError(`gatherRows: index ${r} out of range for ${rows} rows`)
      const n = idx.length * cols
      return produce(n, (o) => {
        if (n === 0) return
        const ib = rawBuffer("gatherRows indices", idx.byteLength, new Uint32Array(idx.buffer, idx.byteOffset, idx.length))
        dispatch1d("gatherRows", gatherRowsKernel, [x, ib, o], [u(cols), u(n)], n, [ib])
      })
    },

    scatterAddRows(grad, rows, cols, idx) {
      const g = get(grad)
      checkLen("scatterAddRows", grad, idx.length * cols)
      for (const r of idx) if (r < 0 || r >= rows) throw new ShapeError(`scatterAddRows: index ${r} out of range for ${rows} rows`)
      const n = rows * cols
      return produce(n, (o) => {
        if (n === 0 || idx.length === 0) return
        // CSR of the inverse map: sources for row r are list[offsets[r] .. offsets[r+1]), ascending.
        const offsets = new Uint32Array(rows + 1)
        for (const r of idx) offsets[r + 1]!++
        for (let r = 0; r < rows; r++) offsets[r + 1]! += offsets[r]!
        const fill = offsets.slice(0, rows)
        const list = new Uint32Array(idx.length)
        idx.forEach((r, j) => (list[fill[r]!++] = j))
        const ob = rawBuffer("scatterAddRows offsets", offsets.byteLength, offsets)
        const lb = rawBuffer("scatterAddRows list", list.byteLength, list)
        dispatch1d("scatterAddRows", scatterAddRowsKernel, [g, ob, lb, o], [u(cols), u(n)], n, [ob, lb])
      })
    },

    adam(p, g, m, v, lr, b1, b2, eps, t) {
      const pb = get(p)
      const gb = get(g)
      const mb = get(m)
      const vb = get(v)
      if (g.length !== p.length || m.length !== p.length || v.length !== p.length) {
        throw new ShapeError("adam: p, g, m, v must have equal length")
      }
      if (new Set([p, g, m, v]).size !== 4) throw new ShapeError("adam: p, g, m, v must be distinct storages")
      const c1 = 1 - Math.pow(b1, t)
      const c2 = 1 - Math.pow(b2, t)
      const fields = [u(p.length), ...[lr, b1, 1 - b1, b2, 1 - b2, eps, c1, c2].map(fl)]
      dispatch1d("adam", adamKernel, [pb, gb, mb, vb], fields, p.length)
      stats.flops += 10 * p.length
    },
  }
  return backend
}
