import { ShapeError } from "./backend"
import type { Backend, Storage } from "./backend"
import { createCpuBackend } from "./cpu"

/** Graph node: the inputs of the op that produced a tensor and its backward rule (filled in by autograd). */
export interface GradNode {
  parents: Tensor[]
  /** Given d(loss)/d(this), returns d(loss)/d(parent) for each parent, in order. */
  backward: (g: Tensor) => Tensor[]
}

export interface TensorOptions {
  requiresGrad?: boolean
  backend?: Backend
}

const sizeOf = (shape: readonly number[]) => shape.reduce((p, d) => p * d, 1)

export class Tensor {
  readonly shape: readonly number[]
  readonly size: number
  readonly storage: Storage
  readonly backend: Backend
  requiresGrad: boolean
  grad: Tensor | null = null
  _node?: GradNode

  constructor(storage: Storage, shape: readonly number[], requiresGrad = false) {
    for (const d of shape) {
      if (!Number.isInteger(d) || d < 0) throw new ShapeError(`invalid shape [${shape.join(",")}]`)
    }
    this.size = sizeOf(shape)
    if (storage.length !== this.size) {
      throw new ShapeError(`storage length ${storage.length} does not match shape [${shape.join(",")}]`)
    }
    this.shape = [...shape]
    this.storage = storage
    this.backend = storage.backend
    this.requiresGrad = requiresGrad
  }

  read(): Promise<Float32Array> {
    return this.backend.read(this.storage)
  }

  dispose(): void {
    this.backend.dispose(this.storage)
  }
}

let defaultBackend: Backend | null = null

export function setDefaultBackend(b: Backend): void {
  defaultBackend = b
}

/** The CPU backend until `setDefaultBackend` is called. */
export function getDefaultBackend(): Backend {
  defaultBackend ??= createCpuBackend()
  return defaultBackend
}

export function tensor(data: Float32Array | number[], shape: number[], opts: TensorOptions = {}): Tensor {
  if (data.length !== sizeOf(shape)) {
    throw new ShapeError(`data length ${data.length} does not match shape [${shape.join(",")}]`)
  }
  const backend = opts.backend ?? getDefaultBackend()
  const arr = data instanceof Float32Array ? data : Float32Array.from(data)
  return new Tensor(backend.upload(arr), shape, opts.requiresGrad ?? false)
}

export function zeros(shape: number[], opts: TensorOptions = {}): Tensor {
  const backend = opts.backend ?? getDefaultBackend()
  return new Tensor(backend.zeros(sizeOf(shape)), shape, opts.requiresGrad ?? false)
}

/** A trainable leaf tensor (requiresGrad true). */
export function param(data: Float32Array, shape: number[]): Tensor {
  return tensor(data, shape, { requiresGrad: true })
}

let gradEnabled = true

export function isGradEnabled(): boolean {
  return gradEnabled
}

/** Runs `fn` with graph recording off (restored afterwards, even on throw). */
export function noGrad<T>(fn: () => T): T {
  const prev = gradEnabled
  gradEnabled = false
  try {
    return fn()
  } finally {
    gradEnabled = prev
  }
}
