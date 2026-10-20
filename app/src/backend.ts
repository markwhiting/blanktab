// Which tensor backend the learner runs on: WebGPU on a hardware adapter at GPU_WIDTHS, else the CPU at half width.
import { CPU_WIDTHS, GPU_WIDTHS, type Widths } from "@blanktab/learner"
import { createCpuBackend, isSoftwareAdapter, type Backend, type WebGpuBackend } from "@blanktab/tensor"

export type BackendKind = "webgpu" | "cpu"

export interface Chosen {
  kind: BackendKind
  backend: Backend
  /** For a newborn; a resumed learner keeps the widths in its checkpoint. */
  widths: Widths
  /** What the page shows, e.g. "WebGPU · Apple M4" or "CPU at half width · no WebGPU". */
  label: string
}

const cpu = (why: string, widths?: Widths): Chosen => ({
  kind: "cpu", backend: createCpuBackend(), widths: widths ?? CPU_WIDTHS, label: `CPU${widths ? "" : " at half width"} · ${why}`,
})

/** The CPU backend for a life that moves off the GPU: it keeps its widths (a GPU checkpoint is valid on the CPU, only slower). */
export const cpuFallback = (widths: Widths): Chosen => ({ kind: "cpu", backend: createCpuBackend(), widths, label: "CPU · the GPU failed, same widths" })

/**
 * WebGPU only on a hardware adapter: a software one (SwiftShader, llvmpipe, ...) is slower than the CPU backend.
 * `widths` overrides both defaults (tests); `force: "cpu"` never asks for WebGPU.
 */
export async function chooseBackend(o: { createGpu: () => Promise<WebGpuBackend | null>; force?: "cpu"; widths?: Widths }): Promise<Chosen> {
  if (o.force === "cpu") return cpu("WebGPU skipped", o.widths)
  let gpu: WebGpuBackend | null
  try {
    gpu = await o.createGpu()
  } catch (e) {
    return cpu(`WebGPU failed to start (${e instanceof Error ? e.message : String(e)})`, o.widths)
  }
  if (!gpu) return cpu("no WebGPU", o.widths)
  const info = gpu.adapterInfo
  const name = info.description || [info.vendor, info.architecture, info.device].filter(Boolean).join(" ") || "unknown adapter"
  if (isSoftwareAdapter(info)) return cpu(`software WebGPU adapter (${name})`, o.widths)
  return { kind: "webgpu", backend: gpu, widths: o.widths ?? GPU_WIDTHS, label: `WebGPU · ${name}` }
}
