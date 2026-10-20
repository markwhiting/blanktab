// Plain DOM rendering. No framework: a handful of labelled fields and two canvases.
import { FOVEA, PERIPHERY } from "@blanktab/interface"
import type { ContributorStatus, Knowledge } from "@blanktab/harness"

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel)
  if (!el) throw new Error(`missing ${sel}`)
  return el
}
const byTest = (id: string) => $<HTMLElement>(`[data-testid="${id}"]`)

function painter(canvas: HTMLCanvasElement, w: number, h: number): (rgb: Uint8Array) => void {
  const ctx = canvas.getContext("2d")
  if (!ctx) throw new Error("2d context unavailable")
  const img = ctx.createImageData(w, h)
  return rgb => {
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
      img.data[j] = rgb[i]!; img.data[j + 1] = rgb[i + 1]!; img.data[j + 2] = rgb[i + 2]!; img.data[j + 3] = 255
    }
    ctx.putImageData(img, 0, 0)
  }
}

const fmt = (v: number | string) => (typeof v === "number" ? String(Math.round(v * 1000) / 1000) : v)

export interface Ui {
  knowledge(k: Knowledge): void
  learner(o: { id: string; settings: Record<string, number | string>; sessionsBefore: number; lifetimeSteps: number }): void
  frame(steps: number, periphery: Uint8Array, fovea: Uint8Array): void
  steps(steps: number): void
  contribution(s: ContributorStatus): void
  lifeComplete(id: string): void
  error(message: string): void
  tokenPanel(visible: boolean): void
  /** The handler returns an error message to show, or null on success. */
  onTokenSubmit(handler: (token: string) => Promise<string | null>): void
}

export function createUi(): Ui {
  const paintPeriphery = painter($("#periphery"), PERIPHERY.width, PERIPHERY.height)
  const paintFovea = painter($("#fovea"), FOVEA.width, FOVEA.height)
  const panel = byTest("token-panel"), input = $<HTMLInputElement>("#token"), button = $<HTMLButtonElement>("#token-save"), err = $("#token-error")
  const life = byTest("life")
  let lifetime = 0

  return {
    knowledge(k) {
      byTest("knowledge").textContent = k.offline
        ? "offline: the shared knowledge could not be loaded, so this learner starts from nothing."
        : `loaded (knowledgeSha ${k.manifest.knowledgeSha.slice(0, 12)}…)`
    },
    learner({ id, settings, sessionsBefore, lifetimeSteps }) {
      lifetime = lifetimeSteps
      byTest("learner-id").textContent = id
      byTest("sessions").textContent = `${sessionsBefore} session${sessionsBefore === 1 ? "" : "s"} before this one`
      const entries = Object.entries(settings)
      byTest("setting").textContent = entries.length ? entries.map(([k, v]) => `${k} = ${fmt(v)}`).join(", ") : "none"
      this.steps(0)
    },
    frame(steps, periphery, fovea) {
      paintPeriphery(periphery)
      paintFovea(fovea)
      this.steps(steps)
    },
    steps(steps) {
      byTest("steps").textContent = `${steps} of ${lifetime}`
    },
    contribution(s) {
      const waiting = s.pending > 0 ? ` (${s.pending} log${s.pending === 1 ? "" : "s"} waiting in this browser)` : ""
      byTest("contribution").textContent = s.message + waiting
    },
    lifeComplete(id) {
      life.hidden = false
      life.textContent = `This learner's life is complete (${id}). A newborn takes its place.`
    },
    error(message) {
      life.hidden = false
      life.textContent = `The learner stopped: ${message}`
    },
    tokenPanel(visible) {
      panel.hidden = !visible
    },
    onTokenSubmit(handler) {
      const submit = async () => {
        const token = input.value.trim()
        if (!token) { err.textContent = "Paste a token first."; return }
        button.disabled = true
        err.textContent = "Checking with GitHub…"
        try {
          const problem = await handler(token)
          err.textContent = problem ?? ""
          if (problem === null) input.value = ""
        } finally {
          button.disabled = false
        }
      }
      button.addEventListener("click", () => void submit())
      input.addEventListener("keydown", e => { if (e.key === "Enter") void submit() })
    },
  }
}
