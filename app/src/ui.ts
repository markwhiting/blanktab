// Plain DOM rendering. No framework: three tabs, a handful of labelled fields, two canvases and a table.
import { FOVEA, PERIPHERY } from "@blanktab/interface"
import { REPO, type ContributorStatus, type Knowledge } from "@blanktab/harness"
import { buildLabel, fmtSettings, historyRows, siteLinks } from "./site"

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

/** Client-side tabs: real buttons with aria-selected, arrow keys move between them. Idempotent. */
function setupTabs(): void {
  const tabs = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"]')]
  const select = (t: HTMLButtonElement, focus: boolean) => {
    for (const x of tabs) {
      const on = x === t
      x.setAttribute("aria-selected", String(on))
      x.tabIndex = on ? 0 : -1
      $(`#${x.getAttribute("aria-controls")!}`).hidden = !on
    }
    if (focus) t.focus()
  }
  tabs.forEach((t, i) => {
    t.onclick = () => select(t, false)
    t.onkeydown = e => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0
      if (d) { e.preventDefault(); select(tabs[(i + d + tabs.length) % tabs.length]!, true) }
    }
  })
}

/** Everything that doesn't depend on the learner: links, repo name, build id. */
function renderStatic(): void {
  for (const ul of document.querySelectorAll("[data-links]")) {
    ul.replaceChildren(...siteLinks(REPO).map(({ label, href }) => {
      const li = document.createElement("li"), a = document.createElement("a")
      a.href = href; a.textContent = label; a.target = "_blank"; a.rel = "noopener noreferrer"
      li.append(a)
      return li
    }))
  }
  for (const el of document.querySelectorAll("[data-repo]")) el.textContent = `${REPO.owner}/${REPO.name}`
  const b = buildLabel(__CODE_SHA__, __BUILD_TIME__, REPO), build = byTest("build") as HTMLAnchorElement
  build.textContent = b.text
  build.href = b.href
}

function renderHistory(k: Knowledge): void {
  const box = byTest("history"), rows = historyRows(k.summary, REPO)
  box.querySelector("tbody")!.replaceChildren(...rows.map(r => {
    const tr = document.createElement("tr")
    const a = document.createElement("a")
    a.href = r.href; a.textContent = r.learnerId; a.target = "_blank"; a.rel = "noopener noreferrer"
    const cells: [string, string | Node][] = [["Learner", a], ["Steps", r.steps], ["Final", r.final], ["Setting", r.setting], ["Host", r.host]]
    for (const [i, [label, c]] of cells.entries()) {
      const td = document.createElement("td")
      if (i === 1) td.className = "num"
      td.dataset.label = label // the stacked mobile layout shows it
      td.append(c)
      tr.append(td)
    }
    return tr
  }))
  box.querySelector("table")!.hidden = rows.length === 0
  box.querySelector<HTMLElement>(".empty")!.hidden = rows.length > 0
}

export interface Ui {
  knowledge(k: Knowledge): void
  learner(o: { id: string; settings: Record<string, number | string>; sessionsBefore: number; lifetimeSteps: number }): void
  frame(steps: number, periphery: Uint8Array, fovea: Uint8Array): void
  steps(steps: number): void
  contribution(s: ContributorStatus): void
  lifeComplete(id: string): void
  error(message: string): void
  tokenPanel(visible: boolean): void
  /** The small "change token" control, shown whenever a token is stored. */
  changeToken(visible: boolean): void
  onChangeToken(handler: () => void): void
  /** The handler returns an error message to show, or null on success. */
  onTokenSubmit(handler: (token: string) => Promise<string | null>): void
}

export function createUi(): Ui {
  const paintPeriphery = painter($("#periphery"), PERIPHERY.width, PERIPHERY.height)
  const paintFovea = painter($("#fovea"), FOVEA.width, FOVEA.height)
  const panel = byTest("token-panel"), input = $<HTMLInputElement>("#token"), button = $<HTMLButtonElement>("#token-save"), err = $("#token-error")
  const life = byTest("life"), change = byTest("change-token") as HTMLButtonElement
  let lifetime = 0
  setupTabs()
  renderStatic()

  return {
    knowledge(k) {
      byTest("knowledge").textContent = k.offline
        ? "offline: the shared knowledge could not be loaded, so this learner starts from nothing."
        : `loaded (knowledgeSha ${k.manifest.knowledgeSha.slice(0, 12)}…)`
      renderHistory(k)
    },
    learner({ id, settings, sessionsBefore, lifetimeSteps }) {
      lifetime = lifetimeSteps
      byTest("learner-id").textContent = id
      byTest("sessions").textContent = `${sessionsBefore} session${sessionsBefore === 1 ? "" : "s"} before this one`
      byTest("setting").textContent = fmtSettings(settings)
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
      life.classList.remove("error")
      life.textContent = `This learner's life is complete (${id}). A newborn takes its place.`
    },
    error(message) {
      life.hidden = false
      life.classList.add("error")
      life.textContent = `The learner stopped: ${message}`
    },
    tokenPanel(visible) {
      panel.hidden = !visible
    },
    changeToken(visible) {
      change.hidden = !visible
    },
    onChangeToken(handler) {
      change.addEventListener("click", handler)
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
        } catch {
          // Never leave "Checking with GitHub…" up forever. The cause is not shown: it could echo the token.
          err.textContent = "Something went wrong while saving the token; try again."
        } finally {
          button.disabled = false
        }
      }
      button.addEventListener("click", () => void submit())
      input.addEventListener("keydown", e => { if (e.key === "Enter") void submit() })
    },
  }
}
