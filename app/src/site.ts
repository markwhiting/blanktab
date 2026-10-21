// Pure helpers for the page's text: the history table, the links, the build label. No DOM here.
import type { SessionRow, Summary } from "@blanktab/harness"

export interface Repo { readonly owner: string; readonly name: string }

const gh = (repo: Repo) => `https://github.com/${repo.owner}/${repo.name}`

/**
 * A setting value as shown on the page: numbers to at most 3 decimals, except that a nonzero value below 1 keeps 3
 * significant digits (lr 5e-4 is "0.0005", not "0.001"; rounding to decimals would misstate small settings).
 */
export const fmt = (v: number | string) => {
  if (typeof v !== "number") return v
  return v !== 0 && Math.abs(v) < 1 ? String(Number(v.toPrecision(3))) : String(Math.round(v * 1000) / 1000)
}

export const fmtSettings = (settings: Record<string, number | string>) => {
  const entries = Object.entries(settings)
  return entries.length ? entries.map(([k, v]) => `${k} = ${fmt(v)}`).join(", ") : "none"
}

export interface HistoryRow {
  learnerId: string; href: string; steps: string; final: string; setting: string; host: string
  /** Compute lived (CU) and the last loss ratio (predLoss ÷ the lazy guess's), "—" for a v1 (flailer) learner. */
  cu: string; lossRatio: string
  /** The latest held-out ratio (decision 0015 §3), "—" for a learner from before the held-out check. */
  heldOutRatio: string
}

const orDash = (v: number | null | undefined) => (typeof v === "number" ? fmt(v) : "—")

/**
 * Past sessions, newest first by lastWrittenAt (ISO strings sort as times). Rows without it (older summaries)
 * go last, ordered by sessionsSeenAtBirth descending, then the later summary entry first.
 */
export function historyRows(summary: Summary, repo: Repo): HistoryRow[] {
  return summary.sessions
    .map((s, i) => ({ s, i }))
    .sort((a, b) => {
      const ta = a.s.lastWrittenAt, tb = b.s.lastWrittenAt
      if (ta !== undefined || tb !== undefined) {
        if (ta === undefined) return 1
        if (tb === undefined) return -1
        if (ta !== tb) return ta < tb ? 1 : -1
      }
      return b.s.sessionsSeenAtBirth - a.s.sessionsSeenAtBirth || b.i - a.i
    })
    .map(({ s }: { s: SessionRow }) => ({
      learnerId: s.learnerId,
      href: `${gh(repo)}/tree/main/data/runs/${s.learnerId}`,
      steps: String(s.stepsTotal),
      final: s.final ? "yes" : "no",
      setting: fmtSettings(s.settings),
      host: `${s.hostSig.cores} cores · ${s.hostSig.webgpu ? "webgpu" : "no webgpu"}`,
      cu: orDash(s.cu),
      lossRatio: orDash(s.lossRatio),
      heldOutRatio: orDash(s.heldOutRatio),
    }))
}

export interface Link { label: string; href: string }

export function siteLinks(repo: Repo): Link[] {
  const r = gh(repo)
  return [
    { label: "Source", href: r },
    { label: "Design", href: `${r}/blob/main/docs/specs/2026-10-19-blanktab-v1-design.md` },
    { label: "Decisions", href: `${r}/tree/main/docs/decisions` },
    { label: "Research notes", href: `${r}/blob/main/docs/research/2026-10-16-research-synthesis-v0.1.md` },
    { label: "How to check it works", href: `${r}/blob/main/docs/validate-plan-01.md` },
    { label: "Session pull requests", href: `${r}/pulls` },
    { label: "Actions", href: `${r}/actions` },
    { label: "Session data", href: `${r}/tree/main/data/runs` },
    { label: "Related: life", href: "https://markwhiting.github.io/life/" },
  ]
}

/** "<short sha> built <time>", linking to the commit (or to the repo for a dev build). */
export function buildLabel(codeSha: string, builtAt: string, repo: Repo): { text: string; href: string } {
  const when = `${builtAt.slice(0, 10)} ${builtAt.slice(11, 16)} UTC`
  const dev = codeSha === "dev"
  return { text: `${dev ? "dev" : codeSha.slice(0, 7)} built ${when}`, href: dev ? gh(repo) : `${gh(repo)}/commit/${codeSha}` }
}
