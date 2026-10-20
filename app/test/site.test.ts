import { describe, it, expect } from "vitest"
import type { SessionRow } from "@blanktab/harness"
import { buildLabel, historyRows, siteLinks } from "../src/site"

const REPO = { owner: "markwhiting", name: "learner" } as const
const row = (id: string, seen: number, over: Partial<SessionRow> = {}): SessionRow => ({
  learnerId: id, settings: { moveProbability: 0.123456 }, hostSig: { cores: 8, deviceMemoryGb: null, webgpu: true, tabsOnMachine: 1 },
  stepsTotal: 3000, final: true, codeSha: "a".repeat(40), knowledgeSha: "b".repeat(64), sessionsSeenAtBirth: seen, ...over,
})

describe("historyRows", () => {
  it("is empty for no sessions", () => {
    expect(historyRows({ schemaVersion: 1, sessions: [] }, REPO)).toEqual([])
  })
  it("lists newest first by lastWrittenAt; rows without it go last (born after more sessions first, then later entry first)", () => {
    const rows = historyRows({ schemaVersion: 1, sessions: [
      row("lrn_1111111111111111", 5),
      row("lrn_2222222222222222", 0, { lastWrittenAt: "2026-10-19T12:00:00.000Z" }),
      row("lrn_3333333333333333", 1),
      row("lrn_4444444444444444", 0, { lastWrittenAt: "2026-10-20T08:00:00.000Z" }),
      row("lrn_5555555555555555", 1),
    ] }, REPO)
    expect(rows.map(r => r.learnerId)).toEqual(["lrn_4444444444444444", "lrn_2222222222222222", "lrn_1111111111111111", "lrn_5555555555555555", "lrn_3333333333333333"])
  })
  it("formats each column and links the learner to its data folder", () => {
    const [r] = historyRows({ schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa", 0, { final: false, stepsTotal: 1500, hostSig: { cores: 4, deviceMemoryGb: 8, webgpu: false, tabsOnMachine: 2 } })] }, REPO)
    expect(r).toEqual({
      learnerId: "lrn_aaaaaaaaaaaaaaaa",
      href: "https://github.com/markwhiting/learner/tree/main/data/runs/lrn_aaaaaaaaaaaaaaaa",
      steps: "1500", final: "no", setting: "moveProbability = 0.123", host: "4 cores · no webgpu",
    })
    expect(historyRows({ schemaVersion: 1, sessions: [row("lrn_aaaaaaaaaaaaaaaa", 0, { settings: {} })] }, REPO)[0]).toMatchObject({ final: "yes", setting: "none", host: "8 cores · webgpu" })
  })
})

describe("siteLinks", () => {
  it("builds every GitHub link from the repo, plus life", () => {
    const links = siteLinks(REPO)
    expect(links.map(l => l.label)).toEqual([
      "Source", "Design", "Decisions", "Research notes", "How to check it works", "Session pull requests", "Actions", "Session data", "Related: life",
    ])
    const gh = "https://github.com/markwhiting/learner"
    expect(Object.fromEntries(links.map(l => [l.label, l.href]))).toEqual({
      "Source": gh,
      "Design": `${gh}/blob/main/docs/specs/2026-10-19-blanktab-v1-design.md`,
      "Decisions": `${gh}/tree/main/docs/decisions`,
      "Research notes": `${gh}/blob/main/docs/research/2026-10-16-research-synthesis-v0.1.md`,
      "How to check it works": `${gh}/blob/main/docs/validate-plan-01.md`,
      "Session pull requests": `${gh}/pulls`,
      "Actions": `${gh}/actions`,
      "Session data": `${gh}/tree/main/data/runs`,
      "Related: life": "https://markwhiting.github.io/life/",
    })
  })
})

describe("buildLabel", () => {
  const time = "2026-10-20T01:45:07.000Z"
  it("shows the short sha and links to its commit", () => {
    expect(buildLabel("0123456789abcdef0123456789abcdef01234567", time, REPO)).toEqual({
      text: "0123456 built 2026-10-20 01:45 UTC",
      href: "https://github.com/markwhiting/learner/commit/0123456789abcdef0123456789abcdef01234567",
    })
  })
  it("a dev build links to the repo", () => {
    expect(buildLabel("dev", time, REPO)).toEqual({ text: "dev built 2026-10-20 01:45 UTC", href: "https://github.com/markwhiting/learner" })
  })
})
