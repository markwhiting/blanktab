// Newborn reference mode (spec §2.3; plan 05a Task 12): ?newborns=a-b runs newborn reference lives k = a…b in this tab,
// in place of a life. A production feature (not a test knob): the reference is measured on the deployed code. Their
// files stay in this browser for tools/eval/collect-newborns.ts (controller ruling B), never sent as PRs.
import { NEWBORN_COUNT } from "@blanktab/harness/newborn"

const INDEX = /^(0|[1-9][0-9]*)$/

/** The newborns ?newborns=a-b asks for (0 ≤ a ≤ b < NEWBORN_COUNT), or null: not newborn mode. */
export function readNewborns(search: string): { from: number; to: number } | null {
  const v = new URLSearchParams(search).get("newborns")
  const parts = v?.split("-") ?? []
  if (parts.length !== 2 || !parts.every(p => INDEX.test(p))) return null
  const [from, to] = parts.map(Number) as [number, number]
  return from <= to && to < NEWBORN_COUNT ? { from, to } : null
}

/** The page's line for newborn k of a run of a…b. */
export const newbornText = (k: number, r: { from: number; to: number }) => `Newborn reference ${k} of ${r.from}–${r.to}`
