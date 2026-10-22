// The milestone curve on the LEARNER tab (decision 0016): one bar per checkup, how many of the six short tests it passed
// in fresh rooms with its learning paused; and the caregiver's line. Pure helpers, no DOM.
import { FIRST_LOG_CU, MILESTONE_IDS, MILESTONE_SUITE_VERSION, type CheckupResult } from "@blanktab/harness/log"

/** Spec §8.2's names, in lower case, for the page's sentences. */
export const MILESTONE_NAMES: Record<string, string> = {
  M1: "predicts its world", M2: "moves with purpose", M3: "prefers things that respond",
  M4: "uncovers hidden content", M5: "explores places efficiently", M6: "uses the taskbar",
}

/** The number of milestones in a checkup (the current suite's). */
export const SUITE_SIZE = MILESTONE_IDS[MILESTONE_SUITE_VERSION]!.length

/** A CU level as written on the page: "512", "1,024". */
export const fmtLevel = (level: number) => level.toLocaleString("en-US")

/** Adds the results ahead (in level) of the last one; a level the page already has is never drawn twice. */
export function appendCheckups(prev: readonly CheckupResult[], add: readonly CheckupResult[]): CheckupResult[] {
  const out = [...prev]
  for (const r of [...add].sort((a, b) => a.level - b.level)) if (out.length === 0 || r.level > out.at(-1)!.level) out.push(r)
  return out
}

export interface Bar { x: number; y: number; w: number; h: number; score: number }

/** Widest a bar is drawn (px), so one early checkup is not a slab across the box. */
const MAX_BAR_W = 18

/**
 * Canvas rectangles in a w×h box with `pad` pixels of margin: one bar per level, oldest on the left, each in an equal
 * slot, standing on the bottom edge; its height is score / SUITE_SIZE of the box (a zero score is a 1 px baseline, so a
 * checkup that passed nothing still shows).
 */
export function milestoneBars(results: readonly CheckupResult[], w: number, h: number, pad = 2): Bar[] {
  const rs = [...results].sort((a, b) => a.level - b.level)
  if (rs.length === 0) return []
  const slot = (w - 2 * pad) / rs.length
  const bw = Math.max(0.5, Math.min(MAX_BAR_W, slot * 0.7))
  const inner = h - 2 * pad
  return rs.map((r, i) => {
    const score = r.passed.length
    const bh = Math.max(1, (Math.min(score, SUITE_SIZE) / SUITE_SIZE) * inner)
    return { x: pad + i * slot + (slot - bw) / 2, y: h - pad - bh, w: bw, h: bh, score }
  })
}

/** The latest checkup in plain words. */
export function milestoneText(results: readonly CheckupResult[]): string {
  const last = results.reduce<CheckupResult | undefined>((a, r) => (!a || r.level > a.level ? r : a), undefined)
  if (!last) return `no checkup yet (the first comes soon after it has lived ${FIRST_LOG_CU} CU)`
  const names = last.passed.map(id => MILESTONE_NAMES[id] ?? id)
  return `Latest checkup at ${fmtLevel(last.level)} CU: passed ${last.passed.length} of ${SUITE_SIZE}${names.length ? ` (${names.join(", ")})` : ""}.`
}

/** The Status line while a checkup runs. */
export const checkupStatus = (level: number) => `having a checkup at ${fmtLevel(level)} CU (its learning is paused)`

/** Room ids (the world's RoomKind) as said on the page. */
const ROOM_WORDS: Record<string, string> = { crib: "crib", toys: "toys", hideseek: "hide-and-seek", places: "places" }

/** What the caregiver has done: `carries` in this life so far, `to` the room of the last one (null if not known). */
export interface CaregiverLine { carries: number; to: string | null; safetyNet: boolean }

export function caregiverText({ carries, to, safetyNet }: CaregiverLine): string {
  const times = carries === 0 ? "has not carried it yet" : `has carried it ${carries === 1 ? "once" : `${carries} times`}`
  const last = carries > 0 && to !== null ? `; last to the ${ROOM_WORDS[to] ?? to} room` : ""
  const net = safetyNet ? "; it now uses the taskbar itself, so the caregiver only steps in when it is stuck" : ""
  return times + last + net
}

/** A log file's share of the caregiver line: its carries (since the file before it) and its safety-net mode. */
export interface CarryFile { seq: number; carries: number; safetyNet: boolean }

/**
 * The caregiver's carries in a life, as the page counts them. Every carry belongs to the log file that will hold it: the
 * one after the last logged (`nextSeq`). Once a file is known its own count replaces the carries seen for its seq (it
 * also holds any a reload's checkpoint carried over, and none a restore undid), so a count is never added twice, never
 * goes below what the files hold, and a file seen again (or out of order) changes nothing. Files once seen are kept,
 * even if the queue later deletes them.
 */
export interface CarryTally {
  /** A carry, to room `to`, during the file that comes after the last logged. */
  carried(to: string): void
  /** File `seq` was logged: later carries belong to the next one. */
  logged(seq: number): void
  /** A log file's own count (from the queue). */
  file(f: CarryFile): void
  line(): CaregiverLine
  /** The files known so far (to seed the next run's tally: the queue may have deleted some since). */
  files(): CarryFile[]
}

/** `files`: those already written; `nextSeq`: the seq of the file the coming carries go into. */
export function createCarryTally(files: readonly CarryFile[], nextSeq: number): CarryTally {
  const known = new Map<number, CarryFile>()
  const seen = new Map<number, number>()
  let next = nextSeq
  let to: string | null = null
  const tally: CarryTally = {
    carried(room) {
      seen.set(next, (seen.get(next) ?? 0) + 1)
      to = room
    },
    logged(seq) {
      next = Math.max(next, seq + 1)
    },
    file(f) {
      known.set(f.seq, f)
    },
    line() {
      let carries = 0
      for (const f of known.values()) carries += f.carries
      for (const [seq, n] of seen) if (!known.has(seq)) carries += n
      const latest = [...known.values()].reduce<CarryFile | undefined>((a, f) => (!a || f.seq > a.seq ? f : a), undefined)
      return { carries, to, safetyNet: latest?.safetyNet ?? false }
    },
    files: () => [...known.values()],
  }
  for (const f of files) tally.file(f)
  return tally
}
