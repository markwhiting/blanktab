// Synthetic lives for the evaluator tests: LogFileV2 objects that parseLogFile accepts (the tests check it).
import type { CheckupResult, CurvePoint, LogFileV2, MetaBlock } from "@blanktab/harness/log"

export const GPU = { conv1: 16, conv2: 32, enc: 128, hidden: 256 }
export const CPU = { conv1: 8, conv2: 16, enc: 64, hidden: 128 }
export const IDS = ["M1", "M2", "M3", "M4", "M5", "M6"] as const
export type Mid = typeof IDS[number]

/** A valid learnerId for an integer k. */
export const lid = (k: number): string => "lrn_" + k.toString(16).padStart(16, "0")

export function meta(over: Partial<MetaBlock> = {}): MetaBlock {
  return {
    v: 1, role: "search", method: "calibrate", genomeSchema: 1, genome: { lr: 0.001, gamma: 0.9 }, genomeSha: "ab".repeat(32),
    drawSeed: 7, posteriorSha: "cd".repeat(32), evalPanel: 3, evalSeed: null, confirmation: null, inheritance: { filters: "none" },
    ...over,
  }
}

/** A checkup at `level`: absent measures are null; `passed` defaults to none. */
export function checkup(level: number, measures: Partial<Record<Mid, number | null>>, passed: Mid[] = []): CheckupResult {
  return {
    level, steps: level * 10, passed: [...passed],
    measures: Object.fromEntries(IDS.map(id => [id, measures[id] ?? null])), evalCU: 1,
  }
}

/** A curve point at `cu`; with `ratio`, a held-out score of loss `ratio` against baseline 1. */
export const point = (cu: number, ratio?: number): CurvePoint => ({
  cu, steps: Math.round(cu * 10), predLoss: 1, baselineLoss: 1, ...(ratio !== undefined ? { heldOut: { loss: ratio, baseline: 1 } } : {}),
})

export interface FileSpec {
  learnerId: string; seq: number; cu: number; final?: boolean; curve?: CurvePoint[]; checkups?: CheckupResult[]
  meta?: MetaBlock | null; widths?: LogFileV2["widths"]; suite?: number
}

export function file(s: FileSpec): LogFileV2 {
  const checkups = s.checkups ?? []
  const m = s.meta === undefined ? meta() : s.meta
  return {
    schemaVersion: 2, learnerId: s.learnerId, seq: s.seq, codeSha: "dev", knowledgeSha: "ef".repeat(32),
    writtenAt: "2026-10-23T00:00:00.000Z", stepsTotal: Math.round(s.cu * 10), settings: { lr: 0.001 },
    hostSig: { cores: 8, deviceMemoryGb: 8, webgpu: true, tabsOnMachine: 1 }, sessionsSeenAtBirth: 0, final: s.final ?? false,
    cu: s.cu, condition: "A", widths: s.widths ?? GPU, curve: s.curve ?? [], skippedUpdates: 0, restores: 0, lostCU: 0,
    evalCU: 10 * (s.seq + 1),
    ...(checkups.length ? { checkups, milestoneSuiteVersion: s.suite ?? 1 } : {}),
    ...(m ? { meta: m } : {}),
  }
}

/**
 * A typical eligible search life: logs at 512, 1024, 2048 and 3072 CU, checkups at 512 and 1024 in their own files and
 * the 2048 checkup in the 3072 file (it runs at the first sleep after crossing 2,048), held-out points in (1024, 2048].
 */
export function typicalLife(learnerId: string, over: { meta?: MetaBlock; widths?: LogFileV2["widths"] } = {}): LogFileV2[] {
  const ms = { M1: 0.9, M2: 1, M3: 0.2, M4: 0.3, M5: 0.8, M6: 0 }
  const f = (s: Omit<FileSpec, "learnerId">) => file({ learnerId, ...(over.meta ? { meta: over.meta } : {}), ...(over.widths ? { widths: over.widths } : {}), ...s })
  return [
    f({ seq: 0, cu: 512, curve: [point(256, 1), point(512, 0.99)], checkups: [checkup(512, ms)] }),
    f({ seq: 1, cu: 1024, curve: [point(1024, 0.97)], checkups: [checkup(1024, ms)] }),
    f({ seq: 2, cu: 2048, curve: [point(1200, 0.95), point(1600, 0.94), point(2048, 0.93)] }),
    f({ seq: 3, cu: 3072, curve: [point(3000, 0.9)], checkups: [checkup(2048, ms)] }),
  ]
}
