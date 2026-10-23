export function stubStorage(init: Record<string, string> = {}): Storage {
  const m = new Map(Object.entries(init))
  return {
    get length() { return m.size },
    clear: () => m.clear(),
    getItem: k => m.get(k) ?? null,
    key: i => [...m.keys()][i] ?? null,
    removeItem: k => void m.delete(k),
    setItem: (k, v) => void m.set(k, String(v)),
  }
}

import type { LogFileV1 } from "../src/log"

export const mkFile = (learnerId: string, seq: number, over: Partial<LogFileV1> = {}): LogFileV1 => ({
  schemaVersion: 1, learnerId, seq, codeSha: "dev", knowledgeSha: "0".repeat(64),
  writtenAt: new Date(0).toISOString(), stepsTotal: (seq + 1) * 500, settings: {},
  hostSig: { cores: 4, deviceMemoryGb: null, webgpu: false, tabsOnMachine: 1 }, sessionsSeenAtBirth: 0, final: false, ...over,
})

import genomeSpaceJson from "../../../knowledge/genome-space.json" with { type: "json" }
import genomeJson from "../../../knowledge/genome.json" with { type: "json" }
import { birthDraw, genomeSha, legacyGenome, parseCentre, parseGenomeSpace, type Genome } from "@blanktab/meta"
import type { MetaBlock } from "../src/log"

/** The repo's knowledge/genome-space.json, as the gate reads it from main: every validateSessionPr call here passes it. */
export const GENOME_SPACE = parseGenomeSpace(genomeSpaceJson)
/** The repo's knowledge/genome.json values (the centre). */
export const CENTRE = parseCentre(genomeJson, GENOME_SPACE)

/** A meta block of the genome space's schema, with its genomeSha. */
export async function metaBlock(o: Pick<MetaBlock, "role" | "method" | "drawSeed" | "posteriorSha" | "evalPanel"> & { genome: Genome; missing?: string[] }): Promise<MetaBlock> {
  const s = GENOME_SPACE.schemaVersion
  return {
    v: 1, role: o.role, method: o.method, genomeSchema: s, genome: { ...o.genome }, genomeSha: await genomeSha(s, o.genome),
    drawSeed: o.drawSeed, posteriorSha: o.posteriorSha, evalPanel: o.evalPanel, evalSeed: null, confirmation: null, inheritance: { filters: "none" },
    ...(o.missing ? { missing: [...o.missing] } : {}),
  }
}

/** A newborn's block as the page draws it offline (no posterior): method centre, its learnerId's panel. */
export async function birthBlock(learnerId: string): Promise<MetaBlock> {
  const centreSha = await genomeSha(GENOME_SPACE.schemaVersion, CENTRE)
  const d = birthDraw({ learnerId, space: GENOME_SPACE, centre: CENTRE, centreSha, widthsClass: "cpu", posterior: null, posteriorSha: null })
  return metaBlock({ role: d.role, method: d.method, drawSeed: d.drawSeed, posteriorSha: d.posteriorSha, evalPanel: d.evalPanel, genome: d.genome })
}

/** A pre-05a life's block from its flat settings (legacyGenome, its `missing` carried), as the page builds it. */
export async function legacyBlock(flat: Record<string, number | string>): Promise<MetaBlock> {
  const { genome, missing } = legacyGenome(flat, GENOME_SPACE)
  return metaBlock({ role: "legacy", method: "legacy", drawSeed: null, posteriorSha: null, evalPanel: null, genome, missing })
}
