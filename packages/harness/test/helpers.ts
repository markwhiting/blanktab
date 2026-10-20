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
