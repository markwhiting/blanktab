import { describe, it, expect } from "vitest"
import { KEYS, MOVE_OPTIONS } from "@blanktab/interface"
import { mulberry32 } from "../src/rng"
import { act, makeWorld } from "./helpers"

const STEPS = 2000
const CARRY_EVERY = 97
/** Picked (from 1-12) so this random run reaches every trace kind, links and widgets included; asserted below. */
const ACTION_SEED = 11

const enc = new TextEncoder()
const hex = (b: ArrayBuffer) => Array.from(new Uint8Array(b), v => v.toString(16).padStart(2, "0")).join("")

/** A running SHA-256 chain (Web Crypto: the world package has no Node types): h' = sha256(h || parts). */
async function chain(prev: Uint8Array, parts: Uint8Array[]): Promise<Uint8Array> {
  const buf = new Uint8Array(prev.length + parts.reduce((n, p) => n + p.length, 0))
  let o = 0
  for (const p of [prev, ...parts]) { buf.set(p, o); o += p.length }
  return new Uint8Array(await crypto.subtle.digest("SHA-256", buf))
}

/**
 * The default four-room world under seeded random actions (moves, presses, releases, scrolls, keys) with a
 * caregiver carry every 97 steps. Hashes every observation, every drained trace and every state().
 */
async function run(worldSeed: number, steps = STEPS, actionSeed = ACTION_SEED) {
  const w = makeWorld(worldSeed)
  const rnd = mulberry32(actionSeed)
  const pick = (n: number) => Math.floor(rnd() * n)
  let hash: Uint8Array = new Uint8Array(32)
  const kinds = new Set<string>()
  const rooms = new Set<string>()
  for (let i = 1; i <= steps; i++) {
    const b = rnd()
    w.act(act(
      pick(MOVE_OPTIONS),
      b < 0.2 ? 1 : b < 0.4 ? 2 : 0,
      rnd() < 0.15 ? 1 + pick(2) : 0,
      rnd() < 0.05 ? 1 + pick(KEYS.length - 1) : 0,
    ))
    if (i % CARRY_EVERY === 0) w.bringToFront(w.rooms()[pick(w.rooms().length)]!)
    const o = w.observe()
    const trace = w.drainTrace()
    for (const e of trace) kinds.add(e.kind)
    const st = w.state()
    rooms.add(st.room)
    hash = await chain(hash, [o.periphery, o.fovea, enc.encode(JSON.stringify([o.cursor, o.buttonHeld, trace, st]))])
  }
  return { digest: hex(hash.buffer as ArrayBuffer), kinds, rooms }
}

describe("determinism of the default four-room world", { timeout: 120_000 }, () => {
  it(`${STEPS} seeded random steps (scroll, buttons, keys, carries): same seed same hash, other seed differs`, async () => {
    const a = await run(1), b = await run(1)
    // the run really exercises the world: every room and every trace kind
    expect([...a.rooms].sort()).toEqual(["crib", "hideseek", "places", "toys"])
    for (const k of ["hover", "click", "respond", "scroll", "focus", "navigate", "reveal", "widget"]) expect(a.kinds, k).toContain(k)
    expect(b.digest).toBe(a.digest)
    // another world seed differs (compared over equal-length runs; 300 steps keeps it quick)
    expect((await run(2, 300)).digest).not.toBe((await run(1, 300)).digest)
  })
})
