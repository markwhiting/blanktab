/// <reference types="node" />
import { createHash } from "node:crypto"
import { readFileSync, writeFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"
import { createCpuBackend } from "@blanktab/tensor"
import { DEFAULT_SETTINGS, createLearner } from "../src/learner"
import type { UpdateInfo } from "../src/learner"
import { createSyntheticStream, heldOutSet } from "./synthetic"

// Pins the learner's CPU trajectory (Plan 04c Task 2): the fixture was recorded on the learner code from before the
// step-speed change (5658165), and any later change that is meant to leave learning untouched must keep this hash.
// GOLDEN_WRITE=1 rewrites the fixture: only ever on code whose learning is known to be the reference.

const FIXTURE = fileURLToPath(new URL("./fixtures/golden-cpu.json", import.meta.url))
const STEPS = 400
const SLEEP_EVERY = 100

/** Feeds numbers to the hash as float64 bytes (so NaN, −0 and every last bit count). */
function numbers(h: ReturnType<typeof createHash>, xs: number[]) {
  h.update(new Uint8Array(Float64Array.from(xs).buffer))
}

/** Feeds a value: typed arrays as their bytes (in traversal order), everything else as JSON with non-finite numbers as strings. */
function value(h: ReturnType<typeof createHash>, v: unknown) {
  const json = JSON.stringify(v, (_k, x: unknown) => {
    if (ArrayBuffer.isView(x)) {
      h.update(new Uint8Array(x.buffer, x.byteOffset, x.byteLength))
      return { bytes: x.constructor.name, length: x.byteLength }
    }
    return typeof x === "number" && !Number.isFinite(x) ? String(x) : x
  })
  h.update(json)
}

describe("golden CPU trajectory (Plan 04c Task 2)", { timeout: 120_000 }, () => {
  it("CPU trajectory is bit-identical to the recorded golden", async () => {
    const h = createHash("sha256")
    const updates: UpdateInfo[] = []
    const learner = await createLearner({
      backend: createCpuBackend(), seed: 1, widths: { conv1: 4, conv2: 4, enc: 16, hidden: 16 }, settings: DEFAULT_SETTINGS,
      heldOut: heldOutSet(), onUpdate: (u) => {
        updates.push(u)
        numbers(h, [u.step, u.lossBefore, u.lossAfter, u.reward, u.advantage, u.skipped ? 1 : 0])
      },
    })
    const world = createSyntheticStream()
    for (let i = 1; i <= STEPS; i++) {
      const a = await learner.step(world.observe())
      numbers(h, [a.move, a.button, a.scroll, a.key])
      world.act(a)
      if (i % SLEEP_EVERY === 0) await learner.sleep()
    }
    // The saved settings are hashed without gamma (Plan 05a Task 2): the fixture predates that key, and gamma must
    // still be the constant the fixture's learning used, the literal 0.9 (not GAMMA, which a change could move along
    // with it). Everything else in the state is hashed as before.
    const state = await learner.save()
    const { gamma, ...settings } = state.settings
    expect(gamma).toBe(0.9)
    value(h, { ...state, settings })
    learner.dispose()
    const record = {
      sha256: h.digest("hex"), steps: STEPS, firstLossBefore: updates[0]!.lossBefore, lastLossBefore: updates.at(-1)!.lossBefore,
    }
    if (process.env.GOLDEN_WRITE === "1") writeFileSync(FIXTURE, `${JSON.stringify(record, null, 2)}\n`)
    const golden = JSON.parse(readFileSync(FIXTURE, "utf8")) as typeof record
    expect(updates).toHaveLength(STEPS - 1)
    expect(record.steps).toBe(golden.steps)
    expect(record.sha256).toBe(golden.sha256)
  })
})
