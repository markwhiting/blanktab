// Runs inside a dedicated Worker: OffscreenCanvas(320,200) + createWorld, timed per room.
import { createWorld, ROOM_KINDS, type CanvasLike } from "../src/index"
import { BUTTON_OPTIONS, KEYS, MOVE_OPTIONS, SCROLL_OPTIONS, WORLD, type Action } from "@blanktab/interface"

const ITERATIONS = 600
const SCALE = 3

// Bench-local seeded rng (harness code, outside src).
function rng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

async function png(canvas: OffscreenCanvas): Promise<string> {
  const big = new OffscreenCanvas(WORLD.width * SCALE, WORLD.height * SCALE)
  const c = big.getContext("2d")!
  c.imageSmoothingEnabled = false
  c.drawImage(canvas, 0, 0, big.width, big.height)
  const buf = new Uint8Array(await (await big.convertToBlob({ type: "image/png" })).arrayBuffer())
  let s = ""
  for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000))
  return btoa(s)
}

self.onmessage = async () => {
  const canvas = new OffscreenCanvas(WORLD.width, WORLD.height)
  const world = createWorld({ seed: 1, canvas: canvas as unknown as CanvasLike })
  const rnd = rng(7)
  const pick = (n: number) => Math.floor(rnd() * n)
  const randomAction = (): Action => ({
    move: pick(MOVE_OPTIONS), button: pick(BUTTON_OPTIONS), scroll: pick(SCROLL_OPTIONS),
    key: rnd() < 0.1 ? pick(KEYS.length) : 0,
  })
  const rows = []
  for (const room of ROOM_KINDS) {
    world.bringToFront(room)
    world.observe()
    const fresh = await png(canvas)
    for (let i = 0; i < 30; i++) { world.act(randomAction()); world.observe() } // warm-up
    const t0 = performance.now()
    for (let i = 0; i < ITERATIONS; i++) { world.act(randomAction()); world.observe() }
    const ms = performance.now() - t0
    const msPerStep = ms / ITERATIONS
    rows.push({ room, msPerStep, stepsPerSec: 1000 / msPerStep, fresh, after: await png(canvas) })
  }
  self.postMessage(rows)
}
