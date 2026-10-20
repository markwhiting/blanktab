import { describe, it, expect } from "vitest"
import { buffersOf } from "../src/transfer"

describe("buffersOf", () => {
  it("lists each ArrayBuffer once, however deep or shared", () => {
    const shared = new Float32Array(4)
    const view = new Uint8Array(shared.buffer, 4, 4)
    const other = new Uint8Array(3)
    const v = { a: [shared, { b: view }], c: { d: other, e: 1, f: "x", g: null }, h: shared }
    expect(buffersOf(v).sort((x, y) => x.byteLength - y.byteLength)).toEqual([other.buffer, shared.buffer])
    expect(buffersOf(null)).toEqual([])
  })
  it("the listed buffers can all be transferred together (a structured clone with them moves, not copies)", () => {
    const rec = { learnerState: { params: [new Float32Array(1000)], frames: { data: new Uint8Array(500) } } }
    const moved = structuredClone(rec, { transfer: buffersOf(rec) })
    expect(rec.learnerState.params[0]!.byteLength).toBe(0) // detached: moved
    expect(moved.learnerState.params[0]!.length).toBe(1000)
  })
})
