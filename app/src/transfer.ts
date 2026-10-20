// Moving a big value (a ~29 MB checkpoint record) to the worker without copying it.

/** Every distinct ArrayBuffer inside `value` (typed arrays, nested in objects and arrays): its transfer list. */
export function buffersOf(value: unknown): ArrayBuffer[] {
  const out = new Set<ArrayBuffer>()
  const seen = new Set<object>()
  const walk = (v: unknown) => {
    if (typeof v !== "object" || v === null || seen.has(v)) return
    seen.add(v)
    if (ArrayBuffer.isView(v)) {
      if (v.buffer instanceof ArrayBuffer) out.add(v.buffer)
      return
    }
    if (v instanceof ArrayBuffer) { out.add(v); return }
    for (const x of Object.values(v)) walk(x)
  }
  walk(value)
  return [...out]
}
