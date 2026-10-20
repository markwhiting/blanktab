// Which steps' frames the worker posts to the page. Kept apart from worker.ts so it is testable.

/**
 * At most one frame every `minMs`, and none while the page is hidden (a background tab draws nothing, so copying and
 * posting ~15 kB a frame there is waste). The learner itself keeps living either way.
 */
export function frameGate(minMs: number, hidden = false) {
  let last = Number.NEGATIVE_INFINITY
  return {
    /** The page's document.hidden, as the page reports it. */
    setHidden(h: boolean): void { hidden = h },
    /** Whether to post a frame now (`t`: performance.now()); if so, counts it as posted. */
    due(t: number): boolean {
      if (hidden || t - last < minMs) return false
      last = t
      return true
    },
  }
}
