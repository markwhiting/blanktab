// How often the page talks to GitHub. Kept apart from main.ts so it is testable.

/** Pending logs that are not final wait at most this long before they are sent. */
export const FLUSH_EVERY_MS = 10 * 60_000

/**
 * When to flush the outbox: when a final log is queued, when a token is saved, and every FLUSH_EVERY_MS.
 * Not after every queued log: each flush is a commit, and each commit runs the validate workflow.
 * (A flush with nothing pending makes no GitHub requests.)
 */
export function flushPolicy(
  flush: () => void,
  every: (fn: () => void, ms: number) => unknown = (fn, ms) => setInterval(fn, ms),
  report: () => void = () => {},
) {
  every(flush, FLUSH_EVERY_MS)
  return {
    /** A final log is sent at once; any other log just updates the "waiting" count shown on the page. */
    queued(final: boolean): void { if (final) flush(); else report() },
    tokenSaved(): void { flush() },
  }
}
