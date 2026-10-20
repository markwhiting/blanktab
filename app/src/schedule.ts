// How fast the learner lives and how often the page talks to GitHub. Kept apart from main.ts so it is testable.

export const LIFETIME_STEPS = 3000
export const LOG_EVERY_STEPS = 500
/** Real time: a 3000-step life takes 25 minutes, so one tab sends a handful of commits per life, not a stream. */
export const STEPS_PER_SECOND = 2
/** Pending logs that are not final wait at most this long before they are sent. */
export const FLUSH_EVERY_MS = 10 * 60_000

/**
 * When to flush the outbox: when a final log is queued, when a token is saved, and every FLUSH_EVERY_MS.
 * Not after every queued log: each flush is a commit, and each commit runs the validate workflow.
 * (A flush with nothing pending makes no GitHub requests.)
 */
export function flushPolicy(flush: () => void, every: (fn: () => void, ms: number) => unknown = (fn, ms) => setInterval(fn, ms)) {
  every(flush, FLUSH_EVERY_MS)
  return {
    queued(final: boolean): void { if (final) flush() },
    tokenSaved(): void { flush() },
  }
}
