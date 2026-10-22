# Hardening, live: before and after Plan 04c

**Date:** 2026-10-22. **Builds:** before e12b89f (Plan 04b), after 1c104b1 (Plan 04c). **Where:** https://markwhiting.github.io/learner/, headless Chromium on an Apple M4 (WebGPU, metal-3), no token, 10 minutes each, `caffeinate` on, load average about 4. Each run is a fresh learner with its own draw of settings.

## Summary
- **About 20% more learning in the same wall time.** After the change, the learner lived 3,615 steps (301.5 CU) in 10 minutes; before, 3,000 steps (252.7 CU).
  - Part of this is the one GPU round trip removed per step. Uncapped, that is 26.2–27.7 → 28.7–30.9 steps/s in the harness ([decision 0017](../decisions/0017-hardening-calls.md)).
  - The rest is noise. The two runs drew different settings, and their checkups fell at different times.
  - Live speed is capped at 30 steps/s, so the harness numbers are the real measure.
- **Learning is unchanged.** Held-out ratio ended at 0.920 in both runs. On CPU, learning is proven bit-identical to before by the golden-trajectory test.
- **Checkups still pass nothing:** 0 of 6 at the latest level in both runs, as in run 3.
- **The logs are sound.** Every queued file from both runs (5 before, 6 after) passes the gate's `validateSessionPr` (`ok: true`).
- **The gate runs on GitHub's runner.** A manual `gate` run on the new code succeeded on git 2.55.0. There were no open PRs, so the new git-based PR reading will first meet a real PR on the next contribution. If it fails there, the PR is left open, never merged unvalidated.

## Samples
| min | before steps | before CU | before held-out | after steps | after CU | after held-out |
|---|---|---|---|---|---|---|
| 2 | 500 | 38.5 | 0.972 | 499 | 38.0 | 0.965 |
| 4 | 750 | 59.9 | 1.06 | 1,000 | 80.0 | 0.909 |
| 6 | 1,500 | 124.2 | 0.931 | 1,750 | 142.9 | 0.921 |
| 8 | 2,126 | 179.9 | 0.904 | 2,766 | 231.8 | 0.907 |
| 10 | 3,000 | 252.7 | 0.920 | 3,615 | 301.5 | 0.920 |

## Not checked live
- **Resuming in the same room after a reload.** This was verified on the dev page in Task 5 (screenshots in that task's report) and by session tests, not on the live site.
- **The gate's byte bound** (`ulimit -f` under bash 5 on Ubuntu). It is unit-tested on macOS bash only.
