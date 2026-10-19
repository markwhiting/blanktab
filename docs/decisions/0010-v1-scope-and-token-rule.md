# 0010 — v1 is a bounded build; the token never leaves localStorage

- **Date:** 2026-10-19
- **Status:** accepted
- **Decided by:** user

## Decision
1. v1 is done when the whole system runs end to end: tabs learn unattended, checkups run, logs reach GitHub, and the picker uses them. Honest zero-milestone results are acceptable. Getting learners to pass milestones is the next phase.
2. A broader classic `public_repo` token is acceptable if the least-access option doesn't work, **as long as the token stays in `localStorage` only** and is never copied to any other storage, log, payload, or destination.
3. The v1 design (`specs/2026-10-19-blanktab-v1-design.md`) is approved.
