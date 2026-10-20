# 0012 — The public name is "learner"

- **Date:** 2026-10-20
- **Status:** accepted
- **Decided by:** user (URL `markwhiting.github.io/learner`); agent (what else changes)

## Decision
- The repo becomes `markwhiting/learner`, served at `https://markwhiting.github.io/learner/`. The page's wordmark is **learner**, styled like `markwhiting.github.io/life`.
- **Internal names stay** (agent's call): npm scope `@blanktab/*`, storage keys `blanktab:*` and the IndexedDB database `blanktab`. Changing the storage keys would orphan tokens and queued logs already in browsers. "blanktab" remains the project's working name in docs.
- GitHub redirects the old repo URL. The old Pages URL `markwhiting.github.io/blanktab/` stops working.
