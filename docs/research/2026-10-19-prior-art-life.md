# Prior art: markwhiting/life

`https://github.com/markwhiting/life`: a distributed artificial-life system by the same author, with a browser prototype. Read 2026-10-19.

## Directly reusable ideas
- **Cross-session experiment loop** (`prototype/src/experiment/`): every session appends `{params, hostSig, metric, schemaVersion, ...}` to a shared log. New sessions fit a Gaussian process over the log and pick their own params by UCB, picking randomly until enough data exists. This is the "what do I do given what I know" loop in its simplest form.
- **Device-relative parameters:** params are unitless fractions of host capacity (memory, compute), so phones and workstations pool into one model.
- **Schema versioning:** observations from older schemas are filtered out so changed semantics don't poison the fit. Essential once our architecture evolves.
- **Telemetry is adjacent:** organisms can't read telemetry, and it never affects their fate. This matches our learner/instrumentation invariant.
- **Stack:** Vite + plain JS, Web Worker for the simulation loop, Vitest, Playwright e2e, GitHub Pages deploy.

## Differences
- `life` syncs live through an Automerge CRDT relay (`sync.automerge.org`), not PRs. We want GitHub PRs as the validated inheritance channel. A live relay could later complement PRs for a fast, unvalidated log.
- `life` refuses fitness functions on principle. We do have objectives (learning efficiency, retention, transfer), but we keep the same discipline: the learner can't see or game the measurement.
- `life` offspring inherit code, not lifetime state. This parallels our "little or no weights in GitHub" stance: what's inherited is the *capacity to learn*.
