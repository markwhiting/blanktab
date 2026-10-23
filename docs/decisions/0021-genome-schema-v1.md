# 0021 — The learning genome, schema v1

- **Date:** 2026-10-23
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (agent's call, not yet confirmed). The direction, one meta-learner over a learning genome with today's learner at the centre, is the user's (0018).
- **Resolves:** the genome format left open by 0018 (spec `docs/specs/2026-10-22-meta-learner-v1-design.md` §3)

## Context

0018 makes the learner one point in a parameterized learning genome, searched by one meta-learner. The search, the gate and the page all need one shared definition: which genes there are, their ranges, how a point in the search's unit cube becomes the values a learner runs with, and how a genome is identified. Today's settings file (`knowledge/settings-space.json`) has none of this. It has no types, no owners and no centre, and its spans are too narrow to search (0022).

## Decision

**Agent's calls (not yet confirmed):**

- **Two knowledge files, both in `knowledgeSha`.** They join `PRIOR_FILES` in `tools/gate/manifest.ts`, so changing either starts a new cohort. Running tabs write one boundary log when they change (0014).
  - `knowledge/genome-space.json`, `schemaVersion` 1: the genes. Each gene has a `name` (the log's `SETTING_KEY_RE`), a `type` (`continuous`, `integer` or `categorical`), a `scale` (`linear`, `log` or `horizon`) with `min` and `max`, or `values` for a categorical. It also has a `centre`, an `owner` (`gp` searched, `fixed` held at the centre; `bandit` is reserved and rejected in v1), an optional `implemented: false`, and a `target` (`learner`, `caregiver` or `harness`, the only place its value goes). There are at most 40 genes.
  - `knowledge/genome.json`: the promoted genome, `{ schemaVersion: 1, genomeSchema: 1, values, provenance }`. At v1 every value is its gene's centre, and `provenance` is `{ source: "initial", decision: this file }`.
- **The v1 genes (spec §3.2).** There are 29 genes, and the 12 with owner `gp` are searched: lr, headLrScale, residualDecay, entropyWeight, curiosityWeight, exploreEpsilon, bufferSize, replayPerSleep, gamma, caregiverPeriod, caregiverTau and caregiverEpsilon. Four are fixed and live: caregiverThreshold, caregiverStuckSteps, maxStepsPerSec and sleepInterval. Thirteen are reserved slots, `implemented: false` at an "off" centre: latentWeight, ctrlWeight, noveltyWeight, targetTau, gaeLambda, replayBeta, plasticAlpha, plasticEta, lrClockSlope, creditMode, sleepSeqLen, modulator and ctrlGate. Every centre is today's learner, which `tools/genome-centre.test.ts` checks against `DEFAULT_SETTINGS`, `CAREGIVER_DEFAULTS`, `DEFAULT_MAX_STEPS_PER_SEC` and `DEFAULT_SLEEP_INTERVAL`. The same test checks that every live range lies inside the learner's and caregiver's clamps. The ranges are in 0022.
- **The unit-cube maps**, one coordinate per searched gene in file order:
  - linear: u = (v − min)/(max − min);
  - log: u = ln(v/min)/ln(max/min);
  - horizon: u = ln(h/h_min)/ln(h_max/h_min), on h = 1/(1 − γ), with h_min and h_max from min and max (γ 0.6–0.975 is h 2.5–40).
  - Decoding uses `@blanktab/detmath`'s `dexp`/`dlog` only, never the engine's `Math.exp`, `Math.log` or `Math.pow`, so every browser decodes a point to the same bytes. Integers are rounded, and a decoded value is clamped into [min, max]. The point a search records for an integer gene is re-encoded from the value that ran.
- **Validity.** `checkGenome` requires the genome's keys to be exactly the genes. Each value must be of its gene's type and inside [min, max]; integer genes must be integral and categorical genes among their values. A reserved (`implemented: false`) gene must be exactly at its centre. The genome of a life born before 05a (`legacyGenome`: its flat settings, absent genes at their centres) is not bounds-checked, because today's picker spans exceed the v1 ranges.
- **Decoding** (`decodeGenome`) emits the implemented genes only, grouped by target: the learner's settings (`LearnerSettings`, with `gamma`), the caregiver's settings, and the harness's `maxStepsPerSec` and `sleepInterval`. A reserved slot never reaches code. The learner never sees its genome (AGENTS.md invariant 2).
- **Integrity.** `genomeSha` = sha256 (Web Crypto) of `JSON.stringify({ schemaVersion, values })`, with `values` keys sorted, so key order never changes it.
- **A slot goes live only with a bit-exact-at-centre golden test, under a new schemaVersion.** The mechanism's code lands with a golden test proving that the learner at the slot's centre is bit-identical to the learner without it. The gene then becomes implemented under a new genome-space `schemaVersion`, which means a fresh fit.
- **Code.** The schema, maps and decoding live in `packages/meta` (`@blanktab/meta`), which imports only `@blanktab/detmath` and is pure (the lint purity bans of plan 05a). Its one Web Crypto call, in `sha256Hex`, is the only permitted use of the `crypto` global there, because sha256 is a pure function of its input. The exception lives in `eslint.config.js` (`webCryptoException`). It drops only `crypto`, only for `packages/meta/src/sha.ts`, and keeps every other ban; `tools/walls.test.ts` pins it.
- `knowledge/settings-space.json` stays, unchanged, for cached older pages.

## Alternatives considered

- **Extending `settings-space.json`.** Older cached pages read it and would break on new fields. It also has no notion of a centre or owner.
- **Engine `Math.log`/`Math.exp` in the maps.** Engines may round them differently. The gate must redo every draw byte for byte (spec §7.4).
- **Plain JSON key order for `genomeSha`.** Two writers with a different key order would give the same genome different hashes.
- **Admitting reserved slots at any value.** A slot whose mechanism is not built would be searched without effect, and the fit would learn noise.

## Consequences

- Every later 05a task (birth draw, log `meta` block, gate checks, page) reads genes through `parseGenomeSpace` and `parseCentre`.
- Changing either file changes `knowledgeSha`. A promotion (05b) rewrites `genome.json` and so starts a new cohort.
- Adding or activating a gene needs a new `schemaVersion`, and its rows are fitted separately.
- The ranges are guesses (0022). Widening them later is a schema change.
