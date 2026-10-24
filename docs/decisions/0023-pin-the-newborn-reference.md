# 0023 — Pin the first newborn reference

- **Date:** 2026-10-24
- **Status:** accepted
- **Decided by:** the user approved the run ("yes", 2026-10-24); the controller pins the reference under decision 0022's ruling C (agent's call, under the user's delegation)
- **Resolves:** the 05a step "pin the newborn reference" (spec §2.3, §10)

## Context

`search-composite@1` scores a life against what an untrained learner of the current recipe does on the same checkups. Until a reference is pinned, every search life is ineligible ("no newborn reference"), so calibration cannot fill up and the signal check cannot run.

## Decision

The reference is pinned at `config/references/newborn-baseline@1-gpu-suite1-genome1.json`. Facts about it:

- **What it was built from.** 64 newborns, k = 0–63, with learner seed 1000 + k and panel 1 + (k mod 16).
- **Where they ran.** On the live site (code `f5f260c`), in a single Chromium tab on WebGPU (apple metal-3), on an Apple M4. The run took about 55 minutes.
- **Their logs are committed.** They are in `data/newborns/`, 64 files. Each was checked with `parseLogFile` and `validateSessionPr` against `knowledge/genome-space.json` before it was written (ruling B). Anyone can rebuild the reference with `npx tsx tools/eval/newborn-reference.ts data/newborns --check`, which reproduces the committed file byte for byte.
- **What it says:**
  - **Held-out ratio median:** 1.00. That is exactly copy-last, as it should be at birth.
  - **Pass rates by milestone:**

    | Milestone | Median measure | Pass rate | Sustained pass rate |
    |---|---|---|---|
    | M1 | 1 | 0 | 0 |
    | M2 | 0.90 | 0.04 | 0 |
    | M3 | 0.86 | 0.22 | 0.11 |
    | M4 | 0.36 | 0.19 | 0.14 |
    | M5 | 0.89 | 0.10 | 0.03 |
    | M6 | 0 | 0.03 | 0 |

  - **Close to the earlier measurement.** These rates are close to the CPU-width rates measured for decision 0016, which were M3 13% and M4 18% of newborns passing.

## Alternatives considered

- **64 contribution PRs through the gate.** Rejected in ruling B: it needs a token in a headless browser and would flood the PR list. It would also add nothing, because the same validators already ran on each file.
- **Waiting for many machines.** One machine and browser class is a known limit, recorded below. The reference is per widths class and suite, and can be re-pinned under a new decision.

## Consequences

- **Search lives become scorable.** Once their logs are merged, search lives can be scored, and calibration starts counting rows.
- **Changing the reference changes scores.** Re-pinning, for example on more machines or after a genome or suite change, is a recorded decision. Old lives can be rescored with `tools/eval/rescore.ts`.
- **Limit:** one machine, one browser.
