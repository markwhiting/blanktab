# Plan 04a.2 — The learner learns in the real world

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## In plain terms
The first live run showed the learner never beating the "nothing changes" guess (`docs/research/2026-10-21-does-it-learn.md`). The diagnosis (`docs/research/2026-10-21-why-not-learning.md`) found three problems:
- the learner can't express "my view shifted when I moved";
- its prediction units then shut down for good;
- the page's progress number can fool us.

This plan fixes all three (decision 0015) and runs live again.

**Spec:** `docs/specs/2026-10-19-blanktab-v1-design.md` §5. Also decisions 0013, 0015, the diagnosis note, and the probe `tools/probe/learn-probe.ts`.

## Global Constraints
- Every rule of Plans 04a and 04a.1 still applies: walls, tensor lifecycle, cold start, the token rule, the log and checkpoint invariants.
- `LEARNER_STATE_VERSION` = 2. Old checkpoints are rejected with the existing failure path (final log, then a newborn). The manifest publishes 2, so open tabs on old code keep running with the "newer version" message.
- **Determinism:** CPU save/load stays bit-exact, and the GPU/CPU learner parity test stays within its tolerance.
- **Held-out compute** is billed separately: logs gain `evalCU`, and it is not included in `cu`.

## Review Focus
1. A newborn's prediction equals copy-last exactly. The shift gate and the residual outputs start at zero.
2. The predictor's hidden units cannot all die: after 5,000 steps at lr 0.002, at least 10% are alive on all 3 seeds.
3. **The held-out set:**
   - it is fixed for a learner's whole life, and survives save/load and reload;
   - it never feeds training;
   - it is billed to `evalCU`, not `cu`.
4. **Logs v2 compatibility:** the new fields `heldOut` and `evalCU` are optional in the parser, and strict when present. The validator accepts old v2 files without them.

---

### Task 1: The fovea-shift predictor and live predictor units
- **Start point:** cherry-pick the experiment commit `06c2ec7` from branch `diag-learning`. It is reverted on `main`.
- **Review and harden it:**
  - tests that a newborn prediction is exactly copy-last;
  - a gradCheck of the shift path;
  - shift kernels that sum to 1;
  - an exact replay of the oracle case: a kernel set to a one-hot pulse shifts a known image by exactly k pixels.
- **Replace ReLU in the predictor's z** with leaky ReLU (0.01).
- **Measure with the probe:** `npx tsx tools/probe/learn-probe.ts --steps 5000 --seed {1,2,3} --settings {live,default}`. Report:
  - held-out ratio at 1k, 3k and 5k steps;
  - the fraction of z units alive.
- **Bar:**
  - held-out ratio ≤ 0.95 at 5k steps on at least 5 of 6 runs;
  - at least 10% of z units alive in all runs (Review Focus 2).
  If the bar isn't met, report and stop; don't loosen it.
- Bump `LEARNER_STATE_VERSION` to 2.
- Commit: `feat(learner): learned fovea shift; leaky predictor units`

### Task 2: The held-out check
- **At birth:**
  - create a second world instance seeded `hashSeed(learnerSeed, "heldout")`;
  - run 65 steps of uniformly random actions with its own RNG;
  - record the 64 transitions (observations and actions) into `LearnerState.heldOut`, which is saved and restored.
- **At each sleep,** score the current weights on the held-out set with a forward pass only, no learning, from the zero recurrent state:
  - `heldOutLoss` and `heldOutBaseline` (copy-last on the same transitions);
  - the ratio.
  - The FLOPs go to `evalCU` (separate counters, using deltas on the same backend).
- **Logs (v2, optional fields):**
  - each curve point gains `heldOut: { loss, baseline }`;
  - each file gains `evalCU`;
  - the validator accepts them as optional, finite and bounded, and keeps the strict-keys rule;
  - the summary row gains `heldOutRatio` (the latest).
- **Page:**
  - the "Is it learning?" chart and ratio use the held-out ratio;
  - the label says "on a fixed test it never trains on";
  - the own-stream ratio is shown as a secondary line of text.
- The `learner` package must not import `world` (wall). The worker builds the held-out set (it can import both) and passes the transitions to `createLearner` as data.
- Commit: `feat: progress measured on a fixed held-out set`

### Task 3: Live again
Same protocol as Plan 04a.1 Task 4:
- merge, push, CI and deploy green;
- check the load average is under 10;
- `caffeinate`, then 20 minutes on WebGPU;
- samples of the held-out ratio and the own-stream ratio.

Write `docs/research/2026-10-21-does-it-learn-2.md`, leading with the summary, and commit and push.
