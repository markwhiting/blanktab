# 0015 — The predictor gets a learned fovea-shift prior; progress is measured on held-out data

- **Date:** 2026-10-21
- **Status:** accepted (agent's call under the user's "keep going" instruction; flagged to the user as a design choice)
- **Decided by:** agent, from `docs/research/2026-10-21-why-not-learning.md`
- **Amends:** decision 0013 §1 (the residual predictor)

## Context
In the real world the learner never beat copy-last. Most of the copy-last error is the fovea *translating* as the cursor moves. The residual predictor `base + tanh(W·z)` cannot express "this image, shifted". With nothing learnable, its hidden units died and its output froze at copy-last.

Separately, the learner's own-stream loss ratio misleads, because a curiosity-driven policy picks its own data. In one run the own-stream ratio was 0.41 while the held-out ratio was 1.40.

## Decision
1. **Fovea-shift prior.** The fovea prediction becomes `base + g(a)·(shift(base; kx(a)⊗ky(a)) − base) + tanh(W·z)`:
   - `kx` and `ky` are learned per-action 17-tap softmax kernels, one per axis, with a centre prior.
   - `g` is a per-action gate, zero-initialised, so a newborn still predicts exactly copy-last.
   - This is an architectural prior ("the next view may be a translated copy of this one", as in CDNA, Finn et al. 2016). Which action shifts what, and by how much, is learned, so AGENTS.md invariant 1 holds.
   - It adds about 2.7k parameters and about 3% compute. `LEARNER_STATE_VERSION` becomes 2, so existing learners restart once.
2. **No dead predictor units.** The predictor's hidden activation must not die permanently. Use leaky ReLU (slope 0.01), unless measurement shows a better choice.
3. **Held-out check.** Learning progress shown on the page and written to logs is measured on a **fixed held-out set**:
   - 64 transitions recorded at birth from a separate, seeded instance of the world under uniformly random actions.
   - Scored at each sleep: forward pass only, no learning, billed separately from training compute.
   - The own-stream ratio remains as a secondary number.
   - This is the basis for milestone M1 in Plan 04b.

## Alternatives considered
- **Let the learner discover translation with no built-in prior.** Purer, but this architecture shows no sign of discovering it within a feasible budget. It remains a future research track.
- **Make the fovea world-fixed.** That would remove the shift, but would also remove the cursor-centred attention the spec chose (§5.2).

## Amendment (2026-10-21, Plan 04a.2 Task 1; controller rulings, agent-measured)

Measurement: `tools/probe/learn-probe.ts`, CPU_WIDTHS, 5,000 steps. Full tables: the Task 1 report (`.superpowers/sdd/2026-10-21-04a2-learns-in-the-real-world/task-1-report.md`).

- **The "alive" bar is replaced.** With leaky units, a negative pre-activation still passes gradient, so "fraction of z > 0" measured the wrong thing (1.6–3.8% at lr 0.002, even though the gradient was not zero).
  - The new bar is on the mean |∂L/∂W| of the residual output layers' **weight** tensors at 5k steps: it must be > 1e-8 in every run.
  - The predictor's input layer (predH) is reported beside it.
- **ε-uniform exploration** (`exploreEpsilon`, default 0.1, clamped [0, 0.5]).
  - With probability ε, each action factor is drawn uniformly from the learner's RNG.
  - The actor-critic uses the action actually taken and its log-probability under π, with no importance weighting.
  - The resulting off-policy bias is **accepted for v1**.
  - ε = 0 is bit-exact to the learner without exploration.
- **residualDecay** (default 1e-3, clamped [0, 1e-2]). This is decoupled weight decay toward zero on the parameters that express deviation from copy-last: the shift gate g and the four zero-initialised output layers.
  - Its semantics are **per update**, not scaled by lr: p ← (1 − λ)·p − Adam step.
  - This sets an equilibrium ceiling of |p| ≲ lr_eff / λ: lr/λ = 0.5 for the gate, and lr·headLrScale/λ = 0.05 for the output layers, at the defaults.
  - Rationale: a residual must keep being re-earned, which counters the late drift toward the policy's own data.
  - λ = 0 is bit-exact to the learner without decay.
- **Defaults:** lr 5e-4, exploreEpsilon 0.1, residualDecay 1e-3 (headLrScale 0.1 as before).
  - `knowledge/settings-space.json` is narrowed to the measured neighbourhood: lr [4e-4, 6e-4], headLrScale [0.08, 0.12], exploreEpsilon [0.08, 0.12], residualDecay [7e-4, 1.5e-3].
- **Held-out set, corrected.** The probe's held-out world was the training world's layout, and is now seeded `hashSeed(seed, "heldout")`, as the live check will be.
  - Re-measured at the defaults, 6 seeds: the held-out ratio at 5k is 0.924, 0.919, 0.928, 0.915, **1.222**, 0.921. That is ≤ 0.95 on **5/6**. The gradient bars are met in all runs (W 8e-6 to 3.3e-5; predH 3e-6 to 3.4e-5).
  - Caveats:
    - this is a 5k snapshot;
    - seed 1 passes at 5k (0.924) but was at 1.30 at 3k;
    - seed 5 drifts to 1.22;
    - the periphery is above copy-last on held-out in every run (1.01–1.07); all of the gain is in the fovea;
    - the live configuration (GPU_WIDTHS, about 28k steps) is untested.
  - The remaining drift is what consolidation conditions B and C (Plan 05) are meant to address.
