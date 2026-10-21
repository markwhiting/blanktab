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
