# Why the learner does not beat copy-last in the real world

**Date:** 2026-10-21. **Branch:** `diag-learning` (from eabb123). **Follows:** [2026-10-21-does-it-learn.md](2026-10-21-does-it-learn.md).
**Probe:** `tools/probe/learn-probe.ts`. It runs the real learner on the CPU backend in the real default four-room world, using a Node canvas. Run it with `npx tsx tools/probe/learn-probe.ts --steps 5000 --seed 1 --settings live|default --log 500`.

## Summary

The ratio of 1.00 reproduces offline, on CPU_WIDTHS, with all 3 seeds at the live settings: loss / copy-last = 1.0002–1.0003. There are two causes, and the first one produces the second.

1. **The predictor cannot express what changes.** In the real world, 70–99% of the copy-last error is in the fovea (median about 95%). Most of that error comes from the fovea *translating*, because it is a window centred on the cursor. The untrained policy moves the cursor on 91–94% of steps, and a third of those moves are 16 px, a shift of 8 target pixels. The old predictor is `base + tanh(W·z)`, where z is a 64-unit vector from h and the action. That adds a pattern that does not depend on the image, so it cannot produce "the current fovea, shifted". An oracle that only knows "the fovea moves with the cursor" gets **0.60–0.78 of copy-last**. The learner gets 1.000.
2. **The predictor's hidden layer then dies, and stays dead.** There is nothing it can learn, so the gradient pushes every ReLU unit of z to ≤ 0 on every input. This happened in 3 of 3 seeds at the live lr of 0.002, by step 1,000–2,000. From then on the gradient into the output layers is **exactly 0** (it was 3.6e-3 at step 500). The prediction is frozen at copy-last plus a learned constant bias. That is why the two curves overlap at 3 significant digits, and why they would stay that way even if learnable signal appeared later.

These are **not** the causes:
- the head learning rate;
- replay fighting the online update;
- the EMA metric;
- a misalignment or weighting bug. The probe's shadow recomputation matches the learner's `lossBefore` exactly (max relative error 0), and the replay transition alignment is checked every step.

**Recommended fix** (commit "experiment: action-conditioned fovea translation in the predictor"). Add an action-conditioned translation of the current fovea to the fovea prediction:

`base + g(a)·(shift(base; kx(a) ⊗ ky(a)) − base) + tanh(W·z)`

- `kx` and `ky` are 17-tap softmax kernels, one per axis, from action-indexed logits with a centre prior. `g` is a per-action gate, zero-initialised.
- It is convex and separable, applied as two banded matmuls, and adds 2,660 parameters and about 3% CU.
- It is an architectural prior only: "the next image may be a translated copy of this one", as in CDNA (Finn et al. 2016). Which action shifts where is learned.
- At 5,000 steps (3 seeds × 2 settings), loss / copy-last is **0.65–0.85 on the learner's own stream** over the run, against 1.000 before. On a fixed held-out random-action trajectory it is **0.73–0.91 in 5 of 6 runs**, against 1.000 / 1.00–1.40 before.
- The WebGPU learner test still matches CPU: relative predLoss difference 1.3e-8, 300/300 identical moves.

## Measurements

All of these are on CPU_WIDTHS (8/16/64/128) in the default world (crib, toys, hideseek, places; world seed = learner seed). The learner sleeps every 250 steps, as in the harness.
- **Live settings:** lr 0.002, headLrScale 0.086, entropy 0.025, curiosity 1.32, buffer 1491, replay 48.
- **Default settings:** lr 1e-3, headLrScale 0.1, entropy 0.01, curiosity 1, buffer 2000, replay 64.
- **Ratio:** prediction loss before the update ÷ copy-last loss, at full precision.
- **Own stream:** the transitions the learner itself experiences.
- **Held-out:** the current weights scored without learning on a fixed 400-transition uniform-random-action trajectory in a second instance of the same world. The core runs along it from the zero state. This removes the confound of the policy choosing its own data.

### What the world's transitions look like (old code, live settings, 5,000 steps)

| seed | cursor moved | periphery target changed | fovea target changed | fovea share of copy-last error | shift oracle / copy-last |
|---|---|---|---|---|---|
| 1 | 0.911 | 0.911 | 0.695 | 0.77 | 0.717 |
| 2 | 0.940 | 0.920 | 0.694 | 0.98 | 0.669 |
| 3 | 0.934 | 0.912 | 0.740 | 0.82 | 0.706 |

- **Action mix:** the same for all seeds, close to uniform over each factor. No move on 3.5–4% of steps; a button press or release on 65%; a key on 98–100%. Move sizes were roughly a third each of 1, 4 and 16 px.
- **Copy-last error:** the fovea's MSE per element is about 0.02–0.07; the periphery's is about 1e-4.
- The **shift oracle** shifts the previous native 32×32 fovea by the observed cursor move, keeps the cursor sprite in place, and downsamples like the target. It is approximate: it ignores content changes and sub-target-pixel moves, and it is worse than copy-last on a few windows with room switches.

### Before vs after the fix (5,000 steps)

| code | settings | seed | own stream, whole run | own stream, last quarter | final EMA ratio (the UI's number) | held-out @1k | held-out @3k | held-out @5k | z units alive @5k |
|---|---|---|---|---|---|---|---|---|---|
| old | live | 1 | 1.000 | 1.000 | 1.0002 | 1.004 | 1.000 | 1.000 | 0.000 |
| old | live | 2 | 1.000 | 1.000 | 1.0002 | 1.000 | 1.000 | 1.000 | 0.000 |
| old | live | 3 | 0.991 | 1.000 | 1.0003 | 1.000 | 1.000 | 1.000 | 0.000 |
| old | default | 1 | 0.710 | 0.605 | 0.407 | 1.060 | 0.980 | **1.403** | 0.144 |
| old | default | 2 | 0.858 | 0.842 | 0.721 | 1.010 | 1.175 | **1.274** | 0.173 |
| old | default | 3 | 1.001 | 1.000 | 1.0003 | 1.001 | 1.000 | 1.000 | 0.000 |
| **fix** | live | 1 | **0.845** | 0.814 | 0.781 | 0.856 | 0.887 | **0.855** | 0.016 |
| **fix** | live | 2 | **0.800** | n/a¹ | 1.059¹ | 0.841 | 0.819 | **0.874** | 0.017 |
| **fix** | live | 3 | **0.797** | 0.742 | 0.606 | 0.925 | 0.822 | **0.912** | 0.028 |
| **fix** | default | 1 | **0.647** | 0.488 | 0.382 | 0.916 | 0.909 | 1.031² | 0.354 |
| **fix** | default | 2 | **0.789** | 0.810 | 0.860 | 0.822 | 0.759 | **0.730** | 0.105 |
| **fix** | default | 3 | **0.787** | 0.803 | 0.757 | 0.968 | 0.858 | **0.873** | 0.063 |

¹ From step about 3,500 the policy parks the cursor in a corner. Every move is clamped and nothing changes, so copy-last error is exactly 0 and the EMA ratio is meaningless: it peaked at 634. See caveats.
² Held-out reached 0.87–0.93 up to step 3,000, then drifted to 1.20–1.31 and back to 1.03. This matches the old-code default runs: the z head learns patterns specific to the policy's own data that hurt elsewhere.

- **Fovea vs periphery after the fix**, own stream, whole run: fovea 0.76–0.82. Periphery 1.00–1.07: it gains nothing, and is slightly worse than copy-last.
- **Cost:** CU at 5,000 steps was 129.1 vs 125.2 (live) and 134.2 vs 130.2 (default), about +3%.
- **Implementation check:** a first implementation (im2col with a block-diagonal 17×17 kernel) gave the same losses to 5 digits at a 38% higher CU.

### Gradient and weight evidence for the collapse (old code, live settings)

| seed | z units alive @500 / @1k / @1.5k | ‖∂L/∂W_headF‖ @500 → @1.5k onward | ‖W_headF‖ after collapse |
|---|---|---|---|
| 1 | 0.165 / 0.015 / 0.001, then 0 | 3.6e-3 → 3.4e-5 → 0 from 2k | frozen at 0.653 from 2.5k |
| 2 | 0.014 / 0, then 0 | 1.6e-4 → 0 (a brief revival at 3–3.5k, ≤ 2e-5) | 0.226, then 0.329 after the revival, frozen |
| 3 | 0.160 / 0, then 0 | 9.4e-3 → 0 | frozen at 0.850 from 1k |

After the collapse only the output biases still move. The output layers did move before it, so the head learning rate was not what stopped them.

## Hypotheses tested

Single changes against the old code, all on seed 1 with the live settings for 1,500 steps unless marked. The figures are window ratios on the own stream, at steps 250, 500, … 1,500.

| | hypothesis | test | result | verdict |
|---|---|---|---|---|
| a | Gradients to the output layers are tiny, or the head lr is too small | Logged ‖grad‖ and ‖W‖; set headLrScale 1 (head lr 0.002) | The gradient is non-zero (1e-4–1e-2) and the weights move (‖W_F‖ 0.2 → 0.65) until z dies; then it is exactly 0. With headLrScale 1: one window at 0.53 at step 500, then z dies (alive 0.000 by 1,500) and the ratio returns to 1.003 | **No** (a symptom of 2) |
| b | The signal is fovea translation, which a linear readout of z cannot express | Oracle; then translation kernels (rows below) | The oracle gets 0.60–0.78. A stable translation kernel gets 0.72 by step 1,500, and 0.65–0.85 over 5k steps in 6/6 runs | **Yes, primary** |
| b′ | — unnormalised kernel from z + action, lr × 0.086 / 0.25 / 1 | | 0.81–0.95 / 0.91–3.8 / 6–219 | Unstable: Adam moves all 289 taps together, so brightness drifts |
| b″ | — action-only kernel, mass-preserving (δ + K − δΣK), lr × 0.086 / 0.25 / 1 | | 0.95–1.11 / 1.06–4.65 / 7–44 | Unstable |
| b‴ | — **convex separable kernel softmax ⊗ softmax plus gate (final)**, lr × 1 / × 5 / × 5 with centre logit 2 | | **0.99, 0.92, 0.89, 0.79, 0.74, 0.72** / 0.85–1.03, then the policy parked (static) / 0.69–0.94 | **Works** at lr × 1 |
| c | Most transitions are no-ops, so copy-last is near-perfect | Change fractions | The cursor moves on 91–94% of steps and the fovea target changes on 69–74%. The achievable gain is about 25–40% (oracle) | **No** |
| d | Online updates and replay fight | No sleeps at all | 1.003, 1.002, 1.000, 0.92, then 0.18 and 3.52 (a policy-driven episode); z still decays (0.08) | **No** |
| e | The EMA hides learning | Window means and held-out set beside the EMA | With the old code the EMA, the window means and the held-out set all read 1.000. But the own-stream metric *can mislead in the other direction*: old code at the default settings shows an EMA of 0.41 / 0.72 while held-out is 1.40 / 1.27 | **No** for the symptom; a separate caveat |
| f | A bug (base not the current frame, targets off by one, action unused, weighting) | Shadow model rebuilt from the pre-step weights on the newest replay transition, compared with `UpdateInfo.lossBefore` every 3–5 steps; replay `next` checked equal to the observation | Relative error 0 on every check; alignment as documented (y_{t−1} base, a_{t−1}, y_t target); the size weighting is as documented | **No bug found** |
| g | Dead ReLU alone (activation choice) | z = tanh instead of relu | z stays alive (0.47–0.50), but the ratio is 0.999–1.011 | Real, but not sufficient on its own |
| h | Learning rate drives the collapse (from the 5k-step runs) | live lr 0.002 vs default 1e-3, old code | z died in 3/3 seeds at 0.002 and 1/3 at 1e-3 | Contributes. The live run's drawn lr made the collapse certain |

## Recommended fix

Commit `experiment: action-conditioned fovea translation in the predictor` (`packages/learner/src/model.ts`):
- three new parameter tensors `shiftX`, `shiftY` [76,17] and `shiftGate` [76,1], all zero;
- constant banded selection matrices;
- `LEARNER_STATE_VERSION` 1 → 2, because Model.params gained three tensors before `predH`;
- a model test that the term translates the base exactly (x + 3, y − 2), and gradCheck now covers the term.

All 662 node tests pass, plus the WebGPU learner test, typecheck and lint.

Why this shape:
- **It removes the instability.** It is convex (softmax), so the shifted image stays within the range of the current one and cannot drift in brightness. That is the failure of b′ and b″.
- **Few parameters to learn.** It is separable: a translation is kx ⊗ ky, so 34 logits per action instead of 289.
- **It stays exact at birth.** The gate starts at zero, so a fresh model still predicts exactly copy-last.
- **It cannot die.** It reads the action one-hot directly, not z, so it learns even after z dies.
- **No world knowledge.** Nothing tells it which action moves the cursor or by how much. It learns that, for example, a move of 16 px east selects tap +8 on x.

## Caveats and what is not fixed

- **The periphery is not helped** (ratio 1.00–1.07). The cursor sprite moving in the periphery is the same kind of signal, but it is a small part of the error.
- **z still dies at lr 0.002** with the fix: alive 0.016–0.028 at 5k. The rest of the predictor is still copy-last plus a bias. Separately, when z is alive (default settings), it can learn patterns specific to the policy's own data that make held-out *worse*: old-default 1.27–1.40, and fix-default seed 1 drifting to 1.03. Next candidates, untested:
  - an activation that cannot die;
  - a predictor that sees the current frames directly;
  - more diverse replay.
- **The display's ratio of EMAs is fragile.**
  - When the policy parks the cursor where nothing changes, copy-last is about 0 and the ratio explodes (fix-live seed 2: 634).
  - When the policy finds its own niche, the ratio can look good without generalising: old-default EMA 0.41 vs held-out 1.40.
  - Recommendation: report a held-out ratio alongside it, or the loss difference.
- **Scope of the evidence.** CPU_WIDTHS, 5,000 steps, 3 seeds × 2 settings. The live configuration (GPU_WIDTHS, WebGPU, 28k steps) has not been re-run with the fix. The WebGPU equivalence test passes, but a live run is the real confirmation.
- **The held-out trajectory uses uniformly random actions**, a different distribution from the trained policy's. Its 0.73–0.91 is a generalisation measure, not an upper bound. The shift oracle is approximate (see above).
- **Checkpoints from version 1 do not load** into version 2. That is intended: `version.ts` says to bump whenever Model.params changes. But existing lives restart.
- **Design-change provenance:** this change was made by a research agent (AGENTS.md invariant 5), not discovered by the learner. Its compute cost is about +3% CU per step.
