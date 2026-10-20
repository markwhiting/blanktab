# 0013 — Learner v1: design changes made while building Plan 04a

- **Date:** 2026-10-20
- **Status:** accepted
- **Decided by:** agent, under the user's instruction to make best choices and keep going. Proposed by implementers and reviewers, ruled on by the controlling agent. Open to revision.
- **Amends:** spec §5.4–5.5, §10; Plan 04a's learning criterion

These are design improvements made by the research agent, not things the learner discovered (AGENTS.md invariant 5). They change the learner every session uses, so they are recorded here.

## Learner
1. **Residual predictor.** The learner predicts *what changes*: `next = current + tanh(f(state, action))`. The output layers start at zero, so a newborn's guess is exactly "nothing changes" (the copy-last-frame baseline), and any learning shows as beating it.
   - **Why:** reconstructing the whole frame from a latent code never beat copy-last at small sizes, because frames are mostly static.
   - **Status:** this is an architectural prior (spec §5.1 allows priors), not pretrained knowledge.
2. **Separate learning rate for the residual output layers.** They train at `lr × headLrScale`, default 0.1, set by the new `headLrScale` setting (range [0.03, 0.3]).
   - **Why:** Adam moves every zero-initialised weight by about `lr` per step even when the true gradient is near zero. That jitter cost more than the learning gained.
3. **Relative curiosity.** The reward is `r = clamp((L_before − L_after) / predLossEMA, −1, 1)`, with the moving average taken from before this step.
   - **Why:** the absolute learning progress was about 1e-5 to 1e-3, too small to steer the policy against an entropy bonus of 0.01.
4. **Split update and truncated backpropagation through time of length 1.** The prediction update runs first, then the actor-critic update on a detached state. Only the prediction loss trains the core, and no computation graph lives across steps.
   - **Why:** the curiosity reward needs the post-update weights. Truncating at one step keeps memory use safe.
5. **Learning criterion (Plan 04a test).** It changed from "predLoss at step 1500 < 0.5 × predLoss at step 100" to "after 3000 steps, predLoss < copy-last baseline, on seeds 1–3".
   - **Why:** the first criterion was easy to pass while the learner was still worse than copy-last. The new one is the honest bar for milestone M1 in spirit.
   - **Measured:** ratios of 0.883, 0.874 and 0.786 at tiny widths.
6. **GPU and CPU agree.** A 300-step parity test shows WebGPU and CPU produce identical actions and predLoss values (relative difference 1.4e-8).

## Lifecycle and storage
7. **Checkpoints.**
   - Written at most every 5 minutes, always together with each log file in one IndexedDB transaction, and removed with the final log.
   - Replay frames are gzip-compressed.
   - A reload can lose up to 5 minutes of learning. That loss is not counted in `lostCU`, because it can't be measured without extra writes.
8. **Logs.** Written when compute lived crosses 2^k CU for k ≥ 3 (8, 16, …, 1024), plus the final log. That is at most 8 files per life.
9. **Restores stay on the record.** After 20 consecutive skipped updates, or a failed GPU read, the learner is restored from its checkpoint. Logs carry `restores` and `lostCU`, and `skippedUpdates` includes skips lost to a restore. A life that fails gets a final log built from its checkpoint.
10. **Orphaned learners.**
    - On startup, a tab adopts the newest eligible orphaned learner: its lock is free, it is not final, its state version matches, and it is less than 24 hours old.
    - Checkpoints that are stale, invalid, or beyond the newest 3 orphans are deleted.
11. **Resume uses birth facts from the checkpoint.** Seed, widths, settings, knowledgeSha and sessionsSeenAtBirth come from the checkpoint, not from the current page load.

## Known limitations
- The world is not checkpointed. On resume it restarts from its seed while the learner continues.
- `lossRatio` in the summary is a single, noisy point.
- About 19 steps per second on an M4 against a cap of 30. There are four serial GPU readbacks per step, to be optimised in a later plan.
