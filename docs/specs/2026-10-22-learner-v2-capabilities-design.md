# Learner v2: capabilities first, then directives — design

**Date:** 2026-10-22 · **Status:** draft for the user's review; nothing here is built or decided yet · **Supersedes:** nothing (amends spec §5 once approved; the approval becomes decision 0018)

Background: `docs/research/2026-10-16-research-synthesis-v0.1.md` ([SYN]). The v1 design: `docs/specs/2026-10-19-blanktab-v1-design.md` ([V1]). Learner changes so far: decisions 0013, 0015; milestones and checkups: 0016; lives and live updates: 0014; hardening: 0017. Live evidence: `docs/research/2026-10-21-does-it-learn*.md`, `2026-10-21-why-not-learning.md`, `2026-10-22-growing-up-live.md`, `2026-10-22-hardening-live.md`.

Labels: **[user]** = the user's words or approval. **[agent]** = the agent's call under the user's "make best choices"; not yet confirmed. **[guess]** = a number or expectation with no measurement behind it yet. External papers are cited from the agent's knowledge of the literature and were not re-checked for this document.

---

## 1. Intent

**The long-term target [user]:** a learner that can be given a directive and carry it out ("explain this email", "play minesweeper"), and eventually a strategic reasoner that uses online resources. No directive channel exists yet.

**The approved order [user]:**
1. **Capabilities.** A sense of "I caused that"; memory of what it has seen and where it has been; behaviour that spans several steps; learning in features, not raw pixels.
2. **A tiny on-screen directive language** with a harness success reward ("click the red one", "open the second page"). Directives arrive as pixels (cold start: no pretrained language model). Adding them is a new modality and reward channel, so it needs a recorded decision (AGENTS.md invariant 2).
3. **Goal games**, minesweeper first. Reading and writing come much later.

Milestones grow toward directive-following in rooms the learner has never seen.

This document specifies stage 1 in enough detail to plan from. It sketches stages 2 and 3 only far enough that stage 1 doesn't paint them into a corner.

### Why stage 1 is needed, from the evidence
- **The learner predicts a little and does nothing.** The held-out ratio is 0.86–0.95 against copy-last. It plateaus and drifts, and all of the gain is in the fovea ([does-it-learn-2], 0015). Checkups score at newborn chance: 1, 1, 1, 1, 0, 0 of 6 ([growing-up-live]). The only milestone getting closer to its bar is M1 (0.987 → 0.865, bar < 0.8).
- **Its only drive is pixel learning progress**, and that is dominated by the fovea shifting as the cursor moves ([why-not-learning]). Once the shift is learned, little drive is left. Nothing in the reward distinguishes "the world changed because I clicked" from "my eye moved".
- **It has no memory beyond a 256-d GRU trained with one-step truncation**, and a one-step TD critic with γ = 0.9 (`learner.ts`). So it can't value anything more than about 10 steps away: reaching a link and clicking it, uncovering a section, switching rooms.
- **A prediction-only learner can prefer predictable inactivity** [SYN §7.3]. One run parked the cursor in a corner ([why-not-learning], seed 2).

### How it serves the research question
The project asks whether learning to retain, consolidate and reuse experience makes later learning cheaper [SYN §1, §7.2]. Stage 1 serves that question in three ways:
- **Something to retain.** A consolidation comparison (conditions A/B/C, [V1 §5.6]) on a learner that has learned nothing compares noise.
- **Signals to consolidate by.** Stage 1 makes per-experience signals that a consolidation rule can use: novelty, controllability, latent surprise. These are the "candidate inputs" of [SYN §3.2].
- **A first test of "cheaper later learning".** Stage 2 gives the first clean measure of later learning: how much compute it takes to learn to follow directives, for a learner that grew up with stage 1 against a newborn running the same code. That is rung 4 of the evidence ladder [SYN §7.2].

## 2. Goals, non-goals, anti-goals

**Goals (stage 1 is done when all hold) [agent]:**
1. **The four mechanisms of §4 run in the live learner,** each one switched by a setting, so ablations need no code change.
2. **A pre-registered offline comparison (§6) has run and its results are written up, favourable or not.**
3. **The held-out pixel check (0015) is no worse than v1,** at the same CU.
4. **The stage-1 bar (§6.3): at least one of M2–M6 survives the primary test against the newborn base rate,** or the null result is published with a finding on why and what comes next. A null result is an acceptable outcome. Hiding one is not.
5. **Live speed stays at or above 24 steps/s uncapped** on the M4 reference machine **[guess at the bar]**, with no new serial GPU readback per step.

**Non-goals for stage 1:**
- The directive channel, any extrinsic reward, or goal games (stages 2 and 3).
- Consolidation conditions B/C, the GP picker and knowledge PRs (Plan 05; see §7).
- Retrieval from episodic memory into the core. Stage 1 memory only scores novelty; retrieval is a later option (§4.3).
- Planning by imagination with the latent model (a stage-3 candidate, §9).

**Anti-goals (failure even if the numbers go up):**
- Any pretrained weight, embedding, tokenizer or learned feature entering the learner (invariant 1). Fixed random projections and architectural structure are priors; learned values are not.
- Any privileged state reaching the learner, its rewards included (invariant 2). Controllability and novelty are computed from the learner's own predictions and features. The privileged trace may score diagnostics in the harness only (§5).
- Tuning designs, weights or thresholds to the milestone variants. Thresholds and `milestoneSuiteVersion` 1 stay fixed (0016). Development uses a separate seed range (§6.2).
- Claiming progress from lower loss, a higher reward or bigger memory alone (invariant 8, [SYN §7.3]). Milestones on fresh variants, read against the newborn base rate, are the evidence.
- Weakening the held-out check or the checkups to make v2 look better (invariant 7).
- Hard-coding the v1 action vocabulary or room set into the new mechanisms (invariant 9). The action-factor groups of §4.1 are read from `interface` constants.

## 3. Constraints
- **Browser compute:** one WebGPU tab (CPU fallback at half width), our own tensor library (`packages/tensor`), no ML libraries [V1 §6].
  - Any new op (a row write into a GPU ring, say) is added there, with CPU/GPU parity tests and gradient checks.
- **Speed:** live 26–30 steps/s on an M4 against a cap of 30 (0017).
  - A v1 step is about 0.075–0.084 CU (2,301 CU over 27,394 steps in run 2).
  - The step is bound by latency, not FLOPs. Pure compute runs at about 108 steps/s [webgpu-throughput]; the rest is 3 serial readbacks and dispatch overhead.
  - So **new work must ride on the existing reads.** Extra FLOPs are cheap; extra round trips are not.
- **Scale:** about 1–10 ordinary computers in total [SYN §7.1].
- **Lifecycle:**
  - Forever lives, live code reload when compatible (0014).
  - Frozen checkups that never write learner state (0016).
  - Rollback must not end lives where that can be avoided (0017).
- **Walls:** `learner` imports only `interface` and `tensor` [V1 §4].

## 4. Stage 1 design

### 4.0 One step, before and after

**v1 step** (`learner.ts`):
1. A prediction update on (o_{t-1}, a_{t-1}) → o_t, through the core from the detached h_{t-2}.
2. Curiosity r = relative pixel learning progress.
3. Act from h_t.
4. A one-step actor-critic on a_{t-1}.
5. Push the transition to FIFO replay.

**v2 step [agent]:**
1. **Prediction update.** The core runs from h_{t-2} as before. The predictor runs twice: once with the real action a_{t-1}, and once with the **manipulation factors dropped** (§4.1). The loss has a pixel term (unchanged, the anchor) and a **latent term** against a stop-gradient EMA target encoding of o_t (§4.2). The read in this round returns L_before and L_drop together, so it adds no round trip. Then one Adam step and EMA updates of the target encoder and the slower key encoder.
2. **Act.** h_t, the policy logits, **one value per reward term**, L_after, and the **novelty** of o_t against episodic memory (§4.3). All of it comes back in the existing single read.
3. **Rewards.** r_lp (learning progress, now on the latent loss), r_ctrl (controllability) and r_nov (novelty), each normalised (§4.5). They are appended to an on-policy segment.
4. **Credit.** Every n steps, an actor-critic update over the segment with λ-returns, one per term (§4.4).
5. **Memory.** o_t's key (from the key encoder) goes into the episodic memory; the transition goes into replay. In sleep, replay trains the core on **sequences**, with truncated backprop through time over k steps (TBPTT-k).

### 4.1 "I caused that": controllability [agent]

**Chosen: action-conditioned versus manipulation-marginal prediction.**
- **Training.** The predictor is trained on two inputs every step: the real action one-hot, and a copy with the button, scroll and key blocks zeroed. Zero means "unknown"; the move block stays.
  - The zeroed input learns the prediction averaged over what the policy does with its hands, given where it moved.
  - This is conditioning dropout, as in classifier-free guidance: one predictor, no new network.
- **The signal.** On the transition just experienced, before the update, **on the pixel loss** (it is grounded in the frame that came; settled, agent's call):
  `c_t = (L_drop − L_cond) / EMA(L_cond)`, clamped to [−1, 1].
  It is positive when knowing what it pressed explains what changed. The same difference on the latent loss is logged as a diagnostic only; it never enters the reward.
- **Why this shape:**
  - **It ignores self-motion.** The fovea shift and the cursor sprite are explained by the move factor, which both predictions share, so cursor motion cancels out. That matters because 70–99% of the copy-last error is the fovea shifting ([why-not-learning]). An all-factors counterfactual would reward wiggling the cursor.
  - **It resists the noisy TV.** Changes the learner did not cause (animations, noise) are equally predictable, or equally unpredictable, under both predictions, so they cancel too.
  - **It is grounded.** It is scored against the frame that actually came, not against the model's own belief.
  - **It is cheap:** one extra predictor forward and backward (§8).
- **Habituation.** A learner can farm controllability by toggling one toggle forever. So the controllability reward is **gated by novelty**: `r_ctrl = c_t⁺ · min(1, ν_t / EMA(ν))`, where ν_t is the novelty of the outcome (§4.3). The reward is for "I caused something I haven't seen lately". The additive form is an ablation arm (§6.1).
- **Factor groups.** The groups are taken from `interface` (`MOVE_OPTIONS` versus the rest), so a larger action vocabulary later needs no change here (invariant 9).
  - **Settled (agent's call, not yet confirmed):** that an action splits into "moving" and "manipulating" is a built-in architectural prior, like the fovea-shift prior (0015). Invariant 1 allows it: it is structure, not learned competence. Which manipulations do what is learned. It is recorded in decision 0018 when 06a lands.
  - Without the split, the reward would favour wiggling the cursor (see "It ignores self-motion" above).
- **Alternatives:**
  - **Inverse-dynamics features (ICM, Pathak et al. 2017).** These shape features toward what the action explains. Here the most action-predictive thing is the cursor, which proprioception already gives. Deferred: available as an auxiliary feature loss if features prove poor.
  - **Empowerment (Klyubin et al. 2005; Mohamed & Rezende 2015).** It needs a variational channel-capacity estimate per step. Rejected for cost at 30 steps/s.
  - **A KL between conditioned and marginal predictions (Seitzer et al. 2021).** This is the belief-only version of c_t. Rejected because it is not grounded in the actual next frame.

### 4.2 Learning in features: latent prediction [agent]

**Chosen: an EMA target encoder (as in BYOL, Grill et al. 2020, and SPR, Schwarzer et al. 2021; JEPA-style, LeCun 2022).**
- **The target encoder** is a copy of both encoders whose weights track the online encoders by EMA (τ = 0.995 per update **[guess]**). It has no gradient. Its output for o_t, `ē_t` (256-d, layer-normalised), is the latent target.
- **The latent head.** The predictor gets a third head, residual like the pixel heads: `ê_t = sg(ē_{t-1}) + W·z`, with W zero-initialised, so a newborn predicts "the features don't change". Without this, a newborn would not start exactly at copy-last.
  - Its loss is `L_lat = 1 − cos(ê_t, sg(ē_t))`, weighted by λ_lat (setting, default 1 **[guess]**).
- **Learning progress moves to the latent loss.** r_lp = clamp((L_lat,before − L_lat,after) / EMA(L_lat), −1, 1). The v1 pixel learning-progress reward remains as an ablation arm.
  - **Why:** pixel learning progress is spent once the shift is learned, and it rewards texture. Feature progress rewards learning what the encoders find predictable at the scale they see.
- **The pixel loss stays** as an anchor and a diagnostic. It still trains the encoders through the core, and it cannot collapse [V1 §5.4]. With the pixel anchor in place, the latent loss cannot drive the encoders to a constant without hurting the pixel loss.
  - The **held-out pixel check (0015) and M1 are unchanged**. A held-out *latent* ratio (against "features don't change") is added as a secondary number.
- **Collapse alarm.** The running per-dimension standard deviation of ē and its effective rank are logged. Falling below a floor marks the learner in its log; it does not change behaviour (§5).
- **Why an EMA target rather than the online encoder with a stop-gradient (SimSiam):**
  - The target forward costs one encoder pass, about 8 MFLOP, +10% CU **[estimate from layer shapes]**.
  - In return, the targets move slowly. With batch size 1 and a continually trained encoder, a stop-gradient on the online encoder chases itself.
- **Alternatives:**
  - **Pixel prediction only (v1).** Kept as the anchor.
  - **Contrastive objectives.** They need negatives; batch 1 makes that awkward.
  - **VICReg-style variance terms (Bardes et al. 2022).** They need batch statistics. Held in reserve as a sleep-time regulariser over replay minibatches if the alarm fires.

### 4.3 Memory of where it has been: episodic novelty [agent]

**Chosen: a bounded kNN-style memory over slowly drifting keys, with a kernel count (as in NGU, Badia et al. 2020; see also episodic curiosity, Savinov et al. 2019).**
- **A separate, slower key encoder (settled, agent's call).** Keys come from a third copy of the encoders, tracking the online ones by EMA at **τ_key = 0.9999 per update** (half-life ≈ 7,000 updates, about 4 minutes at 30 steps/s). The BYOL-style target (τ 0.995, half-life ≈ 140 updates) moves too fast: a key written a minute ago would no longer be comparable with today's. A 2,048-key memory spans about 70 s, so key drift across the memory is small.
- **Keys.** A fixed random projection (an architectural prior, seeded at birth) of [k_t ; cursor x, y] down to 64-d, where k_t is the key encoder's output for o_t. The periphery part says roughly "where I am", the fovea part "what I'm looking at", and the cursor "where on screen".
- **Memory.** A GPU ring of M = 2,048 keys **[guess]**, about 70 s of experience at 30 steps/s. One key is written per step, through a new tensor op for writing a row (§3).
- **Novelty.**
  - Kernel: `K(x, m) = ε / (d²(x, m)/d̄² + ε)`, where d̄² is a running mean of squared distances to the nearest few keys.
  - Novelty: `ν_t = 1 / sqrt(Σ_m K(x_t, m) + c)`, summed over **all** of memory, so no top-k op and no readback is needed. One [1×64]·[64×2048] matmul, an elementwise pass and a sum, reduced to a scalar inside the existing read.
  - With a sharp kernel (ε = 1e-3 **[guess]**), the full sum behaves like a soft count of near neighbours. An exact top-k version is an ablation if the soft count proves too blunt.
- **`r_nov = ν_t`**, normalised (§4.5). It rewards going where it hasn't been lately: new pages, new rooms, revealed content. The lifelong part of NGU's design (RND) is not added. Latent learning progress already plays the lifelong role.
- **Lifecycle.** The memory is saved in the checkpoint (2,048 × 64 floats, 0.5 MB) so a reload doesn't cause a novelty spike. Frozen checkups run with a **fresh, empty memory of their own**, so they read nothing of the live memory and write nothing to it.
- **Room to grow.** Each memory slot can later carry a payload (h, the action, the step) for retrieval into the core: "what was behind that pop-up", "which page was second". Stage 1 stores keys only, but the layout keeps a payload slot (§9).
- **Link to consolidation.** Novelty and controllability at insert time are new per-item signals for the B/C replay scorer [V1 §5.6]. Replay items record both.
- **Alternatives:**
  - **Count-based bonuses on hashed pixels.** Brittle under cursor motion.
  - **RND alone (Burda et al. 2019).** A second network to train, and it measures lifelong novelty, not "lately".
  - **Exact kNN on CPU.** It needs a readback of 2,048 distances every step. Rejected: §3 rules out extra round trips.

### 4.4 Longer credit assignment [agent]

**Chosen:**
- **On-policy n-step segments with GAE(λ) (Schulman et al. 2016) for the actor-critic online.**
  - The learner keeps the last n = 16 steps' detached h, the action, its log-probability under π, and each reward term.
  - Every n steps it runs one batched update: the heads on [n, hidden], with λ-returns per reward term at γ = 0.97 and λ = 0.9 **[guess]**. A horizon of about 30 steps (3 s) covers moving to a link and clicking it, or reaching the taskbar.
  - This changes the cadence of the actor-critic update but not the online prediction update. The ε-uniform off-policy bias stays as accepted in 0015.
- **TBPTT-k sequences in sleep for the core.**
  - Sleep replays R/k contiguous sequences of k = 8 transitions **[guess]**, each starting from its stored h_{t-2}. The budget is the same R transitions as now.
  - The prediction losses (pixel, latent, dropped) backpropagate through the core over the sequence. This is how the GRU learns to carry information across steps.
  - Online prediction stays TBPTT-1, because the learning-progress reward needs before and after losses on the transition just learned (0013 §4).
  - A sequence is contiguous when `next` of one transition is `obs` of the following one (the buffer already shares frames). It may cross a room switch: learning what a taskbar click does is wanted.
- **Alternatives:**
  - **Online TBPTT-k.** It breaks the per-transition learning-progress reward and holds k graphs alive.
  - **Off-policy actor-critic from replay with V-trace (Espeholt et al. 2018).** Intrinsic rewards change as the model learns, so replayed rewards go stale. Deferred.
  - **Simply raising γ in one-step TD.** Bootstrapping through one step propagates too slowly at batch size 1.

### 4.5 One reward without one term dominating [agent]
- **Per-term normalisation.** Each term is divided by its running RMS (EMA of r², floored) and clipped to ±5, so the terms are on comparable scales whatever their raw units.
- **A value output per term.** The value head outputs [V_lp, V_ctrl, V_nov]. Each term gets its own λ-return; the advantage is `A = Σ_i w_i · A_i / Σ_i w_i` (as RND's two value heads did; Burda et al. 2019).
  - This is the **slot stage 2 uses**: an extrinsic term is one more output with its own γ, not a redesign.
- **Weights** w_lp, w_ctrl, w_nov are settings, default 1 / 1 / 1 **[guess]**. `curiosityWeight` keeps its meaning as w_lp. A weight of 0 turns its term off and must be bit-exact to the code without that term, as was done for ε and residualDecay (0015).
- **A dominance diagnostic.** Each log records every term's share of Σ_i |w_i A_i| over the logged window. A share above 0.8 sustained across two logs is flagged in the summary. That is information for the picker and the write-up; the learner is not changed by it.
- **Entropy and ε-exploration** stay as in v1.

## 5. How each mechanism is measured

| mechanism | milestones it should move (0016) | learner-side diagnostic (logged) | harness-only check (privileged trace; never reaches the learner) |
|---|---|---|---|
| controllability (4.1) | M3 (prefers responsive), M2, M6 | mean c_t (pixel); the latent c_t, as a diagnostic only; share of reward | AUC of c_t for "the element responded" against "it did not", on held-out transitions |
| latent prediction (4.2) | M1 (via the anchor), all others indirectly | held-out latent ratio; target std and effective rank; collapse flag | a linear probe from ē to the room, trained and scored by the harness only, never fed back: are the features about the world? |
| episodic novelty (4.3) | M4 (reveals), M5 (places), M6 (taskbar) | mean ν; memory fill | AUC of ν for "new page or room" against "revisit" |
| credit (4.4) | M5, M6, M4 | value explained-variance per term; sleep sequence loss | none |
| combination (4.5) | all | per-term advantage shares; dominance flag | none |

- **Unchanged:**
  - the held-out pixel ratio (0015), the primary prediction number;
  - checkups at the log schedule, frozen, on fresh variants;
  - `milestoneSuiteVersion` 1 thresholds.
- **New diagnostics are for interpretation only.** None of them is a pass criterion. Each is billed to `evalCU`, never `cu`.

## 6. Experiment protocol [agent; all of §6 is pre-registered and fixed before the confirmation run]

### 6.1 Arms (pre-registered definitions)

| arm | what changes | lives |
|---|---|---|
| v1 | v2 code, every new weight 0, pixel learning progress, γ 0.9, n = 1, k = 1. It must match v1's behaviour (a golden-trajectory test on CPU, like 0017's) | 24 |
| v2-full | everything on | 24 |
| −ctrl | w_ctrl = 0. The dropped-action predictor still trains, so only the reward is removed | 12 |
| −novelty | r_nov = 0, and the controllability gate is fixed at 1 (ν/EMA(ν) treated as 1). The memory still runs, for diagnostics | 12 |
| −latent | λ_lat = 0, curiosity on pixel progress (as v1). The target and key encoders still run | 12 |
| −credit | n = 1, k = 1, γ = 0.9 (as v1). The per-term value heads are kept | 12 |
| ctrl-additive | `r_ctrl = c_t⁺` without the novelty gate | 12 |

That is 108 lives in all. A "v1" arm in the new code keeps the implementation fixed within the comparison (invariant 7, [SYN §7.3]).

### 6.2 Seeds, budgets, data
- **Development and confirmation are separate.**
  - Designs and defaults are tuned on development seeds 1–99, with their own checkup variants.
  - The confirmation run uses fresh learner ids and seeds 10,000 and up, written down before it starts, so the variants it meets are new to the design as well as to the learner.
- **Budget.** Every life runs to the same **2,048 CU** of training compute, at GPU widths, with `evalCU` reported separately. CU is the shared unit; steps and wall time are reported beside it [V1 §2 goal 6].
  - v2 costs about 35% more per step **[estimate, §8]**, so equal CU gives it fewer steps. That is intended.
- **Machine time [estimate].** One life is about 19–26k steps, about 12–16 min at 27 steps/s. Its 8 checkups (16 … 512, 1,024, 2,048 CU) take about 11 min at up to 80 s each. So a life is about 30 min.
  - 108 lives is about 54 h. The newborn re-measure (§6.4) adds about 4 h.
  - In total that is **about 2.5 machine-days** on one M4 running one life at a time, or about 1 day on 3 machines. Running two lives at once on one machine is unmeasured.
  - This needs a small headless batch runner (§12).
- **What is kept.**
  - Every arm's logs and the raw summary go in a research folder beside the write-up, `docs/research/2026-1x-xx-v2-capabilities/`.
  - Unfavourable arms are reported in full (invariant 7).
  - Lives that fail stay in the denominator, as non-passes.

### 6.3 The primary test, and the stage-1 bar
- **Unit.** One life is one data point. A life "passes milestone m" if it passes m at **at least 2 of its last 3 checkup levels** (512, 1,024, 2,048 CU). That is the "sustained" reading rule of 0016, made exact.
- **Primary test.** For each of M2–M6: v2-full (24 lives) against the newborn base rate (§6.4, 60 newborns). One-sided Fisher exact test, Holm-corrected over the 5 milestones, family α = 0.05.
- **The bar.** **Stage 1 is done if and only if at least one milestone survives Holm.** M1 is not in the family: it is a prediction measure, and the held-out check covers prediction.
- **Power, briefly [computed for the doc from the 0016 CPU-width single-checkup newborn rates; the builder recomputes on the re-measured rates before the run].** At the first Holm step (α/5 = 0.01), v2-full needs:
  - M5 (2/60 newborns): 6 of 24 lives;
  - M2, M6 (3/60): 7 of 24;
  - M3 (8/60): 10 of 24;
  - M4 (11/60): 12 of 24.
  At 12 lives it would need 4–7 of 12, a 33–58% pass rate, which a first learner is unlikely to reach. At 24 lives the bar is 25–50%. Hence 24 lives for the two arms that carry the decision and 12 for the ablations, which are descriptive.
  - The ≥2-of-3 rule should lower the newborn rates, and so the counts needed.
- **Secondary (reported with effect sizes and confidence intervals; not part of the bar):**
  - v2-full against the v1 arm, per milestone;
  - each ablation against v2-full;
  - held-out pixel ratio at 2,048 CU;
  - CU and wall time per step;
  - drift of the held-out ratio after its best reading.

### 6.4 Newborn base rate
- The v2 network has new heads and initial biases, so the 60-newborn measurement of 0016 is re-run **for v2-full at GPU widths** (seeds 1,000–1,059) before the confirmation run.
- Each newborn takes frozen checkups at levels 512, 1,024 and 2,048, with the variant seeds a life at those levels would get, and is scored by the same ≥2-of-3 rule.
- Its row is the primary bar.

## 7. Fit with consolidation (B/C) and the picker (Plan 05)

**A fact the order depends on:** the code is condition A only. There are no low-rank adapters, no fast tier and no B/C (`learner.ts`, `log.ts` `condition: "A"`). Plan 05 has to build the fast tier as well as the scorer.

**Proposed order [agent]:**
1. **Plan 06a: v2 foundation.**
   - The latent head and target encoder, the per-term value head and normaliser, n-step GAE, and sleep sequences.
   - The settings switches and the v1-arm golden test.
   - One `LEARNER_STATE_VERSION` bump.
2. **Plan 06b: controllability and episodic novelty,** plus the harness-only diagnostics.
3. **Plan 06c: the offline protocol (§6)** and its write-up.

**How v2 reaches live tabs (settled, agent's call).**
- **Offline until the comparison has run.** 06a and 06b are developed and evaluated offline, on a long-lived branch (`learner-v2`), with the headless batch runner. Nothing merges to `main` before 06c's comparison has run: no state-version bump, no restart of live learners. Live tabs keep running v1 meanwhile.
- **Then one merge, one bump, one restart,** following 0015's one restart per change.
- **The merge rule.**
  - If stage 1 passes its bar (§6.3), v2 merges.
  - If the result is null, v2 merges only if it is **no worse than the v1 arm** on both of these:
    - the held-out pixel ratio at 2,048 CU: v2-full's median no more than 0.02 above the v1 arm's;
    - checkups: v2-full's mean count of milestones passed per life (≥2-of-3 rule, M2–M6) not lower than the v1 arm's.
  - Otherwise the branch stays unmerged and live tabs keep v1.
- **The result is published either way.**
4. **Plan 05: picker, B/C and A/B/C,** on the v2 learner.
   - B/C's scorer gains the new per-item signals (novelty and controllability at insert).
   - The fast tier (adapters) wraps the v2 core and predictor.
5. **Stage 2 (directives)** after Plan 05, or in parallel with it once 06c lands. It needs only the harness and the world, plus one new value output.

**Why v2 comes before B/C:**
- **There is something to retain.** A/B/C on a learner at chance would compare noise. Consolidation's value shows only when there is competence to keep.
- **The signals are settled first.** B/C's signal set would otherwise be built twice.
- **The comparison stays clean.** The picker's GP would otherwise fit settings for an architecture about to change. `settings-space.json` gets a new `schemaVersion`, so v1 rows are ignored rather than mixed in.

**What could go in parallel:** the GP picker alone is harness-only and could be built now. The cost is a fit over v1 rows that becomes obsolete. *Agent's call: not worth it.*

**The drift (0015) is not ignored meanwhile.** Sleep sequences and the latent target change what replay trains. Whether drift shrinks is a secondary readout of §6.

## 8. Compute cost per step (estimates from layer shapes at GPU widths, to be measured)

| addition | FLOPs per step | readbacks |
|---|---|---|
| EMA target encoder forward on o_t | ≈ 8 M | 0 (scalars join the existing reads) |
| EMA weight update of the target (~2.19 M params; could run every 2–4 steps) | ≈ 6.6 M (3 FLOPs per parameter) | 0 |
| key encoder: forward on o_t, plus its EMA update at τ_key (could run every 4 steps, at τ_key⁴) | ≈ 8 M + 3 M | 0 |
| dropped-action predictor forward and backward | ≈ 4–5 M | 0 (L_drop is read with L_before) |
| latent head, per-term value outputs | < 1 M | 0 |
| novelty: projection, 64 × 2,048 kernel, sum, ring write | ≈ 1 M | 0 (with L_after and the heads) |
| n-step GAE (batched heads on [16, 256] every 16 steps) | ≈ the same as now, amortised | 1 per 16 steps |
| sleep sequences (same transition budget) | ≈ the same as now | as now |
| **total** | **≈ +0.03 CU on ≈ 0.08 (≈ +35%)** | **3 per step, as now** |

**Wall time.**
- About 60–100 more GPU dispatches per step **[guess]**, at a few µs of issue time each after the uniform-ring fix [webgpu-throughput]: roughly +3–6 ms on a step of about 33 ms.
- Expected 25–28 steps/s uncapped on the M4 **[guess]**. Goal 5 sets the bar at 24.
- If it falls short, cut in this order: both EMA updates every 4 steps, M = 1,024, the projection to 32-d. Widths come last, as in [V1 §6].
- The CPU fallback (half width) is about 15–20 steps/s today. Expect it to drop proportionally; it stays supported.

## 9. Versioning and live learners
- **One bump.** `LEARNER_STATE_VERSION` goes from 2 to 3, once, at the single merge to `main` (§7). On the branch, 06a fixes the layout and 06b must add its parameters and state (the projection seed, the memory ring, the per-term normalisers) in 06a's layout, so no second bump is needed.
  - `LearnerState` gains:
    - the target-encoder params;
    - the key-encoder params, with τ_key;
    - the projection seed;
    - the memory ring and its write position;
    - the kernel's running d̄²;
    - the normaliser state;
    - the segment buffer. Replay items gain novelty and controllability at insert.
- **Lives across the bump (0014).**
  - A running v2-state learner sees a code version it cannot resume, so it keeps running on its old code and the page says so.
  - New tabs are born on v3.
  - A v2 orphan swept by v3 code ends on the record with a final log built from its latest file (0017, "version orphans").
  - This is the same once-per-change restart the user accepted in 0015. It happens once, at the single merge (§7).
- **Logs.**
  - Log `schemaVersion` 2 → 3 adds the per-term shares, the diagnostics of §5 and an `arch: "v2"` field. Comparisons never mix architectures (0014's constant-segment rule).
  - The parser keeps reading v2 files, and the gate validates both.
- **Rollback.** Code before 06a rejects v3 states as another version (`LearnerStateVersionError`), which already ends safely. No new optional fields are added to the v2 shape.

## 10. Stage 2 sketch: directives on screen

**The channel.**
- The world renders an **instruction strip**: the top 12 px of the 320×200 screen, above every room. It shows the current directive as pixels.
- The learner gets no flag, token or id. It sees the strip only as part of the periphery, and in the fovea when it looks there.
- Directives are drawn with the world's bitmap font and swatches (Plan 03). Nothing pretrained is involved.

**The reward.**
- The harness checks success from the privileged trace (for example, the clicked element's colour equals the directive's) and passes one scalar `reward` to the learner with the next observation.
- This changes `interface`: `Learner.step(o, reward?)`.
- In the learner, it becomes a fourth value output with its own γ and normaliser (§4.5). Its weight w_ext is a setting.
- **The decision needed (invariant 2):**
  - a new reward channel, plus the strip as a curriculum element;
  - what the harness may reward;
  - the trial structure (a new directive after success or after T steps);
  - that the trace decides success but never reaches the learner;
  - how directive results are scored: their own suite and version, so the M-suite is not redefined.

**The first directive room, "Pick" [agent].**
- **Settled for now (agent's call, not yet confirmed; revisitable when stage 2 is designed): picture first.**
- A few coloured shapes. The strip shows **a sample swatch** ("click the one like this"), before any words. This tests following a directive separately from reading one.
- Matching a sample needs only visual comparison, not symbol grounding. Words ("red") come next, then ordinals ("the second page", in places).
- Evaluation is frozen checkups on fresh variants, with **held-out combinations** (colour × shape pairs never shown in training), so a pass means generalisation and not memorising.
- D1 passes at a success rate of at least 2× the newborn rate **[guess at the bar]**.

**The research payoff.** Compare the CU needed to reach D1 for a learner aged 2,048 CU against a newborn with the same code. If the older one needs less, prior experience made later learning cheaper (§1).

**What stage 1 must have left in place:**
- the per-term value head;
- a reward normaliser that admits a new term;
- recurrent memory trained over sequences (for "second");
- novelty memory with a payload slot (for "which page was second");
- no learner-side assumption that rewards are intrinsic.

## 11. Stage 3 sketch: minesweeper
- **The room.**
  - An 8×8 grid of 16 px cells, with about 10 mines.
  - A click reveals a cell. Numbers are drawn as digit pixels.
  - A flag is a key ("f") pressed over a cell, so no right button is needed.
- **The reward:** harness-scored, +1 for a win and −1 for hitting a mine, plus optional per-safe-reveal shaping behind its own recorded decision.
- **The directive:** the strip shows the minesweeper icon, so "play minesweeper" means find the room on the taskbar and win.
- **What it newly needs:**
  - reading digits from pixels;
  - inference across cells;
  - horizons of 10–40 clicks, hundreds of steps.
- **Candidate mechanisms** (not designed here): planning in imagination with the action-conditioned latent model of §4.2 (as in Dreamer, Hafner et al., and MuZero, Schrittwieser et al. 2020), and retrieval from episodic memory.
- **Why stage 1 doesn't block it:** the latent predictor is already an action-conditioned model in feature space, which is what such planning needs.

## 12. Left to the builder
- **Numbers:** the EMA τ, M, the projection size, ε, c and d̄ smoothing, the normaliser floors and clip, n, k, γ and λ, all within the ranges above. Put them in `settings-space.json` where the picker should search them, and in constants elsewhere.
- **Implementation:**
  - the row-write op and how the novelty and controllability scalars join the existing reads (as `selL`/`selV` do now);
  - batching the n-step update;
  - sequence sampling from the FIFO ring.
- **Testing:**
  - the v1-arm golden-trajectory test and the bit-exactness of each weight-0 switch;
  - gradient checks for the latent and dropped paths;
  - CPU/GPU parity on the new ops;
  - a frozen-checkup test showing the live memory is untouched.
- **Tooling and display:**
  - the offline batch runner, extending `tools/probe/learn-probe.ts` or the headless page harness;
  - page UI for the new numbers: the reward shares and the held-out latent ratio, beside the held-out pixel ratio.
- **Process:** the exact plan split of 06a–c, and a decision file (0018) recording the approved form of this document.

## 13. Open questions for the user
Only questions whose answer changes what gets built. Each has the default the agent will use if there is no answer.

1. **Order: should v2 stage 1 come before Plan 05 (B/C and the picker)?** *Default: yes,* for the reasons in §7. Plan 05 follows on the v2 learner. The alternative is to build B/C first on the v1 learner to attack drift now.
2. **The "stage 1 done" bar: is §6.3 the right finish line?** That is: one life is one data point, and a life passes milestone m at ≥2 of its last 3 checkup levels. v2-full (24 lives) is tested against the re-measured newborn base rate, one-sided Fisher exact, Holm over M2–M6, α 0.05. Stage 1 is done if and only if at least one milestone survives. A null result is published and accepted. v2 against the v1 arm is secondary. *Default: yes.* The alternative is a stricter bar, such as v2 also beating the v1 arm, before moving to directives.
3. **Offline batch runs on your machines.** May the §6 protocol run as headless batch lives on your computers, with the results committed as a research note? It is 108 lives plus 60 newborns: **about 2.5 machine-days on one M4, or about 1 day on 3 machines** [estimate]. Live tabs alone would take weeks. *Default: yes.* Live tabs get v2 only through the single merge of §7.

The agent settled the earlier questions about the "moving versus manipulating" prior (§4.1) and the picture-first directive (§10) as its own calls. They remain open to reversal.

## Sources
- **Project:** [SYN] `docs/research/2026-10-16-research-synthesis-v0.1.md` §1, §3.2, §7; [V1] `docs/specs/2026-10-19-blanktab-v1-design.md` §2, §5, §6, §8, §11; decisions 0013–0017; [does-it-learn] / [does-it-learn-2] `docs/research/2026-10-21-does-it-learn*.md`; [why-not-learning] `docs/research/2026-10-21-why-not-learning.md`; [growing-up-live] `docs/research/2026-10-22-growing-up-live.md`; [hardening-live] `docs/research/2026-10-22-hardening-live.md`; [webgpu-throughput] `docs/research/2026-10-20-webgpu-step-throughput.md`; code `packages/learner/src/{learner,model,shift,encode,replay,version}.ts`, `packages/interface/src/`.
- **Literature (from the agent's knowledge, not re-checked):** Pathak et al. 2017, curiosity by self-supervised prediction (ICM); Burda et al. 2019, exploration by random network distillation; Badia et al. 2020, Never Give Up; Savinov et al. 2019, episodic curiosity through reachability; Grill et al. 2020, BYOL; Schwarzer et al. 2021, SPR; LeCun 2022, "A path towards autonomous machine intelligence" (JEPA); Bardes et al. 2022, VICReg; Klyubin et al. 2005 and Mohamed & Rezende 2015, empowerment; Seitzer et al. 2021, causal influence detection; Schulman et al. 2016, GAE; Espeholt et al. 2018, IMPALA / V-trace; Hafner et al., Dreamer; Schrittwieser et al. 2020, MuZero; Finn et al. 2016, CDNA; Oudeyer & Kaplan 2007, intrinsic motivation and learning progress.
