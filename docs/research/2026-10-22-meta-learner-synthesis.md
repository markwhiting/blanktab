# One meta-learner over a learning genome: what the literature says

**Synthesis | 22 October 2026 | agent-written; a proposal, not a decision**

This note pulls four literature notes together around the user's plan. The user's view, paraphrased faithfully:
- Improvements should not be theorised by us. They should come from a design space that **one meta-learner** explores with adaptive sampling. The user's default is a Gaussian process (GP) over the parameters that define a learning behaviour, drawing better samples over time. No coding agent is in the loop at first; maybe later.
- The pipeline is "all data pipeline": **one statistical representation** from the meta-learner at the top down to an individual's finest behaviour. **How it learns** is the shared core, inherited through GitHub. **What it learns in a run** is local.
- It is hierarchical. Learned material that many lives acquire (recognising pixels or buttons) should **move up** strategically, into shared weights or into the algorithm itself.

Source notes and the key prefixes used here (each note numbers its own sources, so the prefix is needed):

| prefix | note | keys |
|---|---|---|
| **OLS** | [`2026-10-22-lit-outer-loop-search.md`](2026-10-22-lit-outer-loop-search.md) | O1–O25 |
| **LLR** | [`2026-10-22-lit-learned-learning-rules.md`](2026-10-22-lit-learned-learning-rules.md) | L1–L17 |
| **DLA** | [`2026-10-22-lit-discovered-learning-algorithms.md`](2026-10-22-lit-discovered-learning-algorithms.md) | L1–L13 |
| **KUH** | [`2026-10-22-lit-knowledge-up-the-hierarchy.md`](2026-10-22-lit-knowledge-up-the-hierarchy.md) | K1–K28 |

Project references: [SYN] `2026-10-16-research-synthesis-v0.1.md`; [V1] `docs/specs/2026-10-19-blanktab-v1-design.md`; [V2] `docs/specs/2026-10-22-learner-v2-capabilities-design.md`; code `packages/learner/src/learner.ts`, `model.ts`; `knowledge/settings-space.json`. Labels: **[guess]** = a number with no measurement behind it; **(agent's call)** = a design choice the user has not confirmed.

---

## 1. Summary

**Well supported.**
- **Inherit how it learns, not what it learned (Baldwinian).** Select on how well a fresh life learns, and pass on only the settings it was born with. This is the user's split, and it has a name and a direct test. Fernando et al.: "The Baldwin effect is superior to Lamarckian learning when the distribution of tasks is broad or quickly changing" [LLR L12, KUH K12]. Copying learned weights (Lamarckian) won only on narrow task sets, and it drove evolved learning rates to the floor. Our world is meant to broaden (invariant 9).
- **A black-box outer loop over whole lives.** Every method that worked without backpropagating through the inner loop scored a whole life and searched by population (ES, GA or BO) [LLR §"What this suggests" 1; DLA L4, L10, L13]. For lifetime-spanning schedules, ES found what meta-gradients could not: "Commonly used meta-gradient approaches fail to discover such adaptive objective functions while evolution strategies discover highly dynamic learning rules" [DLA L13]. Tabs cannot share gradients, so this is both the method we can afford and the right one.
- **A GP picker fits, with one change: Thompson sampling instead of UCB.** UCB is deterministic, so every tab born in the same gate hour would run the same settings [OLS §1]. Thompson sampling (TS) is "particularly appropriate for batch or delayed feedback settings where many selections … are based on the identical posterior" [OLS O2 §II.A]. Asynchronous TS loses little against running the same lives in sequence [OLS O5].
- **The genome must be small.** BO is "best-suited for optimization over continuous domains of less than 20 dimensions" [OLS O3]. CMA-ES needs about 300 lives for 10 numbers and about 13,000 for 64 [OLS §7]. A rule *shared* per layer generalises across widths; one rule *per synapse* is far too large [LLR L1, L3, L9, L10].
- **The cheapest template that has worked:** a small, constrained component inside a known-good learner, warm-started at the hand design, then distilled into a readable rule. LPO/DPO did this on one machine with 4 GPUs in about 2 days [DLA L10]. Lion and Co-Reyes et al. also warm-started from the existing design [DLA L8, L9].

**Risky.**
- **Noise against budget.** One life is about 30 min and gives one noisy number. On a binary milestone, telling 18 % from 28 % takes about 216 lives per arm [OLS §0]. One machine gives about 110–340 lives a week. A pass-count fitness is hopeless for a GP, so search needs a continuous fitness.
- **Nothing to climb yet.** Checkups currently score at newborn chance at every level [V2 §1]. Searching a flat landscape optimises noise. Every published success searched *around a learner that already worked* [DLA §"What this suggests" 2]. Nothing in the four notes shows discovery working at ~500 noisy 30-minute evaluations a day [DLA §"What this suggests" 10].
- **Meta-overfitting.** Selection on noisy scores is biased upward, "often of comparable magnitude to differences in performance" [OLS O21]. Lion measured it: "the search fitness keeps growing, but the meta-validation metric declines" [DLA L8]. Search and promotion therefore need different seeds and, later, different worlds.
- **Horizon.** Learned rules stall or degrade past the life length they were selected on [LLR L2, L10; DLA L4]. Our lives are endless (0014).
- **Moving weights up is hard across seeds.** Naive averaging of independently seeded networks fails [KUH K1 Fig. 1]. Permutation alignment fixes this only for wide, well-trained models, and ours are narrow [KUH K3, K4]. Any weight-shaped shared artifact therefore implies an **inherited, labelled condition** that starts from it.

**Cheapest promising first version (agent's call).**
1. **Genome v1** = today's learner and caregiver settings, plus the v2 mechanisms as genes whose centre is "off" (today's learner), so about **12–17 searched continuous numbers and 1–2 switches**. Everything else stays fixed at the centre.
2. **GP plus TS.** The hourly gate fits the GP deterministically. Each tab draws its own seeded sample. One life in five stays uniformly random forever.
3. **A continuous search fitness** that includes M1's held-out ratio, the one measure that is moving. The binary milestone test (Fisher exact against newborns) is kept for **confirmation on fresh seeds ≥ 10,000** only.
4. **In parallel, a harness-only probe** of whether first-layer filters converge across fresh lives (CKA). It is the gate for the first "moving up" artifact (§4.5).

---

## 2. Reading order for a newcomer

Read the four notes' "what this suggests" sections first. Then read these, in order. Every URL below was fetched by the note authors, and those marked ✓ were re-checked for this synthesis (§8).

| # | Read | Why | Note section |
|---|---|---|---|
| 1 | Hinton & Nowlan 1987, *How Learning Can Guide Evolution* — https://www.cs.toronto.edu/~hinton/absps/evolution.htm | Six pages: why learning makes evolution work even though acquired traits are not inherited; the "?" allele | LLR L13, KUH K11 |
| 2 | Fernando et al. 2018, *Meta-Learning by the Baldwin Effect* — https://arxiv.org/abs/1806.07917 ✓ | The user's plan in miniature: evolve initial settings, learn in life, inherit only the settings; Baldwinian vs Lamarckian measured | LLR L12, KUH K12 |
| 3 | Frazier 2018, *A Tutorial on Bayesian Optimization* — https://arxiv.org/abs/1807.02811 ✓ | What a GP picker is and where it stops working (≈20 dims) | OLS O3 |
| 4 | Russo et al. 2018, *A Tutorial on Thompson Sampling*, §6.4 "Concurrence" — https://arxiv.org/abs/1707.02038 | Why randomised picks suit many tabs sharing one posterior | OLS O4 |
| 5 | Kandasamy et al. 2018, *Parallelised BO via Thompson Sampling* — https://arxiv.org/abs/1705.09236 ✓ | Asynchronous TS, the exact shape of our tabs | OLS O5 |
| 6 | Miconi et al. 2018, *Differentiable plasticity* — https://arxiv.org/abs/1804.02464 ✓ | The cleanest genome/lifetime split for plasticity: α, η inherited; the Hebbian trace is "purely a lifetime quantity" | LLR L7 |
| 7 | Najarro & Risi 2020, *Hebbian Plasticity in Random Networks* — https://arxiv.org/abs/2007.02686 | Only learning rules evolved, from pixels; also why per-synapse genomes are too big | LLR L9, DLA L11 |
| 8 | Lu et al. 2022, *Discovered Policy Optimisation* — https://arxiv.org/abs/2210.05639 ✓ | The template for our compute: a small, valid-by-construction learned piece inside PPO, ES on one machine, distilled to a formula | DLA L10 |
| 9 | Jackson et al. 2024, *Temporally-Aware RL Algorithms* — https://arxiv.org/abs/2402.05828 ✓ | Rules that read the life's clock; ES beats meta-gradients for whole-life schedules | DLA L13 |
| 10 | Alet et al. 2020, *Meta-learning curiosity algorithms* — https://arxiv.org/abs/2003.05325 | Closest published analogue: a small typed graph of modules, a surrogate choosing what to try, ~2 GPU-weeks | DLA L5 |
| 11 | Chen et al. 2023, *Symbolic Discovery of Optimization Algorithms* (Lion) — https://arxiv.org/abs/2302.06675 ✓ | Warm-start from the hand design; meta-overfitting and a held-out funnel | DLA L8 |
| 12 | McMahan et al. 2017 (FedAvg), Fig. 1 — https://arxiv.org/abs/1602.05629 ✓; then Li et al. 2016, *Convergent Learning* — https://arxiv.org/abs/1511.07543 ✓ | Why averaging differently seeded lives fails, and how to measure what lives do learn in common | KUH K1, K26 |

If you have time for one more, read Cawley & Talbot 2010 on selection bias (https://www.jmlr.org/papers/v11/cawley10a.html; OLS O21).

---

## 3. Concepts in plain words

- **Baldwinian vs Lamarckian inheritance.** Baldwinian: a child inherits only what its parent was *born with* (settings, rules, an initial state), and the parent's learning only decides which parents are selected. Lamarckian: a child also inherits what the parent *learned* (its trained weights). Baldwinian selection keeps pressure on learning well; Lamarckian copying tends to switch learning off [LLR L12]. Here, "Lamarckian" includes resuming or copying a checkpoint, and it is always a labelled condition (invariant 1).
- **Meta-gradient vs black-box outer loop.** A meta-gradient differentiates the final score with respect to the learning rule, through the inner learning steps. It needs the whole life in one differentiable graph, and in practice it only reaches a short window [LLR L4; DLA L2, L13]. A black-box outer loop (GP, ES, a GA) just tries settings, watches the score, and tries better ones. Tabs can only do the latter.
- **GP + Thompson sampling.** A GP is a smooth guess at "score as a function of settings", with error bars that shrink near tried points. Thompson sampling draws one plausible score surface from the GP and runs its best point. Different tabs draw different surfaces, so they spread out by themselves, and good regions get tried more often over time [OLS O2, O4, O5].
- **Permutation symmetry.** Swapping two hidden units (and their weights) gives the same network. Two independently seeded networks can learn the same features in a different order, so averaging them unit by unit mixes unrelated features. Aligning units first (Git Re-Basin, matching) can fix this, but only for wide, well-trained networks [KUH K3, K4, K26].
- **Distillation.** Pass on *predictions* rather than weights: a student trains to match a teacher's outputs on shared inputs. It needs no shared initialisation or architecture [KUH K7–K10]. The targets are still learned competence, so a learner trained on them is inherited-labelled.
- **Plasticity rule.** A small formula for how a connection changes during life, for example w_eff = w + α·trace, with trace ← (1−η)·trace + η·pre·post, optionally gated by a "neuromodulator" signal M(t) such as surprise [LLR L7, L8]. The coefficients (α, η, the gate source) are genes. The trace is local.
- **Held-out ladder.** Successively harder tests that search never sees: fresh seeds of the same world, then a later world stage, then longer lives. A candidate must keep improving up the ladder, not just on what it was selected on [DLA L8 funnel, L3; OLS §9].
- **Multi-fidelity.** Use cheap, partial measurements (an early checkup) as noisy readings of the expensive one (the full-life score), and stop poor candidates early. It pays only if early scores predict late ones [OLS O12, O13]. Today they don't: early checkups are at chance [OLS §5].

---

## 4. The proposed representation: a "learning genome" (agent's proposal)

### 4.1 One chain, top to bottom

```
GP posterior over genomes   (meta-learner state; rebuilt hourly from merged logs; data)
   │  Thompson draw, seeded hash(learnerId, posteriorSha), logged
   ▼
genome g  (≈12–30 searched numbers + switches; the rest fixed at the promoted centre)
   │  interpreted by the learner as weights and switches of a fixed dataflow graph
   ▼
dataflow graph: signals → mixers → update targets → memory / replay policy
   │  drives within-life learning
   ▼
network weights, plastic traces, episodic memory, replay buffer   (local; never shared by default)
   ▼
actions, one step at a time
```

This is the user's "one statistical representation". Each level is a distribution over, or a parameter of, the level below. The **promoted genome** (the centre that new searches start from and that confirmations compare against) is inheritance: it lives in `knowledge/` and counts toward `knowledgeSha`. The **posterior** is data, like `summary.json`, and stays out of `knowledgeSha` so cohorts stay stable [OLS §10.5; V1 §5.1].

### 4.2 The dataflow view

The learner becomes an interpreter of a small, declarative graph. The graph's *vocabulary* (which signals and targets exist) is code, changed only by people or agents, with a `schemaVersion` bump (invariant 5). Its *weights and switches* are the genome.

- **Signals** are per-step scalars the learner computes from its own predictions, never from pixels directly or from privileged state. This follows the domain-free-interface lesson [DLA L2, L3, L5] and invariant 2. Each is normalised by its running RMS [V2 §4.5].
  - s_lp: relative pixel learning progress (today's curiosity reward)
  - s_lat: latent learning progress [V2 §4.2]
  - s_ctrl: controllability c_t [V2 §4.1]
  - s_nov: episodic novelty ν_t [V2 §4.3]
  - s_sur: surprise, L_before / EMA(L)
  - s_clk: the life's clock, log₂(1 + CU since birth) / 11, so "forever" lives still have a defined clock [DLA L13, §"What this suggests" 4]
- **Mixers.** Each update target reads a weighted sum of a few allowed signals. Optionally, each weight has a clock slope: w(t) = w₀ + w₁·s_clk.
- **Update targets.**
  - **T1 reward** to the actor-critic, r = Σ wᵢ·sᵢ (today: curiosityWeight·s_lp).
  - **T2 replay priority** at insert: sleep samples transitions ∝ exp(β·p), with p = Σ vᵢ·sᵢ (today: β = 0, so uniform).
  - **T3 plasticity modulator** M(t) for fast-weight sites (today: α = 0, so off).
  - **T4 online learning-rate multiplier**, exp(u·s_clk) clamped to [0.25, 4] (today: u = 0, so ×1) **[guess at the clamp]**.
- **Memory and replay policy.** bufferSize, replayPerSleep, the sleep sequence length k, the episodic memory size M, and an eviction switch (FIFO or priority).

**Every point is a valid learner.** Weights are bounded and clamped. A gene at 0 must be bit-exact to the code without that path, as [V2 §4.5] and decision 0015 already require. This is the LPO lesson: a valid-by-construction space wastes no lives on broken learners [DLA L10].

### 4.3 Sections, genes and owners

The levels follow [SYN §3.1]. "Centre" is today's learner (`DEFAULT_SETTINGS` in `learner.ts`; `GAMMA` = 0.9 is a constant there today). "Owner" is the search method, following [OLS §10.3].

| level → section | genes (type, count) | centre = today | owner | inherited / local |
|---|---|---|---|---|
| **Working state** (GRU core) | γ (cont), GAE λ (cont), segment n (disc: 1/16), sleep TBPTT k (disc: 1/8) | γ 0.9, n 1, k 1 | GP for γ, λ; n and k as one switch "credit on/off" (bandit or race) | genes inherited; hidden state h local |
| **Episodic memory** | bufferSize, replayPerSleep (cont, log scale); replay mixer v over {s_sur, s_nov, s_ctrl} and β (4 cont); memory M (disc) | 2000, 64; β 0 (uniform) | GP; replay mixer enters in phase B | genes inherited; buffer and memory contents local |
| **Fast plastic state** | per-site switch, 3 states {off, on, learnable} [LLR L13]; shared α scale, η (2 cont); modulator source (cat: none/s_sur/s_lp/s_ctrl/s_nov) | all off (α = 0) | sites and source: bandit (CoCaBO pattern [OLS O10]); α, η: GP | rule inherited; traces local [LLR L7] |
| **Slow parameters** | lr, headLrScale, residualDecay, entropyWeight, exploreEpsilon (5 cont); λ_lat, target EMA τ (2 cont) | 5e-4, 0.1, 1e-3, 0.01, 0.1; λ_lat 0 | GP | genes inherited; weights local |
| **Learning policies** | reward mixer w over {s_lp, s_lat, s_ctrl, s_nov} (4 cont) + clock slopes (≤ 4 cont); lr clock slope u (1 cont); controllability gate form (cat: gated/additive) | w = (1, 0, 0, 0); slopes 0 | weights: GP; slopes: phase B, low-dim CMA-ES block [OLS O17, O19]; gate form: race | inherited |
| **Curriculum** (harness, a recorded prior; not the learner) | 5 caregiver settings (cont) | `caregiver-prior.json` | GP, as now | inherited prior, logged |
| **Shared inheritance** | artifact switches, e.g. `filters: none / inherited / matched-control` | none | **not searched**: assigned by pre-declared condition shares | the artifact is inherited and labelled |

**Dimension budget, with the numbers from the notes:**
- **Phase A, the first version: ≤ 17 searched continuous genes** (the 8 learner settings, γ, λ_lat, w_ctrl, w_nov and the 5 caregiver genes; fewer if some are fixed) **plus 1–2 switches**. This sits just inside "less than 20 dimensions" [OLS O3]. BOHB's d + 1 rule means about 15–30 random lives first [OLS O13].
- **Phase B: up to ~30 dimensions,** only with a sparsity prior (SAASBO, which handles "hundreds of variables" in a few hundred queries if few matter [OLS O9]). Small numeric blocks (the clock slopes, ≤ 10 numbers) are searched by noisy CMA-ES, at about 300 lives each [OLS §7].
- **Never searched directly:** anything weight-shaped (≥ 64 numbers costs about 13,000 lives [OLS §7]), per-synapse rules [LLR L9], or a GRU meta-trained as the algorithm (RL², which puts competence in the genome [LLR L14]).
- **Ranges.** `settings-space.json` ranges are narrow today (lr 4e-4 to 6e-4). Searching means widening them, on a log scale, around the centre. That widening is a design choice to record.

### 4.4 Inherited vs local, in one line each

- **Inherited (GitHub, through the gate):** the promoted genome, the graph vocabulary (code), curriculum priors, and labelled artifacts.
- **Data, not inheritance:** merged logs, `summary.json`, and the posterior.
- **Local (never leaves the tab unless a labelled artifact is built from it offline):** weights, traces, memories, replay, normaliser state.

Decision 0004's live peer layer stays out of scope here.

### 4.5 The "moving up" path

There are two kinds of moving up.

**(a) Genes move into the core.** When the posterior shows a gene does not matter (a long length-scale under SAASBO), or that its optimum is stable across schema refits, it is fixed at its posterior mean in the promoted genome and dropped from search. That frees dimensions for new genes. This is Hinton & Nowlan's "little pressure to fix the last few genes" turned into a rule [LLR L13] (agent's call).

**(b) Learned material moves into a shared artifact.** The first candidate comes from [KUH §"Concrete first candidate"]: the **first-layer conv filters**, periphery 5×5×3→16 and fovea 3×3×3→16.
- Size: **1,664 parameters, about 6.7 KB** at GPU widths (832 at CPU widths; re-measured for this note with `.local/paramcount.ts`).
- Built by the harness offline, never by the learner:
  - **Step 0, a probe.** Run about 10 fresh lives to a fixed CU, then compute first-layer CKA and one-to-one unit matching on a fixed probe set of frames [KUH K26, K27]. If the filters do not converge across lives, stop; that is a finding.
  - **Generation 0.** Align output channels by Hungarian matching, then average. With RGB input, only the output permutation has to be solved [KUH K4, K6].
  - **Later generations.** A Reptile-style step toward the mean end state of lives that *started from* the artifact; no alignment is needed because they share a start [KUH K14, K1].
  - **Admission.** A new version is admitted only if a protected held-out check improves (the greedy-soup rule) [KUH K5].
- **Measured as three pre-declared arms on the same sampler and widths:**
  - **F (fresh):** all weights random. This is the baseline.
  - **I (inherited):** the filters are loaded and stay trainable.
  - **C (matched control):** random filters with matched per-filter norms and biases.
  - **Primary:** CU to the M1 bar, and the held-out ratio at 512 and 1,024 CU, on world variants no contributing life saw.
  - **Also required:** I must not be worse than F at a long budget, because warm starts can end worse [KUH K16]. Its learning slope on a novel room must be checked, because inherited weights can carry lost plasticity [KUH K17].
  - **Cost:** the population CU spent building the artifact is reported beside the gain (invariant 5).
- **Keeping inherited lives labelled (invariant 1).**
  - Every log carries `inheritance: {filters: "none" | "<sha>" | "control:<sha>"}`, and the artifact is part of `knowledgeSha`.
  - The GP treats the inheritance condition as a fixed context, never as a searched gene, and confirmation tests never pool conditions.
  - A minimum share of fresh lives is held permanently (say ≥ 40 % [guess]), so the fresh condition never disappears.
- **If it fails,** the next candidates are a distilled frame-and-target set (no shared init; works across widths) and then a shared replay or curriculum file [KUH K7, K10, K23, K28].

---

## 5. The meta-learner loop, end to end

1. **Birth.** The tab reads `posterior.json` (its hash is `posteriorSha`) and the promoted genome.
   - It picks a **method**:
     - a confirmation life, if the gate has posted a confirmation queue;
     - a uniform-random genome with probability 1/5, forever [OLS §10.1];
     - otherwise a Thompson draw.
   - The draw is seeded with `hash(learnerId, posteriorSha)`, so it is reproducible.
   - The inheritance condition is assigned by its pre-declared shares.
   - The tab logs the method, seed, `posteriorSha`, `knowledgeSha`, the genome, the inheritance labels and `hostSig`.
2. **Life.** The learner runs the dataflow graph with that genome. The **search segment is the first 2,048 CU**: only checkups up to there count as this life's fitness. The life may go on afterwards (0014), but later data is a separate, labelled longevity reading [OLS §0].
3. **Logs.** The tab commits logs through the PR gate as now [V1 §9].
4. **Hourly gate.**
   - Merge, then rebuild `summary.json`.
   - **Refit the GP deterministically:** rows sorted by `(learnerId, seq)`, hyperparameters fitted from a fixed seed, per-dimension length scales, a fitted noise term [OLS O7].
   - Write `posterior.json` with its content hash. A CI fixture checks that the bytes reproduce [OLS §10.5].
   - Rows from confirmation seeds and from other `schemaVersion`s or inheritance conditions are excluded.
5. **Promotion rule (protected; the rule is code-reviewed, never searched).**
   - **Trigger:** the posterior mean of the best genome beats the promoted centre on the search fitness, with posterior probability ≥ 0.95 [guess].
   - **Confirmation:** the gate posts a confirmation queue of fresh learner ids, seeds ≥ 10,000 declared in advance, N lives per arm (candidate, current centre, newborn). New tabs serve the queue first.
   - **Pass:** the candidate beats newborns on at least one of M2–M6 (one-sided Fisher exact, Holm, family α 0.05, sustained ≥ 2 of the last 3 levels [V2 §6.3]). It also beats the current centre on the search fitness by its pre-declared effect size, measured on these fresh lives, never on the search lives [OLS O21].
   - **On a pass:** the gate opens a knowledge PR that updates `knowledge/genome.json`, which gives a new `knowledgeSha` and a new cohort. On a fail, the result is logged and kept (invariant 7).
   - Any public "best" is shown Ladder-style, only on significant improvement [OLS O23].
6. **Knowledge.** The promoted genome is the new centre. A change to the graph vocabulary or the learner code bumps `schemaVersion` and starts a fresh fit, because learned settings can overfit implementation details [DLA L10, §"What this suggests" 9].

**Fitness.**
- **Search fitness (continuous; to be fixed by decision before search starts):** for each of M2–M6, log(measure / newborn median), clipped to ±2, averaged over the 512, 1,024 and 2,048 CU levels. Add −log(M1 held-out ratio). Rank-transform across rows before fitting [DLA L4]. Score candidates on **shared checkup-variant seeds**, so comparisons are blocked on the same variants [OLS O20]. This needs a shared seed set for search lives, unlike today's per-learner `hash(learnerId, k, milestoneId)`.
- **Confirmation fitness:** binary milestones against newborns, as above.
- **Later:** a Baldwinian "learning cost" fitness, CU-to-criterion on a held-out stage, once any criterion is reachable. That is the research question itself [KUH §"What this suggests" 4].

**Budget (agent's estimates, from [OLS §0]; one tab per machine; ~30 min per life including checkups):**

| machines | lives/week overnight (8 h) | always on (24 h) | what that buys |
|---|---|---|---|
| 1 | ~110 | ~340 | Phase A only. About 30 random lives, then TS. Beating the centre on the continuous fitness takes **~2–5 weeks**. One large-effect confirmation (~26/arm) takes ~1 week. The filter probe (10 lives) fits in one night. |
| 3 | ~340 | ~1,000 | Phase A in ~1–2 weeks. Room for one race of 3–4 v2 switch arms at ~12–24 lives each. Or the F/I/C filter experiment at ~30 lives per arm. |
| 10 | ~1,100 | ~3,400 | Phase B: SAASBO over ~30 dimensions, one 10-number CMA block (~300 lives), modest-effect confirmations (~216/arm), and the F/I/C experiment, all in parallel. TuRBO becomes plausible past ~1,000 lives per schema [OLS O8]. |

With 10 machines, about 10–20 lives are always pending [OLS §0]; asynchronous TS handles that without fantasies [OLS O5]. Headless batch lives on the user's machines [V2 §6.2] would multiply these numbers if they count as search lives (open question 5).

---

## 6. Risks and open questions, each with a proposed default

1. **The search fitness composite.** It defines success, so it must be fixed before search starts (invariant 7). *Default:* the composite of §5, recorded as a decision. Revisions bump the fitness version and restart the fit.
2. **A flat landscape.** If the learner learns nothing, the GP fits noise. *Default:* after the 30 random lives, the gate runs **10 repeat lives of the centre and 10 of a random far point**. If the fitness difference is not distinguishable from repeat noise, do not add genes. Spend effort on the v2 mechanisms (as genes) instead. Report this either way.
3. **Do the v2 mechanisms become genes rather than a fixed redesign?** *Default:* yes. They are built with centre = off, so the v1 golden test still holds [V2 §6.1]. The pre-registered v2 comparison runs as a **race** over those switches on shared seeds [OLS O20]. The GP owns their continuous weights.
4. **Endless lives vs one data point.** *Default:* only the first 2,048 CU counts as search fitness. The promoted genome alone gets a periodic long-life check (for example at 8,192 CU [guess]), for the horizon failure [LLR L2, L10; DLA L4].
5. **Do headless batch lives count?** *Default:* yes, as search lives with `hostSig` marking them. They must use the same code, widths and checkups. GPU and CPU widths are fitted separately, because layer shapes differ [KUH §"Measured"].
6. **Meta-overfitting to the dev seeds and the one world.** *Default:* a held-out ladder. Search uses dev seeds; confirmation uses fresh seeds ≥ 10,000. A later world stage is a second rung once it exists. `milestoneSuiteVersion` stays fixed [OLS O24]. Which worlds a genome is scored on is a recorded prior [DLA L6, invariant 2].
7. **Gaming learner-side signals.** Reward-mixer genes can find settings that farm a proxy [OLS O25]. *Default:* fitness only ever comes from frozen harness checkups. Genes can never touch checkup code, variants or thresholds (0016).
8. **Lamarckian channels by accident.** Forever lives, checkpoint resume, PBT and inherited filters are all Lamarckian. *Default:* no PBT [OLS O15]. Every life carries its inheritance labels. A permanent fresh share is kept (§4.5).
9. **Throughput.** Plastic sites and mixers add GPU work. *Default:* each new graph path must pass the existing "no new serial readback" rule and a steps/s probe (≥ 24 steps/s [V2 §2]) before it becomes a gene.
10. **Is weight-shaped inheritance acceptable at all?** The user said "fine with no or limited weights in GH" [V1 §5.1]. *Default:* the 6.7 KB filter bank only, after the CKA probe, always alongside F and C arms.
11. **Automatic promotion.** *Default:* automatic within the declared rule (§5.5), with an auto-written record per promotion. A change to the rule itself needs the user.

---

## 7. Where a coding agent could plug in later (brief)

- **As a mutation operator on the graph vocabulary.** It proposes a new signal, target or edge (for example "replay priority reads s_clk"). The candidate enters as a new gene at centre = off, in a new `schemaVersion`, and the same loop evaluates it. That is AutoML-Zero or Lion's program search, with an agent proposing instead of random edits. Those searches cost 20,000 to billions of evaluations [DLA L1, L8, L9], so the agent's value is in proposing few, good candidates.
- **As a distiller.** It turns a found mixer or plasticity setting into a readable rule, as LPO became DPO [DLA L10].
- **As an analyst.** It reads the posterior, flags genes to freeze (§4.5a), and writes up results.
- **The fence (invariants 5 and 7).** Agent changes are design improvements, labelled with their compute cost. They never edit the fitness, the checkups, the promotion rule or the confirmation seeds.

---

## 8. Verification log (spot-check for this synthesis)

**Method.** Each source PDF was downloaded on 2026-10-22 into `.local/spotcheck/pdfs/` and converted with `pdftotext`. Line breaks and hyphenation were joined, and each quote was searched as an exact, case-insensitive string. The Shahriari et al. text is the author copy cited by OLS. "Found" means the exact words occur in the source.

| # | Note | Quote (as in the note) | Source | Found | Location |
|---|---|---|---|---|---|
| 1 | OLS O2 | "the randomization in Thompson sampling makes it particularly appropriate for batch or delayed feedback settings where many selections … are based on the identical posterior" | Shahriari et al. 2016, author PDF | yes | §II.A, the fourth reason listed |
| 2 | OLS O5 | "making n evaluations distributed among M workers is essentially equivalent to performing n evaluations in sequence" | arXiv 1705.09236 | yes | abstract |
| 3 | OLS O5 | "about the same or slightly worse than asyTS" (experiments section) | arXiv 1705.09236 | **wording differs** | The source reads "asyHTS performs either about the same as or slightly worse than asyTS on many problems we consider". The meaning is the same, but it sits in the method section (§3), not the experiments |
| 4 | OLS O3 | "continuous domains of less than 20 dimensions" / "tolerates stochastic noise in function evaluations" | arXiv 1807.02811 | yes | abstract |
| 5 | LLR L12 / KUH K12 | "The Baldwin effect is superior to Lamarckian learning when the distribution of tasks is broad or quickly changing, whereas Lamarckian learning is superior when the task distribution is narrow" | arXiv 1806.07917 | yes | concluding section. Also checked: "a very clear loser", and learning rates of "0.005 to 0.01" vs "0.00001" (results) |
| 6 | LLR L7 | "it is purely a lifetime quantity" … "structural parameters of the network that are conserved across lifetimes" | arXiv 1804.02464 | yes | §2 |
| 7 | LLR L10 | "simple weight-sharing and sparsity in an NN is sufficient to express powerful learning algorithms" | arXiv 2012.14905 | yes | abstract |
| 8 | DLA L10 | "Meta-training was only done on a single machine with 4 V100 GPU's with synchronous updates"; population 32; 672 generations; no biases | arXiv 2210.05639 | yes | App. A.2 and Table 1. The text says "approximately 700 outer-updates", Table 1 says 672. Biases are excluded so the network satisfies the drift conditions (§4) |
| 9 | DLA L13 | "Commonly used meta-gradient approaches fail to discover such adaptive objective functions while evolution strategies discover highly dynamic learning rules" | arXiv 2402.05828 | yes | abstract |
| 10 | DLA L8 | "the search fitness keeps growing, but the meta-validation metric declines"; "∼3K TPU V2 days"; "100 TPU V2 chips and runs for ∼72h" | arXiv 2302.06675 | yes | §2.3; §2.2 |
| 11 | KUH K1 | "With shared initialization, averaging the models produces a significant reduction in the loss"; "initialized using different random seeds" | arXiv 1602.05629 | yes | Figure 1 caption and text, §2 |
| 12 | KUH K4 | "relatively thin models do not seem to obey linear mode connectivity"; "1×-sized models did not seem to exhibit linear mode connectivity"; "we were unable to uncover it early in training" | arXiv 2209.04836 | yes | §5.2–5.3 area |
| 13 | KUH K26 | "some features are learned reliably in multiple networks, yet other features are not consistently learned"; "for many units a one-to-one assignment is possible" | arXiv 1511.07543 | yes | abstract; §3.1 |

That is 13 checks across 12 sources: 3–4 per note. **All found, with one wording mismatch (#3).**

**Other mismatches noticed while reading (not quotes; reported, not fixed):**
- **Parameter count.** LLR (intro, question 2) says the network has "about 1.1 M parameters", and [V2 §8] uses ~1.1 M as well. The measured count at GPU widths is **2,185,769**: re-run here with `.local/paramcount.ts`, matching KUH's table. The encoders alone are 1,338,304. The LLR figure, and any v2 cost estimates built on it, are low by about 2×.
- **`consolidation-prior.json` does not exist yet.** OLS §10.3 and KUH K22 refer to it as existing. `knowledge/` holds only `settings-space.json` and `caregiver-prior.json`. The file is specified in [V1 §9.1] but not created; condition C is not built [V2 §7].
- **The picker is not built.** OLS describes the v1 picker as GP-UCB [V1 §9.2]. That is the spec. No picker code exists in `packages/harness/src` today. This synthesis's TS change therefore costs nothing to adopt.
