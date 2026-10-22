# Outer-loop search under expensive, noisy, asynchronous evaluations

**Literature note | 22 October 2026 | agent-written; every source fetched and checked on this date**

Question: how should the meta-learner choose what learning-workflow variant each new tab runs? One evaluation is one ~30-minute life. Its fitness is very noisy: a newborn scores ≥ 1 milestone by luck on ~42 % of learners (0016). Few tabs run at once, asynchronously, and results reach `main` hourly through the gate. The space mixes continuous settings, discrete switches and possibly small weight matrices. The user's default is a Gaussian process (GP) over the settings that define a learning behaviour, drawing better samples over time, with no coding agent in the loop. The v1 spec already has this as the picker: GP plus UCB, fitted by each tab at birth, uniform until 8 usable rows ([V1 §9.2]). It mirrors `markwhiting/life` ([prior-art-life]).

Local references: [V1] `docs/specs/2026-10-19-blanktab-v1-design.md`; [V2] `docs/specs/2026-10-22-learner-v2-capabilities-design.md`; [prior-art-life] `docs/research/2026-10-19-prior-art-life.md`; decisions 0014 (forever lives), 0016 (milestones, newborn base rate); `knowledge/settings-space.json` (15 named settings in schemaVersion 1, one of them fixed).

How sources were checked: each source's abstract page (arXiv, PMLR, NeurIPS, JMLR, Crossref) was fetched for title, authors, year and venue. Where a claim comes from inside a paper, the PDF was fetched and its text extracted. Quotes are exact. Section labels are given where the extracted text showed them, and otherwise the location is described in words.

---

## 0. Our budget, worked out first (agent's estimates)

Everything below depends on how many evaluations we can afford.

- **One evaluation = one life to 2,048 CU.** That is about 12–16 min of training plus about 11 min of checkups, so ~30 min ([V2 §6], an estimate there too).
  - Lives are endless now (0014). A search evaluation therefore needs a rule that it ends at 2,048 CU, or that only the segment up to that point counts. Otherwise one tab yields one data point forever.
- **Tab-lives per week [guess: one tab per machine, because the GPU is the bottleneck].**

| machines | overnight only (8 h/day) | always on (24 h/day) |
|---|---|---|
| 1 | ~110 | ~340 |
| 3 | ~340 | ~1,000 |
| 10 | ~1,100 | ~3,400 |

- **Results are delayed.** A pick made at birth is informed only by logs merged at least one gate cycle earlier. With 10 machines, about 10–20 lives are always "pending", running but not yet in the data.
- **The noise floor, using the 0016 newborn rates** (M2 3/60, M3 8/60, M4 11/60, M5 2/60, M6 3/60).
  - Telling one setting from another on a single binary milestone (one-sided α = 0.05, power 0.8, normal approximation, computed for this note):
    - a large effect, 5 % → 30 % pass rate, needs **~26 lives per arm**;
    - a modest effect, 18 % → 28 % (M4-like), needs **~216 lives per arm**.
  - So a binary "milestones passed" fitness can only separate large effects within a week on one machine. That shapes every recommendation below.
- **Dimensionality.** The current space has 14 free continuous settings, plus `condition` and whatever v2 ablation switches become genes.

---

## 1. GP-UCB (the v1 picker's rule)

**[O1] Srinivas, Krause, Kakade, Seeger. "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design." ICML 2010** (Proceedings of the 27th ICML, Haifa). The PDF's first-page footer reads "Appearing in Proceedings of the 27th International Conference on Machine Learning, Haifa, Israel, 2010". arXiv: https://arxiv.org/abs/0912.3995 ; PDF: https://icml.cc/Conferences/2010/papers/422.pdf

- **Method.** Model the unknown fitness as a GP. Pick the point with the highest optimistic estimate, mean plus a multiple of the standard deviation.
  - §2, eq. 6: "xt = argmax µt−1(x) + βt^1/2 σt−1(x)".
  - The paper notes that pure exploitation is "too greedy too soon and tends to get stuck in shallow local optima" (same section).
- **Assumptions.** The problem is sequential, one point at a time. The fitness is either a GP sample or has low RKHS norm, with noisy observations. The abstract says: "optimizing an unknown, noisy function that is expensive to evaluate".
- **Sample efficiency.** The result is a regret bound, not a number of evaluations. For the squared-exponential kernel it grows like "T (log T)^(d+1)" (Fig. 1), which is why the abstract says the bounds have "surprisingly weak dependence on the dimensionality" in some cases.
- **When it fails.** It has no batch or asynchronous mechanism.
  - UCB is deterministic. Every tab that fits the same posterior picks the same point.
  - The posterior changes only hourly in our setup, so concurrent newborns would duplicate one another. Our inference from the method; no paper tested this for our case.
- **Relevance.** The v1 picker as specified (UCB, each tab fitting at birth on the same `summary.json`) **would send every tab born within the same gate hour to the same settings.** That wastes the parallel lives. The fix is a randomised acquisition (§3).

## 2. Bayesian-optimisation reviews

**[O2] Shahriari, Swersky, Wang, Adams, de Freitas. "Taking the Human Out of the Loop: A Review of Bayesian Optimization." Proceedings of the IEEE 104(1):148–175, 2016.** Venue checked via Crossref (https://api.crossref.org/works/10.1109/JPROC.2015.2494218); text read from the author copy at https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf

- **Method.** A review of surrogate models and acquisition functions (Thompson sampling, PI, EI, UCB, entropy search, portfolios) and of high-dimensional variants.
- **Key claim for us.** §II.A gives four reasons to prefer Thompson sampling. One is that "the randomization in Thompson sampling makes it particularly appropriate for batch or delayed feedback settings where many selections … are based on the identical posterior". That is exactly our hourly-posterior situation.
- **High dimensions (§ "High-dimensional problems").** The review discusses random embeddings, which optimise "g(z) = f(Az) in the lower dimensional space" when only a few dimensions matter.

**[O3] Frazier. "A Tutorial on Bayesian Optimization." arXiv:1807.02811, 2018.** https://arxiv.org/abs/1807.02811

- **Scope** (abstract): BO "is best-suited for optimization over continuous domains of less than 20 dimensions, and tolerates stochastic noise in function evaluations". It targets objectives that take "a long time (minutes or hours) to evaluate".
- **Relevance.** Our 14 continuous settings sit inside this envelope. The envelope does not stretch to weight matrices (hundreds of numbers).

## 3. Thompson sampling, batch and asynchronous BO

**[O4] Russo, Van Roy, Kazerouni, Osband, Wen. "A Tutorial on Thompson Sampling." Foundations and Trends in Machine Learning 11(1):1–96, 2018.** https://arxiv.org/abs/1707.02038

- **Method.** Sample a hypothesis from the posterior, then act as if it were true. The abstract says it balances "exploiting what is known to maximize immediate performance" against exploration.
- **Concurrency (§6.4 "Concurrence").** "TS naturally suits this concurrent mode of operation". Multiple independent samples are drawn from the same posterior, one per agent.
  - The cost is stated too: regret per action decays more slowly "because the posterior distribution is updated only after K concurrent actions are completed".
- **Why randomisation matters (§8.1).** A deterministic stationary strategy can get stuck. To get sublinear regret, "a stationary strategy must randomize actions". The alternative is a deterministic but nonstationary rule, "as is the case with typical UCB algorithms".
- **Limitations (§8.2).** TS is "a poor choice for problems where learning does not require active exploration". That does not apply to us.

**[O5] Kandasamy, Krishnamurthy, Schneider, Póczos. "Parallelised Bayesian Optimisation via Thompson Sampling." AISTATS 2018, PMLR 84:133–142.** https://proceedings.mlr.press/v84/kandasamy18a.html (arXiv title: "Asynchronous Parallel Bayesian Optimisation via Thompson Sampling", https://arxiv.org/abs/1705.09236)

- **Method (§3, Algorithm 2 "asyTS").** "when a worker finishes an evaluation, we update the posterior with the query-feedback pair, sample g from the updated posterior, and re-deploy the worker".
  - Pending evaluations need no special handling.
  - Variants that add "hallucinated" observations perform "either about the same as or slightly worse than asyTS on many problems we consider" (§3).
- **Claims** (abstract):
  - "making n evaluations distributed among M workers is essentially equivalent to performing n evaluations in sequence";
  - "asynchronously parallel TS achieves asymptotically lower regret than both the synchronous and sequential versions".
- **Relevance.** This is the cleanest match to our setup: few workers, asynchronous, variable life length. The theory says parallel tabs lose little against running the same number of lives in sequence.

**[O6] Snoek, Larochelle, Adams. "Practical Bayesian Optimization of Machine Learning Algorithms." NIPS 2012 (Advances in NIPS 25).** https://papers.nips.cc/paper/2012/hash/05311655a15b75fab86956663e1819cd-Abstract.html ; https://arxiv.org/abs/1206.2944

- **The alternative to TS for pending work.** Integrate the acquisition over "fantasies" for pending evaluations, using "Monte Carlo estimates of the acquisiton function under different possible results from pending function evaluations" (the parallel-experiments section).
- **Cost.** The method also accounts for "the variable cost (duration) of learning experiments" (abstract).
- **Relevance.** This is more machinery than TS. It is worth it only if TS's duplicate-avoidance by chance proves too weak at our batch size.

**[O7] Letham, Karrer, Ottoni, Bakshy. "Constrained Bayesian Optimization with Noisy Experiments." Bayesian Analysis 14(2):495–519, 2019** (venue via Crossref). https://arxiv.org/abs/1706.07094

- **Noise.** Existing methods "degrade in performance when the noise level is high" (abstract).
- **Method.** EI is derived "under greedy batch optimization with noisy observations", optimised by quasi-Monte Carlo.
- **Relevance.** This is the reference model when the per-life fitness has high, measured noise. Estimate the noise level rather than assuming it is zero, and judge the incumbent by its posterior mean, not its luckiest life.

## 4. High-dimensional and mixed or categorical BO

**[O8] Eriksson, Pearce, Gardner, Turner, Poloczek. "Scalable Global Optimization via Local Bayesian Optimization" (TuRBO). NeurIPS 2019.** https://arxiv.org/abs/1910.01739

- **Method (§2).** Run local GPs inside trust regions. "After τsucc consecutive successes, we double the size of the TR … After τfail consecutive failures, we halve" it. Batches are chosen by Thompson sampling. Under noise, the centre is "the observation with the smallest posterior mean".
- **Budgets used (§3).** These are large: "10K evaluations and batch size of q = 50"; 200 steps at q = 100; a 200-D synthetic problem.
- **When it fails for us.** It was designed for "several thousand observations". At ~100–3,000 lives per week it is a later tool, not a first one.

**[O9] Eriksson, Jankowiak. "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces" (SAASBO). UAI 2021** (the arXiv page says "To appear in UAI 2021"). https://arxiv.org/abs/2103.00349

- **Method.** A GP with a strong sparsity prior on the length scales, fitted by Hamiltonian Monte Carlo (HMC). The model "quickly identifies the most relevant low-dimensional" subspace (§1).
- **Regime (§1).** "hundreds of variables" and "limited to a few hundred queries", with problems of "as many as 388 dimensions".
- **Cost.** A posterior sample costs "O(N^3 D)" (inference section). That is fine at N in the hundreds.
- **When it fails.** When many dimensions matter roughly equally, the sparsity assumption is wrong.
- **Relevance.** This is the best-matched GP if settings grow past ~15–20, as long as only a few of them matter. It is also a direct test of "which settings matter".

**[O10] Ru, Alvi, Nguyen, Osborne, Roberts. "Bayesian Optimisation over Multiple Continuous and Categorical Inputs" (CoCaBO). ICML 2020, PMLR 119:8276–8285.** https://proceedings.mlr.press/v119/ru20a.html ; https://arxiv.org/abs/1906.08878

- **Method** (abstract). It "combines the strengths of multi-armed bandits and Bayesian optimisation". A bandit picks the categorical values, and a GP with a mixed kernel optimises the continuous ones.
- **Relevance.** This is the template for `condition` and v2 ablation switches together with continuous settings.

**[O11] Baptista, Poloczek. "Bayesian Optimization of Combinatorial Structures" (BOCS). ICML 2018.** https://arxiv.org/abs/1806.08838

- **Method** (abstract). "an adaptive, scalable model that identifies useful combinatorial structure even when data is scarce". The acquisition "pioneers the use of semidefinite programming".
- **Relevance.** Only relevant if we end up with many binary switches, more than ~10. Not now.

## 5. Multi-fidelity and early stopping (checkup levels as fidelities)

**[O12] Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar. "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization." JMLR 18:1–52, 2018.** https://arxiv.org/abs/1603.06560

- **Method.** Successive halving: "throw out the worst half, and repeat until one configuration remains" (§3.1). Hyperband hedges over how aggressive to be by running several brackets. The last bracket "simply performs classical random search" (§3).
- **Efficiency claim.** "5× to 30× faster than popular Bayesian optimization algorithms" (§1).
- **When it fails (§3).** If configurations need many resources "before configurations can differentiate themselves … then it would be reasonable to work with a small number of configurations". In plain terms, early stopping pays only when early scores predict late ones.

**[O13] Falkner, Klein, Hutter. "BOHB: Robust and Efficient Hyperparameter Optimization at Scale." ICML 2018, PMLR 80:1437–1446.** https://proceedings.mlr.press/v80/falkner18a.html

- **Method (§4).**
  - Hyperband's budget schedule is kept, with model-guided proposals: a TPE-style kernel density estimator (KDE), which "easily supports mixed continuous" and discrete spaces.
  - The model needs "Nmin … set to d + 1" points.
  - It "always uses the model for the largest budget for which enough observations are available", so it can "overcome wrong conclusions drawn on smaller budgets".
  - A fraction ρ of proposals stays uniformly random, so "in the worst case (when the lower fidelities are misleading)" it is a bounded factor slower than random search.
- **Claim** (abstract): it "consistently outperforms both Bayesian optimization and Hyperband".

**[O14] Swersky, Snoek, Adams. "Freeze-Thaw Bayesian Optimization." arXiv:1406.3896, 2014** (arXiv preprint; no venue on the page). https://arxiv.org/abs/1406.3896

- **Method.** Model learning curves and "decide whether to pause training and start a new model, or resume the training of a previously-considered model" (abstract).
- **Assumption (§1).** The training loss "roughly follows an exponential decay towards an unknown final value".
- **Relevance.** Our lives checkpoint and are endless (0014), so pause and resume are mechanically possible. But milestone curves are step-like near chance, not exponential decays, so its curve prior does not fit our fitness. The held-out prediction ratio (M1) might fit it.

**Relevance of §5 to us.**
- Checkup levels 16 … 2,048 CU are natural fidelities.
- Live evidence shows checkups at newborn chance at every level so far ([V2 §2], citing growing-up-live). The low fidelities are therefore uninformative today.
- Before any early stopping, measure the rank correlation between level-k and 2,048-CU scores on merged logs. Use BOHB's safeguard: fit on the highest fidelity with enough rows, and keep a random fraction.

## 6. Population Based Training

**[O15] Jaderberg et al. "Population Based Training of Neural Networks." arXiv:1711.09846, 2017.** https://arxiv.org/abs/1711.09846

- **Method.**
  - Workers train in parallel. When a worker is "ready" (a minimum number of steps), it exploits by copying the weights of a better member (truncation selection, §4.1.1).
  - It then explores: "each hyperparameter independently is randomly perturbed by a factor of 1.2 or 0.8".
  - It is "decentralised and asynchronous" (§1), and it finds a "schedule of hyperparameter settings" (abstract).
- **Scale.** "population sizes between 10 and 80" (§4.1). On DeepMind Lab, UNREAL went "from 93% to 106% human performance" with 40 workers.
- **When it fails.** The paper calls itself "inherently greedy" (§1). It needs a population of tens of workers running at the same time.
- **Relevance.**
  - We have 1–10 concurrent tabs, below its range.
  - Exploit copies a checkpoint. That turns a fresh individual into an **inherited-checkpoint condition** (AGENTS invariant 1). It also mixes lifetime learning with design search (invariant 5).
  - Usable only as a separate, labelled track.

## 7. Evolution strategies (for weight-matrix genes)

**[O16] Salimans, Ho, Chen, Sidor, Sutskever. "Evolution Strategies as a Scalable Alternative to Reinforcement Learning." arXiv:1703.03864, 2017.** https://arxiv.org/abs/1703.03864

- **Method.** Perturb the parameters with Gaussian noise, score each perturbation by its return, and step along the score-weighted noise. It uses antithetic ("mirrored") sampling and fitness shaping (§2). Workers "only need to communicate scalars" (abstract).
- **Numbers (§1).** "using 1,440 workers … MuJoCo 3D humanoid task in under 10 minutes". On Atari it needed "between 3x and 10x as much data" as A3C.
- **When it fails for us.** Its strength is massive parallelism over cheap episodes. Ours are 30-minute lives, and we have ≤ 10 workers.

**[O17] Hansen. "The CMA Evolution Strategy: A Tutorial." arXiv:1604.00772, 2016 (revised 2023).** https://arxiv.org/abs/1604.00772

- **Method.** Sample a population from a multivariate normal. Adapt its mean, step size and full covariance from the ranked offspring.
- **Default population.** "4 + ⌊3 ln n⌋" (the default-parameters table).
- **Cost in dimensions.** The covariance learning rate is "cµ ≈ µeff /n²", so the "characteristic time horizon … is roughly n²/µeff" generations (rank-µ update section).
- **Noise.** Path-length step-size control "fails to adapt nearly optimal step-sizes on very noisy objective functions" (step-size section). Small populations converge faster; large ones avoid local optima.
- **Relevance, computed for this note.**
  - n = 64 (one small weight matrix) gives λ = 16. The horizon is about 64²/µeff, roughly 800 generations, so about 13k lives. That is out of reach.
  - n = 10 gives λ = 10 and a horizon of about 30 generations, so about 300 lives. That is feasible.
  - **Weight-matrix genes must be low-dimensional (≤ ~10–20 numbers) or parameterised compactly.**

## 8. Noisy fitness and racing

**[O18] Jin, Branke. "Evolutionary Optimization in Uncertain Environments—A Survey." IEEE Transactions on Evolutionary Computation 9(3):303–317, 2005** (venue via Crossref, 10.1109/TEVC.2005.846356). *Only the metadata was verified; no content claim is used.*

**[O19] Hansen, Niederberger, Guzzella, Koumoutsakos. "A Method for Handling Uncertainty in Evolutionary Optimization With an Application to Feedback Control of Combustion." IEEE TEC 13(1):180–197, 2009** (via Crossref, 10.1109/TEVC.2008.924423). The abstract was read at https://researchportal.ip-paris.fr/en/publications/a-method-for-handling-uncertainty-in-evolutionary-optimization-wi/

- **Method.** UH-CMA-ES "relies on the rank based selection operator" to detect when noise reorders candidates, and "requires only a small number of additional function evaluations" (abstract).
- **Relevance.** This is the noisy-CMA variant to use if §7's small weight genes are searched.

**[O20] Birattari, Stützle, Paquete, Varrentrapp. "A Racing Algorithm for Configuring Metaheuristics." GECCO 2002, pp. 11–18.**
- The venue was confirmed from the TU Darmstadt and Springer reference listings found by web search. The PDF's running header shows the GECCO track and page 11. PDF: https://cs.ubc.ca/labs/algorithms/Courses/CPSC536H-10/Materials/BirEtAl02.pdf
- **Method** (abstract). It "empirically evaluates a set of candidate configurations by discarding bad ones as soon as statistically sufficient evidence is gathered against them". The test is the Friedman test, "nonparametric and two-way", with problem instances as blocks (§3).
- **Relevance.**
  - Racing fits a *small, discrete* set of variants (v2 ablation arms) better than a GP does.
  - Blocking on instances maps onto **shared checkup-variant seeds**. Candidates would be compared on the same variants, which cancels variant difficulty out of the comparison.
  - Today variant seeds come from `hash(learnerId, k, milestoneId)` ([V1 §8.3]), so they differ per learner. Search runs would need a shared seed set to get this benefit.

## 9. Overfitting the outer loop; protected evaluation

**[O21] Cawley, Talbot. "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation." JMLR 11:2079–2107, 2010.** https://www.jmlr.org/papers/v11/cawley10a.html

- Abstract: a selection criterion with "non-negligible variance introduces the potential for over-fitting in model selection". The effect "can be surprisingly large", "often of comparable magnitude to differences in performance" between methods.
- **For us.** The best of many noisy lives is biased upward. Re-measure the winner on fresh lives.

**[O22] Dwork, Feldman, Hardt, Pitassi, Reingold, Roth. "Generalization in Adaptive Data Analysis and Holdout Reuse." arXiv:1506.02629, 2015.** https://arxiv.org/abs/1506.02629 ; companion paper "The reusable holdout: Preserving validity in adaptive data analysis", Science 349(6248):636–638, 2015 (via Crossref)

- Abstract: "Reusing a holdout set adaptively multiple times can easily lead to overfitting to the holdout set itself". The paper gives "a simple and practical method for reusing a holdout".

**[O23] Blum, Hardt. "The Ladder: A Reliable Leaderboard for Machine Learning Competitions." arXiv:1502.04585, 2015.** https://arxiv.org/abs/1502.04585

- Abstract: participants can "overfit to the holdout data that supports the leaderboard". The Ladder reports only significant improvements.
- **For us.** If the gate ever shows a running "best" score on held-out lives, show it Ladder-style, coarsened and only on improvement.

**[O24] Recht, Roelofs, Schmidt, Shankar. "Do ImageNet Classifiers Generalize to ImageNet?" ICML 2019, PMLR 97:5389–5400.** https://proceedings.mlr.press/v97/recht19a.html

- Abstract: "accuracy drops of 3% - 15% on CIFAR-10 and 11% - 14% on ImageNet" on fresh test sets. However, "the accuracy drops are not caused by adaptivity"; they come from distribution shift.
- **For us.** Fresh variants drawn from the *same* generator are the right held-out set. Generator drift between suite versions is a separate risk, so keep `milestoneSuiteVersion` fixed.

**[O25] Krakovna et al. "Specification gaming: the flip side of AI ingenuity." DeepMind blog, 21 April 2020.** https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/

- Definition: "a behaviour that satisfies the literal specification of an objective without achieving the intended outcome". The post collects about 60 examples.
- **For us.** An outer loop given settings that touch rewards (curiosity weight, controllability gate) can find a setting that games a learner-side proxy. The fitness must stay the frozen harness checkups, never a learner-side signal.

---

## 10. Recommendation (agent's reading, not a user decision)

**Keep the user's idea, a GP over learning-behaviour settings with no coding agent in the loop, with five changes.**

1. **Thompson sampling instead of UCB** [O4, O5, O2 §II.A].
   - Each newborn draws its own posterior sample, seeded by `hash(learnerId, summarySha)`, and runs at that sample's argmax. Tabs born in the same gate hour then diverge instead of duplicating.
   - Pending lives need no fantasies (asyTS performed as well as the hallucination variants) [O5].
   - Keep the v1 floors: ≥ 20 % share per condition, and uniform picks until enough rows exist. Use BOHB's rule of at least d + 1 rows [O13]; for d = 14 that is ~15–30 random lives. Keep a fixed random fraction (say 1 in 5) forever, as BOHB's ρ does.

2. **A continuous, lower-noise search fitness. Keep the binary milestone test for confirmation only.**
   - Search on the continuous checkup measures, normalised against newborn medians and averaged over the last 3 levels, plus M1's held-out ratio. Do not search on the pass count. At ~26–216 lives per comparison (§0), a pass-count fitness is hopeless for a GP.
   - Fit a noise term rather than assuming none, and report the incumbent by its posterior mean [O7].
   - Which composite to use is a design choice that needs a recorded decision. It must be fixed before search starts (invariant 7).

3. **Split the representation by gene type.**
   - **Continuous settings (≤ ~15):** GP with per-dimension length scales and TS (§1–3).
     - Grow to 20 or more only with a sparsity prior (SAASBO-style) [O9].
     - TuRBO is for later, past ~1,000 lives per schema [O8].
   - **Discrete switches (`condition`, v2 ablation arms; a handful):** a bandit over categories with the GP for the continuous rest (CoCaBO pattern) [O10].
     - For a fixed small set of arms that must be compared rather than optimised, race them with a Friedman test on shared variant seeds [O20].
     - BOCS only if switches exceed ~10 [O11].
   - **Weight-matrix genes:** not GP.
     - Use a low-dimensional parameterisation (≤ ~10–20 numbers, such as a few scalars or a rank-1 factor) searched by UH-CMA-ES [O17, O19]. A full 64-number matrix needs ~10⁴ lives (§7).
     - The `consolidation-prior.json` rule specified in the v1 spec (not yet implemented) (`prior ← prior + 0.3·(candidate − prior)`, [V1 §9.1]) is already a crude (1, λ)-ES-like step; keep it until a gene needs more.
     - Evolved weight-valued priors are population-learned values. Record them as an *inherited* condition, distinct from a fresh individual (invariant 1).
   - **PBT:** not now. It is below its population size, and it creates inherited-checkpoint lives [O15].

4. **Early checkup levels as fidelities: measure before use.**
   - Compute the rank correlation between each level k and the 2,048-CU fitness on merged logs.
   - Only if it is clearly positive, retire search lives that fall in the bottom half at a mid level, Hyperband-style. Always keep a full-length bracket [O12], and fit the GP on the highest level with enough rows [O13].
   - Today, with checkups at chance at all levels, that rule would discard by luck. So use full lives for now.

5. **Fit the posterior in the hourly gate; draw the samples in the tab.**
   - **The gate.** After merging, the gate deterministically rebuilds `summary.json` from `data/runs/`, as now. It adds the fitted GP posterior: training rows sorted by `(learnerId, seq)`, hyperparameters fitted from a fixed seed, and the result written with its own content hash.
     - This is data, not inheritance: like `summary.json`, it stays out of `knowledgeSha`, so cohorts stay stable ([V1 §6]).
     - Anyone can rerun the fit from the merged logs and get the same bytes. A CI test should check this on a fixture.
   - **The tab.** The tab draws its TS sample with the logged seed, so its pick is reproducible as well.
     - Log the method (random, TS or race), the seed, the posterior hash and the chosen settings.
   - **Bounds.** The gate already checks values against `settings-space.json` bounds ([V1 §10]).
   - **Schema changes.** A new `schemaVersion` starts a fresh fit, as specified.

**Keeping the fitness protected** (invariant 7; [O21–O25]).
- The search reads only checkups on **development seeds**.
- The confirmation run uses fresh learner ids and seeds ≥ 10,000, declared in advance ([V2 §6.2]). The gate excludes those rows from the search posterior.
- Report how many configurations the search tried. Re-measure the chosen setting on fresh lives, because the incumbent's own score is biased upward [O21].
- If a held-out "best" is ever shown on the page, report it Ladder-style, only on significant improvement [O23].
- Settings can never change checkup code, variant generation or thresholds. Those stay code-reviewed (0016).
- A GP-chosen improvement is population-level design search inside the harness. Label it as such, distinct from within-lifetime learning (invariant 5).

**Expected evaluations (agent's guess, to be replaced by measurement).**

| phase | lives | 1 machine, overnight | 10 machines, overnight |
|---|---|---|---|
| random initialisation (d = 14) | ~30 | ~2 days | <1 day |
| TS search until the posterior mean clearly beats the defaults on the continuous fitness | ~200–500 | ~2–5 weeks | ~2–4 days |
| confirmation of one chosen setting against newborns or the v1 arm, on binary milestones | ~25 per arm (large effect) to ~200 per arm (modest effect) | ~0.5–4 weeks | ~1–4 days |

- These lives also count toward the GP fit, so the phases overlap.
- With one machine, the outer loop is realistic for ~10–15 continuous settings and a few switches, not for weight matrices.

---

## Source status

**Verified (fetched and checked on 2026-10-22):** O1–O17 and O19–O25 as cited above.
- O18 (Jin & Branke 2005): metadata only; no content claim is used.
- For O20, the GECCO venue comes from secondary listings found by search plus the PDF's running header, not from a publisher page.

**Unverified or unused:**
- ASHA (asynchronous successive halving) and multi-fidelity GP methods such as BOCA: not fetched, so not cited.
- The irace package paper (López-Ibáñez et al. 2016): the publisher page returned 403, so it is not cited.
- The IEEE Xplore pages for O2 and O18 did not render. Crossref metadata was used instead.
- Claims about common random numbers (sharing seeds across arms) beyond Birattari's blocking design: no separate source checked. That part of §8 and §10 is the agent's reasoning.
