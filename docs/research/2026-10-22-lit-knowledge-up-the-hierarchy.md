# Literature note: moving learned knowledge up the hierarchy and across a population

**Date:** 2026-10-22. **Author:** research agent, on request. **Status:** literature note, not a decision.

**The question.** Many independent learners each acquire similar things in their lives, for example pixel and button detectors. Can that shared learning be moved "up" the hierarchy, out of individual lives and into an inherited artifact or into the algorithm itself, and aggregated through GitHub? The hierarchy is working state → episodic memory → fast plastic state → slow parameters → learning policies → shared inheritance [SYN §3.1]. The setting adds three constraints:
- Every learner is born from its own random seed.
- Learners run in browser tabs.
- Only small artifacts can be shared through the PR gate.

**Constraints from this repo that frame everything below:**
- **Invariant 1.** An inherited checkpoint or initialization is a different, labelled condition from a fresh individual.
- **Invariant 6.** GitHub carries selected, evaluated artifacts. It is not the learning loop.
- **Invariant 5.** If the harness or an offline job builds the shared artifact, that is population-level or design-level learning, with its own compute cost. It is not within-life learning.
- **The current spec.** "No weights are inherited", and the user's position is that they are "fine with no or limited weights in GH" [v1 spec §5.1]. Today `knowledge/` holds only `settings-space.json` (735 B) and `caregiver-prior.json` (151 B).
- **Decision 0004.** It allows a live peer layer and a PR layer, and requires that "a learner that has absorbed state from a peer is labelled as such".

**Measured, not looked up: how big the learner's pieces are.** I counted these by building the model on the CPU backend (`createModel` in `packages/learner/src/model.ts`, run via `npx tsx .local/paramcount.ts`, seed 1).

| Part (GPU widths 16/32/128/256) | Parameters | float32 size |
|---|---|---|
| Whole network | 2,185,769 | ~8.7 MB |
| Both encoders, including their linear projections | 1,338,304 | ~5.4 MB |
| All four conv layers (periphery and fovea, weights and biases) | 10,944 | ~44 KB |
| First conv layers only (periphery 5×5, fovea 3×3) | 1,664 | ~6.7 KB |

At CPU widths (8/16/64/128) the whole network is 680,137 parameters and the four conv layers are 3,168. **The layer shapes differ between GPU and CPU widths.** Any weight-shaped artifact therefore has to be keyed by width.

The "features that recognise pixels" (the conv filters) are tiny and fit in a GitHub file easily. The whole network does not fit under the 1 MB per-file log limit [v1 spec §9] without compression or quantisation.

---

## 1. Averaging weights across a population

### [K1] Federated averaging (FedAvg)
McMahan, Moore, Ramage, Hampson, Agüera y Arcas. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017 (PMLR/JMLR W&CP 54). https://arxiv.org/abs/1602.05629. Verified from the arXiv page and the PDF text.

- **What is transferred:** weights, or weight deltas. Clients train locally from a shared starting model, and the server averages the results. In the abstract's words, a shared model is learned by "aggregating locally-computed updates".
- **Conditions:**
  - **A shared initialization is essential.**
    - Figure 1 (§2) averages two MNIST models trained on different small datasets. In one case the two "were initialized using different random seeds"; in the other, "a shared seed was used".
    - Only the second case works: "With shared initialization, averaging the models produces a significant reduction in the loss".
    - With different seeds, they "see exactly this bad behavior".
  - Each FedAvg round restarts every client from the current shared model ("a shared starting model wt is used for each round of FedAvg"). Averaging happens every round, not after whole independent lives.
- **Evidence:** "a reduction in required communication rounds by 10-100x as compared to synchronized stochastic gradient descent", and the method is described as "robust to the unbalanced and non-IID data distributions" (abstract).
- **Limitations:**
  - Synchronous rounds.
  - It needs the same architecture and a shared start.
  - The Figure 1 experiment is MNIST-scale.
- **Relevance to us:**
  - Our learners are born from different seeds and live for a long time, so **naive averaging of finished lives is exactly the failure case in Figure 1**.
  - FedAvg-style aggregation only fits if the inherited condition starts every life from a shared inherited init. Then we would average deltas (θ_end − θ_init) or partial lives.
  - The fresh condition keeps random seeds and stays the baseline.
  - Shipping whole-network deltas (~8.7 MB) through GitHub at every sleep would violate invariant 6. Infrequent, gated releases of a subset of layers would not.

### [K2] Linear mode connectivity needs a shared trajectory
Frankle, Dziugaite, Roy, Carbin. "Linear Mode Connectivity and the Lottery Ticket Hypothesis." ICML 2020. https://arxiv.org/abs/1912.05671. Verified from the arXiv abstract page.

- **What it shows:**
  - The question is "whether a neural network optimizes to the same, linearly connected minimum under different samples of SGD noise".
  - "Standard vision models become stable to SGD noise in this way early in training. From then on, the outcome of optimization is determined to a linearly connected region."
- **Conditions:** the same initialization *and* a shared early stretch of training. The soups paper's summary of Frankle et al. is that two from-scratch ImageNet runs with only a different data order interpolate to "no better than random accuracy", unless they "share a portion of their optimization trajectory" [K4, §6].
- **Relevance to us:** a cheap way to make lives averageable is to have them share a **common early trajectory**. That trajectory would be an inherited "infancy" checkpoint, from which individual lives fork. This is a labelled inherited condition, not cold start.

### [K3] Permutation symmetry: Entezari et al.
Entezari, Sedghi, Saukh, Neyshabur. "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks." ICLR 2022. https://arxiv.org/abs/2110.06296. The venue was confirmed on mlanthology (https://mlanthology.org/iclr/2022/entezari2022iclr-role/). The quotes are from the arXiv abstract and the ar5iv full text.

- **Claim:** this is a conjecture. "If the permutation invariance of neural networks is taken into account", then "SGD solutions will likely have no barrier in the linear interpolation between them" (abstract).
- **Conditions found:**
  - "for large values of width the barrier becomes small". The effect is non-monotonic: the barrier "increases with width up to a point".
  - "increasing depth increases the barrier size significantly".
- **Limitations:**
  - "Simulated Annealing could not find the optimal permutations for all cases".
  - "The biggest limiting factor of our study is the size of the search space."
- **Relevance to us:** independently seeded lives may in principle be alignable, but our networks are **narrow** (16 and 32 conv channels). Narrow is the regime where the barrier is not reliably removed.

### [K4] Git Re-Basin
Ainsworth, Hayase, Srinivasa. "Git Re-Basin: Merging Models modulo Permutation Symmetries." ICLR 2023. The PDF header reads "Published as a conference paper at ICLR 2023". https://arxiv.org/abs/2209.04836. Verified from the arXiv page and the PDF text (v6).

- **What is transferred:** weights, after permuting one model's hidden units to match a reference model. The paper gives three matching algorithms. "Weight matching" (Algorithm 1) uses only the weights: "remarkable how well Algorithm 1 performs without access to the input data at all" (§5.1).
- **Evidence:**
  - The paper reports "zero-barrier linear mode connectivity between independently trained ResNet models on CIFAR-10" (abstract).
  - **Split-data merging (§5.4).** Two ResNet20s were trained on disjoint, biased halves of CIFAR-100 and merged by weight matching. The merged model beat both inputs on test loss. However, it "did not match the performance of an omniscient model trained on the full dataset or an ensemble of the two models".
- **Conditions and limitations (§5.2, §5.3, App. A.1):**
  - "linear mode connectivity is an emergent property of training, and we were unable to uncover it early in training".
  - "1×-sized models did not seem to exhibit linear mode connectivity". Zero barrier needed wider models.
  - "relatively thin models do not seem to obey linear mode connectivity".
  - The known failure modes include "Models of insufficient width" and "Models in the initial stages of training".
  - The ethics section asks: "Are rare examples forgotten when merging?"
  - An originality dispute against OTFusion was raised in the OpenReview discussion (reported secondhand by The Gradient). I did not verify it, and nothing here depends on it.
- **Relevance to us:**
  - Re-Basin is the only route on this list to merge **independently seeded** lives, which keeps the fresh-seed condition intact. Weight matching is data-free and cheap, so the harness could run it offline over checkpoints.
  - Our two strikes are narrow width and short or early lives. Both are listed failure modes.
  - **Agent's reading:** treat this as a hypothesis to probe. Use a minimal probe: two lives, measure the loss barrier on held-out frames before and after matching. Do not use it as a foundation until that probe works.

### [K5] Model soups
Wortsman, Ilharco, Gadre, Roelofs, Gontijo-Lopes, Morcos, Namkoong, Farhadi, Carmon, Kornblith, Schmidt. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." ICML 2022. https://arxiv.org/abs/2203.05482. Verified from the arXiv page and the PDF text.

- **What is transferred:** an average of fine-tuned weights. The greedy soup adds models "sequentially to the soup if they improve accuracy on held-out data" (§1).
- **Why soups need a shared init:**
  - The paper's own account is that "fine-tuned models optimized independently from the same pre-trained initialization lie in the same basin of the error landscape, inspiring our method" (§1).
  - Table 2 defines every ingredient as "a model found through fine-tuning from a shared initialization".
  - Greedy selection "avoid[s] adding in models which may lie in a different basin".
- **Limitations (§5):**
  - The experiments "mainly explored models pre-trained on large, heterogeneous datasets". With an ImageNet-22k pre-trained model, the gains were "less substantial".
  - Soups do not improve calibration the way ensembles do.
- **Relevance to us:**
  - Soups suggest a workable *gate rule*: accept a contribution into the shared artifact only if adding it **improves a held-out check**. This is the greedy soup recipe, applied by the harness.
  - Soups also say plainly that the shared starting point is what makes averaging safe.
  - We have no pretrained model, so our "shared initialization" would itself have to be grown by the population. That makes it a labelled inherited condition.

### [K6] FedMA: matched averaging in federated learning
Wang, Yurochkin, Sun, Papailiopoulos, Khazaeni. "Federated Learning with Matched Averaging." ICLR 2020 ("Accepted by ICLR 2020" on the arXiv page). https://arxiv.org/abs/2002.06440. Verified from the arXiv abstract.

- **What is transferred:** layer weights, after "matching and averaging hidden elements" with "similar feature extraction signatures", one layer at a time. The paper reports that this "reduces the overall communication burden".
- **Relevance to us:** a **layer-wise** precedent. Align and merge the first conv layer, freeze it, then do the next layer. This suits our case, where the first-layer filters are only 1,664 parameters.

---

## 2. Sharing predictions instead of weights (distillation)

### [K7] Knowledge distillation
Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network." NIPS 2014 Deep Learning Workshop. https://arxiv.org/abs/1503.02531. Verified from the arXiv page and the ar5iv text.

- **What is transferred:** targets (softened outputs), not weights. The paper works in the setting where "it is possible to compress the knowledge in an ensemble into a single model".
- **Mechanism:**
  - "When the soft targets have high entropy, they provide much more information per training case than hard targets" (§1).
  - The transfer set "could consist entirely of unlabeled data" (§1).
- **Conditions:** no shared initialization or architecture is needed. Teacher and student need only agree on an input space and an output space.
- **Relevance to us:**
  - This is the cleanest way to share knowledge between **differently seeded** (and differently sized, GPU vs CPU) learners.
  - The artifact would be a small set of frames plus teacher outputs, for example next-frame predictions or encoder features. It would not be weights.
  - The catch is the targets: they are learned competence. A learner trained on them is in an inherited condition and must be labelled (invariant 1).

### [K8] Deep Mutual Learning
Zhang, Xiang, Hospedales, Lu. "Deep Mutual Learning." CVPR 2018, pp. 4320–4328 (DOI 10.1109/CVPR.2018.00454, confirmed via Crossref). https://arxiv.org/abs/1706.00384. Verified from the arXiv abstract, the ar5iv text and the Crossref record.

- **What is transferred:** each peer's predictions, through a "KL mimicry loss" (§2.1). "an ensemble of students learn collaboratively and teach each other throughout the training process" (abstract).
- **Conditions:**
  - It works across "heterogeneous cohorts consisting of mixed big and small networks".
  - Peers differ because "each network starts from a different initial condition" (§1).
- **Evidence:** "no prior powerful teacher network is necessary", and mutual learning of simple students "outperforms distillation from a more powerful yet static teacher" (abstract). The tasks were supervised: CIFAR-100 and person re-identification.
- **Relevance to us:** this is a fit for decision 0004's **live peer layer**. Concurrent tabs could regularise toward each other's predictions on shared frames. It also fits seed diversity, since the method relies on different starts.

### [K9] Online distillation / codistillation
Anil, Pereyra, Passos, Ormandi, Dahl, Hinton. "Large scale distributed neural network training through online distillation." ICLR 2018 (venue confirmed on mlanthology). https://arxiv.org/abs/1804.03235. Verified from the arXiv abstract and the ar5iv text.

- **What is transferred:** predictions, computed from rarely shipped peer checkpoints. Peers trained on disjoint data share knowledge by "encouraging each model to agree with the predictions the other model would have made". "These predictions can come from a stale version of the other model", using "weights that only rarely get transmitted" (abstract).
- **Conditions:** as defined in §2, codistillation uses "the same architecture for all the models". Distillation helps when models are "sufficiently different (say, by having different initializations ...)".
- **Evidence:**
  - "codistillation reaches the same error in 2X fewer steps".
  - It is "a cost-effective way to make the exact predictions of a model dramatically more reproducible" (abstract).
  - Checkpoint exchange delays beyond about 50 steps slightly degraded learning.
- **Relevance to us:** stale, infrequent peer checkpoints are the GitHub cadence. A learner could distill from **a frozen published "elder" model's predictions** on its own frames, without ever loading the elder's weights into its own. The artifact is then a full model (MBs), which conflicts with "limited weights in GH". A distilled dataset is the smaller alternative.

### [K10] FedDF: ensemble distillation for model fusion
Lin, Kong, Stich, Jaggi. "Ensemble Distillation for Robust Model Fusion in Federated Learning." NeurIPS 2020 (confirmed via the proceedings listing). https://arxiv.org/abs/2006.07242. Verified from the arXiv abstract.

- **What is transferred:** the client models' outputs on unlabeled data, used to train a central model. This "allows flexible aggregation over heterogeneous client models that can differ e.g. in size".
- **Relevance to us:** this would let us fuse GPU-width and CPU-width lives into one artifact. The cost is that the fusion step needs a shared unlabeled frame set, which the harness can render from the world.

---

## 3. Inheriting initializations versus inheriting learned weights

### [K11] Hinton & Nowlan: learning guides evolution (Baldwin effect)
Hinton, Nowlan. "How Learning Can Guide Evolution." *Complex Systems* 1, 495–502, 1987. https://www.cs.toronto.edu/~hinton/absps/evolution.htm. Verified from the abstract page.

- **Claim:**
  - "Learning alters the shape of the search space in which evolution operates", which "provides good evolutionary paths towards sets of co-adapted alleles".
  - "This effect allows learning organisms to evolve much faster than their nonlearning equivalents".
  - None of this requires acquired traits to be inherited.
- **Relevance to us:** selecting on *how well a life learned*, while inheriting only what it was born with, is a coherent population mechanism. Our existing `settings-space.json` loop is already a small instance of this.

### [K12] Fernando et al.: meta-learning by the Baldwin effect
Fernando, Sygnowski, Osindero, Wang, Schaul, Teplyashin, Sprechmann, Pritzel, Rusu. "Meta-learning by the Baldwin effect." GECCO '18 Companion, pp. 109–110 (confirmed via Crossref, DOI 10.1145/3205651.3205763). https://arxiv.org/abs/1806.07917. Verified from the arXiv abstract and the ar5iv text.

- **What is transferred:** initial parameters and hyperparameters, never learned weights. Figure 1: "only the original initial parameters PP and hyperparameters hh are inherited".
- **Lamarckian versus Baldwinian (§1):**
  - In Baldwinian evolution, "these updated weights are forgotten by the next generation".
  - "In Lamarckian evolution, these final, learned weights are evolved and passed on."
- **Evidence and conditions:**
  - "the Baldwin effect is capable of evolving few-shot supervised and reinforcement learning mechanisms by shaping the hyperparameters and the initial parameters".
  - Versus MAML: "in simple cases MAML is more data efficient", but the Baldwin effect "does not require gradients to be backpropagated to the reference parameters" (abstract).
  - **§5:** "The Baldwin effect is superior to Lamarckian learning when the distribution of tasks is broad or quickly changing", "whereas Lamarckian learning is superior when the task distribution is narrow". In one task, "Lamarckian evolution gets stuck in a local optimum".
- **Relevance to us:**
  - This is the key framing. Moving pixel features into "shared, inherited weights" can be done **Lamarckianly** (copy what lives learned) or **Baldwinianly** (evolve a birth state that learns fast, with fitness measured as learning speed).
  - Our world is narrow now but meant to broaden (invariant 9), which by §5 argues for Baldwinian selection, or at least for testing both.
  - Baldwinian fitness lines up with our actual research question: is later learning cheaper?

### [K13] MAML
Finn, Abbeel, Levine. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017. https://arxiv.org/abs/1703.03400. Verified from the arXiv abstract.

- **What is transferred:** an initialization. "the parameters of the model are explicitly trained such that a small number of gradient steps with a small amount of training data from a new task will produce good generalization performance on that task".
- **Conditions:** the inner loop must be differentiable and short, which Fernando §1 lists as limitations. Lives running for thousands of steps in browser tabs cannot backpropagate through a whole life.
- **Relevance to us:** MAML states the objective we want (an init that makes learning cheap), but it is impractical as the mechanism.

### [K14] Reptile
Nichol, Achiam, Schulman. "On First-Order Meta-Learning Algorithms." arXiv 2018. I found no venue listed on the arXiv page. https://arxiv.org/abs/1803.02999. Verified from the arXiv abstract.

- **What is transferred:** an initialization, updated by "repeatedly sampling a task, training on it, and moving the initialization towards the trained weights", "using only first-order derivatives".
- **Relevance to us:** this is the practical bridge between FedAvg and MAML for our setting.
  - The inherited init is moved toward the average of what lives *that started from it* ended up with.
  - It needs a shared init, which is the inherited condition.
  - It needs no second-order gradients, and the harness can compute it offline from a few gated checkpoints.
  - Restricted to the conv layers, the artifact is about 44 KB.

### [K15] Population Based Training (Lamarckian weight copying)
Jaderberg, Dalibard, Osindero, … Kavukcuoglu. "Population Based Training of Neural Networks." arXiv 2017 (no venue on the arXiv page). https://arxiv.org/abs/1711.09846. Verified from the arXiv abstract and the ar5iv text.

- **What is transferred:**
  - Weights and hyperparameters: "exploit selects another member of the population to copy the weights and hyperparameters from" (§3).
  - "PBT discovers a schedule of hyperparameter settings rather than following the generally sub-optimal strategy" of one fixed set (abstract).
- **Relevance to us:**
  - Our `settings-space.json` already does the hyperparameter half, across lifetimes.
  - The weight-copying half is Lamarckian, and under invariant 1 it would produce a heavily labelled "adopted" condition.
  - A PBT-style copy also destroys the fresh condition for that individual, so the population must keep a fresh-seed arm running alongside it.

### [K16] Warm-starting can hurt
Ash, Adams. "On Warm-Starting Neural Network Training." NeurIPS 2020 (journal-ref on arXiv). https://arxiv.org/abs/1910.08475. Verified from the arXiv abstract and the ar5iv text.

- **Finding:**
  - "warm-starting seems to yield poorer generalization performance than models that have fresh random initializations" (abstract).
  - The proposed remedy (§4) is "shrinking the weights found in the previous round of optimization towards zero" and "then adding a small amount of parameter noise".
- **Relevance to us:** an inherited Lamarckian checkpoint may *start* better and *end* worse. The comparison must run to a fixed later budget, not just compare early curves. Shrink-and-perturb is a cheap knob for any inherited init.

### [K17] Loss of plasticity
Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton. "Loss of plasticity in deep continual learning." *Nature* 632(8026), 768–774, 2024. https://doi.org/10.1038/s41586-024-07711-7. Metadata and abstract verified via Europe PMC. nature.com redirected to a login page.

- **Finding:**
  - "standard deep-learning methods gradually lose plasticity in continual-learning settings".
  - Continual backprop keeps plasticity because "a small fraction of less-used units are continually and randomly reinitialized".
- **Relevance to us:**
  - Inherited weights come from lives that may already have lost plasticity. Our lives are also "forever lives" (decision 0014).
  - Any inherited condition should be checked for **learning speed on new material**, not just starting skill.

---

## 4. Consolidation within a life (the lower rungs)

### [K18] Complementary learning systems (1995)
McClelland, McNaughton, O'Reilly. "Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory." *Psychological Review* 102(3), 419–457, 1995. DOI 10.1037/0033-295X.102.3.419. Verified via the Europe PMC record (PMID 7624455). PubMed showed only a cookie wall, and the Stanford PDF is a scan.

- **Claim (abstract):**
  - Networks "discover the structure in ensembles of items if learning of each item is gradual and interleaved".
  - "The hippocampal system permits rapid learning of new items without disrupting this structure".
- **Relevance to us:** this is the theory behind our episodic buffer plus slow weights plus sleep replay [v1 spec §5.5]. It says slow parameters must learn slowly and interleaved. That is also a warning against overwriting a life's slow weights with an aggregated artifact in one step.

### [K19] CLS theory updated (2016)
Kumaran, Hassabis, McClelland. "What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated." *Trends in Cognitive Sciences* 20(7), 512–534, 2016. DOI 10.1016/j.tics.2016.05.004. Verified from the PDF text (Stanford course copy).

- **Claims (summary and Trends box):**
  - Replay "allows goal-dependent weighting of experience statistics".
  - "neocortical learning can be rapid for information that is consistent with known structure".
  - Replay "can be modulated by reward or novelty, which acts to rebalance the general statistics of the environment towards the goals of the agent".
- **Relevance to us:** the second claim is the strongest literature argument for inheritance. **Once structure exists, consistent new learning is fast.** An inherited "known structure", such as shared low-level filters, should make later learning cheaper if CLS theory transfers to our setting. That is a hypothesis to test, not a result.

### [K20] Elastic Weight Consolidation (EWC)
Kirkpatrick, Pascanu, Rabinowitz, Veness, Desjardins, Rusu, Milan, Quan, Ramalho, Grabska-Barwinska, Hassabis, Clopath, Kumaran, Hadsell. "Overcoming catastrophic forgetting in neural networks." *PNAS* 114(13), 3521–3526, 2017. https://arxiv.org/abs/1612.00796. Verified from arXiv, Europe PMC and the ar5iv text.

- **Mechanism:**
  - The method "remembers old tasks by selectively slowing down learning on the weights important for those tasks" (abstract).
  - It uses "the diagonal of the Fisher information matrix", applied as "a spring anchoring the parameters to the previous solution" (§2).
- **Limitations (§2.2):**
  - On Atari, "it does not reach the score that would have been obtained by training ten separate DQNs".
  - "the chief limitation of the current implementation is that it under-estimates parameter uncertainty".
- **Relevance to us:**
  - A per-parameter importance vector (Fisher diagonal) is a natural companion to a shared artifact. Across many lives, it records which weights everyone found important.
  - It could serve two purposes:
    - weight a merge, as in Matena and Raffel's Fisher merging, cited in [K5] §6 and not checked by me;
    - anchor inherited weights softly rather than freezing them.
  - For the conv layers, it is the same size as the weights.

### [K21] Continual-learning survey
De Lange, Aljundi, Masana, Parisot, Jia, Leonardis, Slabaugh, Tuytelaars. "A continual learning survey: Defying forgetting in classification tasks." IEEE TPAMI (accepted, DOI 10.1109/TPAMI.2021.3057446). https://arxiv.org/abs/1909.08383. Verified from the arXiv page.

- **Scope:** "task incremental classification, where tasks arrive sequentially and are delineated by clear boundaries", plus a framework for the "stability-plasticity trade-off".
- **Relevance to us:** background only. Our lives have no task boundaries, so most of the surveyed protocols do not apply directly.

---

## 5. Moving knowledge into the algorithm, or through culture

### [K22] Learned Policy Gradient (LPG)
Oh, Hessel, Czarnecki, Xu, van Hasselt, Singh, Silver. "Discovering Reinforcement Learning Algorithms." NeurIPS 2020 (confirmed via the proceedings listing). https://arxiv.org/abs/2007.08794. Verified from the arXiv abstract and the ar5iv text.

- **What is transferred:** a learned **update rule**. The method "discovers an entire update rule" from "multiple learning agents, each of which interacts with a different environment".
- **Evidence:**
  - Trained "solely on toy environments", LPG "generalises effectively to complex Atari games" (abstract).
  - The meta-training environments were tabular grid worlds and delayed chain MDPs.
- **Limitations:**
  - "LPG is still behind the advanced RL algorithms such as A2C".
  - "meta-training is very hard and unstable".
  - It took "approximately 24 hours to converge using a 16-core TPU-v2".
- **Relevance to us:**
  - This is the strongest precedent for "moving into the algorithmic flow". A population of short, fresh-seeded lives in small worlds produced a rule that transfers.
  - A rule artifact is also **seed-independent and tiny**, and every newborn can use it while still starting from random weights. That makes it the cleanest fit with invariant 1: architecture or learning-rule priors are allowed, and the newborn is still cold-start in weights.
  - However, a learned rule is learned competence at the population level. It must be versioned and recorded like any prior, and LPG's compute cost is far beyond our budget.
  - The planned `consolidation-prior` (the consolidation scorer, specified in the v1 spec but not yet built) is the small-scale version of this idea.

### [K23] Iterated learning in the lab
Kirby, Cornish, Smith. "Cumulative cultural evolution in the laboratory: An experimental approach to the origins of structure in human language." *PNAS* 105(31), 10681–10686, 2008. DOI 10.1073/pnas.0707835105. Verified from PMC (PMC2504810). PMC notes a later correction (PMC5373391).

- **What is transferred:** data, meaning one generation's output becomes the next generation's training input. "languages transmitted culturally evolve in such a way as to maximize their own transmissibility" (abstract). The Discussion attributes this to "a bottleneck on transmission".
- **Relevance to us:**
  - Passing **data through a bottleneck** makes the transmitted material easier to learn. The analogue is a small, shared curriculum or replay set.
  - The same paper also shows "The transmission process cumulatively introduces ambiguity" (Results, Exp. 1). Structure can come at the cost of expressivity.

### [K24] Neural iterated learning
Ren, Guo, Labeau, Cohen, Kirby. "Compositional Languages Emerge in a Neural Iterated Learning Model." ICLR 2020. https://arxiv.org/abs/2002.01365. Verified from the arXiv page.

- **Claim:** NIL "facilitates the emergence of a more structured type of language", and "these languages provide learning speed advantages to neural agents during training", which "can be incrementally amplified via NIL" (abstract).
- **Relevance to us:** a neural-agent precedent for **re-initialising learners each generation** and passing on only data or targets, which keeps every generation's weights fresh. It also measures learning speed as the benefit, which is our metric.

### [K25] Few-shot imitation as cultural transmission
Bhoopchand et al., 18 authors. "Learning few-shot imitation as cultural transmission." *Nature Communications* 14, 7536, 2023. DOI 10.1038/s41467-023-42875-2. Verified from PMC (PMC10684502).

- **What is transferred:** behaviour, within an episode, from a visible expert co-player.
  - Agents "succeed at real-time imitation of a human in novel contexts without using any pre-collected human data" (abstract).
  - They "recall demonstrations within a single episode long after the expert has departed".
- **Conditions:**
  - Deep RL with memory, "hard-coded bots" as experts during training, an attention loss that predicts the co-player's position, and expert dropout ("the expert is rendered visible or hidden from the agent").
  - The Discussion calls this "memory-based meta-learning".
- **Limitations:**
  - "we did not test cultural transmission from a range of humans".
  - The task is navigation-only.
  - They "do not distinguish whether our trained agents memorise a geographical path".
- **Relevance to us:**
  - This is far downstream, matching decision 0004's "expected far downstream".
  - It shows that imitation can be *learned* as a skill, rather than hard-coded, which is what 0004 asks for.
  - Its privileged training signal (the attention loss on co-player position) would be an explicit, recorded modality choice under invariant 2.

---

## 6. Measuring whether lives learned the same thing

### [K26] Convergent learning
Li, Yosinski, Clune, Lipson, Hopcroft. "Convergent Learning: Do different neural networks learn the same representations?" ICLR 2016. https://arxiv.org/abs/1511.07543. Verified from the arXiv page and the ar5iv text.

- **Setup:** "neural networks with the same architecture trained from different random initializations" (§1).
- **Finding:**
  - "some features are learned reliably in multiple networks, yet other features are not consistently learned" (abstract).
  - "a core representation is shared, but some rare features are learned in one network but not another" (§1).
  - "for many units a one-to-one assignment is possible" (§3.1).
  - Shared subspaces are common, but "the specific basis vectors learned are not" (abstract).
- **Relevance to us:** this is direct evidence for the user's intuition that seeds converge on shared features. It is also a method: bipartite matching of units, using activations on shared frames. "Not the same basis vectors" is exactly why naive averaging fails.

### [K27] CKA similarity
Kornblith, Norouzi, Lee, Hinton. "Similarity of Neural Network Representations Revisited." ICML 2019. https://arxiv.org/abs/1905.00414. Verified from the arXiv page.

- **Claim:** CKA "can reliably identify correspondences between representations in networks trained from different initializations".
- **Relevance to us:** CKA is a cheap harness metric for deciding *what* is ready to move up. If layer-L representations of independent lives on a fixed probe set have high CKA, that layer is a candidate for sharing. Low CKA means the layer is still individual.

### [K28] Ape-X: a shared replay memory
Horgan, Quan, Budden, Barth-Maron, Hessel, van Hasselt, Silver. "Distributed Prioritized Experience Replay." ICLR 2018. https://arxiv.org/abs/1803.00933. Verified from the arXiv page.

- **What is transferred:** experience. "the actors interact with their own instances of the environment" and "accumulate the resulting experience in a shared experience replay memory" (abstract).
- **Relevance to us:**
  - This is a precedent for a shared replay set, but Ape-X shares *one* learner's network and streams experience continuously. That is the learning loop, which must not run through GitHub (invariant 6).
  - What would fit is a small, curated, slowly changing **shared replay or curriculum file** of frames (compressed, from the world, no privileged fields). Newborns could replay it during sleep.
  - It is a data prior and must be recorded as curriculum (invariant 2). It is also a weaker form of inheritance than weights, because the learner still has to learn from it.

---

## What this suggests for our design (agent's reading, not decided)

1. **Naive averaging of independently seeded lives will fail.** This is the clearest finding: FedAvg Figure 1 [K1], Frankle et al. [K2], and soups' shared-init premise [K5]. Permutation alignment [K3, K4] is the only escape that keeps seeds independent, and our networks sit in its known failure regime (thin, early). So any *weight-shaped* shared artifact implies an **inherited condition that starts from it**. That condition is labelled, and logged with its artifact hash the same way `knowledgeSha` is now.

2. **Choose what to share based on what already converges, and measure it first.** Before building any aggregation, the harness should measure cross-life similarity, layer by layer, on a fixed probe set of world frames. It would compute CKA [K27] and one-to-one unit matching [K26] over many finished fresh lives. Layers with high similarity are the ones where "everyone learns the same thing". This is a cheap, offline, harness-side probe and needs no change to the learner.

3. **Weights, initializations, rules, or data: rank by how well they preserve a fresh condition and how small they are.**
   - **(a) Rules and settings** are seed-independent, tiny, and leave newborn weights cold-start. Examples are `settings-space`, `consolidation-prior` (planned), and eventually an LPG-like rule [K22].
   - **(b) A shared initialization of a few layers** is small (44 KB for all conv filters) and Baldwinian-compatible, but it creates an inherited arm [K12, K14].
   - **(c) Shared data or targets** need no shared init and work across widths [K7, K9, K10, K28]. Examples are a distilled frame set or a replay or curriculum file. They are larger, and they inherit through learning rather than through birth.
   - **(d) Whole-network Lamarckian checkpoints** are MBs in size, risk worse final generalisation and lost plasticity [K16, K17], and give the cleanest "adopted" condition but the least science. These should come last.

4. **Prefer Baldwinian selection, with fitness measured as learning cost.** Fernando §5 [K12] says Baldwinian inheritance wins when tasks are broad or changing, which is our trajectory. A Baldwinian loop selects inherited birth states by **how cheaply lives that start from them reach milestones**. Its fitness is therefore our research question, and it cannot be gamed by a high starting score.

5. **Use the greedy soup rule at the gate.** A candidate contribution is admitted to a shared artifact only if adding it improves a protected held-out check [K5]. This fits the existing gate design and invariant 7.

6. **Keep both conditions measurable, always.** Every cohort should contain fresh-seed newborns (random init, the current condition) alongside inherited-init newborns. Both should be logged with the artifact hash, and compared at the same CU on the same held-out variants. A **matched-statistics control** separates "learned features" from "better-scaled random init": the same layer filled with random filters whose per-filter norms match the inherited ones, or the inherited filters with their spatial taps shuffled.

7. **The cost of building an artifact is population compute, and it is reported separately** (invariant 5). The CU every contributing life spent counts toward the cost of the inherited condition. Saying "inherited lives reach M1 in fewer CU" is only half the result. The other half is how many fresh-life CU were spent producing the artifact.

8. **A risk to watch: inheritance can make individuals worse learners.** Warm-start [K16] and loss of plasticity [K17] show this. Measure both the CU to reach a milestone *and* the held-out learning speed on a world variant the artifact never saw. Keep shrink-and-perturb [K16] available as a knob.

### Concrete first candidate: an inherited first-layer filter bank (the agent's proposal)

**What moves up.** The two first conv layers, periphery 5×5×3→16 and fovea 3×3×3→16, at GPU widths.
- That is **1,664 parameters, about 6.7 KB** as float32 (measured above). A CPU-width variant has 832 parameters.
- It would be stored as e.g. `knowledge/filters-gpu-v1.json`, with a version and the provenance of the contributing lives.
- This is the literal "features that recognise pixels" layer. Early-layer filters are where convergence across seeds is most plausible [K26]. This is the agent's expectation, to be checked by step 0 below.

**How it is built.** The harness builds it offline, from gated checkpoints. The learner never sees the process.
- **Step 0 (probe; no artifact yet).** Run N≈10 fresh lives to a fixed CU. Measure first-layer CKA and unit-matching correlations across lives on a fixed probe set. If first-layer filters do not converge, stop here. That is a finding in itself.
- **Step 1 (generation 0).** Align each life's first-layer filters to a reference life by one-to-one matching, using Hungarian assignment on weight or activation correlation [K4 weight matching, K6, K26]. Average the aligned filters. This one-layer case is exactly where alignment is easiest: the input channels are fixed RGB pixels, so only the output permutation needs solving.
- **Step 2 (later generations, Baldwinian/Reptile).** Lives in the inherited arm start from the artifact. The next version moves toward the mean end-state of the arm's own lives [K14]. These lives share an init, so no alignment is needed [K1, K2]. A new version is accepted only if a protected held-out check improves [K5].

**Conditions to run:** all at the same widths and settings sampler, with seeds written down in advance per §6.2 of the v2 spec.
- **F (fresh):** the current newborns, with all weights random.
- **I (inherited filters):** the first-layer filters are loaded from the artifact. Everything else is random from the seed. The filters stay trainable; an optional EWC-style soft anchor [K20] would be a separate arm.
- **C (matched control):** random filters with per-filter norms and bias statistics matched to the artifact. This tests whether any gain comes from scale rather than from learned features.

**What counts as "later learning is cheaper":**
- **Primary:** CU to first pass of the M1 bar (prediction loss ÷ copy-last < 0.8), and the held-out prediction ratio at fixed CU (e.g. 512 and 1,024 CU). Both are measured on **world variants no contributing life saw**.
- **Secondary:**
  - the full checkup curve;
  - learning speed after a novel room is introduced mid-life (the slope of the held-out ratio over the following K CU), which is the plasticity check [K17];
  - the final held-out ratio at a long budget, which is the warm-start check [K16].
- **Pass bar (agent's guess):** I beats both F and C on the primary metric with a pre-registered effect size, and I is not worse than F at the long budget.
- **Reporting:**
  - the population CU spent building the artifact;
  - interaction and compute costs separately;
  - unfavourable results, if they occur, kept and reported (invariant 7).

**Why this first:**
- It is the smallest weight-shaped thing that literally is "pixel features".
- It fits in a few KB.
- The alignment problem is at its easiest here.
- It leaves the fresh condition untouched.
- It tests the CLS claim that existing structure makes consistent new learning fast [K19], in our own setting.

If it fails, the next candidates in order are a distilled frame-and-target set (no shared init, works across widths) [K7, K10], then a shared replay or curriculum file [K23, K28].

---

## Sources

**Verified.** Each source's page was fetched in this session, and the title, authors, year and venue were checked against it. Quotes are from the fetched text.
- **ML papers:** [K1] McMahan et al. 2017; [K2] Frankle et al. 2020; [K3] Entezari et al. 2022; [K4] Ainsworth et al. 2023; [K5] Wortsman et al. 2022; [K6] Wang et al. 2020; [K7] Hinton et al. 2015; [K8] Zhang et al. 2018; [K9] Anil et al. 2018; [K10] Lin et al. 2020.
- **Evolution and meta-learning:** [K11] Hinton & Nowlan 1987; [K12] Fernando et al. 2018; [K13] Finn et al. 2017; [K14] Nichol et al. 2018; [K15] Jaderberg et al. 2017.
- **Plasticity and consolidation:** [K16] Ash & Adams 2020; [K17] Dohare et al. 2024; [K18] McClelland et al. 1995; [K19] Kumaran et al. 2016; [K20] Kirkpatrick et al. 2017; [K21] De Lange et al. 2021.
- **Algorithms, culture and measurement:** [K22] Oh et al. 2020; [K23] Kirby et al. 2008; [K24] Ren et al. 2020; [K25] Bhoopchand et al. 2023; [K26] Li et al. 2016; [K27] Kornblith et al. 2019; [K28] Horgan et al. 2018.

**Verification caveats:**
- **Venues confirmed by a secondary record rather than the publisher page:** [K3] mlanthology; [K8] Crossref (the CVF page returned 403); [K9] mlanthology; [K10] and [K22] proceedings listings via search; [K12] Crossref (ACM returned 403).
- **Partial access:** [K17] metadata and abstract via Europe PMC (nature.com requires login). [K18] has abstract-level quotes only, via Europe PMC.
- **No venue found:** [K14] and [K15] have no venue on the arXiv page, so they are cited as arXiv.

**Unverified, and not relied on:**
- the OpenReview originality dispute about Git Re-Basin (seen only in a secondary report);
- Neyshabur et al. 2020, Matena & Raffel 2021 (Fisher merging), Singh & Jaggi 2020 (OTFusion), and Nagarajan & Kolter 2019, all cited only as they appear inside [K5]/[K4] and not fetched;
- Fontanari & Santos 2017 (critique of Hinton & Nowlan), seen in search results only.

**Repo sources:**
- [SYN] `docs/research/2026-10-16-research-synthesis-v0.1.md`
- [v1 spec] `docs/specs/2026-10-19-blanktab-v1-design.md` §5.1, §5.5, §9
- `docs/specs/2026-10-22-learner-v2-capabilities-design.md` §6.2
- `docs/decisions/0004-two-sharing-layers.md`, `0014-forever-lives-and-live-updates.md`
- Parameter counts from `packages/learner/src/model.ts`, measured with `.local/paramcount.ts`, which is local scratch and not committed.
