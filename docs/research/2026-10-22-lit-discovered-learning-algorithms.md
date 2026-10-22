# Literature: systems that discover learning algorithms, objectives and curiosity rules

**Literature note | 22 October 2026 | theme: automatic discovery of "how to learn"**

## Why this note exists

The user wants the learner's whole way of learning written down as data: a "genome" of matrices, switches and small learned rules. A population-level meta-learner would sample that genome and improve it automatically, for example with a Gaussian process doing adaptive sampling. "How it learns" would be shared through GitHub. "What it learns in a life" would stay local. This matches the cascade in [SYN §1, §3] and the two sharing layers in decision 0004.

This note looks at published systems where a learning algorithm, objective or curiosity rule was **found by search** rather than designed by hand. For each one it records what was searched over, how, against what fitness, at what cost, and what generalized. It ends with what that implies for our genome.

**How the sources were checked.** Every source was fetched on 2026-10-22, either the arXiv PDF (converted to text locally in `.local/litdl/`, which is gitignored) or the publisher page. Title, authors, year and venue were confirmed against that copy. The venues for EPG, Enhanced POET and Lion were confirmed against the proceedings or conference pages. Quotes are exact and short, with the section they come from. The compute figures are the papers' own. Arithmetic I added is marked *(my arithmetic)*.

**Our evaluation budget, as the reference point.** In the v2 design one life is about 30 minutes, including its checkups. One M4 runs one life at a time [V2 §6]. So one machine gives about 48 lives a day, and 10 machines about 480 *(my arithmetic)*. Every fitness reading below has to be weighed against that. We get one noisy number per 30-minute life, and only a few lives run at once.

---

## 1. AutoML-Zero (Real et al., 2020)

**Citation.** Esteban Real, Chen Liang, David R. So, Quoc V. Le. *AutoML-Zero: Evolving Machine Learning Algorithms From Scratch.* ICML 2020 (PMLR 119). arXiv:2003.03384. `https://arxiv.org/abs/2003.03384` [L1]

**Problem.** Can a search find a complete ML algorithm, including initialization, prediction and the learning rule, from basic maths ops, with very little human design in the search space?

**What is searched.** A program made of three component functions, `Setup`, `Predict` and `Learn`. Each is a list of instructions over a small memory of scalars, vectors and matrices. The space has 65 ops in total (Suppl. S2). The op and the memory addresses each instruction uses are free parameters, and so is the length of each function. One example is processed at a time, with no loops (§3.1).

**Outer loop.** Regularized evolution: tournament selection, mutate the parent, remove the oldest member. There are three mutation types: insert or remove an instruction, randomize a whole function, or change one argument (§3.2). Random search barely works here. Good algorithms "can be as rare as 1 in 10^12" (§1).

**Fitness and anti-gaming.** The fitness is validation accuracy on small "projected binary" CIFAR-10 and MNIST tasks. Search tasks (T_search) and model-selection tasks (T_select) use disjoint class pairs. The CIFAR-10 test set is held back for the final report (§4.2). Other measures:
- **Functional-equivalence checking.** The predictions after 10 training and 10 validation steps are hashed into a fingerprint, so programs that are really duplicates are not re-evaluated (§3.2).
- **Hurdles.** Weak candidates are stopped early.
- **Fixed meta-parameters.** They were not tuned at full scale: "None of the meta-parameters were tuned in the final set of experiments at full compute scale" (§3.2).

**Scale and compute.**
- Throughput of "10,000 models/second/cpu core" (§1).
- Worker processes on commodity CPU cores exchange models by migration. Typically 100 < W < 1000 (§3.2), and W = 10k for the CIFAR experiment (§4.2).
- "Run time: 5 days" (§3.2).

**Transfer.** The best evolved algorithm scored 84.06 ± 0.10% on binary CIFAR-10. Logistic regression scored 77.65% and a 2-layer network 82.22%. The gain carried over to binary SVHN, downsampled ImageNet and Fashion-MNIST (§4.2). Evolution also adapted to the task type: noisy-ReLU or dropout-like moves appeared when data was scarce, and learning-rate decay appeared when training had to be fast (§4.3, Fig. 7).

**Limitations.**
- No batch-level ops (batch-norm) and no convolution in the space (§4.2, §5).
- "Hyperparameter coupling" needed manual decoupling before the evolved programs could be tuned on new tasks. "This required time-consuming analysis that could be automated by future work" (§5).
- The compute is enormous by our standards.

**Relevance to us.** AutoML-Zero is the "everything is a program" end of the spectrum. Its lessons that carry over are the engineering ones: fingerprint candidates to skip duplicates, stop weak ones early with hurdles, keep search and selection tasks separate, and freeze meta-parameters before the full-scale run. Its search space is infeasible for us. It needs billions of cheap evaluations, and we get about 480 expensive ones a day. Even its "trivial" task needed long random-search runs to measure difficulty (§4.1). The coupling problem is also a warning. Once a genome mixes structure with constants, the constants found by search can quietly encode task-specific values.

## 2. Learned Policy Gradient (LPG): "Discovering Reinforcement Learning Algorithms" (Oh et al., 2020)

**Citation.** Junhyuk Oh, Matteo Hessel, Wojciech M. Czarnecki, Zhongwen Xu, Hado van Hasselt, Satinder Singh, David Silver. *Discovering Reinforcement Learning Algorithms.* NeurIPS 2020. arXiv:2007.08794. `https://arxiv.org/abs/2007.08794` [L2]

**Problem.** Can a whole RL update rule, covering both "what to predict" and "how to learn from it" (bootstrapping), be discovered from interaction rather than designed?

**What is searched.** The weights η of a backward LSTM, the "update rule". It reads a trajectory of the agent's outputs and produces targets for the agent's policy (π̂) and for an m-dimensional prediction vector (ŷ). Nothing fixes in advance what ŷ means. The agent minimizes the KL divergence to those targets (§3.2). The rule never sees observations or actions, only rewards, discounts, termination flags and the agent's own outputs. This is what lets it move between environments.

**Outer loop.** Meta-gradient descent. Gradients are backpropagated through K agent updates using a sliding window, and averaged over many parallel "lifetimes", each with its own sampled environment and freshly initialized agent (Fig. 1, Alg. 1).

**Fitness and stability.**
- The fitness is the agent's return within its lifetime.
- Meta-training was "very hard and unstable" (§4.4). It was stabilized by entropy and L2 regularizers on the rule's outputs, and by a bandit that picks agent hyperparameters per environment during meta-training only (§3.3–3.4).
- Meta-test environments (Atari) were never seen during meta-training.

**Scale and compute.** 960 parallel lifetimes, about 10^10 environment steps in total. "It took approximately 24 hours to converge using a 16-core TPU-v2" (App. B.1, B.3). The training environments were tabular and random grid worlds plus delayed-chain MDPs (§4).

**Transfer.** A rule trained only on toy environments transferred to Atari, "achieving super-human performances on 14 games" (§4.5). Adding delayed-chain MDPs to training noticeably improved this generalization. The learned ŷ came to encode value-like information at several discount factors (§4.3).

**Limitations.**
- In the authors' words, "LPG is still behind the advanced RL algorithms such as A2C" on Atari (§4.5).
- Meta-test hyperparameters still had to be chosen by hand (§3.4).
- Meta-gradients only reach K updates into a lifetime (also see [L13]).

**Relevance to us.** LPG is the main evidence that a *small* learned rule, trained on *cheap* worlds, can transfer to much harder pixel worlds, provided its inputs are kept domain-free (no raw observations). That fits "simplify the world, not the architecture's ceiling" (invariant 9). Its method does not fit us:
- Meta-gradients need differentiating through many agent updates on accelerators with large memory.
- It needs hundreds of lifetimes running at once.

We get a few tabs and one noisy scalar per life, so we are in a black-box (ES, Bayesian optimization or GP) setting, not a meta-gradient one.

## 3. DiscoRL: "Discovering state-of-the-art reinforcement learning algorithms" (Oh et al., Nature 2025)

**Citation.** Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, Matteo Hessel, Luisa Zintgraf, Satinder Singh, Hado van Hasselt, David Silver. *Discovering state-of-the-art reinforcement learning algorithms.* Nature 648, 312–319 (2025). Published 22 October 2025, open access. doi:10.1038/s41586-025-09761-x. `https://www.nature.com/articles/s41586-025-09761-x` [L3]. Confirmed to exist. The page was fetched and read in full.

**Problem.** Scale LPG-style discovery up to complex environments and a more expressive rule space, and see whether the result beats hand-designed RL.

**What is searched.**
- The meta-network weights η, an LSTM unrolled backwards in time over the agent's outputs. It produces targets for the policy and for two vectors whose meaning is not fixed: y(s), conditioned on the observation, and z(s,a), conditioned on the action.
- The agent also has fixed-meaning heads, action values q and an auxiliary policy p. These are trained with standard targets (Retrace).
- A "meta-RNN" unrolled across agent updates lets the rule keep statistics over a lifetime, for example for reward normalization (Methods).
- The authors note that "outputting targets is strictly more expressive than outputting a scalar loss function" (Discovery method).

**Outer loop.** Meta-gradients through 20 agent updates in a sliding window. The ∇θJ term is estimated with advantage actor-critic and a meta-value function used only for discovery. Each agent's meta-gradient goes through its own Adam optimizer before averaging, and advantages are normalized per environment (Methods, "Meta-optimization stabilization"). "Each agent's parameters are periodically reset to encourage the update rule to make fast learning progress within a limited agent lifetime."

**Fitness and anti-gaming.**
- The fitness is the collective return of the population of agents.
- Meta-training hyperparameters "were tuned on only training environments (that is, Atari) to prevent the rule from being implicitly optimized for held-out benchmarks" (Generalization).
- Held-out benchmarks: ProcGen, Crafter, NetHack, Sokoban, and DMLab for Disco57.

**Scale and compute.** "Disco57 was discovered using 1,024 TPUv3 cores for 64 hours, and Disco103 was discovered using 2,048 TPUv3 cores for 60 hours" (Methods, Implementation details). That means 128 agents for Disco57 and 206 for Disco103. The best rule was found "within approximately 600 million steps per Atari game" (Efficiency and scalability).

**Transfer.**
- Disco57 reached an Atari IQM of 13.86, above all existing rules. It beat all published methods on unseen ProcGen and placed third on NetHack without domain knowledge.
- Disco103 adds ProcGen and DMLab-30 to the training set. It reached human level on Crafter and improved on every benchmark apart from Atari, where it performed about the same.
- What did **not** work: discovery on 57 grid-world tasks gave "significantly worse performance" on Atari (Fig. 3c). Ablations show that both a value function and learned predictions matter.

**Limitations.** The compute is enormous. The rule still relies on hand-chosen fixed-meaning heads (q, p), and on discrete actions with weights shared across action dimensions.

**Relevance to us.** This paper strengthens two points. First, a rule that only sees the agent's *outputs* transfers across observation and action spaces. Second, the diversity of the discovery environments drives generality. It also partly contradicts LPG's "toy worlds suffice": at scale, discovering in grid worlds did clearly worse. For us, the training distribution of worlds is part of what must be chosen and recorded as a prior (invariant 2: curriculum selection is a prior). The method itself is out of reach: thousands of TPU cores for days. What we can borrow is its *structure*:
- a small rule network that reads the agent's predictions, not its pixels;
- open-meaning prediction slots;
- a lifetime-level recurrent state for normalization;
- periodic resets so the rule is selected for fast early learning.

## 4. Evolved Policy Gradients (EPG) (Houthooft et al., 2018)

**Citation.** Rein Houthooft, Richard Y. Chen, Phillip Isola, Bradly C. Stadie, Filip Wolski, Jonathan Ho, Pieter Abbeel. *Evolved Policy Gradients.* NeurIPS 2018 (Advances in Neural Information Processing Systems 31). arXiv:1802.04821. `https://arxiv.org/abs/1802.04821`, proceedings page `https://papers.nips.cc/paper/2018/hash/7876acb66640bad41f1e1371ef30c180-Abstract.html` [L4]

**Problem.** Learn a differentiable loss function such that an agent minimizing it by SGD gets high returns.

**What is searched.** The parameters φ of a loss network: temporal convolutions over a buffer of the agent's recent experience (s, a, r, done, plus memory-unit outputs). The agent's policy is trained by gradient descent on this loss (§3.3). For the first 500 epochs the loss is blended with PPO or REINFORCE (α annealed from 1 to 0) to make the start of the search easier (§3.3, §4.1).

**Outer loop.** Evolution strategies over φ (Salimans-style). W = 256 inner-loop workers share V = 64 noise vectors, so each perturbation is scored on W/V randomly sampled MDPs. "The actual implementation calculates each loss function's relative rank for the ES update" (§3.2).

**Fitness and anti-gaming.**
- The fitness is the **final return** after a whole inner-loop training run. This is non-differentiable and has a long horizon.
- Sharing each perturbation across several sampled MDPs prevents "promoting loss functions that are assigned to MDPs that consistently generate higher returns" (§3.2).
- Rank-based updates.

**Scale and compute.** 5,000 ES epochs (§4.1). "Each inner-loop worker takes approximately 196 million steps to evolve the loss function used in the RandomReacher experiments" (§6). The text I read reports no hardware figure.

**Transfer.**
- The loss trained on GoalAnt goals to the right generalized to goals to the left. RL² and MAML overfit to rightward goals (§4.3).
- Transfer to different policy architectures worked but was weaker (§4.3).
- The authors are blunt: "EPG gains on performance by losing on generality" (§6). A loss for ant-walking would be "unlikely to be at all effective" on Space Invaders.
- Generalization beyond the inner-loop training horizon was limited. It was fixed by starting inner loops from saved, partly trained policies (§4.3).

**Limitations.** Sequential outer-loop dependence and large compute. Each loss covers one narrow family of tasks.

**Relevance to us.** EPG is structurally closest to our setting: black-box ES over a loss's parameters, with fitness equal to the result of a whole life. Its noise handling carries over: rank-transform fitness, and score each candidate on several sampled worlds or seeds so lucky draws don't win. Two of its findings map onto our open questions. The horizon problem applies to "forever lives" (decision 0014): a rule selected on 30-minute lives may not keep working after that. And the narrowness warning applies to our world: a genome tuned on one staged world may not carry over to the next stage.

## 5. Meta-learning curiosity algorithms (Alet et al., 2020)

**Citation.** Ferran Alet, Martin F. Schneider, Tomás Lozano-Pérez, Leslie Pack Kaelbling. *Meta-learning curiosity algorithms.* ICLR 2020. arXiv:2003.05325. `https://arxiv.org/abs/2003.05325` [L5]

**Problem.** Discover intrinsic-reward ("curiosity") mechanisms that generalize across very different environments. Their framing is that curiosity is something evolution designed (§1).

**What is searched.**
- Programs in a typed DSL: directed acyclic graphs (DAGs) of modules, including neural networks with their own internal SGD, buffers, nearest-neighbour regressors, and losses.
- The programs come in two parts. A curiosity program *I* maps transitions to an intrinsic reward. A combiner χ mixes that intrinsic reward with the extrinsic reward, the time step and the lifetime fraction t/T (§2).
- Programs are capped at 7 modules (§2.2).
- Types are instantiated per environment, so a network from S to F becomes a CNN for image states and an MLP for vectors (§2.2).

**Outer loop.** Enumeration plus a learned predictor:
- remove duplicates and invalid programs (about 52,000 remain);
- evaluate cheap environments first;
- stop runs early if the partial learning curve is significantly below the current top 625;
- use a 10-nearest-neighbour regressor over program-structure features to pick the next program to evaluate, epsilon-greedy with 90% exploit (§3, §4.1).

They evaluated only about 26,000 programs, "half of the whole search space", which they estimated would recover 88% of the top 1% (§4.1).

**Fitness and anti-gaming.** In the first phase the fitness is "the total number of distinct cells visited across the agent's lifetime" in an image-based grid world (§4.1). The agent is trained only on the proxy reward and never sees r_t directly (§2.1). Later phases score mean extrinsic reward on lunar lander, acrobot, hopper and ant.

**Scale and compute.** First phase: 4 machines with 8 K80 GPUs each, for 10 hours. "Searching through this space took a total of 13 GPU days" (§4.1). The conclusion calls the total "relatively modest compute (2 GPU-weeks)" (§6).

**Transfer.** Programs selected only in the grid world transferred to lunar lander and acrobot, which have vector inputs and 15–20× longer horizons. Programs scoring above 400 in the grid world did well on both in more than 90% of cases (§4.2). On hopper and ant the top 16 were statistically equivalent to published curiosity methods (RND, inverse features, disagreement) and beat constant-reward baselines (Table 1). The top program, FAST, has only 5 ops (§4.1).

**Limitations.**
- The combiner was searched separately, on lunar lander, then frozen. Co-adapting it is left to future work (§4.2).
- The search space is medium-sized.
- On MuJoCo the learned programs only *matched* published ones.

**Relevance to us.** This is the closest published analogue to our plan. The genome is a small typed graph of modules. Most of the search cost is avoided by:
- removing duplicates;
- evaluating cheap environments first;
- stopping on learning curves;
- a **surrogate model over the genome that decides what to evaluate next**.

That last item is our GP picker in kNN form. Their budget, about 26,000 evaluations of which most were stopped early, is about 54 machine-days of 30-minute lives *(my arithmetic)*. That is tight but not absurd if most candidates are killed at an early checkup. Their phase-1 fitness (distinct cells visited) is privileged instrumentation used only by the harness, which is exactly how our invariant 2 asks fitness to be handled.

## 6. AI-generating algorithms (AI-GAs) (Clune, 2019)

**Citation.** Jeff Clune. *AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence.* arXiv:1905.10985 (v1 May 2019, v2 1 Feb 2020). Position paper, arXiv preprint. `https://arxiv.org/abs/1905.10985` [L6]

**Problem.** A research-agenda argument. General AI may come sooner from algorithms that *learn to produce* AI than from assembling hand-designed parts by hand.

**Representation and search.** There is no single system. It argues for "Three Pillars": "(1) meta-learning architectures, (2) meta-learning the learning algorithms themselves, and (3) generating effective learning environments" (Abstract). Hebbian plasticity, neuromodulation and differentiable plasticity are discussed as "materials" for pillar 2 (§2.2).

**Compute, fitness and generalization.** No experiments. On cost it says: "Meta-learning is currently computational expensive" (§2.2, sic). Limits on compute, memory and optimization constrain task complexity, task length and how many outer-loop steps are possible. It also raises the question of task distribution: "switching from normal learning to meta-learning changes the burden on the researcher from designing learning algorithms to designing environments" (§2.2).

**Limitations.** It is a position paper. Its claims are hypotheses, and it is explicitly a bet on large future compute.

**Relevance to us.** It frames our genome as pillar 2, and our staged world, curriculum and milestones (decisions 0001, 0005) as a hand-designed pillar 3. Its warning to us is the burden shift. Once the learning rule is searched, *which worlds it is scored on* becomes the main design decision. That decision must be recorded as a prior and kept separate from the evaluation, so the search can't redefine success (invariant 7).

## 7. POET and Enhanced POET (Wang et al., 2019; 2020)

**Citations.**
- Rui Wang, Joel Lehman, Jeff Clune, Kenneth O. Stanley. *Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions.* arXiv:1901.01753 (2019). `https://arxiv.org/abs/1901.01753` [L7a]. No peer-reviewed venue was confirmed, so it is cited as a preprint.
- Rui Wang, Joel Lehman, Aditya Rawal, Jiale Zhi, Yulun Li, Jeff Clune, Kenneth O. Stanley. *Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions.* ICML 2020 (confirmed through the ICML 2020 virtual poster page `https://icml.cc/virtual/2020/poster/53548`). arXiv:2003.08536. `https://arxiv.org/abs/2003.08536` [L7b]

**Problem.** Generate environments and their solutions together, so a population of curricula unfolds without a hand-written sequence.

**What is searched.** POET does not search a learning algorithm. It searches **environment encodings** and agent weights. In POET the encoding is a parameter vector of obstacle settings (stump height, gap width, roughness) for a 2-D bipedal walker (POET §4.2). Enhanced POET replaces it with CPPNs, small evolving networks that draw the terrain (E-POET §4). Each agent's weights are optimized by ES within its environment.

**Outer loop.**
- Mutate environments and admit a child only if it passes a **minimal criterion**: the current agents' scores fall between 50 and 300 (not too easy, not too hard), ranked by novelty (POET §3, §4.2).
- Periodically try **transferring** agents between environments.
- Enhanced POET adds PATA-EC, a domain-general novelty measure. It describes an environment by the rank-normalized scores of all agents on it (E-POET §3).
- It also adds a stricter transfer test that is cheaper and less prone to "false positives" caused by ES noise. The incoming agent must beat the maximum of the incumbent's 5 most recent scores (E-POET §3).

**Fitness and progress.** In POET an environment counts as solved at a score of 230 or more. Enhanced POET introduces ANNECS, the accumulated number of novel environments created and solved. An environment only counts if it passes the minimal criterion against all agents so far and is eventually solved, so producing unsolvable challenges earns no credit (E-POET §5).

**Scale and compute.**
- POET: "each run harnessed 256 parallel CPU Cores" (§3). Runs of up to 25,200 iterations "each take about 10 days to complete on 256 CPU cores" (§4.3).
- Enhanced POET: "completes a 60,000-iteration POET run in about 12 days with 750 CPU cores" (E-POET §6).

**Transfer.**
- Many environments POET created and solved could not be solved by direct ES from scratch, or by a hand-made direct-path curriculum under the same budget (POET Abstract, §4.3).
- Transfer between environments was essential.
- Original POET's ANNECS plateaued after about 20,000 iterations because its encoding was limited. Enhanced POET kept rising (E-POET Fig. 8).

**Limitations.** The domain is a single 2-D walker. The agents are fixed-weight controllers that do not learn within a life. The compute is about hundreds of core-days per run.

**Relevance to us.** Three transferable mechanisms:
1. **Minimal criterion**: only keep a candidate (a world stage, or a genome) that is neither trivially passed nor hopeless. This is an inexpensive filter on noisy candidates.
2. **Transfer tests that compare against the incumbent's best recent scores**, not one draw. This is a direct defence against noise when a whole life is the evaluation.
3. **Measure progress by what is newly solved and stays solved** (ANNECS), not by raw score.

POET also shows that a fixed encoding limits growth (invariant 9). Its compute is CPU-only and runs in parallel without coordination, which suits browser tabs in shape, but not in scale.

## 8. Lion: Symbolic Discovery of Optimization Algorithms (Chen et al., 2023)

**Citation.** Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le. *Symbolic Discovery of Optimization Algorithms.* NeurIPS 2023. arXiv:2302.06675. `https://arxiv.org/abs/2302.06675`, proceedings PDF `https://proceedings.nips.cc/paper_files/paper/2023/file/9a39b4925e35cf447ccba8757137d84f-Paper-Conference.pdf` [L8]

**Problem.** Discover a neural-network optimizer by program search that generalizes from tiny proxy tasks to large state-of-the-art training runs.

**What is searched.**
- A symbolic `train(w, g, m, v, lr)` program: a sequence of assignment statements over NumPy-like functions, with mutable constants (§2.1).
- The search is **warm-started from AdamW**.
- Redundant statements are allowed during search, as stepping stones for later mutations.

**Outer loop.**
- Regularized evolution with warm-start and restarts.
- Abstract execution removes redundancy and catches invalid programs. It hashes programs, which gives about a 10× saving through caching (§2.2).
- After search comes **funnel selection** on meta-validation tasks of growing size, then automatic simplification (§2.3).

**Fitness and anti-gaming.**
- The fitness is validation accuracy or perplexity on small proxies (§2.2).
- They name and measure **meta-overfitting**: "the search fitness keeps growing, but the meta-validation metric declines" (§2.3).
- Selection on a ladder of larger meta-validation tasks is the guard. Runs that meta-overfit later tended to generalize better (Fig. 3).

**Scale and compute.**
- Each proxy evaluation takes up to 20 minutes on one TPU v2 chip.
- "Each search experiment utilizes 100 TPU V2 chips and runs for ∼72h". 200–300K programs are generated per run, of which about 20–30K are actually evaluated.
- "This results in a total cost of ∼3K TPU V2 days" (§2.2).

**Transfer.** Lion transferred to ViT image classification, vision-language contrastive learning, diffusion and language modelling. For example, it raised ViT accuracy on ImageNet by up to 2% (Abstract). The authors list where it does *not* help: gains are not significant on some large language and image-text datasets, and "smaller if using strong augmentations or a small batch size (<64)" (§1, §6).

**Limitations.** The compute is very large. The search space is narrow, starting from AdamW. A good result still needed a multi-stage selection funnel.

**Relevance to us.** Two lessons. First, **warm-start the genome from the current hand-designed learner** (our v1/v2 settings), so search refines a working design rather than starting from noise. Co-Reyes et al. [L9] found the same. Second, **meta-overfitting is real and measurable**. We need a held-out ladder: different world seeds, then a later world stage, then longer lives. Candidates that improve the search fitness but not the held-out ladder should be rejected. This maps directly onto protected evaluation (invariant 7) and onto the PR gate.

## 9. Evolving Reinforcement Learning Algorithms (Co-Reyes et al., 2021)

**Citation.** John D. Co-Reyes, Yingjie Miao, Daiyi Peng, Esteban Real, Sergey Levine, Quoc V. Le, Honglak Lee, Aleksandra Faust. *Evolving Reinforcement Learning Algorithms.* ICLR 2021. arXiv:2101.03958. `https://arxiv.org/abs/2101.03958` [L9]

**Problem.** Search symbolic loss functions for value-based RL that work across domains.

**What is searched.** A typed DAG that computes the loss for a DQN-style agent. Its nodes include maths ops and neural-network modules, and its inputs are MDP variables (§3). The population is started either from scratch or **bootstrapped with DQN**.

**Outer loop.** Regularized evolution with population 300, plus a functional-equivalence check and an **early hurdle on CartPole**: only candidates scoring above 0.6 normalized go on to the harder training environments (§3.3, §4).

**Fitness.** The sum of normalized training returns across training environments. Training uses 3 sampled environments, classic control plus MiniGrid. Test environments include Atari (§4).

**Scale and compute.** "The search is done over 300 CPUs and run for roughly 72 hours, at which point around 20,000 programs have been evaluated" (§4).

**Transfer.**
- Learning from scratch rediscovered TD learning.
- Bootstrapped from DQN, it found DQNReg and DQNClipped, which generalized to unseen classic control, MiniGrid and Atari.
- More training environments gave better algorithms. Bootstrapping sped up convergence and gave higher final performance (Fig. 3).

**Limitations.** "Our work is limited to discrete action and value-based RL algorithms that are close to DQN" (§5).

**Relevance to us.** This is about the smallest compute (CPU-only) at which symbolic RL-rule search has produced transferable results. Even so it needed about 20,000 evaluations, roughly 42 machine-days of our lives *(my arithmetic)*. It supports warm-starting from the hand-designed learner, and a cheap "hurdle" check early in a life that ends bad candidates before the full 30 minutes.

## 10. Discovered Policy Optimisation (LPO/DPO) (Lu et al., 2022)

**Citation.** Chris Lu, Jakub Grudzien Kuba, Alistair Letcher, Luke Metz, Christian Schroeder de Witt, Jakob Foerster. *Discovered Policy Optimisation.* NeurIPS 2022. arXiv:2210.05639. `https://arxiv.org/abs/2210.05639` [L10]

**Problem.** Black-box algorithm discovery tends either to fail or to lose guarantees. Instead, restrict the search to a theoretically safe family (Mirror Learning) and learn one component inside it.

**What is searched.**
- Only the **drift function** inside Mirror Learning (PPO is one instance).
- It is a tiny MLP (1 hidden layer, 128 units, no biases) of the probability ratio r and the advantage A, built so it always satisfies the drift conditions. That guarantees convergence by construction (§4).
- The learned drift was then distilled into a closed-form rule, DPO (§7).

**Outer loop.** ES with antithetic sampling, using evosax, population 32 (App. A.2).

**Fitness.** The final return after a full inner PPO-style training in Brax, estimated from 10,240 evaluation rollouts per training (App. A.1).

**Scale and compute.** "Meta-training was only done on a single machine with 4 V100 GPU's with synchronous updates", over about 2 days, with 672 generations (App. A.2). That is about 21,500 inner trainings at population 32 *(my arithmetic)*. Brax trains each inner agent "within minutes" on an accelerator (§4).

**Transfer.** LPO trained on Ant alone beat PPO on unseen Brax environments and under unseen hyperparameters (§5.2). DPO, the closed-form version, matched LPO.

**Limitations.** "A possible weakness of this approach is that it could overfit to specific code-level details of the implementation" (§8).

**Relevance to us.** This is the most promising template for *our* compute: a **constrained, low-dimensional learned component inside a known-good learner**, searched by ES over whole lives, then *distilled into a readable rule*. A guaranteed-safe parameterization means most genomes are valid, so few evaluations are wasted on broken learners. Its weakness matches our invariant 5. A learned rule can overfit to the specifics of our implementation, so any implementation change should start a new schema version rather than reuse old fitness data.

## 11. Meta-Learning through Hebbian Plasticity in Random Networks (Najarro & Risi, 2020)

**Citation.** Elias Najarro, Sebastian Risi. *Meta-Learning through Hebbian Plasticity in Random Networks.* NeurIPS 2020. arXiv:2007.02686. `https://arxiv.org/abs/2007.02686` [L11]

**Problem.** Instead of evolving weights, evolve per-synapse plasticity rules, so that a randomly initialized network organizes its own weights during its lifetime without a reward signal.

**What is searched.** For every connection, a generalized Hebbian "ABCD" rule, Δw_ij = η_w·(A_w·o_i·o_j + B_w·o_i + C_w·o_j + D_w) (§3, Eq. 1). Each connection has its own η, A, B, C and D. On CarRacing that is 456,640 coefficients for a 92,690-weight network (§4).

**Outer loop.** ES with population 200 (CarRacing) or 500 (quadruped); the hyperparameters were "found by trial-and-error" (§3.1).

**Fitness.** Episode reward (CarRacing, 84×84 RGB pixels), or distance walked over 1,000 steps averaged over two robot shapes. A third damaged shape was held out (§4).

**Scale and compute.** The text I read reports population sizes but not hardware or wall-clock time.

**Transfer.**
- On CarRacing the Hebbian network scored 872 ± 11 against 711 ± 16 for fixed weights. It generalized better to procedurally generated tracks it had not seen (§4.1).
- The quadruped adapted to the held-out leg damage within about 100 steps. Fixed weights did not (Abstract, §4.1).
- The authors say "the Hebbian networks are more general but performance for a particular task/robot morphology can be less" (§5).

**Limitations.** The "lifetime" here is one 1,000-step episode, so this is fast adaptation, not long learning. The rule is local, with no reward and no credit assignment. It needs very many ES evaluations for its parameter count.

**Relevance to us.** This is a direct example of "a genome of small learned rules", and of an inherited *capacity to learn* rather than inherited weights. That is the `life` precedent [prior-art-life] and the stance of decision 0004. But one rule *per synapse* gives a genome far too large for about 480 evaluations a day. The scalable version for us would **share** a few rule coefficients per layer or per module type. That keeps the genome to tens or hundreds of numbers, which is GP territory.

## 12. Discovering Temporally-Aware Reinforcement Learning Algorithms (Jackson et al., 2024)

**Citation.** Matthew T. Jackson, Chris Lu, Louis Kirsch, Robert T. Lange, Shimon Whiteson, Jakob N. Foerster. *Discovering Temporally-Aware Reinforcement Learning Algorithms.* ICLR 2024. arXiv:2402.05828. `https://arxiv.org/abs/2402.05828` [L13]

**Problem.** Learned objectives usually ignore how much of the training budget has passed. Humans change how they learn as a deadline approaches.

**What is searched.** LPG and LPO, with the lifetime fraction n/N and log N added as inputs, giving TA-LPG and TA-LPO (§4.1).

**Outer loop.** It compares meta-gradients with ES. ES uses "antithetic task sampling": each antithetic pair of candidates is evaluated on the *same* sampled task (§4.2).

**Key finding.** "Commonly used meta-gradient approaches fail to discover such adaptive objective functions while evolution strategies discover highly dynamic learning rules" (Abstract). The reason given is that truncated backpropagation is "a myopic proxy objective". ES optimizes "the agent's return at the end of training" (§4.2).

**Scale and compute.** LPO meta-training "was done on 2 A100 GPUs with synchronous updates" (App. A.2).

**Transfer.** TA-LPO, meta-trained only on MinAtar SpaceInvaders, generalized across MinAtar and Brax, and to other training horizons (§5).

**Relevance to us.** The genome's rules should be allowed to read **the life's clock**, as Alet's combiner also does with t/T. Black-box ES or BO over whole lives is not just the method we can afford. Here it is the method that finds schedules spanning a whole life. Antithetic or paired evaluation on the same world seed is a cheap way to reduce noise when there are few tabs. One open tension: our lives are meant to be "forever" (decision 0014), which has no fixed N. A clock input would need a defined meaning, such as CU since birth.

## 13. Open-Endedness is Essential for Artificial Superhuman Intelligence (Hughes et al., 2024)

**Citation.** Edward Hughes, Michael Dennis, Jack Parker-Holder, Feryal Behbahani, Aditi Mavalankar, Yuge Shi, Tom Schaul, Tim Rocktäschel. *Open-Endedness is Essential for Artificial Superhuman Intelligence.* ICML 2024 (PMLR 235), position paper. arXiv:2406.04268. `https://arxiv.org/abs/2406.04268` [L12]

**Problem.** Define open-endedness precisely, and argue that it is necessary for superhuman AI.

**Content.** "From the perspective of an observer, a system is open-ended if and only if the sequence of artifacts it produces is both novel and learnable" (§2.1). Novelty means artifacts become increasingly unpredictable to the observer's model. Learnability means a longer history makes them more predictable. The proposed path is to build open-ended systems on top of foundation models (§3).

**Compute, fitness and generalization.** None. This is a definitions and agenda paper.

**Limitations for us.** Its main path assumes foundation models, which cold start rules out (invariant 1). It is still useful for the definition.

**Relevance to us.** The definition depends on the observer. That gives a harness-side test of whether the population of genomes is *going anywhere*: a fixed observer, our checkups, should find new lineages' behaviour neither already predictable nor unlearnable. This is close in spirit to ANNECS [L7b]. It is a possible lineage-level metric, and is not adopted.

---

## Cross-paper comparison

| # | System | What is searched (representation) | Search method | Fitness signal | Compute (reported) | Generalized / didn't |
|---|---|---|---|---|---|---|
| L1 | AutoML-Zero | Programs (Setup/Predict/Learn) over 65 basic ops | Regularized evolution with migration, FEC, hurdles | Validation accuracy, separate search and select tasks | 10k models/s/core; W = 10k CPU processes, 5 days | Beat a 2-layer NN on binary CIFAR, SVHN, ImageNet; no conv or batch ops; hyperparameter coupling |
| L2 | LPG | Backward-LSTM update rule giving targets for π and open-meaning y | Meta-gradient over K-update windows, 960 lifetimes | Return within the lifetime | ~10^10 steps; 24 h on 16-core TPUv2 | Toy worlds → 14 superhuman Atari games; still behind A2C |
| L3 | DiscoRL | Meta-network targets for π, y, z, plus a lifetime meta-RNN | Meta-gradient (20-update window), 128–206 agents | Collective return; hyperparameters tuned only on training set | 1,024–2,048 TPUv3 cores × 60–64 h | SOTA on Atari; beat all published methods on unseen ProcGen; grid-world discovery much worse |
| L4 | EPG | Temporal-conv loss network over recent experience | ES (W = 256, V = 64), rank-based | Final return after full inner training | 5,000 epochs; ~196M steps per worker | Right→left goals; "gains on performance by losing on generality" |
| L5 | Curiosity programs | Typed DAG DSL (NNs, buffers, kNN, losses), ≤ 7 modules; intrinsic part plus combiner | Enumeration + kNN surrogate (ε-greedy) + cheap-env filter + learning-curve early stop | Distinct cells visited (harness-only), then extrinsic reward | ~26k programs; 13 GPU-days (phase 1), ~2 GPU-weeks total | Grid world → lunar lander, acrobot, hopper, ant; on par with published |
| L6 | AI-GAs | (Agenda) architectures, learning algorithms, environments | n/a | n/a | n/a | Position paper |
| L7 | POET / E-POET | Environment encodings (parameter vector → CPPN) + agent weights | Mutation + minimal criterion + novelty + ES + transfer | Score threshold (230); ANNECS for progress | 256 cores × ~10 days; 750 cores × ~12 days | Solved environments direct ES could not; fixed encoding plateaus |
| L8 | Lion | Symbolic optimizer program, warm-started from AdamW | Regularized evolution + restarts + abstract execution + funnel selection | Proxy validation metric; larger meta-validation ladder | ~3K TPUv2-days | Many large tasks; small or none with batch < 64, strong augmentation, some LM and image-text |
| L9 | Evolving RL algos | Typed loss DAG for value-based RL, optionally seeded with DQN | Regularized evolution + FEC + CartPole hurdle | Normalized training return over 3 environments | 300 CPUs × 72 h, ~20k programs | Unseen classic control, MiniGrid, Atari; DQN-like only |
| L10 | LPO / DPO | Tiny MLP drift function inside Mirror Learning (valid by construction) | ES (antithetic), population 32 | Final return after full training | 4× V100 × ~2 days, 672 generations | Ant-only → unseen Brax environments and hyperparameters; may overfit implementation |
| L11 | Hebbian ABCD | Per-synapse plasticity coefficients (~457k) on random weights | ES, population 200–500 | Episode reward or distance walked | Not stated in text read | Unseen tracks; held-out leg damage; narrower peak score |
| L12 | Open-endedness | (Definition) novelty + learnability to an observer | n/a | n/a | n/a | Position paper; assumes foundation models |
| L13 | TA-LPG / TA-LPO | LPG / LPO with lifetime-fraction inputs | ES vs meta-gradient (ES wins); antithetic task sampling | Final return | 2× A100 (LPO) | Unseen horizons and environments |

**What the table shows.** Every success that transferred used one or both of two ingredients:
- a **narrow, well-structured space**, usually warm-started or valid by construction (L8, L9, L10, L13);
- a **domain-free interface**, where the rule sees the agent's own predictions and rewards rather than raw observations (L2, L3, L5).

The broad, from-scratch spaces (L1, and L5 at small scale) needed either huge numbers of evaluations or enumeration with heavy pruning. Every method that works with a whole-life, black-box fitness used ES or evolution with rank or paired evaluations to handle noise (L4, L10, L13). Note that at scale, discovery in toy worlds did worse (L3). Diversity of training worlds was the strongest lever for generality (L2, L3, L9).

---

## What this suggests for our representation

*This section is the agent's reading of the sources. These are proposals, not decisions, and none is confirmed by the user.*

1. **Keep the searchable genome small and typed. Aim for tens of numbers at first, not thousands.** At about 48 lives per machine-day, even CPU-scale symbolic searches (~20k evaluations [L9], ~26k [L5]) would cost 40–55 machine-days. A GP or Bayesian-optimization picker works well only up to roughly tens of dimensions. So the genome should be continuous knobs and a few switches, with *shared* rule coefficients per module type, not per synapse [L11]. Structural (program-level) search can come later, if ever.

2. **Warm-start from the current hand-designed learner, and build the genome so that every point is a valid learner.** The successes searched *around* a working design (AdamW [L8], DQN [L9], PPO inside Mirror Learning [L10]). Bounded parameterizations (clamped rates, mixing weights that sum to 1, rules that reduce to v2 at a known point) avoid wasting 30-minute lives on broken learners.

3. **Make the learned rules read the agent's own signals, not pixels.** LPG and DiscoRL transfer because their rules see only predictions, rewards and termination [L2, L3]. Alet's curiosity programs are typed so they work on any input [L5]. Our learned consolidation, replay or intrinsic-reward rules should take as input things like prediction errors, latent learning progress, novelty ν_t, memory age and the life's clock. They should not take raw frames. This also keeps learner inputs separate from harness data (invariant 2).

4. **Give rules a clock.** Let schedule-type genes depend on "CU since birth" (or a log of it), as Alet's combiner uses t/T [L5] and TA-LPO uses n/N [L13]. Because lives are "forever" (decision 0014), define the clock as elapsed CU rather than a fraction of a fixed N.

5. **Use the picker as a surrogate, and spend lives sparingly.** Alet's kNN-over-structure surrogate [L5] is the closest precedent for our GP picker. To stretch the budget:
   - remove duplicate candidates by fingerprint [L1, L8, L9];
   - add an early hurdle at the first checkup that ends clearly bad lives [L1, L5, L9];
   - treat early-checkup scores as low-fidelity readings of the full-life score.

   Agent's call: a multi-fidelity GP would fit this. That is a guess, not tested.

6. **Treat fitness as noisy by design.**
   - Rank-transform scores [L4].
   - Run paired comparisons on the same world seed: antithetic or common random numbers [L10, L13].
   - Score each candidate on more than one seed [L4].
   - Only promote a genome over the incumbent if it beats the incumbent's best of several recent scores [L7b].

   This matters more for us than for any paper here, because one life gives one data point [V2 §6.3].

7. **Build a held-out ladder against meta-overfitting before any GitHub inheritance.** Lion's funnel [L8] and DiscoRL's "tune only on training environments" [L3] amount to a rule: the search fitness and the promotion test must differ. For us that means held-out world seeds, then a later world stage, then longer lives. Promotion goes through the PR gate. This is invariant 7 put into practice.

8. **Choosing the worlds matters as much as choosing the genome.** More, and more varied, training worlds was the most consistent lever for generality [L2, L3, L9]. The pure grid-world rule failed at scale [L3]. Clune's "burden shift" [L6] says the same. Which world stages a genome is scored on is a recorded prior (invariant 2). An open-ended stage generator (minimal criterion plus novelty [L7]) is a later possibility, not a v2 item.

9. **Distill what is found into readable rules, and version the schema whenever the implementation changes.** LPO became DPO [L10], and AutoML-Zero and Lion simplified their programs [L1, L8]. Learned rules can overfit implementation details [L10], and constants can hide task-specific values [L1]. So any change to the learner made by a research agent (invariant 5) should bump `schemaVersion` and start a fresh fit, as the `life` prior art already does [prior-art-life].

10. **Keep the expectation of a payoff modest.** Every positive result here used between about 4 GPUs for 2 days [L10] and about 2,000 TPU cores for 2.5 days [L3]. Nothing here shows discovery working at ~500 noisy 30-minute evaluations a day. Whether our genome search beats hand-tuning is a hypothesis (invariant 8). The hand-tuned v2 arm should stay as a permanent control.

---

## Sources

### Verified (fetched 2026-10-22; title, authors, year and venue confirmed)

- **[L1]** Real, Liang, So, Le. AutoML-Zero. ICML 2020. `https://arxiv.org/abs/2003.03384`
- **[L2]** Oh, Hessel, Czarnecki, Xu, van Hasselt, Singh, Silver. Discovering Reinforcement Learning Algorithms. NeurIPS 2020. `https://arxiv.org/abs/2007.08794`
- **[L3]** Oh, Farquhar, Kemaev, Calian, Hessel, Zintgraf, Singh, van Hasselt, Silver. Discovering state-of-the-art reinforcement learning algorithms. *Nature* 648, 312–319 (2025). `https://www.nature.com/articles/s41586-025-09761-x`
- **[L4]** Houthooft, Chen, Isola, Stadie, Wolski, Ho, Abbeel. Evolved Policy Gradients. NeurIPS 2018. `https://arxiv.org/abs/1802.04821`. The NeurIPS proceedings page lists the second author as "Yuhua Chen"; arXiv and the PDF say "Richard Y. Chen".
- **[L5]** Alet, Schneider, Lozano-Pérez, Kaelbling. Meta-learning curiosity algorithms. ICLR 2020. `https://arxiv.org/abs/2003.05325`
- **[L6]** Clune. AI-GAs: AI-generating algorithms. arXiv preprint 1905.10985 (2019/2020). `https://arxiv.org/abs/1905.10985`
- **[L7a]** Wang, Lehman, Clune, Stanley. POET. arXiv preprint 1901.01753 (2019). `https://arxiv.org/abs/1901.01753`
- **[L7b]** Wang, Lehman, Rawal, Zhi, Li, Clune, Stanley. Enhanced POET. ICML 2020. `https://arxiv.org/abs/2003.08536`
- **[L8]** Chen, Liang, Huang, Real, Wang, Pham, Dong, Luong, Hsieh, Lu, Le. Symbolic Discovery of Optimization Algorithms. NeurIPS 2023. `https://arxiv.org/abs/2302.06675`
- **[L9]** Co-Reyes, Miao, Peng, Real, Levine, Le, Lee, Faust. Evolving Reinforcement Learning Algorithms. ICLR 2021. `https://arxiv.org/abs/2101.03958`
- **[L10]** Lu, Kuba, Letcher, Metz, Schroeder de Witt, Foerster. Discovered Policy Optimisation. NeurIPS 2022. `https://arxiv.org/abs/2210.05639`
- **[L11]** Najarro, Risi. Meta-Learning through Hebbian Plasticity in Random Networks. NeurIPS 2020. `https://arxiv.org/abs/2007.02686`
- **[L12]** Hughes, Dennis, Parker-Holder, Behbahani, Mavalankar, Shi, Schaul, Rocktäschel. Open-Endedness is Essential for Artificial Superhuman Intelligence. ICML 2024. `https://arxiv.org/abs/2406.04268`
- **[L13]** Jackson, Lu, Kirsch, Lange, Whiteson, Foerster. Discovering Temporally-Aware Reinforcement Learning Algorithms. ICLR 2024. `https://arxiv.org/abs/2402.05828`

### Unverified

- None of the sources used above. Not covered here, and not checked: MetaGenRL (Kirsch et al., ICLR 2020), cited in [L2]'s Table 1; ML3 (Bechtle et al.); VeLO and other learned-optimizer work (Metz et al.); LLM-driven discovery (for example DiscoPOP). DeepMind's `disco_rl` code repository (`https://github.com/google-deepmind/disco_rl`) returned HTTP 200, but its contents were not read.

### Project references

- **[SYN]** `docs/research/2026-10-16-research-synthesis-v0.1.md`
- **[V2]** `docs/specs/2026-10-22-learner-v2-capabilities-design.md`, §6 (life length, budget, one life is one data point)
- **[prior-art-life]** `docs/research/2026-10-19-prior-art-life.md` (GP-UCB picker, schema versioning)
- Decisions 0001, 0004, 0005, 0014 in `docs/decisions/`
