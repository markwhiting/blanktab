# Literature note: learned learning rules and meta-learned plasticity

**22 October 2026 | agent literature review, verified sources only**

## Why this note exists

The user wants the learner's whole learning workflow written down as data: a "genome" of matrices, switches, and small learned rules. A population meta-learner would sample and improve that genome automatically. **How it learns** would be shared through GitHub. **What it learns in a life** would stay local. This note surveys work in which the update rule itself has parameters that an outer loop optimizes. For each paper it asks three questions:

1. Can the outer loop run without backpropagating through a whole life? A life is about 19–26k steps, roughly 30 minutes [`docs/specs/2026-10-22-learner-v2-capabilities-design.md` §6], and we cannot take meta-gradients across tabs.
2. What would fit our own tensor library on WebGPU at about 30 steps/s? The step is bound by latency, not FLOPs [same spec, §2], and the network has about 2.19 M parameters at GPU widths (2,185,769, counted by building the model; an earlier draft said 2.19 M) with a 256-d GRU core.
3. How does the paper's work map onto inheriting *how* it learns (Baldwinian) versus inheriting *what* it learned (Lamarckian)?

**Method.** For every source, the arXiv abstract page was fetched to confirm title, authors, and date. The full PDF was then downloaded and read as text, and the venue was taken from the PDF header or footer, or from an independent index (Crossref, mlanthology). Every quote below was copied from the paper text, with its section. Numbers come from the papers; anything that is my inference is marked **[agent's reading]**. Scratch copies are in `.local/lit-llr/`, which is gitignored.

Citation keys are `[L1]`–`[L17]`. They are local to this note.

---

## 1. Learned optimizers: the update rule is a small network

### [L1] Andrychowicz et al. 2016: Learning to learn by gradient descent by gradient descent
Marcin Andrychowicz, Misha Denil, Sergio Gomez, Matthew W. Hoffman, David Pfau, Tom Schaul, Brendan Shillingford, Nando de Freitas. *NIPS 2016* (30th Conference on Neural Information Processing Systems, Barcelona; from the PDF footer). https://arxiv.org/abs/1606.04474

- **Parameterization.** The update rule is a two-layer LSTM with 20 hidden units per layer. It is applied separately to each coordinate, takes that coordinate's gradient, and outputs its update. *Shared (meta-learned):* the LSTM weights φ. *Within a lifetime:* the optimizee weights θ and one LSTM hidden state per coordinate. "All LSTMs have shared parameters, but separate hidden states." (§2.1, Fig. 3 caption)
- **Outer loop.** Gradient descent on φ, using BPTT through the unrolled optimization. Second derivatives are dropped: "This assumption allows us to avoid computing second derivatives of f." (§2) On MNIST, each optimization ran 100 steps and "the trained optimizers were unrolled for 20 steps" (§3.2).
- **Cost.** No compute totals are reported.
- **Generalized.** The optimizer kept working when run for 200 steps after being trained on 100. It also handled an MLP with 40 hidden units instead of 20, and two hidden layers instead of one (§3.2).
- **Failed.** "changing the activation function to ReLU makes the dynamics of the learning procedure sufficiently different that the learned optimizer is no longer able to generalize." (§3.2)
- **Limitations.** Very short truncations (20 steps), and the gradients are biased by the dropped second-order terms.
- **Relevance to us.** BPTT through a 30-minute life is ruled out, so we cannot copy the outer loop. The parameterization does carry over: one tiny rule shared across all coordinates, with per-coordinate state. That is a genome whose size does not depend on network width. A per-coordinate LSTM over 2.19 M parameters at 30 steps/s would be a heavy elementwise kernel. A per-coordinate MLP with a handful of units, as in VeLO [L2], is the cheaper form. **[agent's reading]** The ReLU failure is a warning: a rule meta-trained on one learner configuration can break when we change the architecture (invariant 9).

### [L2] Metz et al. 2022: VeLO: Training Versatile Learned Optimizers by Scaling Up
Luke Metz, James Harrison, C. Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, Jascha Sohl-Dickstein. arXiv preprint, 17 Nov 2022; the abstract page lists no venue. https://arxiv.org/abs/2211.09760

- **Parameterization.** A hierarchical hypernetwork. One per-tensor LSTM with 512 hidden units runs on features aggregated from each weight tensor. It *generates the weights* of a per-parameter MLP with 4 hidden units, and that MLP produces each parameter's update. The LSTMs exchange a max-pooled global context signal (§3.1). *Shared:* the LSTM parameters. *Within a lifetime:* the model weights, the LSTM states, and the generated MLP weights.
- **Outer loop.** Evolution strategies with **full-length unrolls**: "Unlike past work …, we use full length unrolls, and train each model to completion for each meta-gradient evaluation." (§3.3) Per-task gradients are normalized to unit length before averaging, and a curriculum grows both the number of steps and the problem size.
- **Cost.** "Meta-trained with approximately four thousand TPU-months of compute" (abstract). It ran on 1K–4K TPU accelerators for about 4 weeks, with batches of 10K–40K tasks (§3.3).
- **Generalized.** On the 83-task VeLOdrome benchmark it beats learning-rate-tuned Adam on every task, and is more than 4× faster on about half of them (Fig. 1). It also did well on NeRF and MLP-Mixer, which were never meta-trained on (§4.3).
- **Failed (§4.4).**
  - Models above about 500M parameters.
  - Runs longer than 200K steps. It was meta-trained on runs of at most 20K steps and fine-tuned up to 200K. "VeLO's performance relative to baseline optimizers worsens as the number of training steps approaches, and then exceeds, 200K."
  - RL (Brax Ant). With ES gradients "it fails to learn a locomotive gait at all."
- **Overhead.** It costs about 46× more per parameter than SGD (§B, time model) and needs "about twice the memory requirements of Adam" (limitations list).
- **Relevance to us.** VeLO's outer loop is ES over whole lives with a fitness taken at the end, which is the shape our tabs allow. Its cost is about seven orders of magnitude beyond our budget. **[agent's reading]** Two lessons carry over:
  - **Horizon.** Learned rules degrade past the horizon they were meta-trained on, and our lives are meant to run on (decision 0014).
  - **Domain.** A rule meta-trained on supervised tasks failed on RL, which is our setting.

### [L3] Metz et al. 2019: Meta-Learning Update Rules for Unsupervised Representation Learning
Luke Metz, Niru Maheswaranathan, Brian Cheung, Jascha Sohl-Dickstein. *ICLR 2019* (PDF header: "Published as a conference paper at ICLR 2019"). https://arxiv.org/abs/1804.00222

- **Parameterization.** A neuron-local unsupervised rule. Every neuron has an "update network" MLP that reads its forward activations and an error signal sent back through *learned* backward weights V, not Wᵀ. The weight change is a product of pre- and post-synaptic signals generated by these networks. "All update networks share meta-parameters θ." (§3.2) *Within a lifetime:* the base weights W and the backward weights V. The labeled meta-objective is used only in meta-training: "this meta-objective is only used during meta-training and not used when applying the learned update rule." (§3.3)
- **Outer loop.** SGD on θ through truncated backprop through time. Stability required batch norm and a cap on the size of the update step (§4.1).
- **Cost.** "Training uses 512 workers … Training takes ∼8 days, and consists of ∼200 thousand updates to θ" (§4.3).
- **Generalized.** Meta-trained on 2–5 layers of 64–512 units, the rule worked up to 11 layers of 10,000 units. Meta-trained only on ReLU, it worked with other activations, including the step function, since it does not use base-model gradients (§5).
- **Failed.** On text (IMDB), a 30-hour checkpoint helped, but with longer meta-training "the learned optimizer “meta-overfits” to the image domain resulting in poor performance." (§5)
- **Relevance to us.** This is the closest prior art for "the learning rule is local, label-free, and shared across neurons", which matches our cold-start, pixels-only setting. It also shows that meta-training longer on a narrow world can make the rule *worse* elsewhere. **[agent's reading]** Fitness should include held-out rooms or worlds, as in the 0015 held-out check, or the genome will overfit to the curriculum.

### [L4] Metz et al. 2019: Understanding and correcting pathologies in the training of learned optimizers
Luke Metz, Niru Maheswaranathan, Jeremy Nixon, C. Daniel Freeman, Jascha Sohl-Dickstein. *ICML 2019*, PMLR 97 (from the PDF footer). https://arxiv.org/abs/1810.10180

- **Main claim.** Unrolled meta-gradients are "either strongly biased (for short truncations) or have exploding norm (for long truncations)" (abstract). The fix is a variational, smoothed outer loss, with a reparameterization estimator and an ES estimator weighted against each other dynamically.
- **Relevance to us.** This paper is the reason to avoid truncated backprop meta-gradients altogether. Smoothing the fitness with ES-style noise is itself a stabilizer.

### [L5] Metz et al. 2020: Tasks, stability, architecture, and compute: Training more effective learned optimizers, and using them to train themselves
Luke Metz, Niru Maheswaranathan, C. Daniel Freeman, Ben Poole, Jascha Sohl-Dickstein. arXiv preprint, 23 Sep 2020; no venue listed. https://arxiv.org/abs/2009.11243

- **Parameterization.** A per-tensor LSTM feeds a per-parameter feedforward network (§3.3), as in [L2] but without the hypernetwork.
- **Outer loop.** ES "obviating the need to compute derivatives through the unrolled optimization process" (§3.1). Unrolls are 240–360 steps, and training then continues with PES [L6]. The ES noise has standard deviation 0.01, with antithetic sampling on "1024 multi-core CPU workers". The task set is about 6,000 tasks.
- **Cost.** "roughly entails 60K CPU cores for around a month, or around 5k CPU years" and "on the order of 200 megawatt hours" (App. H.2).
- **Generalized.** Some out-of-distribution tasks. The learned optimizer can train itself from scratch.
- **Relevance to us.** This is the best documented price of meta-learning a general rule. **[agent's reading]** Our genome must be small enough that hundreds to thousands of lives can move it, not millions.

### [L6] Vicol, Metz, Sohl-Dickstein 2021: Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies
Paul Vicol, Luke Metz, Jascha Sohl-Dickstein. *ICML 2021* (PDF: Proceedings of the 38th ICML). https://arxiv.org/abs/2112.13835

- **Method.** In PES, "we accumulate the perturbations experienced by the outer parameters in each partial unroll—rather than starting perturbations from scratch as in vanilla ES—which yields an unbiased estimate of the gradient even when using truncated sequences." (§1) In practice this means "not resetting the particles between unrolls" and "accumulating the perturbations ξ each particle has experienced over multiple unrolls" (§4).
- **Relevance to us.** In principle PES fits the tab setting: each tab is a persistent particle, and the outer parameters can update every few minutes without bias. **[agent's reading]** The catch is that PES updates θ in the middle of an unroll, so a tab's genome would change during its life. That conflicts with holding the implementation fixed within a run (invariant 7) and complicates comparing lives. Treat it as an option to record as a decision, not a default.

## 2. Meta-learned plasticity: fixed weights plus a learned Hebbian component

### [L7] Miconi, Clune, Stanley 2018: Differentiable plasticity: training plastic neural networks with backpropagation
Thomas Miconi, Jeff Clune, Kenneth O. Stanley. *ICML 2018*, PMLR 80. https://arxiv.org/abs/1804.02464

- **Parameterization.** Each connection's effective weight is w + α·Hebb. The trace follows Hebb(t+1) = η·xᵢ(t−1)·xⱼ(t) + (1−η)·Hebb(t), with Oja's rule as an alternative. η is a single learned scalar for the whole network. The paper is explicit about which part lives where: "The Hebbian trace Hebbi,j is initialized to zero at the beginning of each lifetime/episode: it is purely a lifetime quantity. The parameters wi,j and αi,j, on the other hand, are the structural parameters of the network that are conserved across lifetimes" (§2).
- **Outer loop.** Gradient descent through each episode, between lifetimes.
- **Cost.** Not reported in the passages checked.
- **Results.** On image memorization, the plastic network reached its error target "within 2,000 episodes, which is 250 times faster than the LSTM" (§4, pattern memorization). Per-connection α beat a single shared α (§4.3). On Omniglot, plasticity only in the last layer added 320 parameters to 111,426 and was competitive (§4.4). It also helped on a maze task.
- **Limitations.** Episodes are short (hundreds of steps), and meta-gradients run through the whole episode.
- **Relevance to us.** This is the simplest genome/lifetime split with a direct mapping. α and η are "how it learns" and go into the genome. The Hebbian trace is "what it learns" and stays local. **[agent's reading]** One extra trace tensor per plastic layer, updated elementwise, is cheap in our library and needs no GPU readback. The outer loop would have to be ES or GA rather than backprop.

### [L8] Miconi, Rawal, Clune, Stanley 2019: Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity
Thomas Miconi, Aditya Rawal, Jeff Clune, Kenneth O. Stanley. *ICLR 2019* (arXiv posting 24 Feb 2020). https://arxiv.org/abs/2002.10585

- **Parameterization.** The network computes its own plasticity rate, M(t). Simple version: Hebb(t+1) = Clip(Hebb(t) + M(t)·xᵢ·xⱼ). Retroactive version: the Hebbian product goes into an eligibility trace E with a trainable decay η, and Hebb(t+1) = Clip(Hebb(t) + M(t)·E(t)) (§3.2). The genome holds the weights that compute M, plus α, w, and η. The traces and Hebb are lifetime state.
- **Outer loop.** Gradient descent; A2C for the RL tasks (§4.1).
- **Results.** Cue-reward association and the maze both improved. On Penn Treebank, retroactive neuromodulation gave "about 1.7 perplexity improvement vs. the Baseline LSTM" (§4.3).
- **Relevance to us.** A neuromodulator is exactly the kind of "small learned rule" that can gate *when* local learning happens, for example from surprise or reward. The authors frame it as a step above learning in activations alone: "the system could now determine its own arbitrary weight-modification scheme, which might make use of any signal it can compute (reward predictions, surprise, saliency, etc.)" (§5). **[agent's reading]** An eligibility trace with a scalar gate per layer fits a GPU step cheaply and gives the genome a meaningful switch: which signal drives M.

### [L9] Najarro & Risi 2020: Meta-Learning through Hebbian Plasticity in Random Networks
Elias Najarro, Sebastian Risi. *NeurIPS 2020*. https://arxiv.org/abs/2007.02686

- **Parameterization.** The weights start random in every lifetime, and **only the learning rules are evolved**: the method "only searches for synapse-specific Hebbian learning rules that allow the network to continuously self-organize its weights" (abstract). Each synapse follows the ABCD rule, Δw = η·(A·oᵢoⱼ + B·oᵢ + C·oⱼ + D), so there are 5 coefficients per synapse.
  - On CarRacing from 84×84 RGB pixels, this comes to 456,640 coefficients.
  - The 1,362 conv weights do not change during the lifetime (§4).
- **Outer loop.** An evolution strategy with population 200 (CarRacing) or 500 (quadruped) (§4).
- **Cost.** Generation counts and hardware are not stated in the text checked.
- **Generalized.** On the quadruped, the evolved rules adapted to an unseen leg-damage morphology: 452 ± 95 distance, against 68 ± 56 for static weights (Table 1). This happened with no reward signal during the lifetime.
- **Failed.** Where static weights solved the task, they reached *higher* reward than the Hebbian network, e.g. 1604 against 1051 with no damage (Table 1). The authors note that larger rule spaces "need more generations to be explored by the ES algorithm" (App.).
- **Relevance to us.** This is the purest version of "genome = how it learns, lifetime = what it learns", with no inherited weights at all, from pixels and trained by ES. **[agent's reading]** The per-synapse version is far too large for our population (5 × 2.19 M coefficients). Rules shared per layer or per tensor, as in [L1] and [L3], are the scale that fits.

### [L10] Kirsch & Schmidhuber 2021: Meta Learning Backpropagation And Improving It (VSML)
Louis Kirsch, Jürgen Schmidhuber. *NeurIPS 2021* (camera-ready on arXiv v4). https://arxiv.org/abs/2012.14905

- **Parameterization.** Every weight is replaced by a tiny LSTM, and all of these LSTMs share parameters V_M. Each has its own state V_L, and the LSTMs pass forward and backward messages. Learning happens entirely in the forward dynamics. The paper reports |V_M| ≈ 2,400 meta-variables against |V_L| ≈ 257,200 learned variables (§4). The design principle is that "simple weight-sharing and sparsity in an NN is sufficient to express powerful learning algorithms" (abstract).
- **Outer loop.** ES, which is "significantly more efficient on VSML compared to standard NNs due to the small parameter space V_M" (§3.1). Settings: 128 GPUs, 10k ES steps, population 1024, and one lifetime of 500 online examples per member (App.). Gradient descent was "less stable".
- **Generalized.** MNIST to Fashion-MNIST, unseen input and output sizes, shuffled inputs, and random projections (§4).
- **Failed.** This is a "short horizon bias": learning "is fast in the beginning but flattens out at some horizon", because only 500 examples were meta-optimized (App.). The paper also reports "premature convergence of the solution at meta test time" and a cost of O(W·N²) (§7).
- **Relevance to us.** This is the strongest evidence that a genome of a few thousand numbers can encode a general learning algorithm, and that ES suffices to find it. **[agent's reading]** About 10M member-lifetimes went into it, so our version must have far fewer parameters or reuse prior lives. The short-horizon bias predicts a specific failure: if fitness is scored on early checkups, the genome will stop improving after them.

## 3. Learning the initialization, or learning in activations

### [L11] Finn, Abbeel, Levine 2017: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)
Chelsea Finn, Pieter Abbeel, Sergey Levine. *ICML 2017*, PMLR 70. https://arxiv.org/abs/1703.03400

- **Parameterization.** The meta-learned part is only the initial parameters θ. Within a lifetime, a few SGD steps adapt them. "In effect, our method trains the model to be easy to fine-tune." (abstract)
- **Outer loop.** Second-order meta-gradients through the inner steps. The first-order approximation performed "nearly the same", with a "roughly 33% speed-up" (§5.2). RL uses TRPO as the meta-optimizer.
- **Relevance to us.** MAML needs differentiable, short inner loops (1–5 steps), and a life is 20k steps. **[agent's reading]** Its idea survives in [L12]: evolve the starting point together with the learning hyperparameters, and score after a full lifetime of learning. Note that an evolved initial weight state is learned competence placed in the genome. It is allowed only if it starts from scratch and is labelled (invariant 1).

### [L12] Fernando et al. 2018: Meta-Learning by the Baldwin Effect
Chrisantha Fernando, Jakub Sygnowski, Simon Osindero, Jane Wang, Tom Schaul, Denis Teplyashin, Pablo Sprechmann, Alexander Pritzel, Andrei A. Rusu. *GECCO '18 Companion*, pp. 109–110 (Crossref, DOI 10.1145/3205651.3205763); full version arXiv 1806.07917. https://arxiv.org/abs/1806.07917

- **Parameterization.** The genotype holds the initial parameters P and the hyperparameters h of an ordinary learner (SGD, or A2C for RL). Under **Baldwinian** inheritance, "these learned parameters are not inherited by the next generation; only the original initial parameters P and hyperparameters h are inherited" (Fig. 1). Under **Lamarckian** inheritance, the learned weights are inherited.
- **Outer loop.** Two variants:
  - A generational GA with population 100, or NES with population 25, on sinusoid regression.
  - An asynchronous steady-state GA with population 500 on MuJoCo Cheetah.
  
  One fitness evaluation is 10 episodes of 3,000 steps each (§3). No gradients flow to the genome, which is why it is "more general in that it does not require gradients to be backpropagated to the reference parameters or hyperparameters, and permits effectively any number of gradient updates in the inner loop." (abstract) It can also evolve discrete genes such as macro-actions (§1).
- **Results.**
  - MAML is more data-efficient in simple cases (abstract).
  - On goal direction, Baldwinian evolution won and Lamarckian was "a very clear loser": it got stuck going only backwards.
  - On goal velocity, Lamarckian won, because the previous task's end state is a good start for the next (§4).
  - Evolved hyperparameters diverged. Baldwinian learning rates rose to 0.005–0.01, while Lamarckian learning rates fell to the minimum, 0.00001 (§4).
  - Conclusion: "The Baldwin effect is superior to Lamarckian learning when the distribution of tasks is broad or quickly changing, whereas Lamarckian learning is superior when the task distribution is narrow." (§5)
- **Relevance to us.** This is the closest match to the user's plan, and it is the reference for our Baldwinian/Lamarckian choice. **[agent's reading]** Inheriting only the genome and keeping weights local is Baldwinian. It keeps evolution selecting *for* fast learning. Lamarckian inheritance of checkpoints tends to switch learning off. "Forever lives" and checkpoint inheritance (0014) are Lamarckian channels, so they must stay separately labelled conditions.

### [L13] Hinton & Nowlan 1987: How Learning Can Guide Evolution
Geoffrey E. Hinton, Steven J. Nowlan. *Complex Systems* 1(3), 495–502, 1987. The volume and issue were confirmed on complex-systems.com, and the pages from the reprint header. https://www.complex-systems.com/?p=1656 (reprint: https://www.cs.toronto.edu/~hinton/absps/evolution.htm)

- **Parameterization.** There are 20 genes, each with the allele 1, 0, or **?**. "? specifies a connection containing a switch which can be open or closed. It is left to learning to decide how the switches should be set." (§3) Learning is random search over the ? switches during the lifetime. The outer loop is a genetic algorithm.
- **Result.** Learning lets evolution find a needle-in-a-haystack solution that evolution alone never found, "even though the characteristics acquired by the phenotype are not communicated to the genotype" (abstract). There is little pressure to fix the last few genes, because learning reliably sets them (§3).
- **Relevance to us.** This is the conceptual root of "genome switches". A switch can be *fixed by the genome* or *left to lifetime learning*. Evolution benefits from the "?" option because learning smooths the fitness landscape. **[agent's reading]** Each structural switch in our genome could have three states: on, off, and "learnable". This is a direct transplant, but untested for us.

### [L14] Duan et al. 2016: RL²: Fast Reinforcement Learning via Slow Reinforcement Learning
Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, Pieter Abbeel. arXiv preprint, 9 Nov 2016; the page says "Under review as a conference paper at ICLR 2017". https://arxiv.org/abs/1611.02779

- **Parameterization.** "the algorithm is encoded in the weights of the RNN, which are learned slowly through a general-purpose (“slow”) RL algorithm" (abstract). The network is a GRU with 256 units. Lifetime learning lives in the hidden state, which "is preserved to the next episode, but not preserved between trials" (§2.2).
- **Outer loop.** TRPO. Visual navigation used batches of 50,000 and up to 5,000 policy iterations (App. A.3).
- **Results.** Bandits and tabular MDPs came close to Gittins indices and other algorithms with guarantees. ViZDoom mazes from 40×30 RGB worked.
- **Failed.** The agent "occasionally … forgets about where the target was", and "the outer-loop reinforcement learning algorithm was shown to be an immediate bottleneck" (§3.3; conclusion).
- **Relevance to us.** Our 256-d GRU already *is* an RL²-style core, but it is trained online within the life, not meta-trained across lives. **[agent's reading]** Meta-training a GRU's weights across lives would put learned competence in the genome (invariant 1), and it needs RL across many trials, which we cannot afford. We should not adopt it as the genome mechanism.

### [L15] Wang et al. 2016: Learning to reinforcement learn
Jane X. Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z. Leibo, Remi Munos, Charles Blundell, Dharshan Kumaran, Matt Botvinick. arXiv preprint, 17 Nov 2016; no venue listed. https://arxiv.org/abs/1611.05763

- **Parameterization.** An LSTM trained with A2C/A3C. After training, "the weights are frozen, but the activations are changing", and the recurrent dynamics implement "a second, quite separate RL procedure" (§2, abstract).
- **Results.** Structured task distributions taught priors that transferred within that family, but "previous training on any structured distribution … hurts performance when agents are tested on an independent distribution" (§3.1.2). The paper also covers Harlow's task in 3D.
- **Relevance to us.** Its point for us is a contrast: learning in activations encodes the training distribution's priors. That is the opposite of a general rule. **[agent's reading]**

## 4. Genotype-to-phenotype indirection and in-context learning

### [L16] Ha, Dai, Le 2017: HyperNetworks
David Ha, Andrew M. Dai, Quoc V. Le. *ICLR 2017* (mlanthology). https://arxiv.org/abs/1609.09106

- **Parameterization.** One network generates the weights of another. The authors frame this as "the relationship between a genotype – the hypernetwork – and a phenotype – the main network" (abstract). In one example, a kernel of 12,544 weights is generated from "an embedding vector of only 4 parameters" by a hypernetwork with 4,240 parameters, at 99.24% accuracy against 99.28% for the conventional network (MNIST, §4.1).
- **Outer loop.** Backprop end-to-end, within one training run. It is not a meta-learning method by itself.
- **Relevance to us.** It shows how to keep a genome small while still controlling large tensors: compress per-tensor rules into short embeddings. VeLO [L2] uses exactly this to generate its per-parameter MLPs. **[agent's reading]** Generating rule coefficients from a per-layer code is a cheap way to give ES a low-dimensional genome over many tensors.

### [L17] Kirsch, Harrison, Sohl-Dickstein, Metz 2022: General-Purpose In-Context Learning by Meta-Learning Transformers (GPICL)
Louis Kirsch, James Harrison, Jascha Sohl-Dickstein, Luke Metz. *NeurIPS 2022 Workshop on Meta-Learning*; the full version was under review as of arXiv v2 (9 Jan 2024). https://arxiv.org/abs/2212.04458

- **Findings.**
  - Black-box models move through three phases: memorization, task identification, and general learning-to-learn. "To transition to the learning-to-learn regime, we needed at least 2¹³ = 8192 tasks." (§4.1)
  - "the capabilities of meta-trained algorithms are bottlenecked by the accessible state size (memory)", "unlike standard models which are thought to be bottlenecked by parameter count" (abstract).
  - Meta-training gets stuck on loss plateaus. Larger meta-batches shorten them, following a power law, but "this intervention relies on parallelizability" (§4.3).
- **Relevance to us.** Lifetime learning needs state to store its progress. **[agent's reading]** A 256-d GRU is a tiny state, and plastic fast weights ([L7]–[L9]) add a lot of state for little compute. Also, needing thousands of tasks and large meta-batches argues against meta-learning black-box learners at our scale. Structured, shared rules need far fewer.

---

## Comparison table

| Source | What the genome holds | What a lifetime holds | Outer loop | Meta-training cost (reported) | Generalized | Failed / limit | Fits tabs without cross-life backprop? |
|---|---|---|---|---|---|---|---|
| L1 L2L-GD | Coordinatewise LSTM (2×20) | Weights, LSTM states | BPTT, 20-step truncations | not reported | 2× horizon, wider/deeper MLP | ReLU activations | No (BPTT) |
| L2 VeLO | Per-tensor LSTM, hypernet to per-param MLP | Weights, states | ES, full unrolls | ~4,000 TPU-months | 83 tasks, NeRF, Mixer | >500M params, >200K steps, RL | Outer loop yes; cost no |
| L3 Unsup. rule | Neuron-local MLP, shared θ | W, backward V | Truncated BPTT | 512 workers, ~8 days | Width, depth, activations | Meta-overfit to images | No (BPTT) |
| L5 Tasks/stability | Per-tensor LSTM + per-param FF | Weights | ES then PES | ~5k CPU-years | Some OOD, self-training | Cost | Outer loop yes |
| L6 PES | (estimator) | | ES with persistent particles | n/a | Unbiased with truncations | Genome changes mid-life | Yes, with caveat |
| L7 Diff. plasticity | w, α per connection, scalar η | Hebbian trace | Backprop through episode | not reported | Omniglot, maze, images | Short episodes | Rule yes; loop needs ES |
| L8 Backpropamine | w, α, η, M-computing weights | Hebb, eligibility trace | Backprop / A2C | not reported | PTB, cue-reward, maze | | Rule yes; loop needs ES |
| L9 Random-net Hebbian | ABCD+η per synapse | All weights (random start) | ES, pop 200–500 | not stated | Unseen leg damage | Lower peak than static | Yes, but genome too big |
| L10 VSML | Shared tiny-LSTM params (~2.4k) | Per-weight LSTM states | ES, pop 1024, 128 GPUs | 10k ES steps | MNIST→Fashion, unseen sizes | Short-horizon bias, O(W·N²) | Yes (ES); cost high |
| L11 MAML | Initial weights | SGD-adapted weights | 2nd-order meta-gradient | not quantified | Few-shot | Short inner loops | No |
| L12 Baldwin effect | Init P + hyperparams h (+ discrete genes) | Learned weights (not inherited) | GA / NES, pop 25–500 | not quantified | Broad task distributions | Less data-efficient than MAML | **Yes** |
| L13 Hinton & Nowlan | 1/0/? alleles | Switch settings | GA | toy | Needle in a haystack | Toy | Yes (conceptual) |
| L14 RL² | GRU weights (the algorithm) | Hidden state | TRPO | batch 50k, ≤5k iters | Bandits, MDPs, mazes | Outer loop bottleneck | No; puts competence in genome |
| L15 L2RL | LSTM weights | Activations | A2C/A3C | not quantified | Within family | Hurts on unstructured | No |
| L16 HyperNetworks | Hypernet + small embeddings | (n/a) | Backprop | n/a | Near-parity, fewer params | | Indirection only |
| L17 GPICL | Transformer weights | Context (state) | Gradient | not quantified | Unseen datasets (≥8,192 tasks) | Plateaus; state-size bound | No |

---

## What this suggests for our representation (agent's reading, not decided)

1. **Use a Baldwinian outer loop: evolution or ES over whole lives, scored after learning.** Every method that avoids backprop through the inner loop and still worked uses population search with fitness measured at the end of a lifetime ([L2], [L9], [L10], [L12], [L13]). That matches tabs that cannot share gradients. Lamarckian inheritance of weights or checkpoints stays a separately labelled condition. It is reported to suppress learning rates and to lose on broad task distributions [L12].
2. **Keep the genome small and shared, not per-synapse.** A rule shared per layer or per tensor generalized across widths and depths ([L1], [L3], [L10]). A per-synapse genome [L9] costs 5× the network size. At our budget, about 1–10 machines and roughly 50 lives per machine per day at 30 minutes each **[estimate]**, the genome should be on the order of tens to a few thousand numbers. Hypernetwork-style codes [L16] can expand them.
3. **Start with plasticity, not black-box learners.** A fixed weight plus α·trace with a learned η, gated by a neuromodulatory signal [L7, L8], adds lifetime state at the cost of elementwise ops and no new readbacks. That suits a latency-bound WebGPU step. VeLO-sized per-parameter networks (46× SGD's per-parameter cost [L2]) and per-weight LSTMs (O(W·N²) [L10]) are much heavier. This needs a throughput probe before we commit.
4. **Give genome switches three states, as Hinton and Nowlan's alleles do: on, off, or "left to lifetime learning".** Population search handles discrete genes where meta-gradients cannot ([L12] §1, [L13]).
5. **Score fitness over the full life and across held-out worlds.** Learned rules stall or degrade past their meta-training horizon ([L2] >200K steps; [L10] short-horizon bias), and they meta-overfit to the training domain [L3]. Fitness should weight late checkups and held-out rooms. Lives that run longer than any training horizon should be checked for degradation.
6. **Avoid PES-style mid-life genome updates by default.** PES [L6] would make the population update faster and without bias, but it changes the genome during a life, which conflicts with invariant 7. If wanted, it should be a recorded decision.
7. **Plan for ES noise.** Use antithetic pairs [L5], normalize fitness per task [L2], and accept that plateaus shrink only with larger populations [L17]. A small population will move the genome slowly. Report lives and compute per genome update (invariants 5 and 8).
8. **Keep evolved initial weights out of the genome, or label them.** Evolving initial weights (MAML and Baldwin's P [L11, L12]) places learned competence in the shared layer. Under invariant 1, inheriting initial weights is a labelled condition, not a cold start. "How it learns" should mean rules, rates, switches, and schedules.

## Unverified

None. All 17 sources were fetched and confirmed. Two venue details came from secondary indexes because the primary pages did not show them:
- the GECCO '18 Companion venue for [L12] comes from Crossref, because the ACM page returned 403;
- the ICLR 2017 venue for [L16] comes from mlanthology, because the arXiv page lists no venue.

No sources were dropped. Compute numbers that a paper does not report are marked "not reported" or "not stated" rather than estimated.
