# Continuous Learning System
## Cross-session synthesis: core ideas, execution options, and open decisions

**Research synthesis v0.1 | 16 October 2026**

## At a glance

The project is a continuously learning system that starts without pretrained world knowledge, encounters a computer-like world through pixels and primitive actions, and selectively carries learning into more persistent and more widely shared forms. The central question is whether learning how to retain, consolidate, and reuse experience can make later learning more efficient on a small compute budget. [S1; S2; C1-C3]

There are two related ambitions:

- **The scientific ambition:** demonstrate useful developmental learning, retention, transfer, and eventually improved learning efficiency through hierarchical plasticity and learned consolidation.
- **The deployment ambition:** let someone open a page on an ordinary computer, start or resume an independent learner, and contribute validated discoveries to a shared lineage without supervising every step.

The browser makes the second ambition attractive; it is not the explanation for why the first should work. Nevertheless, browser execution and broad browser compatibility are strong design preferences, not details to discard casually. The clearest flexibility expressed so far is to narrow the available training material before weakening the learner's autonomy or perceptual boundary. [C2]

**GitHub is the higher-level inheritance layer, not the local learning algorithm.** A session can learn privately and continuously. Only selected knowledge, mechanisms, or architectural discoveries need to become shared contributions. [C2]

The latest execution proposal is to use an existing local research harness for durable experiments, a coding agent for implementation and candidate changes, and xot for research coordination and summaries. That division is a proposal, not a verified integration or a selected final stack. [C4]

**The decisive result would be:** under controlled experience and compute budgets, prior experience helps the system acquire unfamiliar competence more cheaply while retaining earlier competence. Neither a larger memory, more commits, nor a lower training loss alone establishes that result.

## 1. The core, reduced to six ideas

| Core idea | Meaning | Priority in the research |
|---|---|---|
| **Continuous individual learning** | A persistent learner changes through ongoing experience rather than only accumulating prompts around a frozen pretrained model. | Essential architecture. |
| **Hierarchical plasticity** | Transient state, episodic memory, fast adaptation, and durable parameters serve different purposes and update at different rates. | Central hypothesis. |
| **Learned consolidation** | The system can eventually improve its decisions about what to retain, replay, compress, specialize, or promote. | Highest-value unresolved mechanism. |
| **Cold-start perception and action** | Learned world knowledge starts empty; the learner develops representations through surface observations, actions, and consequences. | Core experimental condition. |
| **Learning efficiency at small scale** | The target is meaningful acquisition and improving transfer on roughly 1-10 computers, not an assumption of frontier-model breadth. | Main practical motivation and measurement constraint. |
| **Individual-to-population inheritance** | Some discoveries become reusable knowledge or mechanisms for other instances and future descendants. | Higher-level extension of the cascade. |

This priority ordering synthesizes the discussion, not a literature-based novelty assessment. Earlier numerical novelty scores were informal judgments and should not be carried forward as established research claims. The interesting contribution must be demonstrated by a specific mechanism, comparison, and result. [C3]

## 2. What is core, what is preferred, and what remains optional

| Category | Current position |
|---|---|
| **Core research commitments** | Persistent learning; hierarchical plasticity; selective consolidation; a cold-start experimental track; autonomous interaction; explicit separation of learner inputs from privileged instrumentation. |
| **Strong deployment preferences** | An independent learner per browser session; all-modern-browser ambition; local learning; little setup; unattended operation; Markdown project notes; automated contribution evaluation rather than routine human approval. |
| **Explicitly flexible** | Which websites, datasets, applications, videos, or documents supply experience; how complex the starting world is; which controllable environments are supported first. |
| **Alternatives discussed, not adopted** | Native-first research, a separate headless laboratory, a pretrained comparison track, an extension or local helper, or a streamed remote environment. |
| **Still unresolved** | Exact model and plastic-state design; consolidation objective; usable pixel acquisition for each content source; initial runtime; harness integration; fair evaluation and population acceptance rules. |

Cold start permits architectural priors: sensors, memory machinery, objectives, optimizers, recurrence, and exploration mechanisms have to exist. It does not permit quietly supplying pretrained visual or linguistic competence while describing the learner as starting from scratch. A descendant inheriting a trained checkpoint is a different condition from a freshly randomized individual. [S1, sections 1 and 6; C2-C4]

The architecture should have room to grow from the beginning, but the initial world can be small. The recurring principle was to simplify what the learner encounters rather than impose an artificial ceiling on its eventual action vocabulary or capabilities. [S1, sections 1-2 and 9]

## 3. How the learning cascade is meant to work

### 3.1 Different kinds of state, not just a longer prompt

| Level | Intended role | What is still a design choice |
|---|---|---|
| **Working or recurrent state** | Maintain immediate context, predictions, and beliefs. | Representation and recurrence mechanism. |
| **Episodic memory** | Preserve experiences for replay, comparison, and reconsideration. | What to store, retrieve, compress, or forget. |
| **Fast plastic state** | Adapt locally without immediately rewriting the whole learner. | Adapters, writable memory, parameter subsets, or other mechanisms. |
| **Slow parameters** | Make useful regularities and reusable representations durable. | Consolidation objective, timing, scope, and stability controls. |
| **Learning policies** | Improve replay, exploration, consolidation, and allocation of learning effort. | How these policies receive credit and are themselves updated. |
| **Shared inheritance** | Transfer validated discoveries between individuals and generations. | Artifact types, acceptance tests, compatibility, and specialization. |

This is not a mandatory one-way assembly line. An episode can remain an episode, a useful specialist can remain local, and retrieval can bring durable knowledge back into current computation. Two independent questions matter: **how long should this persist, and who or what should it apply to?** [S1, sections 3-5 and 12]

Project notes can be Markdown without requiring the cold-start learner's own memory to be natural language. In early stages, internal memory may be latent state, raw observations, or learned parameters. [S1, section 6; C2]

### 3.2 Working hypothesis for consolidation

The working hypothesis is that consolidation is a **selection and routing problem**: given experience and existing competence, decide what to retain, where it belongs, and whether the expected benefit justifies cost and interference. Candidate signals discussed include recurrence, surprise, uncertainty, utility, compressibility, transfer, and damage to earlier competence. These are candidate inputs, not a finalized formula. [S1, section 5; C3]

The most useful initial contrast is between a fixed consolidation policy and one that learns from experience. The learned policy must have an explicit update mechanism and a feedback signal; adding several timescales does not by itself establish meta-learning.

The difficult questions are delayed credit assignment, distinguishing useful surprise from noise, preventing repeated errors from becoming durable, and measuring whether promotion helps beyond the current task. A proposal that merely promotes frequent or surprising events is a starting heuristic, not the research result.

### 3.3 Why context learning remains relevant

The original discussion began with reflective notes and very large context windows. That approach remains a useful comparison: what can a system achieve by changing accessible state, and what improves when its representations or learning machinery can also change? [C1; C3]

A frozen base model is not limited to reproducing traces it already has; search and external feedback can generate new solutions. Nor does fixed base code alone prove a formal ceiling on learning, since writable state can contain procedures. The question here is empirical efficiency, transfer, and adaptability, not a demonstrated impossibility theorem about context-only systems.

The linked Context Language Models work treats context as an editable file and examines ways to improve its management. It is a relevant reference for persistent state and memory policy, but not evidence that this project's cold-start visual learner or complete consolidation hierarchy already works. [R1]

## 4. The world, instrumentation, and input/output boundary

### 4.1 Three components that should not be conflated

**The learner** receives its declared observations, chooses primitive actions, and updates its state.

**The environment** renders the world and determines what happens when an action is applied. It may be an application, a media viewer, a controlled desktop, or another interactive setting.

**The harness** manages lifecycle, capture, timestamps, action acknowledgements, replay records, checkpoints, evaluation, and provenance. It may know the environment's internal state for verification without exposing that state to the learner. [S1, sections 2, 4, 7-8]

The intended default is pixels in and primitive actions out. DOM structure, accessibility labels, transcripts, object identities, and source-specific meanings should not silently enter the learner. An additional modality or reward channel is possible, but it must be an explicit experimental choice. Curriculum selection is also a prior and should be recorded. [C2]

### 4.2 Why arbitrary websites became a blocker

Embedding permission, readable pixels, and usable action control are separate capabilities. A site can prohibit framing, and a cross-origin iframe does not grant its host unrestricted access to the embedded document. Selecting a smaller list of uncooperative websites does not remove those boundaries. [R3; R4]

Screen capture can provide an observation stream after user selection and permission. It does not, by itself, provide the complete independent action channel this project needs; passive observation was therefore not accepted as a replacement for autonomous interaction. [R5; C2]

### 4.3 Candidate worlds within the intended paradigm

| Source of experience | How it could fit | Condition that must be demonstrated |
|---|---|---|
| **Owned computer-like environments** | Controlled interfaces with pixels and primitive input, starting simple and growing richer. | The observation is genuinely visual, and control does not depend on semantic shortcuts. |
| **Cooperative embedded applications** | An adapter supplies observations and applies actions within an agreed world boundary. | Both pixel acquisition and action execution work; permission to embed alone is insufficient. |
| **Video files or accessible media** | A viewer presents decoded frames and primitive playback controls. | The harness can legally and technically access frames, not just display an opaque embed. |
| **Downloaded text or documents** | Render content visually so symbols must be learned rather than delivered as interpreted meaning. | A permitted acquisition path and a faithful renderer exist. |
| **Privileged local or remote computer** | Broader websites become environments behind the same sensory/action boundary. | Installation or hosting tradeoffs are explicitly accepted and measured. |

These remain environment options, not commitments to a particular implementation. The preference that emerged was to vary the training material before changing the core learning experiment. [C2]

### 4.4 The YouTube correction

YouTube's iframe API supports embedding and playback commands, but its documented player interface does not provide a general raw-frame extraction endpoint. Cross-origin restrictions remain relevant, and drawing unauthorized cross-origin media into a canvas can prevent pixel readback. **A visible YouTube embed is not yet a demonstrated pixel source for the model.** [R3; R6; R7]

A curriculum beginning with video is conceptually compatible with the project. The dependable initial example would be media whose frames the harness can actually access, rendered in a controlled viewer. Direct YouTube integration remains an unresolved source-specific problem, not a completed design.

The learner need not be given a website address or know that it is watching a video. A harness can present the next curriculum item. However, advancing the curriculum is a harness action, and a prerecorded video does not respond to interventions in its depicted world. Passive prediction and interactive control should therefore be measured separately.

### 4.5 Continuous state is not guaranteed uninterrupted execution

The browser ambition needs persistence across pauses and restarts, not an assumption that an inactive tab will compute forever. Browsers can freeze or discard pages. A proposed browser implementation must test that lifecycle and recover learner state; a managed local harness is a separate option for dependable unattended experiments. [R8]

## 5. GitHub as population-level inheritance

The intended population loop is:

**Shared lineage -> independent individual -> local experience and consolidation -> selected contribution -> independent evaluation -> shared release or retained specialization.** [S1, section 12; C2]

Most fast state and ordinary updates remain individual. A higher-level contribution might contain reusable memory, a learned representation, a skill, a consolidation heuristic, a memory policy, or eventually a change to architecture. A packaging service can submit an artifact before the learner itself understands GitHub or can write a natural-language explanation. [C2; C4]

The shared layer should preserve ancestry, evidence, compatibility information, and rollback. Acceptance should assess generalization and retention, not merely whether files merge. The saved specification explicitly allows specialized branches rather than forcing every locally useful update into the same global representation. [S1, sections 8 and 12]

The aim is automated promotion within declared rules, not manual review of every contribution. Automation still needs a protected acceptance mechanism: candidates should not weaken their own tests, redefine success, or gain access to repository credentials. Code and architecture changes need a different evaluation regime from a new episode or memory artifact. [C2; C4]

The social and evolutionary analogies describe different inheritance scopes. They do not establish that knowledge, weights, and architectural changes can all be merged by the same mechanism.

## 6. Potential execution strategies

### 6.1 Runtime choices

| Strategy | Why it was discussed | Tradeoff | Position in the discussion |
|---|---|---|---|
| **Browser-first, controlled worlds** | Closest to opening a URL and creating an independent, locally learning instance. | Requires verified capture, action, storage, and compute support across target browsers. | Strong fit to the deployment preference; exact implementation open. |
| **Native/local research harness first** | Easier to use existing local research machinery and run controlled experiments. Earlier notes favored Apple silicon, MLX, and Swift hosting. | Can drift away from browser portability unless interfaces stay portable. | Earlier concrete path and later proposed alternative, not a final replacement. |
| **Split laboratory and browser target** | A managed laboratory produces evidence while browser execution remains a deployment target. | Two execution paths create maintenance and behavioral-parity work. | Proposed compromise, not settled. |
| **Pretrained comparison track** | Helps investigate consolidation without also solving initial perception and representation acquisition. | Does not answer the same cold-start question. | Discussed alternative or comparator; not a substitute adopted for the core track. |
| **Privileged local or hosted world backend** | Expands access to real websites while preserving surface-level observations for the learner. | Adds installation, permissions, hosting, latency, or cost. | Optional environment path; not necessary when controlled sources suffice. |

The older specifications differ on Python versus Swift hosting and later add a browser variant. Those details should not be combined into an accidental requirement to implement every stack. The browser branch discussed browser-capable compute such as WebGPU/WASM; the runtime and checkpoint boundaries were intended to stay portable. [S1, sections 7 and 12; S2, section 8]

### 6.2 Coding agent, research harness, and xot

The latest proposed division of responsibility was:

| Component | Proposed responsibility |
|---|---|
| **Coding agent** | Build, test, debug, and produce isolated candidate changes. |
| **Existing local research harness** | Own jobs, budgets, checkpoints, recovery, evaluation, and experiment records. |
| **xot** | Maintain the agenda, interpret results, propose bounded next experiments, and surface concise updates or decisions. |
| **GitHub** | Maintain versioned code and shared inheritance, with automated promotion gates. |

These are roles, not a requirement for four separate intelligent agents. The user already has local harnesses; their interfaces and xot's durable execution capabilities still need to be established. No specific coding-agent vendor was selected. The proposal was to reuse the existing machinery rather than first build another general-purpose agent platform. [C4]

The autoresearch reference illustrates an outer loop that edits training code, runs bounded experiments, measures a result, and keeps or discards changes. Its reference training setup is not this browser or MLX learner. The transferable idea is the experiment workflow, not an assumption of drop-in compatibility. [R2]

### 6.3 Two different meanings of improvement

**Within-lifetime learning:** a learner improves through experience under a fixed implementation, including any explicitly learnable consolidation policy.

**Research/design improvement:** a coding agent or researcher changes the implementation between experiments.

Both are useful, but they support different claims. A pretrained assistant inventing a better architecture is not evidence that the cold-start learner discovered that architecture. Keep the source of each change and the associated compute visible. A pretrained coding assistant can help build the experiment without feeding pretrained answers or representations into the learner. [C4]

## 7. Small-scale feasibility, significance, and evaluation

### 7.1 What the scale question actually was

The question was not whether multiple machines can coordinate. It was whether a carefully designed learner can extract enough useful structure from modest experience to make meaningful progress on roughly 1-10 computers, perhaps with input streams on the order of 10,000 files per day rather than hundreds of millions. That is an aspiration and experimental budget, not a measured throughput requirement. [C3]

The motivating hypothesis is that active exploration, selective replay, reusable representations, and consolidation could increase the value extracted from experience. But file count is a poor common unit, replay consumes compute, and a textbook page is not immediately informative to a random visual network in the way it is to a literate person.

No result in the project notes demonstrates that this closes an orders-of-magnitude gap with frontier training. The tractable question is whether the proposed machinery produces measurable acquisition, retention, and transfer advantages under a small, explicit budget.

### 7.2 What would count as a useful result

The proposed evidence ladder is:

1. **Basic developmental learning:** acquired visual or interactive competence generalizes beyond the exact training sequence.
2. **Useful consolidation:** retention and adaptation improve relative to a simpler continual learner at comparable resource use.
3. **Learned consolidation earns its cost:** an adaptive policy beats a fixed policy on unfamiliar variants, not just its tuning tasks.
4. **Improved future acquisition:** an experienced learner needs fewer interactions or less computation to reach a defined criterion on new task families.
5. **Useful inheritance:** another individual benefits from a transferred artifact without unacceptable interference or loss of provenance.

This ladder is a proposed evaluation structure. The saved specifications already identify held-out prediction, retention, transfer, and improving acquisition of later capabilities as success criteria. [S1, sections 8-9; S2, section 10]

### 7.3 What would not establish the claim

A growing memory, decreasing training loss, easier later curriculum, more agent-generated commits, or a larger amount of replay can all produce apparent progress without proving improved learning efficiency. A prediction-only learner can also favor predictable inactivity; interactive evaluation needs to distinguish competence from merely finding an easy stream to predict.

The evaluation proposal is to hold implementation and budgets fixed within a run, report interaction and compute costs separately, preserve unfavorable results, repeat across initializations, and reserve unfamiliar task variants for confirmation. Any outer research-agent compute and inherited training should be accounted for separately. [C3-C4]

## 8. Proposed first research cycle, not a finalized build plan

**First operational milestone:** an unattended cycle returns reproducible measurements, recoverable learner state, and a justified next experiment. It need not produce a better learner every morning. [C4]

The most recent candidate comparison was:

| Condition | Purpose |
|---|---|
| **A. Straightforward continual learning with replay** | Establish a credible simple baseline. |
| **B. Fast/slow state with a fixed consolidation policy** | Test whether the hierarchy adds value. |
| **C. The same hierarchy with a learned consolidation policy** | Test whether adapting the promotion rule adds value beyond a fixed design. |

Earlier notes also proposed frozen-weight/context and replay-without-consolidation comparisons. They remain useful ablations where the architectures and tasks make them interpretable; they do not all have to precede the first measurement. [S1, section 9; C4]

The proposed unattended workflow is: select a bounded question, prepare a candidate, run integrity checks, execute, evaluate independently, retain the evidence, and choose the next experiment. Keep a stable developmental lineage separate from branches where the implementation changes. Protect evaluation and impose explicit compute, storage, and recovery limits. [C4]

The first environment should exercise the observation-action-learning-consolidation boundary without making broad website access a prerequisite. Before training at length, validate that its frames, primitive actions, timing, resets, and checkpoints actually work in the selected runtime. A small initial world should preserve the architecture's ability to grow. [S1, sections 2, 7-9; C4]

Progress reports should answer what was tested, what changed in measured competence, what failed or remains uncertain, and what runs next. They should point to artifacts rather than rely on the research agent's narrative. [C4]

## 9. Open decisions and important corrections

### Decisions still needed

| Decision | What would resolve it |
|---|---|
| **First runtime and harness** | Inspect the existing local harnesses and demonstrate one resumable run; make the browser-versus-native scope explicit. |
| **First world and content sources** | Verify usable pixels, primitive control, and autonomous operation for a small set of supported sources. |
| **Cold-start definition** | Specify all initialized parameters, supplied objectives, curriculum information, and inherited state. |
| **Consolidation mechanism** | Define the decision it makes, its feedback signal, and a fixed-policy comparison. |
| **Meaningful competence** | Choose measurable acquisition, retention, and transfer tasks with declared budgets and held-out variants. |
| **Population contribution unit** | Distinguish data, memory, learned state, code, and architecture; define compatible promotion tests for each. |
| **Autonomy envelope** | Define what runs and merges automatically, what remains protected, and when execution pauses for a decision. |

### Corrections to preserve in future handoffs

- **GitHub was never meant to receive every local weight update.** Its role is higher-level inheritance and shared lineage.
- **Watching a person browse is not the target.** Passive media can contribute experience, but autonomous action remains a core requirement.
- **An embeddable page is not automatically a readable pixel buffer.** YouTube integration remains unresolved at that boundary.
- **Remote hosting was a proposed workaround, not an accepted requirement.** Controlled local content is a distinct path.
- **Architecture-first does not mean implementing every future mechanism immediately.** Preserve extensibility while making the first experiment interpretable.
- **The small-scale question is about learning efficiency, not coordination.** More participating computers do not by themselves answer it.
- **Novelty and feasibility were hypotheses, not established findings.** Informal enthusiasm should not become evidence in the specification.
- **Research-agent improvements and learner-discovered improvements must remain distinguishable.** Automating the scientist is not the same result as teaching the organism to learn better. [C1-C4; R3-R7]

**Bottom line:** preserve the cold-start developmental hypothesis, the autonomous surface-level interface, and higher-level shared inheritance. Choose the smallest supported world and execution path that can test learned consolidation honestly; treat the rest as explicit options rather than quietly settled architecture.

## 10. Source record

### Project specifications

**[S1] Continuous Learning Machine**, living architecture specification v0.1, `continuous_learning_machine_spec.docx`, 7 pages. Principal references: sections 1-2 (objective and world), 3-6 (continuous learning, plasticity, cold start), 7-9 (instrumentation and evaluation), 10 (open questions), and 12 (browser sessions and population consolidation).

**[S2] Developmental Computer Learner**, living architecture and research specification v0.1, `developmental_computer_learner_spec.docx`, 4 pages. Principal references: sections 2-7 (constraints, loop, hierarchy, environment, objectives), 8-10 (local execution and success criteria), and 11-12 (open questions and architectural principle).

### Discussion provenance

**[C1] Early architecture discussions, 3 October 2026:** cold-start world modeling; a local Apple-silicon path; recurrent state, fast/slow plasticity, replay, and an extensible computer-use architecture.

**[C2] Browser and inheritance discussions, 3-5 October 2026:** browser sessions as independent learners; modern-browser and Markdown preferences; automated shared contributions; GitHub as social/population inheritance; willingness to relax content sources; rejection of passive browsing as a substitute for agency; video and rendered-text curricula.

**[C3] Consolidation and efficiency discussions, 5 October and subsequent follow-up:** learned routing and promotion; practical small-compute ambitions; information density versus file count; need for meaningful learning, retention, transfer, and fair comparison. Earlier novelty scores and feasibility statements are treated as exploratory judgments.

**[C4] Execution-strategy follow-up in the conversation:** existing local research harnesses; coding-agent versus xot question; proposed role separation, unattended experiments, bounded autonomy, and the distinction between improving a learner and improving its design.

### External references checked for this synthesis

**[R1] Context Language Models**, official repository. Context-as-file and context-management learning reference, not validation of the proposed cold-start system. `https://github.com/facebookresearch/context-language-models`

**[R2] autoresearch**, official repository. Reference for bounded agent-driven training-code experiments; its reference implementation targets a single NVIDIA GPU. `https://github.com/karpathy/autoresearch`

**[R3] MDN, Same-origin policy.** Cross-origin embedding versus document/script access. `https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Same-origin_policy`

**[R4] MDN, CSP frame-ancestors.** Websites can restrict which parents embed them. `https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Content-Security-Policy/frame-ancestors`

**[R5] MDN, MediaDevices.getDisplayMedia().** User-selected, permissioned screen capture. `https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getDisplayMedia`

**[R6] YouTube IFrame Player API Reference.** Embedding, player commands, and events. `https://developers.google.com/youtube/iframe_api_reference`

**[R7] MDN, Use cross-origin images in a canvas.** Origin-clean media handling and restrictions on tainted-canvas readback. `https://developer.mozilla.org/en-US/docs/Web/HTML/How_to/CORS_enabled_image`

**[R8] Chrome for Developers, Page Lifecycle API.** Freezing, discarding, and lifecycle recovery. `https://developer.chrome.com/docs/web-platform/page-lifecycle-api`
