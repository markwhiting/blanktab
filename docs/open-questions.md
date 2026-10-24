# Open design questions

Settled questions live in `decisions/`. This file lists what is still open.

## Settled so far
World staging (0001) · single repo (0002) · PR flow from day one, built from scratch (0003) · live + PR sharing layers (0004) · milestone ladder (0005) · logs kept separate from knowledge (0006) · per-contributor token (0007) · desktop, caregiver, checkups (0008) · no LLM observer for v1 (0009)

## Still open

| # | Question | Notes / agent's proposed default | Status |
|---|---|---|---|
| Q3 | **Cold-start definition**: exactly what is initialized, what objectives are supplied, what is inherited. | Inheritance = contents of `knowledge/` only (settings, priors, small meta-parameters); weights start random. | open |
| Q12 | **What drives the learner to act**: its built-in objectives. | Predict what happens next, plus curiosity (seek situations where prediction is improving). | open |
| Q13 | **Learner architecture v1**: perception, recurrent state, fast/slow plasticity, action selection. | Small conv encoder → recurrent core → action head; fast weights or adapters as the fast tier. | open |
| Q4 | **Consolidation mechanism**: the decision it makes, its feedback signal, and the fixed-policy comparison (conditions A/B/C). | | open |
| Q14 | **What "knowledge" holds and how a session uses it** ("what do I do given what I know"). | One meta-learner over a learning genome (0018); evaluators, genome schema and 05a calls (0020–0022). | reshaped by 0018; 05a live |
| Q15 | **"From scratch" scope**: our own infra only, or also our own tensor/autodiff and WebGPU code (no ML libraries)? | | open |
| Q16 | **Observation and action spec**: resolution, frame rate, cursor model, key set. | Low-res color frames (e.g. 64–128 px), relative cursor moves, click, scroll, a small key set; room to grow. | open |
| Q17 | **Persistence and lifecycle**: saving learner state across closed tabs and restarts. | IndexedDB/OPFS checkpoints; one learner identity per browser profile, resumable. | open |
| Q18 | **Merge gate rules**: what counts as "beats the current best," the dead-session timeout, plausibility checks. | Checkup milestones per unit of compute; dead after 24 h silent. | open |
| Q19 | **Browser support for v1.** | Any browser with WebGPU; test Chrome first. | open |
| Q20 | **Hosting origin and repo name** (own domain or dedicated account; see 0007). | | open |
| Q7 | **Autonomy envelope**: what is automatic, what is protected, when to pause for a human. | Mostly covered by 0006/0007; write up in the spec. | open |
| Q6 | **Later contribution units** (weights, architecture). | Deferred. | deferred |

## Spikes to run before committing
1. Open a real PR from a page using a pasted token; find the least-access token type; check pre-filled token-creation links.
2. Learner-side compute: a small training step in WebGPU at useful speed.
3. Render a room to a readable pixel buffer at a useful frame rate.
4. Tab lifecycle: survive backgrounding, freezing, and reload with state restored.

## Missing source material
The synthesis cites `continuous_learning_machine_spec.docx` [S1] and `developmental_computer_learner_spec.docx` [S2], which are not in this repo.
