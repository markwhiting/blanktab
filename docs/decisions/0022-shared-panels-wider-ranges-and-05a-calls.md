# 0022 — Shared panels, wider ranges, and plan 05a's other calls

- **Date:** 2026-10-23
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (every call here is an agent's call, not yet confirmed)
- **Resolves:** the choices plan 05a (`docs/plans/2026-10-23-05a-meta-learner-foundations.md`) makes for the user, including its spec gaps

## Context

Plan 05a builds the meta-learner's foundations (0018, 0019; spec `docs/specs/2026-10-22-meta-learner-v1-design.md`). Several choices along the way belong to the user but were not asked. This file records each as an agent's call, so it can be confirmed or reversed. Later 05a tasks add their sections here.

## Decision

### Widened ranges [agent; guesses]

- **Today's spans leave almost nothing to search.** `knowledge/settings-space.json` spans ±20 % on lr (4e-4 – 6e-4), and similarly on headLrScale, exploreEpsilon and residualDecay.
- **Every searched gene now spans ×4 each way around today's value** (`knowledge/genome-space.json`, 0021):

| gene | today | v1 range, scale |
|---|---|---|
| lr | 5e-4 | 1.25e-4 – 2e-3, log |
| headLrScale | 0.1 | 0.025 – 0.4, log |
| residualDecay | 1e-3 | 2.5e-4 – 4e-3, log |
| entropyWeight | 0.01 | 0.0025 – 0.04, log |
| curiosityWeight | 1 | 0.25 – 4, log |
| exploreEpsilon | 0.1 | 0.025 – 0.4, log |
| bufferSize | 2000 | 500 – 4000, integer log |
| replayPerSleep | 64 | 16 – 256, integer log |
| gamma | 0.9 | 0.6 – 0.975, horizon (h = 1/(1 − γ) 2.5 – 40, ×4 around h 10) |
| caregiverPeriod | 500 | 125 – 2000, integer log |
| caregiverTau | 0.05 | 0.0125 – 0.2, log |
| caregiverEpsilon | 0.1 | 0.025 – 0.4, log |

- **The exceptions.** bufferSize goes up only ×2 (to 4000). 8,000 needs a checkpoint-size probe first (spec §3.2). exploreEpsilon's top, 0.4, stays under the learner's clamp of 0.5.
- **The fixed genes are held at today's value.** caregiverThreshold (0.01) and caregiverStuckSteps (2000) are inert: the safety net acts only after two M6 passes in a row, which no life has reached. maxStepsPerSec (30) and sleepInterval (250) are fixed because the cap is a courtesy to the host, and sleeps also time the checkups.
- **Every range lies inside the learner's and caregiver's clamps.** `tools/genome-centre.test.ts` checks this, so a decoded genome runs as drawn.
- **The ranges are guesses.** None has been tested at its ends. A range that is too wide costs search rows on bad learners. One that is too narrow hides the optimum. Changing a range is a new genome-space `schemaVersion` (0021).

### Shared panels [agent; spec §4.1 step 4]

- **What changes.** A life whose `meta.evalPanel` is set (every 05a search life and every newborn reference) meets checkup variants seeded `panelSeed(p, level, milestoneId) = hashSeed(0, "panel/p/level/milestoneId")`, with p = `evalPanel` (1–16). This amends 0016's per-learner variant seeds (`checkupSeed(learnerId, level, milestoneId)`), **for new lives only**.
- **Existing lives keep theirs.** A life without `meta`, and a `legacy` life (born before 05a and resumed on 05a code), keeps `checkupSeed`, so its checkpoints and its record go on as they were.
- **Why.** Lives on one panel meet the same variants, so two genomes on a panel are compared on the same worlds (blocking [OLS O20]). The newborn reference uses the same panels, so a life is measured against newborns that met its worlds.
- **What does not change.** Thresholds, budgets and the suite (0016). The policy seed is still `hashSeed(variant seed, "policy")`.
- **Cost if wrong.** Lives on one panel share their luck in the variants (16 panels spread it). A grinder could pick its panel by its learnerId (a known limit, spec §7.4).

### The meta block's strict rules beyond the spec (Task 5) [agent]

- **A role and its method agree.** `search` has `centre`, `calibrate`, `uniform` or `ts`; `confirm`, `newborn` and `legacy` have the method of their own name.
- **`drawSeed` and `posteriorSha` are null together**, as `birthDraw` makes them.
- **`meta.missing`** (role `legacy` only): legacyGenome's sorted list of genes the life's settings did not record. Their values in `genome` are placeholders, not claims (see "Settings an older life did not record" below). It is constant across the life's files.
- **Confirmation fields belong to confirmation lives** (controller ruling, Task 5 review). `confirmation` and `evalSeed` are set if and only if `role` is `confirm`. `evalSeed` lies in its confirmation's block, [10,000 + 1,000·id, 10,999 + 1,000·id] (spec §4.5). `confirmation.arm` is `candidate`, `centre` or `newborn` (spec §4.5). A confirm life has no `evalPanel`: it meets the confirmation's own variants (spec §7.3 "evalPanel or evalSeed").
- **No genome schema newer than main's** (controller ruling, Task 5 review). A tab is served from main, so the gate rejects `meta.genomeSchema` above `knowledge/genome-space.json`'s `schemaVersion`, for every role. An older schema is still checked for structure only. The parser caps `genomeSchema` at 1,000.
- **The gate never accepts a genome unchecked.** `validateSessionPr` rejects a non-legacy `meta` when it has no genome space; the gate and `validate-cli` read main's `knowledge/genome-space.json` and stop before deciding anything if it is missing or invalid.
- **Cost if wrong.** A parser is easier to relax than to tighten: loosening a rule later accepts every file accepted now.

### A life already past budget when short lives are switched on (controller ruling A) [agent]

- 0019 does not say what happens to a life already past 2,048 CU when the short-lives switch goes on.
- **Call:** the page asks for confirmation. Once confirmed, the life ends at its next clean sleep, as a short life ends at its budget checkup. Switching short lives off never ends a life: without the switch no forever life ends (0014, 0019).
- **Cost if wrong:** one prompt.

### The switch, end reasons and the block in the session (Task 6) [agent]

- **The switch is read at clean sleeps only.** A life with a meta block ends at the first clean sleep where the switch is on, its 2,048 CU checkup is done, and no checkup is still due. Its final file carries `meta.end = "budget"` and the checkups not yet logged. A flip between sleeps does nothing until the next one, so a life switched on and then off before a sleep never ends.
- **A life already past its budget** ends at the next clean sleep once the switch goes on (ruling A above; the confirmation prompt is the page's).
- **A halt waiting at that sleep goes first.** The resumed run ends at its next clean sleep. A halt asked during the 2,048 checkup abandons it, so the resumed run checks it again, and the checkup lands in the final file.
- **Without a meta block the switch does nothing.** Test lives and lives without a genome space are unchanged.
- **Other end reasons.** A failed life's final file has `meta.end = "failed"`. A swept orphan's has `meta.end = "orphan"`, but only when it has a block, from its checkpoint or its latest file (without that file's `filters`).
- **A checkpoint's block is kept** even when the page gives no `cfg.meta` (for example, when the genome space failed to load). If it were dropped, the next resume would relabel a search life `legacy`, and the gate would refuse that because the block must stay constant.
- **A bad legacy block never ends a forever life** (controller ruling, Task 6 review). If the page's `legacy()` throws, or gives a block the parser refuses at the checkpoint's cu and steps (or one not of role `legacy`), that is said on the console. The life then runs on without a block, as before 05a.
- **A newborn's block must describe the learner that runs** (controller ruling, Task 6 review). The block must parse, its genome must lie in the page's genome space at that space's schema, and it must decode to the learner, caregiver and speed-cap settings the life runs and logs (`settingsMismatch`). Otherwise the session throws `BadMetaBlockError` before making a learner, and the page falls back to another draw. The gate checks the same thing for every non-legacy file at main's schema. sleepInterval is not logged, so it is not compared.
- **A first layer of the wrong size is said and its filters left out** (controller ruling, Task 6 review). A missing snapshot costs one data point, while ending the life would cost the whole row.
- **A legacy block is saved at once.** A pre-05a life resumed with `cfg.meta` saves a checkpoint with its new legacy block straight away. So every checkpoint it can be restored or ended from has that block.

### Settings an older life did not record (controller ruling) [agent]

- **Call:** `legacyGenome` returns the genome plus `missing`, the sorted genes absent from a pre-05a life's settings whose value is genuinely unknown (for example sleepInterval, the caregiver genes, and exploreEpsilon or residualDecay on older lives). Those genes get centres only as placeholders, and the page shows "some settings not recorded". An absent mechanism is not mapped to "off", because we do not know what that life ran.
- **Known by construction, filled and not listed:** gamma, at 0.9 (the constant before 05a, not the gene's centre), and the reserved `implemented: false` slots, at their centre (by 0021, the code without the mechanism).

## Alternatives considered

- **Keep today's spans.** The search would have almost nothing to explore.
- **×10 each way.** This is closer to a cold-start sweep, but most rows would be spent on learners that clearly fail at the edges. ×4 is a middle guess.
- **End a past-budget life at once, or never.** Ending it mid-sleep could lose a clean checkpoint. Never ending it would leave the switch silently doing nothing on that tab.

## Consequences

- Each range and call here can be reversed by a later decision. A range change is a new genome-space schema and a fresh fit.
- Later 05a tasks append their calls here: the switch and orphan lines, and the result of the noise check.
