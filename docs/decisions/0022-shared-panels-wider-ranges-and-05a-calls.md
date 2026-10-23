# 0022 — Shared panels, wider ranges, and plan 05a's other calls

- **Date:** 2026-10-23
- **Status:** proposed
- **Decided by:** agent proposal awaiting confirmation (every call here is an agent's call, not yet confirmed)
- **Resolves:** the choices plan 05a (`docs/plans/2026-10-23-05a-meta-learner-foundations.md`) makes for the user, including its spec gaps

## Context

Plan 05a builds the meta-learner's foundations (0018, 0019; spec `docs/specs/2026-10-22-meta-learner-v1-design.md`). Several choices along the way belong to the user but were not asked. This file records each as an agent's call, so it can be confirmed or reversed. Later 05a tasks add their sections here; shared panels are added by Task 5.

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

### A life already past budget when short lives are switched on (controller ruling A) [agent]

- 0019 does not say what happens to a life already past 2,048 CU when the short-lives switch goes on.
- **Call:** the page asks for confirmation. Once confirmed, the life ends at its next clean sleep, as a short life ends at its budget checkup. Switching short lives off never ends a life: without the switch no forever life ends (0014, 0019).
- **Cost if wrong:** one prompt.

### Settings an older life did not record (controller ruling) [agent]

- **Call:** `legacyGenome` returns the genome plus `missing`, the sorted genes absent from a pre-05a life's settings whose value is genuinely unknown (for example sleepInterval, the caregiver genes, and exploreEpsilon or residualDecay on older lives). Those genes get centres only as placeholders, and the page shows "some settings not recorded". An absent mechanism is not mapped to "off", because we do not know what that life ran.
- **Known by construction, filled and not listed:** gamma, at 0.9 (the constant before 05a, not the gene's centre), and the reserved `implemented: false` slots, at their centre (by 0021, the code without the mechanism).

## Alternatives considered

- **Keep today's spans.** The search would have almost nothing to explore.
- **×10 each way.** This is closer to a cold-start sweep, but most rows would be spent on learners that clearly fail at the edges. ×4 is a middle guess.
- **End a past-budget life at once, or never.** Ending it mid-sleep could lose a clean checkpoint. Never ending it would leave the switch silently doing nothing on that tab.

## Consequences

- Each range and call here can be reversed by a later decision. A range change is a new genome-space schema and a fresh fit.
- Later 05a tasks append their calls here: shared panels (Task 5), the switch and orphan lines, and the result of the noise check.
