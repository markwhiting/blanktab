# 0016 — Milestone suite v1 (M1–M6) and how checkups run

- **Date:** 2026-10-22
- **Status:** accepted (agent's call; not yet confirmed by the user)
- **Decided by:** agent (under the user's "make best choices"); not yet confirmed by the user. The plan controller approved the details. The controller also ruled that M5 and M2 be tightened (below); those rulings are agent calls too.
- **Amends:** spec §7.3 (caregiver) and §8 (milestones, checkups, scoring); builds on 0005, 0008 and 0014
- **Plan:** `docs/plans/2026-10-21-04b-growing-up.md` (Task 3)

## Context
The spec (§8.2) named six spontaneous-behaviour milestones and gave their thresholds in words. Before a checkup can produce a score, each milestone needs four things: a world, a budget, an exact measure and a pass rule. These must be precise enough that a learner can't pass by flailing and success can't be quietly redefined (AGENTS.md invariant 7).

The spec also left open:
- when checkups run relative to the log schedule;
- what happens to a checkup that a reload interrupts;
- how a checkup keeps the learner unchanged.

## Decision

### The suite: `milestoneSuiteVersion` 1, ids M1–M6, every module at version 1
The modules are in `packages/harness/src/milestones/`. Each variant is `createWorld({ seed, canvas, rooms, start })`. Everything a measure reads (state, trace, areas) is privileged and harness-only; the learner only acts on observations.

| id | variant world | measure | passes | budget |
|---|---|---|---|---|
| M1 | rooms `["crib","toys"]`, start crib | frozen `predLoss / baselineLoss` | `< 0.8` (NaN, or a zero or non-finite baseline, fails) | 300 |
| M2 | `["crib"]` | over the in-view steps (`cursor.y < 184`): the fraction whose `hovered` is interactive and not `taskbar:*`, ÷ the mean `areas.interactive` over those steps | `≥ 2`, with area > 0 **and in-view steps ≥ 50 % of the budget (≥ 200 of 400)** | 400 |
| M3 | `["toys"]` | `(respClicks / meanResponsive) / (max(unrespClicks, 1) / meanUnresponsive)`. Counts `click` events without `taskbar:*` ids. Areas are means over in-view steps; unresponsive = 1 − responsive | `respClicks ≥ 3` and `≥ 2` | 400 |
| M4 | `["hideseek"]` | `(r − r0) / (1 − r0)`, where r0 = `initial.scroll.revealedFraction` and r = the last state's | `≥ 0.6` | 400 |
| M5 | `["places"]` | the learner's distinct `page` values (start included) ÷ `reference` | `≥ 1.5` **and distinct pages ≥ 4** | 400 |
| M6 | all four rooms, start `ROOM_KINDS[seed % 4]` | the number of `focus` events with `source: "learner"` | `≥ 3` | 500 |

More on the table:
- **NaN measures.** A measure is NaN when there is nothing to measure: no in-view step, zero area, r0 = 1, or a missing reference. NaN never passes, and the log records it as `null`.
- **M5 reference.** The reference is the mean distinct pages (start page included) over 16 uniform-random clickers on the same variant, each run for the same 400 steps. Clicker j draws its actions per factor (`uniformAction`) from the RNG seeded `hashSeed(seed, "random-clicker/" + j)`, j = 0..15. It is a pure function of the seed. It runs without observations, because observing never changes the world; a test checks that the recorded clickers' mean equals the reference exactly.
- **Prerequisites.** M3 needs M2, M4 needs M2, M5 needs M3, and M6 needs M3. In v1 they are metadata for the ladder only; every checkup runs all six.
- **`room` field.** M1 and M6 are labelled `"desktop"` because their worlds have more than one room. The others are labelled with their single room.

### Budgets
The budgets are M1 300, M2–M5 400 each and M6 500: 2,400 steps per suite, plus 16 × 400 world-only steps for the M5 reference.

Why these lengths:
- Each budget lets a cursor that moves at most 16 px per step cross the 320×200 screen many times.
- M6 gets 500 because three switches need repeated trips to the 16 px taskbar strip.
- At the ~26 learning steps/s seen live, a suite would take ~92 s. Frozen steps are forward-only, so they should be faster: ~40 steps/s (~60 s a suite) is an unmeasured estimate, not a measurement. They obey the same `maxStepsPerSec` cap as learning steps, as a courtesy to the visitor's machine. So expect up to ~80 s per suite.
- Early life is checkup-heavy (seven levels by 512 CU) by design: the developmental curve is densest where change is fastest.

### Tightening after the random check (controller rulings; agent's calls)
As first specified, the uniform-random check failed the plan's rule (≤ 1 random pass in seeds 1–10 for each of M2–M5):

- **M5.** The reference was a single clicker, seeded `hashSeed(seed, "random-clicker")`, with no page floor. That reference is 1–6 pages, and it is 1 on 4 of 10 seeds. A random policy passed **4 of 10**, with measures 1.5, 0.5, 0.25, 1, 2, 0.17, 3, 0.5, 0.25 and 2. Averaging over 16 clickers alone would still have left 2 of 10, because the learner's side is noisy too. Hence both changes: the 16-clicker mean and the ≥ 4 page floor.
- **M2.** With only "≥ 1 in-view step", a policy that walked 5 steps straight onto the taskbar and parked there passed on 2 of 3 seeds (measures 4.29 and 2.08), from the shapes it crossed on the way. Hence the ≥ 50 % in-view floor.

The rulings only tighten pass rules; no threshold was loosened.

### Measured random pass rates (`uniformAction` from `createRng(hashSeed(seed, "flailer"))`, seeds 1–10)

| | before the rulings | after | measures after, seeds 1..10 |
|---|---|---|---|
| M2 | 0/10 | 0/10 | 1.76 0.81 0.95 0.70 0.38 1.33 1.16 1.54 0.91 0.91 |
| M3 | 0/10 | 0/10 | 1.78 0.23 0 0 1.47 0 0.53 1.48 0.34 0 |
| M4 | 1/10 | 1/10 | 0.07 0.73 0.16 0.38 0.23 0.24 0.59 0.22 0.54 0.19 |
| M5 | 4/10 | 0/10 | 1.20 0.71 0.62 0.62 0.78 0.39 1.71 0.62 0.46 0.91 |
| M6 | 0/10 | 0/10 | 0 0 0 0 0 0 0 0 0 1 (measured and recorded, not asserted) |

**M5's random false-pass rate is not zero.** Over seeds 1–50 the same random policy passes M5 on 5 of 50 seeds (12, 28, 29, 40, 50). Of the 32 reference clickers for seeds 1–2, 4 pass against their own reference. Read single M5 passes with that ~10 % base rate in mind.

### Scripted competence (test-only policies that read privileged state; they are fake learners)
On every seed tested:
- **M2:** parking on a crib shape passes.
- **M3:** clicking a button or toggle passes.
- **M4:** closing the pop-up, paging through each tab, then scrolling to the end gives r = 1 (seeds 1–5).
- **M5:** a depth-first link explorer passes on seeds 1–20, with ≥ 4 pages on each.
- **M6:** three taskbar switches pass.

Every variant in seeds 1–20 leaves room to pass: interactive and responsive area in (0, 0.5), r0 < 1, ≥ 4 pages reachable, and 4 rooms.

### Newborn base rate (added in Plan 04b Task 4, controller ruling; agent-measured)
An untrained learner is not a uniform-random policy. Its fixed initial biases can repeat one kind of action, such as drifting to the taskbar and clicking there. So newborns pass milestones by luck more often than the uniform-random rates above suggest, including M3 and M6, which random passed 0/10.

**Method.** The real v1 suite, via `runCheckup`, on 60 untrained learners:
- `CPU_WIDTHS` on the CPU backend, `DEFAULT_SETTINGS`;
- seeds 1000–1059, ids `newborn-0`…`newborn-59`, level 8;
- a one-off script, not committed.

**Results:**

| | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---|---|---|---|---|---|
| newborns passing (of 60) | 0 | 3 | 8 | 11 | 2 | 3 |
| rate | 0 % | 5 % | 13 % | 18 % | 3 % | 5 % |
| median measure | 1 | 0.99 | 0.65 | 0.41 | 0.89 | 0 |

- **Scores (passes per newborn):** 0 on 35 learners, 1 on 23, 2 on 2. No learner had a milestone error.
- **M1:** every newborn scored exactly 1, the on-device baseline ruling at work.
- **Cost:** a checkup cost ~7.3 evalCU on average at `CPU_WIDTHS`.

**How to read a pass.** A newborn is not a random policy: it scores 1 or more on ~42 % of learners (the Expected results line under Consequences says so).
- A single milestone pass counts as evidence of learning only when it is above this base rate. For example, it holds across learners at a rate clearly above the newborn row, or it is sustained across a learner's later checkups.
- The first checkup, at 8 CU, is each learner's own baseline. Later checkups are read against it.
- Thresholds are unchanged.

### Checkups
- **Schedule.** Checkups follow the log schedule (0014): 8, 16, …, 512 CU, then every 1,024 CU. So 2⁰–2² CU are never checked. This amends §8.3's "2ᵏ, k = 0, 1, 2…".
- **One per sleep.** Each sleep runs at most one checkup, at the highest level that is due and not yet checked. Early on, 8 and 16 CU can share one; the skipped level is not run separately.
- **Logging.** Results go in the next log file written (`checkups`, `milestoneSuiteVersion`), and checkup FLOPs are billed to `evalCU`, never `cu`.
- **Halts.** A halt (e.g. a code reload) that arrives during a checkup abandons it: no result is recorded and the level stays pending. It runs once after the reload.
- **Variant seeds.** Each milestone's variant seed is derived from the learner id, the level and the milestone id, so the learner has never seen that world.
- **Frozen runs, not save/restore.** The learner acts and predicts with its current weights through a frozen run, which never writes the learner's state (no weight, Adam, RNG, recurrent-state, replay or stats change except `evalCU`). There is therefore nothing to restore.
- **On-device baseline (Task 1 ruling).** The frozen run's copy-last baseline is computed on the learner's own backend, with the same size-weighted loss as its predictions. The host-side baseline differs at ~1e-11, which would let a newborn whose prediction equals copy-last land on either side of the ratio, so M1 could otherwise pass by float noise. Computed this way, a newborn scores exactly 1.
- **Speed cap.** Frozen checkup steps obey `maxStepsPerSec`. `recordRun` awaits the session's pacer after every step. Without a pacer it yields to the event loop every 50 steps.
- **No caregiver.** The caregiver neither acts nor learns during a checkup.

### Caregiver (§7.3)
- **Relative learning progress.** LP per room is `(EMA_slow(loss) − EMA_fast(loss)) / max(EMA_slow, 1e-6)`, rather than the raw difference, so rooms whose loss scales differ compete fairly (Task 5).
- **Settings source.** While the picker still samples randomly, caregiver settings come from `knowledge/caregiver-prior.json` (spec §9.1).
- **When it decides: every `caregiverPeriod` steps, and only then** (final-review ruling; agent's call). This deviates from §7.3's "and after each sleep". With sleeps every 250 steps, a decision after each sleep also reset the period count, so any `caregiverPeriod` ≥ 250 (the default is 500) never took effect and the setting was inert. Now a decision falls exactly every `caregiverPeriod` steps (in safety-net mode, after `caregiverStuckSteps` stuck steps, as before).
- **Which losses it observes: transitions within one room only** (final-review ruling; agent's call). A frame pair that spans two rooms (after a carry, or the learner's own taskbar switch) has a loss about 3× a normal one, and was billed to the new room, seeding both its EMAs and biasing its LP. The session now observes a loss only when the previous frame was seen in the same room as the current one; it also skips a run's first transition and the first after a restore.
- **Carries on the record.** Carries since the last checkpoint are lost from the record when the life restores from that checkpoint or fails (its final file is built from the checkpoint), as its curve points are. The log's carry counts are a lower bound on the carries that happened.

### Module signature (amends §8.1)
- `makeVariant(seed, canvas)` takes the canvas to draw on.
- `measure(trace, stats, reference?)` returns the logged value.
- `passes(trace, stats, reference?)` is the pass rule.
- `reference?(seed, canvas)` exists for M5 only.
- `trace` is a `CheckupTrace`: `{ initial, states, events }`.
- `stats` is the frozen run's `FrozenResult`.
- The modules live in `packages/harness/src/milestones/`, not `packages/harness/milestones/`.

## Alternatives considered
- **Save/restore around a training-mode checkup.** Rejected: it is easy to miss a piece of state, and that is hard to test. A frozen run that never writes is provably side-effect-free.
- **A single-clicker M5 reference.** Rejected after measurement (4/10 random passes).
- **Loosening any threshold.** Not considered; the rule was to tighten only.
- **Gating milestones on prerequisites.** Deferred: with every milestone run, the curve shows passes out of order if they happen.

## Consequences
- **Score.** A checkup score is the number passed of M1–M6. Results compare only within `milestoneSuiteVersion` 1; changing any row above needs a new decision and a version bump.
- **Expected results.** A curve of zeros is an acceptable first result (spec §2). A uniform-random policy scores ~0 (M4 and M5 pass by luck about 1 in 10 each), but an untrained newborn does not: it scores 1 or more on ~42 % of learners (Newborn base rate, above). Read early passes against that base rate and the learner's own 8 CU checkup.
- **Cost.** Checkups cost up to ~80 s per suite at the speed cap, plus ~6,400 cheap world-only steps for the M5 reference.
- **Still open.** The thresholds are initial (spec §8.2: "tune once real data exists"), and so are the M2/M5 floors.
- **Still open.** The M2 and M5 tightenings (the ≥ 50 % in-view floor; the 16-clicker reference and ≥ 4 page floor) were made by the agent under controller rulings, not yet confirmed by the user.
