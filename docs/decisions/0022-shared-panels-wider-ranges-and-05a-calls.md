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

- **The meta block carries its life's evaluation budget** (controller ruling, Task 7 review; detail in 0020).
  - **The field.** `meta.budget` (`{ budgetCU, levels }`) is optional and strict. It is constant across a life's files, as the other constant fields are, and it is not allowed on role `legacy`.
  - **When it is absent,** the block reads as the v1 budget, 2,048 at 512, 1,024 and 2,048.
  - **What reads it.** `meta.filters` levels and a newborn reference's checkup levels come from the block's own budget.
  - **New lives.** A new life's block must carry the current config's budget, in the session and at the gate (seq 0).

### Settings an older life did not record (controller ruling) [agent]

- **Call:** `legacyGenome` returns the genome plus `missing`, the sorted genes absent from a pre-05a life's settings whose value is genuinely unknown (for example sleepInterval, the caregiver genes, and exploreEpsilon or residualDecay on older lives). Those genes get centres only as placeholders, and the page shows "some settings not recorded". An absent mechanism is not mapped to "off", because we do not know what that life ran.
- **Known by construction, filled and not listed:** gamma, at 0.9 (the constant before 05a, not the gene's centre), and the reserved `implemented: false` slots, at their centre (by 0021, the code without the mechanism).

### signal-check@1 and confirm-fisher@1 (Task 8) [agent]

- **Both score lives only through the pinned search evaluator.** `evaluateCohort` takes a third argument, `CohortContext { search }`, which the caller fills with `cohortContext(config)`, that is `REGISTRY[config.search]`. The search evaluator publishes `rankAverage` on the interface. So swapping the search evaluator is a pin edit, with no change to either cohort test. Both report the search evaluator they used (`diagnostics.search`). `loadEvalConfig` now also checks each pin's kind: `search` must be a life evaluator with `rankAverage` and `targets`, and the other three must be cohort evaluators.
- **The search evaluator scores confirmation lives only by a declared capability** (controller ruling, Task 8 review). It is recorded in 0020 ("Declared capability: scoresRoles").
- **signal-check@1.** Lives are counted once (the first by sorted genome, then learnerId), and only under the genome their meta carries. The duplicates, the mislabelled lives and the ones the search evaluator rejects are reported, with the rejection reasons counted. When every life is rejected for one reason (e.g. "no newborn reference"), that reason is the result's reason. Kruskal–Wallis and the ICC run over every genome with at least one eligible life. Signal requires p < 0.05/3 (strictly) and ICC ≥ 0.1.
- **ICC(1).** A negative estimate reads as 0, so groups with identical means give 0.
- **confirm-fisher@1.**
  - **Which lives count.** A life counts in its arm only if every file up to its budget checkup has a meta block of role `confirm`, that arm, one confirmation id, one genome, genome schema 1, inheritance `none`, this version's budget, GPU widths and suite-1 checkups. Other lives are ignored and counted.
  - **The life's end.** A life with its budget checkup is complete. One whose last file is final with `end: "failed"` is failed, which counts as a non-pass. Any other final life is abandoned, excluded and counted. A life still running is pending, excluded and counted.
  - **Ineligible results.** The result is ineligible when an arm has no counted lives, when lives of more than one confirmation are mixed, when an arm mixes genomes, when the newborns are not of the candidate genome, or when the candidate is the centre.
  - **The two tests.** Holm and Mann–Whitney reject at p ≤ 0.05. Test 2 compares the candidate and centre lives that the search evaluator scores as eligible; failed lives are rows, ranked worst.
  - **No rows is no result** (Task 8 review). If the search evaluator scores no life in the candidate or the centre arm (e.g. no newborn reference), test 2 cannot run, so the result is ineligible ("no scored lives in …"), never a failed confirmation. The unscored lives and their reasons are always reported (`unscored`, `unscoredCount`).
  - **`needs` is empty.** confirm-fisher reads no reference itself; it passes refs through to the pinned search evaluator, whose own needs make config load them.
  - **The abandonment flag** is set when the highest abandonment rate across the three arms exceeds twice the lowest, so 0 against any positive rate is flagged. It does not change the score.
- **Mann–Whitney** uses the normal approximation with the tie-corrected variance and a continuity correction of 0.5. When every value is tied, p = 1.

### Summary scores, rescoring and the newborn reference (Task 9) [agent]

- **One scoring for every summary writer** (controller ruling, Task 9 review). `pinnedScoring(root)` is used by both deploy's `summary-cli.ts` and the dev server (`app/vite.config.ts`), and `buildSummary`'s scoring argument is required, so an unscored summary needs an explicit `null`.
- **Summary rows.** With the pinned config, each `summary.json` row carries `evals["<id@version>"] = { eligible, score, C, H }`, scored by the pinned search evaluator over all of the life's merged files. Without a reference every row is `eligible: false` with nulls. The page's parser keeps each well-formed entry and drops a malformed one, never the row. A well-formed entry has an id@version key, finite numbers or null, and a score exactly when eligible; at most 16 entries are kept.
- **How a newborn runs** (`runNewborn`). Its learner is untrained, with seed 1,000 + k and panel 1 + (k mod 16). One `sleep()` on its empty buffer makes no update and scores the held-out set. It then has frozen checkups at the config's levels on that panel. It writes one final file at cu 0, born under the config's budget. A held-out score that is not finite is left out of the file, with a warning, rather than dropping the file.
- **`NewbornConfig` gains `space`** (the genome space). The gate checks that a file's logged settings decode from its genome, and the caregiver genes and speed cap can only be found through the space. The genome is checked (bounds, sha, decodes to `settings`) before any learner is made (`BadMetaBlockError`).
- **Newborns are run locally and collected, not merged through PRs** (controller ruling B). So `newborn-reference.ts` reads a collected directory, not `data/runs`. The default is `data/newborns`, laid out like `data/runs`; the controller commits the collected files there (controller ruling), so anyone can rebuild the reference. Like `data/runs`, it is outside the repo-hygiene scans. Any other directory can be passed as the argument. The gate never sees these files, so the tool stands in for it:
  - every file must pass `validateSessionPr` with main's genome space;
  - every life must be one newborn file of the centre genome (`knowledge/genome.json`), with a seed 1,000 + k for a k in 0–63 that no other life has, and panel 1 + (k mod 16).
  Any failure refuses the whole build. Lives `newborn-baseline@1` ignores (e.g. CPU widths) are reported. Missing k's are reported, and writing a reference with any missing needs `--allow-missing` (controller ruling). The tool never throws: every failure is an error message and exit 1.
- **The reference file gains an optional, strict `provenance`**: `{ codeSha, browser, machine, date }`. No evaluator reads it, so no evaluator version changes. The codeSha must be one for all counted lives ("dev" is allowed with a warning). The date is the UTC date of the latest file. The browser and machine class are given on the command line, because a log cannot say them. `--check` takes them from the committed file.
- **`rescore.ts`** scores only life evaluators. Its references are the pinned ones; a candidate that needs another reference finds it absent.

### The gate's data/meta build (Task 10) [agent]

- **Byte-stable files.** `posterior.json`, `calibration.json` and `diagnostics.json` are `JSON.stringify` with every object's keys sorted, a one-space indent and a final newline. None holds a time. `builtAt` is passed in by `meta-cli.ts` (its one clock read) and goes only into the `posterior-index.jsonl` line.
- **Unchanged content keeps its bytes.** If the new posterior differs from the previous one only in `sourceCommit`, the previous bytes are kept, old `sourceCommit` included, and no index line is written. The posterior changes only when the eligible row set, the mode, the looks or the design change.
- **A calibration built for other pins is refused, never restarted by the gate.** If `calibration.json` names another search evaluator, signal check or genome schema than main's, the build throws and nothing is written. Restarting calibration is a recorded decision that removes the file. The kept design genomes are re-checked against the space and their shas on every build.
- **Per-birth counts live outside the posterior** (controller ruling, Task 10 review). `data/meta/diagnostics.json` (gate-written; tabs never hash it) holds `perGenome` with `born`, `rows`, `failed` and `abandoned`, `verify` (`checked`, the flagged lives, and the genome mismatches among all checked lives) and `unscored`. The posterior keeps only what follows from the rows: `diagnostics.perGenome` with `rows` and failed rows, and `diagnostics.genomeMismatch` among the rows. So a birth that is not yet a row never changes the posterior's sha, which every tab seeds its draw with.
- **Looks** (spacing: controller ruling, Task 10 review). At most one look per build. Look n+1 needs at least max(`SIGNAL_LOOKS[n]`, the last look's lives + (`SIGNAL_LOOKS[n]` − `SIGNAL_LOOKS[n−1]`)) calibration rows. So a burst of merges cannot bring two looks close together: from 40 lives at look 1, look 2 waits for 55. An ineligible signal-check result (too few genomes or lives) records nothing; the next build tries again.
- **Re-verification goes slightly beyond the brief.** Besides an unknown `posteriorSha`, a mismatched `method` or `evalPanel`, and a calibrate genome outside the design, a row is also lost for a mismatched `drawSeed`, a posterior whose bytes do not parse, and a logged genome outside the space or not matching its `genomeSha` (spec §7.4 keeps a genome-mismatch row only when both hold). Flagged lives are listed in `diagnostics.json`'s `verify.ineligible`, with the reason; their logs are kept.
- **What the recomputation uses.** The learner's widths class comes from the widths its first meta-bearing file logged. The space and centre are main's current `knowledge/`. That is exact in 05a, where no promotion moves the centre; 05b, which promotes, must recompute each draw with the centre in force at that birth.
- **Who is checked.** Every life whose first `meta` has role `search` at main's genome schema, CPU lives included (they are draws, though never rows). Other schemas cannot be recomputed against this space and are never rows of this fit.
- **`verify.genomeMismatch` replaces spec §7.2's `drawMismatch {rate, byBrowser}`.** Logs record no browser class, so it gives the count, the rate and the learnerIds.
- **Which posterior versions are read.** `posteriorHistory` resolves only the posteriorShas some life was born with. It walks main's history of `posterior.json`, newest first, until all are found or the history ends; one never found is "unknown". The CLI refuses a shallow checkout.
- **Counts.** `perGenome` (in `diagnostics.json`) has one entry per design genome, over verified `calibrate` lives that log it: `born`, `rows`, `failed` (final `end: "failed"`) and `abandoned` (final, not failed, without its budget checkup). `need` counts rows only, so lives still running are not subtracted. `diagnostics.unscored` counts why verified search lives are not rows (the search evaluator's reasons). `page.calibrationTarget` is the next look's size (30, 45, 60) while calibrating, and the deciding look's once decided.
- **Cost if wrong:** each is a gate-side rule over merged data; changing one rebuilds `data/meta` from the same logs.

### The gate's direct commit of data/meta (Task 11) [agent]

- **Can the Actions token push to main?** Checked on 2026-10-23 with `gh api repos/markwhiting/learner/branches/main/protection`: no required reviews, no required status checks, no push restrictions, `enforce_admins` off, force pushes and deletions blocked. `rules/branches/main` and `rulesets` are both empty, and the default workflow permission is `read` (the gate job asks for `contents: write`). So a `contents: write` token can push a fast-forward to main, and a forced push is refused by GitHub as well as never issued. The plan's live probe (a `workflow_dispatch` pushing an empty commit to a scratch branch) was **not run**: this task was told not to push to the real origin or trigger workflows, so the first real push is the gate's first meta commit. *Agent's call, not yet confirmed.*
- **The file set.** `META_PATH_RE` accepts exactly the four files `meta-cli.ts` writes: `posterior.json`, `calibration.json`, `diagnostics.json` (added in Task 10, after the plan's regex was written) and `posterior-index.jsonl`, all directly under `data/meta/`. Any other staged path, a deletion, an unstaged edit or an untracked file anywhere aborts the run with nothing committed. The commit must also sit directly on `origin/main`.
- **A rejected push.** Only git's `[rejected] … (fetch first | non-fast-forward)` counts as "main moved". Then the gate fetches, hard-resets to `origin/main`, removes untracked files under `data/meta` (an index file the old build created would otherwise survive), and runs `meta-cli.ts` again. At most 3 pushes in all; after that the run fails and the next hourly run rebuilds. Any other push failure (a protection rule, auth, network) fails at once.
- **No build on an old install** (controller ruling, Task 11 review: fail loud, never re-run `npm ci` mid-job). `node_modules` is ignored, so no tree check sees an install older than the code. The gate records the lockfile it installed from (`LOCK_BLOB`, right after `npm ci`). It stops before `meta-cli.ts`, and again before every commit and every rebuild, if main's `package-lock.json` is another blob. The next run reinstalls. The merge step runs before the pull, on the code it installed. Each staged change must be a regular `100644` file, added or modified: no symlink, deletion or mode change.
- **Identity and message.** Author and committer are `github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>`. The message is `meta: <mode>, <page.eligibleLives> eligible lives, posterior <first 8 hex of sha256(posterior.json)>`, the same sha the index line records.
- **Deploy.** A push made with `GITHUB_TOKEN` triggers no workflow, so `gate.yml`'s deploy job also runs when `commit-meta.ts` outputs `meta=committed`. Deploy copies `data/meta/posterior.json` to `knowledge/posterior.json` on the page. It is not a prior file, so `knowledgeSha` and the tabs' reload rule are unchanged.
- **Cost if wrong:** a failed gate run (nothing pushed), or a meta commit that waits for the next run.

## Alternatives considered

- **Keep today's spans.** The search would have almost nothing to explore.
- **×10 each way.** This is closer to a cold-start sweep, but most rows would be spent on learners that clearly fail at the edges. ×4 is a middle guess.
- **End a past-budget life at once, or never.** Ending it mid-sleep could lose a clean checkpoint. Never ending it would leave the switch silently doing nothing on that tab.

## Consequences

- Each range and call here can be reversed by a later decision. A range change is a new genome-space schema and a fresh fit.
- Later 05a tasks append their calls here: the switch and orphan lines, and the result of the noise check.
