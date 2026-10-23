# Meta-learner v1: one meta-learner over a learning genome — design

**Date:** 2026-10-22 · **Status:** draft for the user's review; nothing here is built · **Implements:** decisions 0018 and 0019 (Plan 05) · **Amends on approval:** spec [V1] §5.1, §8.3 (variant seeds for new lives), §9.1–9.2 (the picker); decision 0016 (variant seeds, as above)

Inputs: decision 0018 (binding); [SYN-M] `docs/research/2026-10-22-meta-learner-synthesis.md`; [OLS] `2026-10-22-lit-outer-loop-search.md`; [KUH] `2026-10-22-lit-knowledge-up-the-hierarchy.md`; [V1] `docs/specs/2026-10-19-blanktab-v1-design.md`; [V2] `docs/specs/2026-10-22-learner-v2-capabilities-design.md`; decisions 0011, 0014, 0016, 0017. Code read for this spec: `packages/learner/src/learner.ts`, `packages/harness/src/{session,knowledge,log,checkup,caregiver,final}.ts`, `packages/harness/src/milestones/`, `tools/gate/{gate,summary,manifest,validate}.ts`, `app/src/{main,settings,life}.ts`, `knowledge/*.json`, `.github/workflows/{gate,deploy}.yml`.

Labels: **[user]** = the user's words or decision. **[agent]** = the agent's call, not yet confirmed. **[guess]** = a number with no measurement behind it. Unlabelled text is mechanics that follow from labelled choices.

---

## 1. Intent

**What the user asked for [user, 0018].** Improvements "should not be theorized or conceptualized by us, but be natural explorations of the underlying design space". **One meta-learner** samples variants adaptively, in proportion to how much better they look, as part of normal running. The pipeline is "all data pipeline". **How it learns** (the genome, and the meta-learner's state) is the shared core, inherited through GitHub. **What it learns in a run** stays local. Learned material that many lives acquire moves up the hierarchy over time. The default outer loop is Gaussian-process adaptive sampling with no coding agent in the loop. And: "make sure that all our evaluation functions are easily compartmentalized so we can switch them later."

**In plain words.** Every new tab is one experiment. Its learning recipe (learning rate, curiosity weight, the caregiver's habits, and so on) is drawn from the meta-learner's current belief about which recipes produce learners that learn well. After the learner has lived 2,048 CU, its checkups and held-out prediction are scored, the hourly gate folds that score into the belief, and later tabs draw better recipes. A recipe becomes the inherited default only after it beats newborn luck on fresh worlds it was never searched on.

**Why this shape.** It follows the Baldwinian split: inherit how a learner learns, select on how well a fresh life learns, never copy learned weights by default [SYN-M §1; KUH K12]. GP plus Thompson sampling suits many asynchronous tabs drawing from one posterior [OLS O2, O4, O5]. Its weakness, noise against a small budget, is handled by a continuous score, a noise check before searching, and separate confirmation [OLS §0, §9].

### Goals (v1 is done when all hold) [agent]
1. Every newborn's recipe is a **genome**, drawn by a logged, recomputable method (centre, calibration, uniform, Thompson, confirmation, newborn).
2. **Evaluators** are versioned modules in one package. The gate scores with the pinned ones, and every score names its evaluator id and version. Any life can be rescored under another evaluator.
3. The **repeat-noise check** has run, and its answer (signal or not) is on the page and in a research note.
4. If there is signal, the gate **refits the GP hourly and deterministically**, tabs draw Thompson samples from it, and confirmation and promotion run under the pinned rule. An unfavourable outcome is recorded.
5. The **filter-convergence probe** has run on logged snapshots, and its result is written up.
6. No forever life (0014) ends or restarts because of any phase of this work.

### Non-goals
- Implementing any v2 mechanism [V2 §4]. They appear only as **reserved gene slots**, fixed at "off" (§3.3).
- A coding agent in the loop [user, 0018].
- Inherited weights. The only exception is the filter probe, which reads weights from logs and gives learners nothing (§5).
- Consolidation conditions B/C and the fast tier [V1 §5.6]. Those come later, as genes.
- Resisting deliberately dishonest contributors (a v1 non-goal, [V1 §2]).

### Anti-goals (failure even if the numbers go up)
- **A search that optimises noise.** No Thompson sampling until the repeat-noise check finds signal (§4.6).
- **Evaluators that the genome or a contribution can influence.** Scoring code, its references and its seeds live only in protected paths. Genes never touch checkups, variants or thresholds (0016; invariant 7).
- **Picks that cannot be reproduced.** Every draw is a pure function of logged inputs. Today `app/src/main.ts` draws settings with `Math.random` (`chooseSettings(knowledge, Math.random)`); that is replaced.
- Learner-side signals (reward, loss on its own stream) used as fitness [OLS O25].
- Any privileged field reaching the learner (invariant 2). Genome decoding, scoring and the probe are harness or tooling code.

## 2. Swappable evaluators [user requirement]

### 2.1 One package, one interface [agent]
- **Where:** a new workspace package `packages/evaluators/`. It may import only `@blanktab/harness/log` (types and `parseLogFile`) and its own files. It does no IO, uses no clock and no `Math.random`, and imports no learner, world or meta code. An ESLint rule enforces this, like the existing walls [V1 §4]. `tools/gate` and `tools/eval` import it. The app does not; the page shows numbers the gate computed.
- **The interface:**

```ts
interface LifeInput { learnerId: string; files: LogFile[] }        // one learner's merged files, sorted by seq
interface EvalOutput {
  evaluator: string; version: number                              // e.g. "search-composite", 1
  eligible: boolean; reason?: string                               // why not, e.g. "no 2048 checkup", "spans genome-unaware code"
  score: number | null                                              // the headline number
  components: Record<string, number | null>                         // e.g. { C, H }
  diagnostics: Record<string, unknown>                              // per-milestone q's, nulls seen, levels used, ...
}
interface Evaluator {
  id: string; version: number; kind: "life" | "cohort"
  needs: string[]                                                   // reference ids it reads, e.g. ["newborn"]
  evaluateLife?(life: LifeInput, refs: References): EvalOutput      // kind "life"
  evaluateCohort?(groups: Record<string, LifeInput[]>, refs: References): EvalOutput  // kind "cohort": a test over arms
  targets?(outs: EvalOutput[]): number[]                            // search only: the GP target transform (§2.4)
}
```

- **Pure and deterministic.** The same input bytes give the same output bytes. All maths goes through the shared deterministic math module (§4.3), so results do not depend on the Node version.
- **The registry.** `packages/evaluators/src/index.ts` exports `REGISTRY: Record<"id@version", Evaluator>`. Old versions stay in the registry forever, so any merged life can be rescored under any of them.

### 2.2 Pinned in a protected config [agent]
`config/evaluation.json` (new; not in `knowledgeSha`, because learners never read it):

```json
{ "schemaVersion": 1,
  "search": "search-composite@1", "confirm": "confirm-fisher@1", "signal": "signal-check@1", "baseline": "newborn-baseline@1",
  "budgetCU": 2048, "levels": [512, 1024, 2048],
  "references": { "newborn": "config/references/newborn-baseline@1-gpu-suite1-genome1.json" },
  "decision": "docs/decisions/00NN-evaluators-v1.md" }
```

- **A change needs a recorded decision and a version bump.** Changing evaluator code means a new `version` in the registry. Changing which evaluator is pinned means editing this file. A person or a coding agent makes either change, only under a recorded decision. **The gate never writes `config/`.** The gate never merges a PR touching any non-log path: it labels such PRs `needs-human`, as now.
- **Changing the search evaluator starts a fresh fit.** The posterior records `evaluator@version`. Rows are never mixed across versions; old lives are rescored instead.
- **Every score carries its id and version:** in `summary.json` as `evals: { "search-composite@1": { eligible, score, C, H } }`, in `posterior.json` rows, and in confirmation records.
- **Rescoring:** `tools/eval/rescore.ts <id@version> [runsDir]` writes a table to `.local/`, so a candidate evaluator can be compared with the pinned one on all merged lives before a decision.

### 2.3 `newborn-baseline@1` (kind cohort) [agent]
- **Input:** newborn lives, i.e. log files with `meta.role = "newborn"` (§7.3). A newborn is an untrained learner of the centre genome. It takes frozen checkups at levels 512, 1,024 and 2,048 with the variants a search life on its panel would get, and it scores its held-out set once.
- **Output (the reference file):** per milestone M1–M6, the median measure (null counted as 0), the pass rate, and the sustained pass rate (≥ 2 of 3 levels). Also the newborn held-out ratio's median, and the learnerIds used. Built only from GPU-width newborns, per suite version and genome schema.
- **Pinned by a person.** Newborn logs arrive through ordinary PRs. A human then commits the reference that `newborn-baseline@1` computes from them, and the file names those learnerIds, so anyone can rebuild it from `data/runs/`. The scoring function therefore reads a fixed file that no later contribution can shift.
- **Size [agent]:** 64 newborns, seeds 1,000–1,063, 4 per panel × 16 panels: 192 checkups at ~80 s, ~4.3 h on one machine [estimate from 0016].

### 2.4 `search-composite@1` (kind life): the search score [user: "checkup measures against newborns plus the held-out prediction ratio"; agent: the formula]
**Eligibility.**
- The life is a search life: `meta.role = "search"`, genome schema 1, GPU widths, inheritance `none`.
- A merged file holds its checkup at level 2,048 (it runs at the first sleep after crossing 2,048 CU, so it often lands in the next file; `session.ts`), and at least 2 of the 3 levels have checkups.
- Every file up to that checkup has a `meta` block. A file without one was written by code that does not know the genome, for example after a rollback.
- Only data at cu ≤ 2,048, plus the checkups at those levels, is read.
- Short lives (0019, `end: "budget"`) and forever lives are equally valid rows.
- **Failed lives are rows, ranked worst [agent].** A life that failed before 2,048 CU has a final file with `end: "failed"` (restores exhausted, or a non-recoverable error). It gets C = H = −2 before ranking, ties included, so a genome that breaks learners is penalised rather than dropped.
- **Abandoned lives are not rows [agent].** A life that stopped before 2,048 CU without a failure (tab closed; `end: "orphan"` or no final file) is not a row. The posterior's diagnostics track the abandonment and failure rate per genome and report them (§7.2).

**Behaviour part C.** For each level L ∈ {512, 1024, 2048} present, and each m ∈ {M2, …, M6}:
- x = `checkups[L].measures[m]`, with null read as 0.
- q_{m,L} = clip( ln( (x + κ_m) / (ñ_m + κ_m) ), −2, 2 ).
- ñ_m is the pinned newborn median. κ_m = 0.1 × the pass threshold: M2 0.2, M3 0.2, M4 0.06, M5 0.15, M6 0.3.
- The offset keeps zero measures finite. M6's newborn median is 0 (0016), and many M3 measures are 0.
- **C = mean over levels of (mean over m of q_{m,L}).**
- M1 is left out: it is a prediction measure, and H covers prediction ([V2 §6.3] leaves it out of the family too). M1's q is kept as a diagnostic.

**Prediction part H.**
- ρ̃ = the median of `heldOut.loss / heldOut.baseline` over curve points with 1,024 < cu ≤ 2,048. At least 3 points are needed, else the life is ineligible. Sleeps every 250 steps give about 50 points in that window.
- **H = clip(−ln ρ̃, −2, 2).**
- If the reference shows the newborn held-out median outside 1 ± 0.01, the evaluator uses −ln(ρ̃ / newborn median) instead. The rule is written down now so that it is not a later choice.

**Headline score (population-free, for display and rescoring):** `score = C + H`. Both parts are reported.

**GP targets (`targets`, over the n eligible training rows).**
1. Rank C and H separately (mid-ranks for ties), giving u_C, u_H = (rank − 0.5)/n.
2. Average them: a_i = (u_C + u_H)/2.
3. Re-rank the a_i and map to normal scores: y_i = Φ⁻¹((rank(a_i) − 0.5)/n).

This gives behaviour and prediction equal weight whatever their raw scales [agent]. Today H moves (0.86–0.95) and C sits near chance, so without ranking H would be invisible. Rank transforms follow [SYN-M §5; DLA L4].

**Fields used:** `checkups[].level/measures`, `curve[].cu/heldOut`, `cu`, `widths`, `hostSig.webgpu`, plus the new `meta` block (§7.3). All but `meta` exist today.

### 2.5 `confirm-fisher@1` (kind cohort): the confirmation test [user: "milestone passes against newborn luck"; agent: details from V2 §6.3]
- **Arms:** `candidate`, `centre` and `newborn`. The newborns are of the **candidate** genome, so an innate advantage (e.g. a higher `exploreEpsilon` that flails into M5) cancels out.
- **Sustained pass:** a life passes m if it passes at **≥ 2 of the 3 levels** 512, 1,024 and 2,048 [V2 §6.3]. A newborn's three frozen checkups follow the same rule.
- **Test 1 (behaviour):** for each of M2–M6, candidate against candidate-newborns, one-sided Fisher exact, Holm over the 5, family α 0.05. Test 1 passes if ≥ 1 milestone survives.
- **Test 2 (better than what we have):** candidate against centre on the rank-average a_i, computed over the pooled candidate and centre lives. One-sided Mann–Whitney U, α 0.05.
- **Passes if and only if both pass.** The score is 1 or 0. The diagnostics hold every p-value, count and dropout.
- **Denominators:**
  - Lives that failed before the budget (`meta.end = "failed"`) count as non-passes. Abandoned lives (tab closed, `end = "orphan"`) are excluded and counted. If abandonment rates differ between arms by more than 2×, the result is flagged.

### 2.6 `signal-check@1` (kind cohort): the repeat-noise check [agent]
- **Input:** calibration lives grouped by genome (§4.6).
- **Test:** Kruskal–Wallis on a_i across genomes. It needs ≥ 6 genomes with ≥ 2 lives each and ≥ 20 lives in all.
- **Signal** if p < α_look and the ICC estimate of `score` is ≥ 0.1 [guess]. **Looks:** at most 3 (at 30, 45 and 60 calibration lives), each at α_look = 0.05/3 (Bonferroni), so repeated looks do not manufacture signal.

## 3. The genome

### 3.1 Files and schema [agent]
- `knowledge/genome-space.json`: the genes. It is new and joins `PRIOR_FILES`, so it is in `knowledgeSha`.
- `knowledge/genome.json`: the **promoted genome**, i.e. the centre, with provenance. It also joins `PRIOR_FILES`. A promotion changes it, and so changes `knowledgeSha` and starts a new cohort.
- `settings-space.json` stays, unchanged, for cached older pages.

```json
{ "schemaVersion": 1,
  "genes": [
    { "name": "lr", "type": "continuous", "scale": "log", "min": 1.25e-4, "max": 2e-3, "centre": 5e-4, "owner": "gp", "target": "learner" },
    { "name": "gamma", "type": "continuous", "scale": "horizon", "min": 0.6, "max": 0.975, "centre": 0.9, "owner": "gp", "target": "learner" },
    { "name": "bufferSize", "type": "integer", "scale": "log", "min": 500, "max": 4000, "centre": 2000, "owner": "gp", "target": "learner" },
    { "name": "latentWeight", "type": "continuous", "scale": "linear", "min": 0, "max": 2, "centre": 0, "owner": "fixed", "implemented": false, "target": "learner" },
    { "name": "creditMode", "type": "categorical", "values": ["off", "gae16"], "centre": "off", "owner": "fixed", "implemented": false, "target": "learner" }
  ] }
```

- **Types:** `continuous` or `integer` with `scale` ∈ {`linear`, `log`, `horizon`}; `categorical`/switch with `values`.
- **Unit-cube maps:**
  - linear: u = (v − min)/(max − min).
  - log: u = ln(v/min)/ln(max/min).
  - horizon: u = ln(h/h_min)/ln(h_max/h_min), with h = 1/(1 − γ).
  - Integers are decoded, rounded, then re-encoded, so the GP sees the value that actually ran.
- **Owners:**
  - `gp`: modelled and searched.
  - `bandit`: a categorical whose arms are raced, not modelled (§4.3). None in Phase A.
  - `fixed`: held at the centre.
- **`implemented: false`:** the gene must stay `fixed` at a centre that is bit-exact to the code without the mechanism. The validator rejects any other value.
- **`target`:** `learner`, `caregiver` or `harness`. It is the only place a decoded value goes.
- **Integrity:** `genomeSha` = sha256 of canonical JSON `{schemaVersion, values sorted by name}`.

### 3.2 Phase A genes (all centres = today's learner) [agent; ranges are guesses]
The widening below is a recorded design choice (§9 decision list). Today `settings-space.json` spans ±20 % on lr, so there is almost nothing to search [SYN-M §4.3].

| gene | today (`DEFAULT_SETTINGS`, `CAREGIVER_DEFAULTS`, `GAMMA`) | v1 range, scale | owner |
|---|---|---|---|
| lr | 5e-4 | 1.25e-4 – 2e-3, log (×4 each way) | gp |
| headLrScale | 0.1 | 0.025 – 0.4, log | gp |
| residualDecay | 1e-3 | 2.5e-4 – 4e-3, log | gp |
| entropyWeight | 0.01 | 0.0025 – 0.04, log | gp |
| curiosityWeight | 1 | 0.25 – 4, log | gp |
| exploreEpsilon | 0.1 | 0.025 – 0.4, log (clamp is 0.5) | gp |
| bufferSize | 2000 | 500 – 4000, integer log (8,000 only after a checkpoint-size probe) | gp |
| replayPerSleep | 64 | 16 – 256, integer log | gp |
| gamma | 0.9 (`GAMMA` constant) | 0.6 – 0.975, horizon (h 2.5 – 40) | gp |
| caregiverPeriod | 500 | 125 – 2000, integer log | gp |
| caregiverTau | 0.05 | 0.0125 – 0.2, log | gp |
| caregiverEpsilon | 0.1 | 0.025 – 0.4, log | gp |
| caregiverThreshold, caregiverStuckSteps | 0.01, 2000 | — | fixed: the safety net only acts after two M6 passes in a row, which no life has reached, so the genes are inert |
| maxStepsPerSec, sleepInterval | 30, 250 | — | fixed: a courtesy cap; sleeps also time the checkups |

That is **12 searched genes and no active switch**. It is under the "< 20 dimensions" rule [OLS O3], and the d + 1 rule [OLS O13] means 13+ rows. **Reserved slots** (owner `fixed`, `implemented: false`, centre = off) [V2 §4; SYN-M §4.3]:
- `latentWeight` (λ_lat, 0), `ctrlWeight` (0), `noveltyWeight` (0), `targetTau` (0.995, unused at λ_lat 0);
- `creditMode` {off, gae16} with `gaeLambda` (0.9);
- `sleepSeqLen` {1, 8};
- `replayBeta` (0); `plasticAlpha` (0), `plasticEta`, `modulator` {none, sur, lp, ctrl, nov};
- `ctrlGate` {gated, additive}; `lrClockSlope` (0).

A slot goes live only when its code lands with a bit-exact-at-centre golden test, under a new genome `schemaVersion` (a fresh fit).

### 3.3 How the learner reads a genome [agent]
- **The genome replaces "settings" at birth.**
  - A new browser-safe package, `packages/meta/`, holds the genome schema, `decodeGenome`, the deterministic maths, the GP sampler and the method draw. `decodeGenome(genome, space)` → `{ learnerSettings, caregiverSettings, harness: { maxStepsPerSec, sleepInterval } }`.
  - `app/src/settings.ts` `birthSettings` and `knowledge.ts` `chooseSettings` give way to `birthGenome(knowledge, learnerId)` (§4.1).
  - The learner still takes `LearnerSettings`. It never sees the genome, its draw or its score.
- **`gamma` joins `LearnerSettings`.** `DEFAULT_SETTINGS.gamma = GAMMA` (0.9), `clampSettings` bounds it to [0, 0.999], and the TD target uses `settings.gamma`. A stored state without `gamma` reads as 0.9, as `stats.lastLoss` absent reads as NaN. At 0.9 the CPU golden trajectory (0017) must be bit-identical.
- **`LEARNER_STATE_VERSION` stays 2 (no bump), so no forever life restarts.**
  - Old checkpoints resume with γ 0.9.
  - Code before 05a ignores the extra `gamma` key, since `checkState` checks only `DEFAULT_SETTINGS` keys. A rollback would therefore run a γ ≠ 0.9 learner at 0.9, but it would also write files without `meta`. `search-composite` marks those lives ineligible (§2.4), so the mislabel cannot reach the fit.
- **Recorded in the log.**
  - Every file carries the `meta` block (§7.3), with the full decoded genome and `genomeSha`.
  - `settings` keeps the flat values as now (with `gamma` added), so summary rows and old readers still work.
  - The checkpoint gets a top-level `meta` (0017: old `checkCheckpoint` allows new top-level keys), so a resumed or adopted life keeps writing the same block.
  - A learner born before 05a resumes with `meta.role = "legacy"` and its settings decoded to a genome. It is scored for display, never fitted: its picks used `Math.random` and per-learner variants.

## 4. The meta-learner loop

### 4.1 Birth: which genome this tab runs [user: TS; 1 in 5 random; agent: mechanics]
1. **Load.** The tab loads knowledge as now, plus `posterior.json` (§7.2). `posteriorSha` is the sha256 of its bytes. Offline or malformed means method `centre`.
2. **Seed.** `drawSeed = hashSeed(0, "draw/" + learnerId + "/" + posteriorSha)` and `rng = createRng(drawSeed)`. The first draw u = rng() picks the method.
3. **Method, by posterior mode:**
   - `calibrate`: every search birth is a calibration life (§4.6).
   - `search`: u < 0.2 → `uniform` (a uniform point of the cube, forever [OLS §10.1]); otherwise `ts`.
   - `paused` or `flat`: u < 0.2 → `uniform`, otherwise `centre`.
   - A confirmation in flight takes u ∈ [0.2, 0.7) for `confirm` (§4.5). The rest is unchanged, so random lives stay at 1 in 5.
   - CPU-width tabs (`hostSig.webgpu = false`) use only `uniform` (0.2) and `centre`: their layer shapes differ, so they are fitted apart, and not until a class has 30 eligible rows [SYN-M §6.5].
4. **Panel.** `evalPanel = 1 + (hashSeed(0, "panel/" + learnerId) mod 16)`. A search or calibration life's checkup variant seeds become `hashSeed(0, "panel/p/level/milestoneId")`, so lives on one panel meet the same variants. That is blocking [OLS O20], and the newborn reference uses the same panels. It amends 0016's per-learner variant seeds for new lives only. Existing lives and their checkpoints keep `checkupSeed(learnerId, …)`. Thresholds are unchanged.
5. **Log.** The tab logs method, drawSeed, posteriorSha, posterior source commit, evalPanel or evalSeed, the genome, genomeSha and inheritance, then starts the learner. Given the posterior bytes, the gate can recompute every pick.

### 4.2 The evaluation budget [user: "only the first 2,048 CU counts"]
- A life's search evaluation is its first 2,048 CU plus the checkups at 512, 1,024 and 2,048. Later data is a separate, labelled longevity reading and is never fitted.
- **Throughput is births, not machine time.** A forever life gives one row per birth (0014). So decision 0019 adds a **"short lives" switch** [user: the switch; agent: details].
  - It is per browser (stored in localStorage), defaults to off, and sits on the LEARNER tab. With it on, a life ends once its 2,048 CU checkup is logged. It writes a final log with `meta.end = "budget"`, and a newborn starts in the same tab (roughly every 15–30 min).
  - With it off (the default), lives are forever, as 0014 says.
  - Lives from both modes are valid search rows. The page states plainly that the search runs at a useful rate only where the switch is on.

### 4.3 The gate's hourly refit [user: GP, deterministic; agent: the GP]
**When.** After merging, the gate rebuilds `data/meta/posterior.json` if the set of eligible rows changed, and commits it with an index line (§7.2). Deploy copies it into `knowledge/` beside `summary.json`. It is outside `knowledgeSha` (it is not in `PRIOR_FILES`).

**Rows.**
- Each eligible search life under the pinned search evaluator gives one row, for the genome schema in force and the GPU class.
- Included: methods `calibrate`, `uniform`, `ts` and `centre`.
- Excluded: `confirm`, `newborn`, `legacy`, other schema versions, other inheritance conditions, and draws that fail re-verification (§7.4).
- Rows are sorted by learnerId. Above 1,000 rows, the gate keeps the 1,000 with the largest `hashSeed(0, "subset/" + learnerId)`: a deterministic random subset, which bounds the cost. Past that point the method is reconsidered (TuRBO or SAASBO [OLS O8, O9]).
- Targets are y = `targets(...)`, so roughly N(0, 1).

**Model.**
- Constant mean m.
- Kernel: σ_f² · Matérn-5/2 with ARD length scales ℓ_j on the unit-cube genes, times an overlap kernel exp(−Σ_c 1[x_c ≠ x'_c]/ℓ_c) for active categoricals. There are none in Phase A.
  - The product form is simpler than CoCaBO's mixture kernel [OLS O10]. *Agent's call:* categoricals are modelled jointly and chosen by Thompson sampling over candidates. That needs no bandit state outside the posterior. `owner: "bandit"` is reserved for arms that must be raced on shared seeds [OLS O20].
- Homoscedastic noise σ_n², with a floor of 1e-3.

**Priors (MAP).**
- ln ℓ_j ~ N(√2 + ½ ln d, √3²). This is the dimension-scaled prior of Hvarfner et al. 2024, BoTorch's current default; it is from the agent's knowledge and was not re-checked for this spec. It favours smooth, nearly flat surfaces when data are noisy.
- ln σ_f² ~ N(0, 1).
- ln σ_n² ~ N(ln 0.5, 1): the prior expects half the variance to be noise.
- m ~ N(0, 1).

**Fit (deterministic, plain TypeScript).**
- Maximise log marginal likelihood + log prior over θ = (ln ℓ, ln σ_f², ln σ_n², m), with analytic gradients and L-BFGS (memory 10).
- 4 starts: the prior mode, plus 3 points of a Sobol sequence scrambled with the fixed seed `"gp-fit/1"`, within ±1.5 prior sd. At most 100 iterations per start; stop when |Δobjective| < 1e-6. The best start wins, and ties go to the earliest start.
- Cholesky jitter follows a fixed schedule: 1e-8·mean(diag), ×10 until it succeeds, at most 1e-2.

**Deterministic maths.**
- `packages/meta/src/detmath.ts` implements `exp`, `log` and Φ⁻¹ using only +, −, ×, ÷ and `Math.sqrt`. Those operations are exactly rounded IEEE-754 in every JS engine, and JS never fuses multiply-adds.
- Loops run in a fixed order.
- With this, the gate's fit and every browser's draw are bit-reproducible. This is a claim to probe (Chrome, Safari, Firefox, Node 22) before 05b relies on it, not an assumption.

**Representation the tabs sample from [agent].**
- `posterior.json` ships the hyperparameters, the training inputs (unit-cube x, rounded to 6 decimals before fitting, so the file is the exact training set) and the targets y.
- The tab rebuilds the exact GP posterior and draws one joint sample on a seeded candidate set:
  - 256 uniform candidates, and 256 local ones: 16 perturbations of each of the 16 training rows with the highest posterior mean, N(0, (0.1·ℓ_j)²) per dimension, clipped to [0, 1].
  - Draw f ~ N(μ_K, Σ_K + 1e-6 I) by Cholesky, take the argmax (ties to the lowest index), then decode.
- **Why this over the alternatives:**
  - Shipping a precomputed mean and covariance on fixed candidates needs K² numbers (about 400 KB at K = 256), and every tab would share one candidate set.
  - Random Fourier features approximate the Matérn kernel and suffer variance starvation at small N.
  - The exact posterior from N ≤ 1,000 rows is a small file (about 150 KB at N = 1,000 [estimate]). It costs about N³/3 + K·N² + K³/3 ≈ 1e9 flops, roughly 1–3 s once at birth in a worker [estimate].
- Pending lives are not "fantasised": asynchronous TS loses little without it [OLS O5].

### 4.4 The incumbent and the trigger [agent]
- **Incumbent:** x* = the argmax of the posterior mean over the gate's own candidate set (seed `"incumbent/" + posteriorSha`, same construction). It is reported by its posterior mean [OLS O7].
- **Trigger:** P(f(x*) − f(centre) > 0) ≥ 0.95 [guess, from SYN-M §5.5], computed from the joint posterior of the two points. Three more conditions:
  - ≥ 100 eligible rows since the last confirmation was declared [guess];
  - no confirmation in flight;
  - x* is not the centre.

### 4.5 Confirmation and promotion [user: confirm on milestone passes; auto-promote only on a pass; agent: mechanics]
**Declare first.** On a trigger, the gate commits `data/meta/confirmations/<c>.json` before any tab can serve it. The file holds:
- the candidate genome and the centre genome;
- arm targets: candidate 24, centre 24, candidate-newborn 60 [V2 §6.3 power];
- the seed block evalSeed ∈ [10,000 + 1,000c, 10,999 + 1,000c], disjoint from every search panel by construction (different hash namespaces, `"confirm/"` against `"panel/"`);
- the pinned confirm evaluator id@version;
- a 21-day deadline [guess].

**Serving.**
- A `confirm` life draws j = rng() → integer in [0, 1,000), so evalSeed = base + j.
- Its arm is j mod 4: 0 → candidate, 1 → centre, 2 and 3 → newborn (cheap, about 4 min). A full arm falls back to `ts`.
- The learner seed is evalSeed. Checkup variants are `hashSeed(0, "confirm/evalSeed/level/milestoneId")`.
- Per (arm, evalSeed), the gate counts only the lowest learnerId, so duplicate draws are not double-counted. Each arm counts its first N eligible lives in learnerId order.

**Decision.** When all arms are full, or the deadline passes (then "incomplete", with no promotion), the gate runs the pinned confirm evaluator and appends the result to the confirmation file.
- **Pass:** in the same commit, the gate writes `knowledge/genome.json` (the new centre, with provenance: the confirmation id, its p-values, rows and lives used) and appends `data/meta/promotions.jsonl`. `knowledgeSha` changes, and new births form a new cohort. Running lives keep their birth genome, so the knowledge refresh does not change their behaviour and their rows stay eligible.
- **Fail or incomplete:** nothing else changes. The outcome is on the record and on the page (invariant 7).
- **Ranges do not move** with the centre within a schema. Re-centring the ranges is a new genome schema (a recorded decision).
- **No coding agent and no person in the loop** for an ordinary promotion. Changing the rule itself needs the user [SYN-M §6.11]. Promotion requires both tests of §2.5 (agent's call, settled 2026-10-23).
- **Integrity.** Promotion rests on confirmation lives, whose genomes the gate fixes in the declaration; nothing there is drawn in a browser. A browser draw mismatch (§7.4) can therefore affect only search rows, never promotion.

### 4.6 The repeat-noise check: calibration before search [user: "first ~30 random lives measure whether the search score has signal"; agent: design]
- **Design.** 10 calibration genomes: the centre, plus 9 points of a Sobol sequence with the fixed seed `"calib/1"`. Each is run 3 times, giving 30 lives on mixed panels. The posterior file lists the genomes and how many lives each still needs. A tab picks one with probability proportional to remaining need.
- **Look 1, at 30 eligible lives.** `signal-check@1`. Signal → mode `search`: the GP is fitted on all calibration rows, and TS begins.
- **No signal → `paused`.** Calibration continues to 6 repeats per genome, with looks 2 and 3 at 45 and 60 lives.
- **Still none → `flat`.** Births run centre (80 %) and uniform (20 %). The page says so (§6). A research note records the finding, and the next step is the v2 mechanisms as genes [SYN-M §6.2]. Leaving `flat` requires a recorded decision: a new genome schema or a new evaluator version, which restarts calibration.
- The 05a gate computes the check, so the answer arrives before any GP code ships.

## 5. Moving up the hierarchy: the first step only

### 5.1 Filter snapshots and the convergence probe (harness-only) [user: material that generalises moves up; agent: KUH's first artifact]
- **Logged:** each file at levels 512, 1,024 and 2,048 carries `meta.filters`: the first-layer conv weights and biases of both encoders, as base64 float32 in parameter order.
  - At GPU widths that is periphery 5×5×3→16 and fovea 3×3×3→16: 1,664 numbers, about 9 KB of base64 [SYN-M §4.5]. CPU widths have 832.
  - The validator checks the decoded length against `widths`.
  - The learner exposes it through the existing `readParams()`. Nothing flows into the learner.
- **The probe (`tools/probe/filter-convergence.ts`, offline).**
  - **Probe set:** 256 frames from 16 world seeds in a reserved range (20,000–20,015), 16 steps each, under seeded uniform actions. They are rendered by the world in the tool, never seen by any life, and fixed by a committed manifest.
  - **Per life:** activations of each encoder's first layer on the probe set, at 2,048 CU and at init (init is recomputed from the learner seed; a test checks it equals a newborn's step-0 parameters).
  - **Pairwise metrics, over fresh search lives:**
    - linear CKA [KUH K27];
    - one-to-one unit matching by Hungarian assignment on activation correlation, reported as the median matched correlation [KUH K26];
    - weight motion ‖W − W₀‖/‖W₀‖.
  - **Controls:** the same metrics between independent inits (random filters have non-zero CKA), and between genomes, to separate convergence from recipe.
- **Converged [guess at the bar]:** mean learned-pair CKA minus mean init-pair CKA ≥ 0.1, the median matched correlation ≥ 0.5, and a permutation test over lives at p < 0.05. Measured over ≥ 10 lives of one genome (the centre or calibration repeats), and separately across genomes.
- **Output:** a research note, favourable or not; if filters do not converge, next come a distilled frame set, then a shared replay or curriculum file [KUH K7, K10, K28].

### 5.2 The inherited-filter condition: specified, not built [agent; gated on the probe and a later decision]
- **Arms:**
  - **F (fresh):** the default.
  - **I (inherited):** `knowledge/filters-gpu-v1.json` loaded at birth, still trainable.
  - **C (matched control):** random filters with matched per-filter norms and biases.
- **Building the artifact.** Generation 0 aligns the output channels to a reference life by Hungarian matching, then averages them. Later generations take a Reptile step toward the mean end state of the I arm.
- **Labels.** Every file carries `meta.inheritance.filters` ∈ {`none`, `<sha>`, `control:<sha>`}. The artifact joins `knowledgeSha`. The GP treats the condition as fixed context, never as a gene. Confirmations never pool conditions. Fresh lives keep a share of ≥ 40 % [guess].
- **Readouts:** CU to the M1 bar; the held-out ratio at 512 and 1,024 CU on unseen variants; long-budget no-worse-than-F; the slope on a novel room (plasticity) [KUH K16, K17]. The population CU spent building the artifact is reported beside the gain (invariant 5).

## 6. The page (brief) [agent]
- **"Your learner's recipe".** How it was drawn, in words: "drawn by the meta-learner (Thompson sample from 214 lives)"; "one of the 1-in-5 random explorers"; "a calibration life (recipe 4 of 10, repeat 2)"; "a confirmation life (candidate arm, fresh world 10,347)".
- **The genome.** The three genes furthest from the centre, as ×ratios ("learning rate ×2.6, curiosity ×0.4"), with the full list behind a toggle, plus its `genomeSha` and `posteriorSha`, both short.
- **"The meta-learner".** One line for its state, plus the date of the last promotion: "Calibrating: 18 of 30 lives measured, to learn whether the score can tell recipes apart." / "Searching: 214 lives; its best guess is predicted to beat the current recipe with 72 % confidence." / "Confirming recipe 3f2a… on fresh worlds: 31 of 108 lives." / "Paused: the score can't yet tell recipes apart."
- **Progress.** A small chart of eligible lives' `score` against birth order, coloured by method, with the promoted centre's line. A best recipe is shown Ladder-style: it changes only on a passed confirmation [OLS O23].
- **After 2,048 CU.** Once this life's own score exists, it is shown with its evaluator id@version.

## 7. Data formats, versioning, determinism, protection

### 7.1 Files

| path | what | written by | in `knowledgeSha`? |
|---|---|---|---|
| `knowledge/genome-space.json` | genes, ranges, owners (§3.1) | humans, by decision | yes |
| `knowledge/genome.json` | the promoted genome, with provenance | the gate, only on a confirmation pass; otherwise humans | yes |
| `config/evaluation.json`, `config/references/*.json` | pinned evaluators and references (§2.2) | humans, by decision | no (learners never read them) |
| `data/meta/posterior.json` | the current posterior (§7.2) | the gate | no (data) |
| `data/meta/posterior-index.jsonl` | `{posteriorSha, commit, builtAt}`, one per version | the gate | no |
| `data/meta/calibration.json` | calibration design and status | the gate | no |
| `data/meta/confirmations/<c>.json`, `promotions.jsonl` | declarations, results and promotions, append-only | the gate | no |

**Gate commits [user: 0019].**
- The gate commits directly to `main` with the Actions token, and only for two path sets: `data/meta/**` and `knowledge/genome.json`.
- The contents are always computed by the gate itself from main's merged data and pinned config, never copied from PR content.
- A gate commit of either path triggers the deploy, as a merge does.
- `validate.ts` keeps rejecting any PR path outside `data/runs/<id>/<n>.json`, so contributions cannot reach these paths. `data/` is not a `BUILD_INPUTS` path, so gate commits never reload tabs (0014); a `knowledge/genome.json` commit changes `knowledgeSha` by design.

### 7.2 `posterior.json` (schemaVersion 1)

```json
{ "schemaVersion": 1, "genomeSchema": 1, "evaluator": "search-composite@1", "sourceCommit": "<main sha the rows came from>",
  "mode": "calibrate|search|paused|flat", "centre": { "genomeSha": "…", "values": { … } },
  "calibration": { "genomes": [ … ], "need": [ … ] } ,
  "models": { "gpu": { "rows": [ { "learnerId": "lrn_…", "x": [ … ] } ], "y": [ … ], "hyper": { "lengthscales": [ … ], "outputscale": 1.1, "noise": 0.6, "mean": 0.02 } } },
  "incumbent": { "x": [ … ], "mean": 0.4, "pBeatsCentre": 0.72 },
  "confirmation": { "id": 3, "candidate": { … }, "seedBase": 13000, "need": { "candidate": 7, "centre": 9, "newborn": 22 } } | null,
  "diagnostics": { "perGenome": [ { "genomeSha": "…", "born": 9, "rows": 6, "failed": 1, "abandoned": 2 } ], "drawMismatch": { "rate": 0.0, "byBrowser": { … } } },
  "page": { "eligibleLives": 214, "lastPromotion": "…" } }
```

- **Recomputable.** `posteriorSha` → the commit (from the index) → these bytes in git history, from which any pick can be recomputed. CI also rebuilds a fixture posterior twice, and on Node 22, and compares it byte for byte with a committed golden file.
- **Size.** About 150 KB at 1,000 rows [estimate]; rewritten only when the rows change (git deltas consecutive versions).

### 7.3 New log fields (log `schemaVersion` stays 2; optional, and strict when present, as in 0017)
A new top-level key `meta`, added to `LOG_KEYS_V2`:
- `v`: 1. `role`: `search` | `confirm` | `newborn` | `legacy`. `method`: `centre` | `calibrate` | `uniform` | `ts` | `confirm` | `newborn` | `legacy`.
- `genomeSchema` (integer), `genome` (name → number or string; ≤ 40 keys, `SETTING_KEY_RE` names), `genomeSha` (hex64), `drawSeed` (uint32 | null), `posteriorSha` (hex64 | null).
- `evalPanel` (1–16 | null), `evalSeed` (integer ≥ 10,000 | null), `confirmation` ({id, arm} | null), `inheritance`: `{ filters: "none" }`.
- `filters` (§5.1; only in files at 512, 1,024 or 2,048 CU); `end` (`failed` | `orphan` | `budget`; final files only, set by `final.ts`).

**Validation (`parseLogFile` plus `validate.ts`):**
- Keys and types are strict.
- Every value lies inside `genome-space.json` bounds, integers are integral, and `implemented: false` genes are at centre.
- `genomeSha` matches the values.
- Every field except `filters` and `end` is constant across a learner's files. Absence is allowed (old code); the evaluator handles it (§2.4).
- **`role: "newborn"`:** cu 0, stepsTotal 0, `final: true`, exactly 3 checkups at 512, 1,024 and 2,048. Only here may a checkup level exceed `cu`.
- `settings` gains `gamma`.

### 7.4 Gate determinism and protection (invariant 7)
- The gate runs main's code, config and references only. PR files are data [V1 §9.5].
- **Re-verification.** For each newly eligible life, the gate recomputes `method`, `evalPanel`, `evalSeed`, the arm and (for `uniform`, `calibrate` and `ts`) the genome, from `learnerId` and the posterior bytes named by `posteriorSha`.
  - A `posteriorSha` that is not in the index, or a mismatched `method`, panel, seed or arm, makes the row ineligible. It is flagged in the summary, and logs are kept, never closed for it.
  - **A genome mismatch alone does not exclude the row [agent].** If the recomputed genome differs (for example, a browser's arithmetic differs) but the logged genome is within bounds and its `genomeSha` validates, the GP fits on the logged genome, which is what actually ran. The row's browser is flagged (from `hostSig` and the user-agent class, if logged), and the mismatch rate overall and per browser goes in the posterior diagnostics and on the page.
  - A TS recheck costs one sampler run (about 1–3 s) per life, which is affordable at ≤ 50 lives an hour.
- **Seeds that PR content cannot choose:**
  - Panels and confirmation seeds come from hash namespaces the evaluators define.
  - Confirmation blocks are committed by the gate before they are served, and search never reads `confirm` rows.
  - A dishonest tab could grind learnerIds to pick its panel. That is a known limit (dishonest contributors are a non-goal).
- The rows, sort order, seeds, iteration counts and jitter schedule are all fixed (§4.3), and a fixture test asserts identical bytes across two runs. Every decision the gate takes (the signal check, triggers, results, promotions) is appended to `data/meta/` with its inputs.

## 8. Budget estimates
- **Evaluations.** One per birth (0014), ~30 min to the 2,048 checkup [OLS §0].
  - **Switch off (forever lives, the default):** births happen only when a tab opens with nothing to adopt, a handful a day across all tabs [guess].
  - **Switch on (short lives, 0019):** one row per ~30 min per tab, so ~110–340 per machine per week (8 h or 24 h a day) [0018 estimate].
  - With one machine on short lives: calibration (30–60 lives) takes ~1–4 days; Thompson-sampling search to a trigger (~200–500 lives [OLS §10]) ~2–5 weeks; one confirmation (108 lives, of which 60 are cheap newborns) ~1 week. With 3 machines, divide by about 3. With the switch off everywhere, each stage takes months.
- **The gate.** At N = 300 a fit is about 2e7 flops per objective evaluation × 400 evaluations, ~10 s. At N = 1,000 it is about 1–3 min [estimates]. That is within the job's 30-minute limit, and the row cap keeps it there.
- **The tab.** A draw takes about 1–3 s once at birth [estimate]. Each log file grows by about 2 KB for `meta`, and each life by about 27 KB of filters.
- **The newborn reference.** About 4.3 h once per (widths class, suite, genome schema).

## 9. Risks
- **Flat landscape** (checkups at chance today [V2 §1]). This is the calibration check's job. `flat` is a reported finding, not a failure.
- **Throughput depends on the switch (0019).** See §8. If few tabs turn short lives on, the search takes months.
- **Meta-overfitting to 16 panels** [OLS O21; DLA L8]. Mitigated by disjoint confirmation seeds, rows never reused for confirmation, and reporting how many genomes were tried.
- **Innate rather than learned gains** (e.g. a high ε flailing into M5 passes). Confirmation compares against the candidate's own newborns, and the 8 CU checkup is shown per genome as an "innate" diagnostic.
- **Horizon:** a genome tuned to 2,048 CU may degrade later [LLR L2, L10]. Longevity readings past 2,048 CU are shown for the centre and the incumbent, never fitted.
- **Code changes mid-search.** A behaviour-changing change must bump `genome-space.json` `schemaVersion`, starting a fresh fit. The CPU golden-trajectory test is the guard that catches an unrecorded change. A refactor that keeps the golden test leaves rows comparable across `codeSha`.
- **Cross-engine determinism of draws** is unproven until probed (§4.3). If a browser differs, its draws are still logged and verified as far as possible, and that browser is flagged.
- **Rank targets and selection bias.** Targets depend on the whole row set, so a single life's y changes from one refit to the next. That is expected. The absolute `score` stays the per-life number.
- **A wider lr range can diverge.** Divergence ends in restores or a failed life (0017). A failed search life is a row at the worst rank (C = H = −2, §2.4).
- **Differential abandonment.** If a genome's lives are abandoned more often (for example, they make the page slow), its rows are biased. The per-genome rates in the posterior diagnostics expose this.

## 10. Phased delivery (each phase ships live; no `LEARNER_STATE_VERSION` bump; forever lives continue)
- **05a: evaluators, genome, calibration, logging.**
  - `packages/evaluators` with the four v1 evaluators and `config/evaluation.json`.
  - `knowledge/genome-space.json` and `genome.json`; `packages/meta` with decode, the deterministic maths and the method draw; `gamma` as a setting.
  - Births by method `centre`, `uniform` and `calibrate`, seeded; panels; the newborn mode; the `meta` block and `filters` snapshots; validation; summary `evals` columns; `tools/eval/rescore.ts`.
  - The newborn reference committed; the gate's signal check; the page's recipe and state lines.
  - The short-lives switch (0019) and the `end` reasons; the gate's direct commits of `data/meta/**`, which trigger the deploy.
  - Recorded decisions: the evaluators v1; the genome schema; and a new decision, written during 05a, that records two agent's calls: the shared variant panels for new lives (amending 0016) and the widened gene ranges.
  - **Done when** the signal check has answered.
- **05b: GP, Thompson sampling, confirmation, promotion.**
  - The deterministic GP fit and `posterior.json` in the gate, with index and fixtures; the cross-browser determinism probe; TS draws in the tab; re-verification; incumbent, trigger, confirmation declaration and serving; the confirm evaluator wired to promotion; promotion records and page progress.
  - Ships only if 05a found signal. Otherwise 05b waits behind the v2 genes.
- **05c: filter probe.**
  - `tools/probe/filter-convergence.ts` over the logged snapshots, and the write-up.
  - The §5.2 condition is drafted as a decision for the user, not built.

## 11. Left to the builder
- Module boundaries inside `packages/meta`: whether `hashSeed` and `createRng` move to `interface` or are copied. The learner's wall must not change.
- L-BFGS details within §4.3 (line search, convergence tests; all deterministic); the Sobol generator and scrambling; candidate counts if 512 is too slow on the CPU class.
- The UI layout for §6 and the switch; scoping the gate's push permission to its two path sets; how a newborn-mode tab runs its 3 checkups and held-out scoring; how `final.ts` tells `failed` from `orphan`.
- Whether `filters` uses float32 or float16. Float16 halves the size; the probe tolerance must allow it.

## 12. Open questions for the user
None open. Resolved on 2026-10-23:
- **Throughput:** the short-lives switch, decided by the user in 0019 (§4.2).
- **Gate writes:** direct commits of `data/meta/**` and `knowledge/genome.json`, decided by the user in 0019 (§7.1).
- **Promotion requires both tests** of §2.5, and **the caregiver genes are searched** (§3.2). These two are agent's calls, now settled defaults.

## Sources
Project: decisions 0018, 0019; [SYN-M] §1, §4–§6; [OLS] §0, §1, §4, §5, §8–§10 (O2–O5, O7–O10, O13, O20, O21, O23, O25); [KUH] §6, "Concrete first candidate" (K12, K16, K17, K26, K27); [DLA] L4, L8, L10; [LLR] L2, L10; [V1] §5.1, §8, §9; [V2] §1, §4, §6.2–6.4; decisions 0011, 0014, 0016, 0017. Literature from the agent's knowledge, not re-checked here: Hvarfner, Hellsten, Nardi 2024, "Vanilla Bayesian optimization performs great in high dimensions" (the length-scale prior); Kruskal–Wallis, Fisher exact, Holm and Mann–Whitney are standard tests.
