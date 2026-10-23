# Plan 05a — Meta-learner foundations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## In plain terms
The meta-learner (decision 0018) will treat every new tab as one experiment. Each tab draws a learning recipe, called a **genome**. The tab lives 2,048 CU, its checkups are scored, and the scores teach the meta-learner which recipes produce learners that learn well. This plan builds everything the meta-learner needs except its brain, which is the Gaussian-process fit in Plan 05b:
- **A recipe for every newborn, drawn so anyone can check it.** Today the page rolls `Math.random`. From now on the draw is a pure function of the learner's id and the published meta-learner file. The gate can redo every draw.
- **Scores that can be swapped.** All scoring lives in one walled package of versioned "evaluators". The gate uses the ones pinned in a protected file. Every score says which evaluator, and which version, made it. Any life can be rescored under another evaluator.
- **A noise check before any search.** The first ~30 short lives run 10 fixed recipes three times each. The gate then asks whether the score tells recipes apart better than chance. If it cannot, the search would only chase noise, and the page says so.
- **Short lives on a switch** (decision 0019). With the switch on, a life ends once its 2,048 CU checkup is logged, and a newborn starts in the same tab. With it off (the default), lives go on forever, as now.
- **The gate writes the meta-learner's file itself.** The file is `data/meta/**`. The gate computes every byte from main's merged data and commits it directly. Nothing a pull request contains can reach that path.
- **The page says plainly** what recipe this learner has, how it was drawn, and what the meta-learner is doing.

**Goal:** ship spec §10 phase 05a live. That is evaluators, genome, seeded births, the `meta` log block, newborn reference runs, the noise check computed by the gate, the gate's direct commits of `data/meta/**`, the short-lives switch and the page lines. Learning and forever lives stay unchanged.

**Architecture:**
- **Three new walled packages.**
  - `packages/detmath` has no imports at all. It holds deterministic maths, the seed hashes and the Sobol points.
  - `packages/meta` imports only `detmath`. It holds the genome schema, decoding and the birth draw.
  - `packages/evaluators` imports only `detmath` and `@blanktab/harness/log`. It holds the scoring functions and the registry.
- **Who uses them.**
  - The app uses `meta`.
  - The gate and `tools/eval` use `meta` and `evaluators`.
  - The harness uses neither. It stores and validates the `meta` block's shape in `log.ts`, like any other log field.
- **The gate** gains a step after merging: build `data/meta/**` (`tools/gate/meta.ts`), check that the commit touches only those paths, then push. Deploy then copies `data/meta/posterior.json` into `knowledge/`.

**Tech stack:** TypeScript, Vitest, ESLint flat config, `@blanktab/*` workspaces, Playwright (e2e and live runs), GitHub Actions (`GITHUB_TOKEN`), Web Crypto `crypto.subtle` (sha256, in the browser and in Node 22).

**Spec:** `docs/specs/2026-10-22-meta-learner-v1-design.md` (§2, §3, §4.1–4.2, §4.6, §6, §7, §10 "05a"). Decisions 0018 and 0019 are binding. Also 0014, 0016 and 0017.

**Planned against `main` at `de74619`.** Another agent may be changing `app/index.html` and `tests/e2e/loop.spec.ts` on the `layout-fix` branch. Task 12 touches both files, so it rebases onto that work first (see Task 12).

## Global Constraints
- **Earlier plans still bind.** That covers the walls, the tensor lifecycle rules, cold start, the token rule, and the log and checkpoint invariants of Plans 04a–04c.
- **AGENTS.md invariant 2.** The learner sees pixels and emits actions. Its genome, its draw, its score, the posterior and any reference never reach it. Decoding produces only `LearnerSettings` values (with `gamma` added), and the harness, caregiver and page use the rest.
- **AGENTS.md invariant 5.** This plan is research-agent design work, not learner-discovered improvement. The page and the research note say which changes came from us. The calibration lives are the first learner-side evidence.
- **AGENTS.md invariant 7.** Evaluation is protected:
  - Evaluators are pure and deterministic, versioned, and live in `packages/evaluators`. The gate imports that package.
  - The pinned `id@version`s live in `config/evaluation.json`, which the gate never writes.
  - Changing evaluator code means a new version in `REGISTRY` and a recorded decision. Changing a pin means editing `config/evaluation.json` and a recorded decision.
  - Every score written anywhere carries its `evaluator@version`.
  - Unfavourable results (no signal, flat) are recorded and shown.
- **Evaluators are pure.** They do no IO, read no clock and call no `Math.random`. All their maths goes through `@blanktab/detmath`. The same input bytes give the same output bytes.
- **The learner's checkpoint version is unchanged.** `LEARNER_STATE_VERSION` stays **2** (verified: `packages/learner/src/version.ts`), so forever lives continue through every task. Log `schemaVersion` stays 2. `milestoneSuiteVersion` stays 1.
- **New fields are optional, and strict when present.**
  - The fields: log `meta`, checkpoint `meta`, the learner setting `gamma`, summary `evals`.
  - The writer (`session.ts`, `final.ts`, `newborn.ts`), the parser (`log.ts`) and the gate (`validate.ts`) change in the same task, so they stay in sync.
  - Every file under `data/runs/` still parses (the existing log test).
- **Birth picks are seeded and recomputable.**
  - No `Math.random` on the birth path.
  - `chooseSettings` and `birthSettings` are deleted.
  - Every pick is `birthDraw(input)`, a pure function of `learnerId`, the posterior bytes, the genome space, the centre and the widths class.
- **The gate's commits.**
  - A gate commit contains only files the gate computed, under exactly `data/meta/**`. (`knowledge/genome.json` joins this list in 05b.)
  - It never contains PR content and never force-pushes.
  - `validate.ts` keeps labelling any session PR with a non-`data/runs/<id>/<n>.json` path `needs-human`.
- **Variant thresholds, budgets and the suite are unchanged** (0016). Only *which* variant seeds new lives meet changes (shared panels, Task 5).
- **Agent's calls are labelled.** Every choice this plan makes for the user is recorded in `docs/decisions/0022-shared-panels-wider-ranges-and-05a-calls.md`, marked not yet confirmed.
- **Hygiene applies to docs and to gate output.**
  - The trailing-whitespace test scans `.md`, `.json`, `.yml` and `.ts`, including `docs/` and `data/meta/`.
  - The machine-path test skips `docs/` and `.md`, but no absolute home-directory path goes into any file, docs included.
  - Gate-written JSON ends with a newline and has no trailing spaces.
- **Dependencies.** New workspaces need one `npm install`, with no arguments, to link them. After it, run `npm run dev` and load the page, then `npm test`. Treat "removed/changed N packages" as a red flag (global rule).
- **Gates for every task.**
  - Always: `npm test && npm run typecheck && npm run lint`.
  - When learner or tensor code changed: `npm run test:gpu`.
  - When `app/` changed: `npm run e2e`.
  - **After any push**, open the CI run and check it is green. Linux fonts differ from macOS, so a local e2e pass is not enough.

## Review Focus
1. **A forever life born before 05a resumes on 05a code.** Its checkpoint has no `meta`, its settings have no `gamma`, and its `entropyWeight` (0.001–0.05 under today's picker) may lie outside the new gene ranges. It must:
   - resume at γ 0.9, bit-identically;
   - log `meta.role = "legacy"`;
   - pass the gate (genome bounds are not applied to `legacy`);
   - never become a fitted row.

   (Tasks 2, 5, 6)
2. **The short-lives switch flipped at odd moments.** The moments are: during a checkup, after the 2,048 checkup is already logged, just before a reload, and off again before the checkup. In every case the life ends exactly once, at a clean sleep, with `meta.end = "budget"`, its 2,048 checkup in the final file, and a newborn after it. With the switch off, no life ends. (Task 6)
3. **A gate commit that carries something else.**
   - Examples: a file `npm ci` touched, a `config/` edit, a stray file, a race with a human push.
   - The guard aborts the commit if any staged path is outside `data/meta/`.
   - A rejected push is retried at most 3 times: fetch, rebuild from the new main, push. It never forces.

   (Task 11)
4. **Bytes that differ between runs or engines.**
   - `buildMeta` run twice gives identical bytes.
   - A draw made in the page equals the Node recomputation for the same learnerId and posterior.
   - `detmath` gives committed golden values in Node and in Chromium.

   (Tasks 1, 4, 10, 12)
5. **Missing or malformed meta-learner files on the page.** `posterior.json`, `genome-space.json` or `genome.json` may be absent, a 404, garbage, or offline. The learner still starts:
   - With a valid space and centre but no posterior: method `centre`, `posteriorSha: null`.
   - Without a valid space or centre: `DEFAULT_SETTINGS` and no `meta`, which is a valid but ineligible life.

   The page says which case applies. (Tasks 4, 12)

---

### Task 1: `packages/detmath` and the walls for the three new packages

**Files:**
- Create: `packages/detmath/{package.json,tsconfig.json,src/index.ts,src/math.ts,src/seed.ts,src/sobol.ts}`
- Create: `packages/detmath/test/detmath.test.ts`, `packages/detmath/test/fixtures/golden.json`
- Create: `tools/detmath-copies.test.ts`
- Modify: `eslint.config.js` (`PKGS`, `FORBIDDEN`, a purity block), `tools/walls.test.ts`

**Interfaces:**
- Produces (`@blanktab/detmath`):
  ```ts
  export function dexp(x: number): number             // only + − × ÷ and Math.sqrt; |rel err| ≤ 2e-16 on [−700, 700]
  export function dlog(x: number): number             // x > 0; NaN otherwise
  export function normCdf(x: number): number
  export function normInv(p: number): number          // Wichura AS241 (PPND16); p ∈ (0,1), else NaN
  export function lnGamma(x: number): number          // Lanczos g=7, n=9, via dlog/dexp
  export function chiSquareSf(x: number, df: number): number // regularized upper gamma Q(df/2, x/2): series for x < a+1, else Lentz CF
  export function hashSeed(seed: number, salt: string): number    // byte-identical copy of @blanktab/world hashSeed
  export function createRng(seed: number): () => number           // byte-identical copy of @blanktab/tensor createRng (mulberry32)
  export function sobolPoints(dims: number, n: number, salt: string): number[][] // points 1..n of Joe–Kuo 6.21201, dims ≤ 16, digital shift (XOR) by createRng(hashSeed(0, salt)) one uint32 per dim
  ```
- Walls (`eslint.config.js`):
  - `PKGS` adds `detmath`, `meta` and `evaluators`.
  - `detmath` may import nothing.
  - `meta` may import only `detmath`.
  - `evaluators` may import only `detmath` and `@blanktab/harness/log`: pattern `["@blanktab/harness", "@blanktab/harness/*", "!@blanktab/harness/log"]`.
  - `learner` and `world` forbid all three new packages. This tightens their walls; it opens no new import.
  - A purity block on `packages/{detmath,meta,evaluators}/src/**` reuses the world's `Math.random`, `Date` and `performance` bans, and also bans `node:*` and bare `fs`/`path` imports.

- [ ] **Step 1: Write the failing tests**
  - `detmath.test.ts`:
    - `"dexp/dlog agree with Math.exp/Math.log to 2e-16 relative on 10,000 seeded points"`;
    - `"dexp(0) = 1, dlog(1) = 0, dlog(0) = −Infinity, dlog(−1) NaN"`;
    - `"normInv(normCdf(x)) = x to 1e-12 on [−8, 8]"`, and `"normInv(0.975) = 1.959963984540054"`;
    - `"chiSquareSf(16.919, 9) = 0.05 to 1e-4; chiSquareSf(0, k) = 1"`;
    - `"sobolPoints(12, 9, 'calib/1') are in [0,1)¹², distinct, and equal the golden fixture"`;
    - `"every function's outputs on the fixture inputs equal golden.json bit for bit"`. Generate the fixture with `DETMATH_WRITE=1`.
  - `tools/detmath-copies.test.ts`:
    - `"hashSeed equals @blanktab/world hashSeed on 1,000 (seed, salt) pairs"`;
    - `"createRng streams equal @blanktab/tensor createRng for 1,000 draws on 20 seeds"`.
  - `walls.test.ts`:
    - `"detmath imports nothing; meta only detmath; evaluators only detmath and @blanktab/harness/log"`. `import "@blanktab/harness/session"` in `packages/evaluators/src/x.ts` is one error.
    - `"learner and world cannot import meta, evaluators or detmath"`;
    - `"Math.random, Date, performance and node:fs are errors in packages/{detmath,meta,evaluators}/src"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run packages/detmath tools/walls.test.ts tools/detmath-copies.test.ts`. Expected: FAIL (module not found).
- [ ] **Step 3: Implement.** Write the package and run `npm install` (no arguments) to link it, then check the lockfile diff only adds the workspace link. `dexp`: range reduction x = k·ln2 + r with ln2 split hi/lo, then a degree-13 Taylor series on |r| ≤ ln2/2, then scale by 2^k through repeated multiplication (no `Math.pow`). `dlog`: reduce to m ∈ [√½, √2) by repeated halving or doubling, then `2·atanh((m−1)/(m+1))` by series.
- [ ] **Step 4: Run them; they pass.** Run the same command, then `npm run lint`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(detmath): deterministic exp/log/Φ⁻¹/χ², seed hashes and Sobol points; walls for detmath, meta and evaluators`

### Task 2: `gamma` becomes a learner setting

**Files:**
- Modify: `packages/learner/src/learner.ts` (`LearnerSettings`, `DEFAULT_SETTINGS`, `clampSettings`, `checkState`, the TD target that uses `GAMMA`)
- Modify: `packages/learner/test/golden.test.ts`
- Test: `packages/learner/test/learner.test.ts`, `packages/harness/test/session.test.ts`

**Interfaces:**
- Produces:
  - `LearnerSettings.gamma: number`.
  - `DEFAULT_SETTINGS.gamma = GAMMA`, with 0.9 appended as the last key so earlier key order is kept.
  - `export const GAMMA_RANGE = [0, 0.999] as const`.
- **Rules:**
  - `clampSettings` treats an absent `gamma` as `GAMMA`. That is a state saved before 05a, read as `stats.lastLoss` absent reads as NaN. A non-finite `gamma` is a RangeError, and others are clamped into `GAMMA_RANGE`.
  - `checkState` skips the equality check for `gamma` when the stored settings lack it.
  - `createLearner({ state })` uses `{ ...state.settings, gamma: state.settings.gamma ?? GAMMA }`, so its next `save()` writes `gamma`.
  - The TD target is `settings.curiosityWeight * info.reward + settings.gamma * valueNext`.

- [ ] **Step 1: Make the golden test key-order-proof on unchanged code.**
  - Hash `save()` with `settings.gamma` removed: `const { gamma, ...settings } = state.settings`, then hash `{ ...state, settings }`.
  - Assert `gamma === undefined || gamma === GAMMA`.
  - Run `npx vitest run packages/learner/test/golden.test.ts`. Expected: PASS with the **unchanged** fixture hash `905654ea…`.
  - Commit `test(learner): golden hash ignores a gamma setting key`.
- [ ] **Step 2: Write the failing tests** in `learner.test.ts` under `describe("gamma setting (Plan 05a)")`:
  - `"DEFAULT_SETTINGS.gamma is GAMMA (0.9)"`.
  - `"clampSettings: gamma absent → 0.9; 1.2 → 0.999; −1 → 0; NaN → RangeError"`.
  - `"a state saved without gamma resumes with γ 0.9, and its next 50 steps are bit-identical to a learner whose state had gamma 0.9"`. This is Review Focus 1.
  - `"γ 0.6 gives a different advantage trajectory from γ 0.9 on the same stream"`. The setting is live.
- [ ] **Step 3: Run them; they fail.** Expected: FAIL. `gamma` does not exist yet.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Verify identity.** Run `npx vitest run packages/learner && npm run test:gpu -- learning`. Expected: PASS, with `golden.test.ts` unchanged and no `GOLDEN_WRITE`. **If the golden hash changes, stop and report.** Also run `npx vitest run packages/harness`. The session and checkpoint suites stay green, and logs now carry `settings.gamma`.
- [ ] **Step 6: Commit** `feat(learner): gamma is a setting (default 0.9, bit-identical); states without it resume at 0.9`

### Task 3: The genome — schema, files, decoding (`packages/meta`)

**Files:**
- Create: `packages/meta/{package.json,tsconfig.json,src/index.ts,src/genome.ts,src/sha.ts}`, `packages/meta/test/genome.test.ts`
- Create: `knowledge/genome-space.json`, `knowledge/genome.json`
- Create: `tools/genome-centre.test.ts`
- Modify: `tools/gate/manifest.ts` (`PRIOR_FILES` adds `genome-space.json` and `genome.json`, kept sorted)
- Create: `docs/decisions/0021-genome-schema-v1.md` and `docs/decisions/0022-shared-panels-wider-ranges-and-05a-calls.md` (the ranges section now; Task 5 adds panels, other tasks add their calls)
- Test: `tools/gate/test/manifest.test.ts`

**Interfaces:**
- Produces (`@blanktab/meta`):
  ```ts
  export type GeneValue = number | string
  export type Genome = Record<string, GeneValue>
  export interface GeneSpec {
    name: string; type: "continuous" | "integer" | "categorical"; scale?: "linear" | "log" | "horizon"
    min?: number; max?: number; values?: string[]; centre: GeneValue
    owner: "gp" | "bandit" | "fixed"; implemented?: false; target: "learner" | "caregiver" | "harness"
  }
  export interface GenomeSpace { schemaVersion: number; genes: GeneSpec[] }
  export function parseGenomeSpace(v: unknown): GenomeSpace          // RangeError on any violation (below)
  export function parseCentre(v: unknown, space: GenomeSpace): Genome // knowledge/genome.json → its values; RangeError
  export function searched(space: GenomeSpace): GeneSpec[]            // owner "gp", in file order
  export function toUnit(space: GenomeSpace, g: Genome): number[]     // over searched(space)
  export function fromUnit(space: GenomeSpace, u: readonly number[], centre: Genome): Genome // integers rounded; non-searched genes = centre
  export function checkGenome(space: GenomeSpace, g: Genome): string | null // keys = genes exactly, types, bounds, integral, implemented:false at centre
  export interface Decoded { learner: Record<string, number>; caregiver: Record<string, number>; harness: { maxStepsPerSec: number; sleepInterval: number } }
  export function decodeGenome(g: Genome, space: GenomeSpace): Decoded // implemented genes only, by target
  export function legacyGenome(settings: Record<string, GeneValue>, space: GenomeSpace): Genome // a pre-05a life's flat settings; absent genes at centre; NOT bounds-checked
  export function sha256Hex(bytes: Uint8Array): Promise<string>       // crypto.subtle
  export function genomeSha(schemaVersion: number, g: Genome): Promise<string> // sha256 of JSON.stringify({ schemaVersion, values: <keys sorted> })
  ```
- **Unit-cube maps (spec §3.1):**
  - linear: u = (v − min)/(max − min).
  - log: u = dlog(v/min)/dlog(max/min).
  - horizon: on h = 1/(1−γ), with h_min and h_max derived from min and max.
  - Decoding uses `dexp`/`dlog` only, never `Math.exp`, `Math.log` or `Math.pow`.
- **`parseGenomeSpace` rejects:**
  - duplicate or non-`SETTING_KEY_RE` names;
  - more than 40 genes;
  - a centre outside [min, max];
  - a `gp` gene with min ≥ max, or `log`/`horizon` with min ≤ 0;
  - `implemented: false` with owner ≠ `fixed`;
  - categorical values that are missing or duplicated;
  - an `owner: "bandit"` gene (reserved; none in Phase A).
- **`knowledge/genome-space.json`, schemaVersion 1** (spec §3.2; [agent] ranges are guesses):

| gene | type, scale | min – max | centre | owner | target |
|---|---|---|---|---|---|
| lr | continuous log | 1.25e-4 – 2e-3 | 5e-4 | gp | learner |
| headLrScale | continuous log | 0.025 – 0.4 | 0.1 | gp | learner |
| residualDecay | continuous log | 2.5e-4 – 4e-3 | 1e-3 | gp | learner |
| entropyWeight | continuous log | 0.0025 – 0.04 | 0.01 | gp | learner |
| curiosityWeight | continuous log | 0.25 – 4 | 1 | gp | learner |
| exploreEpsilon | continuous log | 0.025 – 0.4 | 0.1 | gp | learner |
| bufferSize | integer log | 500 – 4000 | 2000 | gp | learner |
| replayPerSleep | integer log | 16 – 256 | 64 | gp | learner |
| gamma | continuous horizon | 0.6 – 0.975 | 0.9 | gp | learner |
| caregiverPeriod | integer log | 125 – 2000 | 500 | gp | caregiver |
| caregiverTau | continuous log | 0.0125 – 0.2 | 0.05 | gp | caregiver |
| caregiverEpsilon | continuous log | 0.025 – 0.4 | 0.1 | gp | caregiver |
| caregiverThreshold | continuous linear | 0.01 – 0.01 | 0.01 | fixed | caregiver |
| caregiverStuckSteps | integer linear | 2000 – 2000 | 2000 | fixed | caregiver |
| maxStepsPerSec | integer linear | 30 – 30 | 30 | fixed | harness |
| sleepInterval | integer linear | 250 – 250 | 250 | fixed | harness |

  Reserved slots, all `owner: "fixed"`, `implemented: false`, `target: "learner"` [agent ranges]:
  - continuous linear:
    - latentWeight 0–2 (0);
    - ctrlWeight 0–2 (0);
    - noveltyWeight 0–2 (0);
    - targetTau 0.9–0.999 (0.995);
    - gaeLambda 0–1 (0.9);
    - replayBeta 0–1 (0);
    - plasticAlpha 0–1 (0);
    - plasticEta 0–0.5 (0.1);
    - lrClockSlope −1–1 (0).
  - categorical:
    - creditMode [off, gae16] (off);
    - sleepSeqLen ["1", "8"] ("1");
    - modulator [none, sur, lp, ctrl, nov] (none);
    - ctrlGate [gated, additive] (gated).

  That is 29 genes, 12 of them searched.
- **`knowledge/genome.json`:** `{ "schemaVersion": 1, "genomeSchema": 1, "values": { …every gene at its centre… }, "provenance": { "source": "initial", "decision": "docs/decisions/0021-genome-schema-v1.md" } }`.

- [ ] **Step 1: Write the failing tests**
  - `genome.test.ts`:
    - `"parseGenomeSpace accepts the committed file and rejects each violation listed (one case each)"`;
    - `"toUnit(fromUnit(u)) = u for searched genes on 1,000 seeded u; integers round-trip as the value that ran"`;
    - `"decodeGenome(centre) gives exactly DEFAULT_SETTINGS-shaped values: lr 5e-4 … gamma 0.9, caregiver CAREGIVER_DEFAULTS, harness 30/250"`;
    - `"decodeGenome never emits an implemented:false gene"`;
    - `"checkGenome rejects a missing key, an extra key, an out-of-range value, a fractional integer and a reserved gene off centre"`;
    - `"legacyGenome maps flat settings (entropyWeight 0.001 kept, out of range) and fills absent genes with centres"`;
    - `"genomeSha is order-independent and stable (golden hex)"`.
  - `tools/genome-centre.test.ts` (tools may import every package):
    - `"genome.json values = genome-space centres = DEFAULT_SETTINGS, CAREGIVER_DEFAULTS, DEFAULT_MAX_STEPS_PER_SEC and DEFAULT_SLEEP_INTERVAL"`;
    - `"every gene range lies inside the learner's and caregiver's clamps (LR_RANGE, HEAD_LR_SCALE_RANGE, EXPLORE_EPSILON_RANGE, RESIDUAL_DECAY_RANGE, GAMMA_RANGE, CAREGIVER_RANGES)"`.
  - `manifest.test.ts`: `"knowledgeSha changes when genome-space.json or genome.json changes"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run packages/meta tools/genome-centre.test.ts tools/gate/test/manifest.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** Write the package and files, then run `npm install` to link the workspace. Write **0021** (the genome schema v1 as above: the files, the maps, `genomeSha`, "a slot goes live only with a bit-exact-at-centre golden test under a new schemaVersion"). Write **0022 §Widened ranges**: today's ±20 % lr span left almost nothing to search, so every searched gene now spans ×4 each way around today's value [agent; guesses]. Each decision says agent's call, not yet confirmed.
- [ ] **Step 4: Run them; they pass.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(meta): genome schema v1, knowledge/genome-space.json and genome.json (in knowledgeSha); decisions 0021, 0022`

### Task 4: The seeded birth draw

**Files:**
- Create: `packages/meta/src/posterior.ts`, `packages/meta/src/draw.ts`, `packages/meta/test/draw.test.ts`, `packages/meta/test/fixtures/draws.json`

**Interfaces:**
- Consumes: `hashSeed` and `createRng` (Task 1); `GenomeSpace`, `Genome`, `fromUnit` and `searched` (Task 3).
- Produces:
  ```ts
  export type Mode = "calibrate" | "search" | "paused" | "flat"
  export interface Posterior {           // data/meta/posterior.json, schemaVersion 1 (spec §7.2); 05a writes models {}, incumbent null, confirmation null
    schemaVersion: 1; genomeSchema: number; evaluator: string; sourceCommit: string; mode: Mode
    centre: { genomeSha: string; values: Genome }
    calibration: { genomes: { genomeSha: string; values: Genome }[]; need: number[] }
    models: Record<string, unknown>; incumbent: unknown | null; confirmation: unknown | null
    diagnostics: Record<string, unknown>; page: { eligibleLives: number; lastPromotion: string | null; calibrationLives: number; calibrationTarget: number; looks: number }
  }
  export function parsePosterior(text: string): Posterior | null     // null on any shape error (the tab then draws centre)
  export type Method = "centre" | "calibrate" | "uniform" | "ts" | "confirm" | "newborn" | "legacy"
  export type Role = "search" | "confirm" | "newborn" | "legacy"
  export interface DrawInput {
    learnerId: string; space: GenomeSpace; centre: Genome; widthsClass: "gpu" | "cpu"
    posterior: Posterior | null; posteriorSha: string | null       // null together: offline, absent or malformed
  }
  export interface Draw { role: Role; method: Method; drawSeed: number | null; posteriorSha: string | null; evalPanel: number; genome: Genome; calibrationIndex?: number }
  export const UNIFORM_SHARE = 0.2
  export const PANELS = 16
  export function evalPanelOf(learnerId: string): number             // 1 + (hashSeed(0, "panel/" + learnerId) % 16)
  export function drawSeedOf(learnerId: string, posteriorSha: string): number // hashSeed(0, "draw/" + learnerId + "/" + posteriorSha)
  export function birthDraw(input: DrawInput): Draw
  ```
- **The draw (spec §4.1).** `rng = createRng(drawSeed)`, and `u0 = rng()` always comes first.
  - **No posterior:** `centre`, with `drawSeed: null`.
  - **`gpu` and mode `calibrate`:** method `calibrate`. Pick a calibration genome: index i with probability ∝ `need[i]`, or uniformly when Σneed = 0, using `u1 = rng()`.
  - **Modes `search`, `paused` and `flat`, and every mode on `cpu`:** if `u0 < UNIFORM_SHARE`, method `uniform`, and the genome is `fromUnit(space, [rng() for each searched gene, in file order], centre)`. Otherwise `centre`. In 05a a `search` posterior has no model, so it draws exactly like `paused`.
  - **For 05b:** it adds the `ts` branch (search, gpu, u0 ≥ 0.2) and the `confirm` branch (u0 ∈ [0.2, 0.7) while a confirmation is in flight). Nothing above changes, so 05a's draws stay valid.
  - **Every 05a draw** has role `search` and `evalPanel = evalPanelOf(learnerId)`.

- [ ] **Step 1: Write the failing tests** (`draw.test.ts`)
  - `"birthDraw is a pure function: the same input gives a deep-equal Draw; 20 draws equal fixtures/draws.json"` (generate it with `DRAW_WRITE=1`).
  - `"no posterior → centre, drawSeed null, genome = centre"`.
  - `"parsePosterior: garbage, wrong schemaVersion, a need/genomes length mismatch → null"`.
  - `"calibrate on gpu: every draw is calibrate; over 2,000 learnerIds the genome frequencies follow need [3,0,…] (only index 0) and [1,1,…] (uniform, χ² p > 0.001)"`.
  - `"calibrate on cpu: only uniform or centre; uniform share 0.2 ± 0.03 over 2,000 ids"`.
  - `"flat/paused/search: uniform share 0.2 ± 0.03; uniform genomes pass checkGenome and keep fixed genes at centre"`.
  - `"evalPanelOf is in 1..16 and roughly uniform over 1,600 ids"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run packages/meta`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them; they pass.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(meta): seeded, recomputable birth draws (centre, calibrate, uniform) and the posterior file's shape`

### Task 5: The `meta` block in logs and checkpoints; shared panels for checkups

**Files:**
- Modify: `packages/harness/src/log.ts` (`LOG_KEYS_V2` adds `"meta"`; new types and `parseMeta`; `parseLogFile`; `parseCheckups` gets `allowAhead`)
- Modify: `packages/harness/src/checkpoint.ts` (`Checkpoint.meta`, `checkCheckpoint`)
- Modify: `packages/harness/src/checkup.ts` (`panelSeed`, `CheckupDeps.variantSeed`)
- Modify: `tools/gate/validate.ts` (`validateSessionPr` options), `tools/gate/gate.ts` and `tools/gate/validate-cli.ts` (load and pass the space)
- Modify: `docs/decisions/0022-…md` (§Shared panels), `docs/decisions/0016-milestones-and-checkups-v1.md` (header line `- **Amended by:** [0022](0022-shared-panels-wider-ranges-and-05a-calls.md) (new lives meet shared variant panels)`)
- Test: `packages/harness/test/log.test.ts`, `checkpoint.test.ts`, `checkup.test.ts`; `tools/gate/test/validate.test.ts`

**Interfaces:**
- Produces, in `log.ts`:
  ```ts
  export type MetaRole = "search" | "confirm" | "newborn" | "legacy"
  export type MetaMethod = "centre" | "calibrate" | "uniform" | "ts" | "confirm" | "newborn" | "legacy"
  export type MetaEnd = "failed" | "orphan" | "budget"
  export interface MetaBlock {
    v: 1; role: MetaRole; method: MetaMethod; genomeSchema: number; genome: Record<string, number | string>; genomeSha: string
    drawSeed: number | null; posteriorSha: string | null; evalPanel: number | null; evalSeed: number | null
    confirmation: { id: number; arm: string } | null; inheritance: { filters: "none" }
    filters?: string      // base64 of 104·widths.conv1 float32 LE: periphery conv1 w,b then fovea conv1 w,b (Model.params order)
    heldOut?: HeldOutScore // role "newborn" only: its one held-out score at birth
    end?: MetaEnd         // final files only
  }
  export const BUDGET_CU = 2048
  export const FILTER_LEVELS: readonly number[] = [512, 1024, 2048]
  export const NEWBORN_LEVELS: readonly number[] = [512, 1024, 2048]
  export function parseMeta(v: unknown, f: { cu: number; stepsTotal: number; final: boolean; widths: LogFileV2["widths"] }): MetaBlock | string
  export interface LogFileV2 { /* as before */ meta?: MetaBlock }
  ```
- In `checkup.ts`:
  ```ts
  export function panelSeed(panel: number, level: number, milestoneId: string): number // hashSeed(0, `panel/${panel}/${level}/${milestoneId}`)
  // CheckupDeps gains: variantSeed?: (level: number, milestoneId: string) => number  (default checkupSeed(learnerId, …))
  ```
- In `validate.ts`: `validateSessionPr(files, previous, opts?: { space?: GenomeSpace })`.
- **`parseMeta` (structure; browser-safe):**
  - Keys are strict.
  - `genome` has ≤ 40 keys matching `SETTING_KEY_RE`, with finite numbers or strings of ≤ 64 characters.
  - `genomeSha` matches `KNOWLEDGE_SHA_RE`; `drawSeed` is a uint32 or null; `evalPanel` is 1–16 or null; `evalSeed` is an integer ≥ 10,000 or null.
  - `filters` is present only when `FILTER_LEVELS.includes(levelAtOrBelow(cu))`, and its decoded length must equal `104 * widths.conv1 * 4` bytes.
  - `heldOut` is allowed only for `role: "newborn"`.
  - `end` is allowed only in a `final: true` file.
- **The `role: "newborn"` rules:**
  - The file has `cu: 0`, `stepsTotal: 0`, `final: true` and `seq: 0`.
  - It holds exactly 3 checkups, at `NEWBORN_LEVELS`.
  - `parseCheckups(…, { allowAhead: true })` allows a level above `cu` for this role only.
  - `method` is `"newborn"` and `evalPanel` is non-null.
- **`validateSessionPr` (needs main's `knowledge/genome-space.json`):**
  - **Bounds.** For roles other than `legacy`, `checkGenome(space, meta.genome) === null` when `meta.genomeSchema === space.schemaVersion`. Another schema is checked for structure only.
  - **Integrity.** `genomeSha` recomputes, across all roles.
  - **Constancy.** Every `meta` field except `filters`, `heldOut` and `end` is equal across all of a learner's files that have `meta`, previous and PR. Absence is allowed in any file (old code).
  - **Filters.** At most one file per `FILTER_LEVELS` level carries `filters`.
  - **Async.** `validateSessionPr` becomes `async` because of the sha. `decide` and `runGate` await it.
- **Panels:**
  - A checkup uses `variantSeed` when the session's meta block has `evalPanel` non-null. That is every 05a search and newborn life.
  - Lives without `meta`, and `legacy` lives, keep `checkupSeed`.
  - Thresholds, budgets and the suite are unchanged.
- `Checkpoint.meta?: MetaBlock` is checked with `parseMeta` against the checkpoint's own `cu`, `steps` and widths. `filters` and `end` are never stored there.

- [ ] **Step 1: Write the failing tests**
  - log:
    - `"meta: a full search block parses; each wrong key or value fails with a named error (one case per field)"`;
    - `"meta.filters only in a file whose levelAtOrBelow(cu) is 512, 1024 or 2048, of exactly 104·conv1 floats"`;
    - `"newborn file: cu 0, three checkups at 512/1024/2048 above cu parse; the same checkups without role newborn fail 'level is ahead'"`;
    - `"meta.end only in a final file; heldOut only for newborn"`;
    - `"every committed log under data/runs still parses"` (existing; it stays green).
  - checkpoint: `"a checkpoint with meta loads; one with a malformed meta is a RangeError; a pre-05a checkpoint (no meta) loads"`.
  - checkup: `"with variantSeed, milestone m meets makeVariant(panelSeed(p, level, m.id)); two learners on panel p meet identical variants; without it, checkupSeed as before"`.
  - validate:
    - `"a search genome out of bounds is rejected; the same values under role legacy are accepted"` (Review Focus 1);
    - `"a genomeSha that does not match its genome is rejected"`;
    - `"meta.method changing between a learner's files is rejected; a file without meta after one with it is accepted"`;
    - `"two files carrying filters for level 512 are rejected"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run packages/harness/test/log.test.ts packages/harness/test/checkpoint.test.ts packages/harness/test/checkup.test.ts tools/gate/test/validate.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `gate.ts` and `validate-cli.ts` read `knowledge/genome-space.json` from main's checkout with `parseGenomeSpace`. If it is missing or invalid, the run aborts before deciding anything; it never merges unchecked. Write **0022 §Shared panels**: they amend 0016's per-learner variant seeds, for new lives only; existing lives keep theirs. Lives on one panel meet the same variants (blocking [OLS O20]), and the newborn reference uses the same panels.
- [ ] **Step 4: Run them; they pass.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(harness): the meta log block (strict, optional), newborn files, shared checkup panels; the gate checks genome bounds, sha and constancy`

### Task 6: The session writes `meta`: birth, legacy, filters, end reasons and short lives

**Files:**
- Modify: `packages/harness/src/session.ts` (`SessionConfig.meta`, `SessionInbox`, `SessionEvent`, `SessionResult`, `checkpointNow`, `logFile`, `checkupAtSleep`, the main loop)
- Modify: `packages/harness/src/final.ts` (`finalLogOf` gains `end`), `packages/harness/src/orphans.ts` (passes `end: "orphan"`)
- Modify: `packages/learner/src/model.ts` (exports `FIRST_LAYER_PARAMS`)
- Test: `packages/harness/test/session.test.ts`, `orphans.test.ts`; `packages/learner/test/model.test.ts`

**Interfaces:**
- Consumes: `MetaBlock`, `BUDGET_CU`, `FILTER_LEVELS`, `panelSeed` and `variantSeed` (Task 5).
- Produces:
  ```ts
  // session.ts
  export interface SessionMeta {
    birth: MetaBlock                                                    // a newborn's draw (role search)
    legacy(flat: Record<string, number | string>): Promise<MetaBlock>  // a resumed pre-05a life's block (role/method "legacy", evalPanel null)
  }
  // SessionConfig gains  meta?: SessionMeta   (absent: no meta block anywhere, as today: existing tests unchanged)
  // SessionInbox gains   setShortLives(on: boolean): void; get shortLives(): boolean
  // SessionEvent gains   { type: "meta"; meta: MetaBlock }   (once per run, after resume or birth)
  // SessionResult gains  ended?: "budget"
  // final.ts: finalLogOf(o) gains  end?: "failed" | "orphan"   (session passes "failed"; orphans.ts "orphan")
  // model.ts
  export const FIRST_LAYER_PARAMS: readonly number[] // indices in Model.params of periphery conv1 w, b, fovea conv1 w, b
  ```
- **Which block.** `cfg.meta === undefined` means no meta. Otherwise:
  - `cp?.meta` if the checkpoint has one;
  - else, when resuming a checkpoint without meta, `await cfg.meta.legacy(flatSettings)`;
  - else `cfg.meta.birth`.

  The block is stored in every checkpoint and every log file, `filters` and `end` excepted.
- **Variant seeds.** `checkupAtSleep` passes `variantSeed: (l, m) => panelSeed(meta.evalPanel, l, m)` when `meta?.evalPanel` is non-null.
- **Filters.** A crossing file whose level is in `FILTER_LEVELS` awaits `learner.readParams()` before `logFile`. It then carries `meta.filters`: the `FIRST_LAYER_PARAMS` arrays concatenated, base64 of float32 LE. `SessionLearner` gains `readParams?`; without it there are no filters.
- **Short lives** (0019; the switch is read at clean sleeps only):
  - **When it ends.** After the sleep, at a clean sleep where `inbox.shortLives` is on, the tail holds no checkup still due, and `lastLevel ≥ BUDGET_CU`, the life ends. This covers the sleep at which the 2,048 checkup just finished, and any later sleep of a life already past it.
  - **The final file.** It is `logFile(true)` with `meta.end = "budget"`, carrying the tail, so the 2,048 checkup is in it. It goes into the queue with `{ remove: id }`, and the run returns `ended: "budget"`.
  - **Halts.** A halt that is waiting goes first. The end then happens on the resumed run.
  - **Without meta.** Without `cfg.meta` the switch does nothing, so test lives are unchanged.
  - A failed life's final file gets `end: "failed"`. A swept orphan's gets `end: "orphan"`, and only when it has a meta block, from its checkpoint or its latest file.

- [ ] **Step 1: Write the failing tests** (fake learners and worlds, as the existing session tests do)
  - `"a newborn with cfg.meta: every log file and checkpoint carries the birth block; the meta event fires once"`.
  - `"a resumed checkpoint without meta gets the legacy block from its flat settings (gamma 0.9 included), logs it, and checkups keep checkupSeed"` (Review Focus 1).
  - `"a resumed checkpoint with meta keeps it, even when cfg.meta.birth differs"`.
  - `"with evalPanel p, every checkup's variants are panelSeed(p, …)"`.
  - `"crossing files at 512, 1024 and 2048 carry filters of 104·conv1 floats; the 8–256 and 3072 files do not"`.
  - **Short lives** (Review Focus 2):
    - `"shortLives on: the life ends at the sleep where the 2048 checkup finished; the final file has end budget and that checkup; ended budget"`;
    - `"shortLives switched on after the 2048 checkup was logged: ends at the next clean sleep"`;
    - `"switched on then off before 2048: no end"`;
    - `"a halt waiting at that sleep wins; the resumed run ends at its next clean sleep"`;
    - `"off (the default): the life passes 4096 CU and never ends"`;
    - `"without cfg.meta, shortLives has no effect"`.
  - `"a failed life's final file has meta.end failed; a swept orphan's has meta.end orphan; without meta neither has end"`.
  - Every file sequence written by these tests passes `await validateSessionPr(files, [], { space })` (the existing `gateAccepts` helper, made async).
  - model: `"FIRST_LAYER_PARAMS name 75·c, c, 27·c, c values at widths conv1 = c (16 and 8)"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run packages/harness packages/learner/test/model.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** `logFile` stays synchronous and takes the filters string as an argument. Add the 0022 lines: the switch read at clean sleeps; a life already past budget ending when the switch goes on; orphan end only with meta.
- [ ] **Step 4: Run the full gates and `npm run test:gpu`.** Expected: PASS. The golden test is unchanged.
- [ ] **Step 5: Commit** `feat(harness): lives log their genome block; filter snapshots at 512/1024/2048; end reasons; short lives end at the budget checkup`

### Task 7: The evaluators package, registry and pinned config; `search-composite@1` and `newborn-baseline@1`

**Files:**
- Create: `packages/evaluators/{package.json,tsconfig.json,src/index.ts,src/types.ts,src/life.ts,src/search-composite.ts,src/newborn-baseline.ts,src/reference.ts}`
- Create: `packages/evaluators/test/search-composite.test.ts`, `newborn-baseline.test.ts`, `packages/evaluators/test/fixtures/` (synthetic lives as LogFile JSON)
- Create: `config/evaluation.json`, `tools/eval/config.ts`, `tools/eval/config.test.ts`
- Create: `docs/decisions/0020-evaluators-v1.md`

**Interfaces:**
- Produces (`@blanktab/evaluators`; spec §2.1 interface verbatim, plus):
  ```ts
  export interface LifeInput { learnerId: string; files: LogFile[] }     // sorted by seq
  export interface EvalOutput { evaluator: string; version: number; eligible: boolean; reason?: string; score: number | null; components: Record<string, number | null>; diagnostics: Record<string, unknown> }
  export interface NewbornReference {
    evaluator: "newborn-baseline@1"; suiteVersion: 1; genomeSchema: 1; widthsClass: "gpu"
    learnerIds: string[]; milestones: Record<"M1"|"M2"|"M3"|"M4"|"M5"|"M6", { median: number; passRate: number; sustainedPassRate: number }>
    heldOutMedian: number
  }
  export interface References { newborn?: NewbornReference }
  export interface Evaluator { id: string; version: number; kind: "life" | "cohort"; needs: string[]
    evaluateLife?(life: LifeInput, refs: References): EvalOutput
    evaluateCohort?(groups: Record<string, LifeInput[]>, refs: References): EvalOutput
    targets?(outs: EvalOutput[]): number[] }
  export const REGISTRY: Readonly<Record<string, Evaluator>>   // "search-composite@1", "newborn-baseline@1", "signal-check@1", "confirm-fisher@1"
  export function getEvaluator(idAtVersion: string): Evaluator // throws on an unknown id
  export function rankAverage(outs: EvalOutput[]): number[]   // a_i of spec §2.4 step 2 (mid-ranks), over eligible outs in order
  export function parseNewbornReference(v: unknown): NewbornReference | null
  export const GPU_WIDTHS_CLASS = { conv1: 16, conv2: 32, enc: 128, hidden: 256 } as const
  // tools/eval/config.ts
  export interface EvalConfig { schemaVersion: 1; search: string; confirm: string; signal: string; baseline: string; budgetCU: number; levels: number[]; references: { newborn: string }; decision: string }
  export function loadEvalConfig(root: string): EvalConfig
  export function loadReferences(root: string, c: EvalConfig): References // an absent reference file → {} (scores then ineligible)
  ```
- **`search-composite@1`** is exactly spec §2.4. Its parts:
  - **Eligibility**, in order, with these `reason` strings:
    - `"not a search life"`;
    - `"no meta block before the checkup"`;
    - `"not gpu widths"`;
    - `"no newborn reference"`;
    - `"no 2048 checkup"`;
    - `"fewer than 2 of 3 levels"`;
    - `"fewer than 3 held-out points in (1024, 2048]"`;
    - `"abandoned"`.
  - **A failed life** (its final file has `meta.end = "failed"` and cu < 2,048) is eligible, with C = H = −2.
  - **The formulas:** κ_m (0.2, 0.2, 0.06, 0.15, 0.3 for M2–M6), q clipped to [−2, 2], C, H, and H's fallback when the reference's held-out median is outside 1 ± 0.01.
  - **The output:** `score = C + H`, with `components` `{ C, H }`.
  - **The diagnostics:** the per-milestone and per-level q's, M1's q, and the levels used.
  - **`targets`** is spec §2.4 steps 1–3, with `normInv` from detmath.
- **`newborn-baseline@1`** (cohort over `{ newborn: LifeInput[] }`) is spec §2.3:
  - each milestone's median, with null read as 0;
  - its pass rate;
  - its sustained pass rate (≥ 2 of 3 levels);
  - the held-out median of `meta.heldOut.loss / baseline`;
  - the sorted learnerIds.

  Only GPU-width newborn lives of suite 1 and schema 1 count. Its `diagnostics.reference` is the `NewbornReference`.
- **`config/evaluation.json`** is spec §2.2 verbatim, with `"decision": "docs/decisions/0020-evaluators-v1.md"`.

- [ ] **Step 1: Write the failing tests**
  - search-composite:
    - `"a hand-built life with known measures gives the hand-computed C, H and score to 1e-12"`;
    - `"each ineligibility reason, one fixture each"`;
    - `"a failed life before 2048 scores C = H = −2; an orphan before 2048 is ineligible 'abandoned'"`;
    - `"only cu ≤ 2048 data is read: appending later files changes nothing"`;
    - `"targets: ties get mid-ranks; output equals normInv((rank−0.5)/n) of the re-ranked average"`;
    - `"the same input JSON bytes give byte-identical JSON output twice"`.
  - newborn-baseline: `"64 synthetic newborns → the medians and rates computed by hand; CPU-width and suite-2 newborns are ignored"`.
  - `tools/eval/config.test.ts`:
    - `"every pinned id@version is in REGISTRY"`;
    - `"budgetCU equals BUDGET_CU and levels equal FILTER_LEVELS"`;
    - `"references.newborn is under config/references/ and, if present, parses"`;
    - `"GPU_WIDTHS_CLASS equals the learner's GPU_WIDTHS"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run packages/evaluators tools/eval`. Expected: FAIL.
- [ ] **Step 3: Implement.** Write the package and run `npm install`. Write **0020**. It records the evaluators v1, the config file, "the gate never writes `config/`", the version-bump rule, and "rows never mix evaluator versions" (agent's calls, not yet confirmed).
- [ ] **Step 4: Run them; they pass.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(evaluators): swappable, versioned evaluators; search-composite@1 and newborn-baseline@1; config/evaluation.json pins them; decision 0020`

### Task 8: `signal-check@1`, `confirm-fisher@1` and their statistics

**Files:**
- Create: `packages/evaluators/src/stats.ts`, `src/signal-check.ts`, `src/confirm-fisher.ts`
- Test: `packages/evaluators/test/stats.test.ts`, `signal-check.test.ts`, `confirm-fisher.test.ts`

**Interfaces:**
- Consumes: `rankAverage`, `searchComposite`, `chiSquareSf`, `normCdf`, `lnGamma`.
- Produces (`stats.ts`):
  ```ts
  export function midRanks(xs: readonly number[]): number[]
  export function kruskalWallis(groups: readonly (readonly number[])[]): { H: number; df: number; p: number } // tie-corrected; p = chiSquareSf(H, k−1)
  export function icc1(groups: readonly (readonly number[])[]): number // (MSB−MSW)/(MSB+(n0−1)MSW), n0 = (N − Σn_g²/N)/(k−1)
  export function fisherOneSided(a: number, b: number, c: number, d: number): number // P(X ≥ a), hypergeometric via lnGamma
  export function holm(ps: readonly number[], alpha: number): boolean[]
  export function mannWhitneyOneSided(x: readonly number[], y: readonly number[]): number // P(x > y), normal approximation, tie-corrected, continuity 0.5 [agent]
  export const SIGNAL_LOOKS = [30, 45, 60] as const
  export const ALPHA_LOOK = 0.05 / 3
  export const ICC_MIN = 0.1
  ```
- **`signal-check@1`** takes `groups` keyed by calibration genomeSha, whose lives are `LifeInput`s.
  - **Its inputs.** It scores each life with `search-composite@1`, keeps the eligible ones, and computes `a = rankAverage(outs)` over all of them, grouped back by genome.
  - **Not enough data.** With fewer than 6 genomes that have ≥ 2 lives each, or fewer than 20 lives, it returns `eligible: false`.
  - **Signal.** `score` is 1 when `kw.p < ALPHA_LOOK` and `icc1(score groups) ≥ ICC_MIN`, else 0.
  - **Diagnostics:** H, df, p, the ICC, n, k and the learnerIds.
- **`confirm-fisher@1`** is spec §2.5, over the groups `candidate`, `centre` and `newborn`. It is built and tested here and wired in 05b.

- [ ] **Step 1: Write the failing tests**
  - stats:
    - `"kruskalWallis on the textbook example (Hollander & Wolfe 6.1: H ≈ 6.4, p ≈ 0.04) to 1e-3"`, checked against an independent hand calculation recorded in the test;
    - `"icc1 = 0 for identical groups' means; → 1 for zero within-group variance"`;
    - `"fisherOneSided(3,0,0,3) = 0.05"`;
    - `"holm rejects in order"`;
    - `"mannWhitneyOneSided(x≫y) < 0.01; identical samples ≈ 0.5"`.
  - signal-check:
    - `"10 genomes × 3 lives with a planted genome effect → score 1"`;
    - `"the same lives with genome labels shuffled by a seeded permutation → score 0 on ≥ 95 of 100 permutations"`;
    - `"5 genomes → ineligible"`.
  - confirm-fisher: `"a candidate that sustains M3 in 20/24 against 6/60 newborns and beats centre on a → 1; failed lives count as non-passes; abandoned excluded and counted; an abandonment ratio > 2× is flagged"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run packages/evaluators`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them; they pass.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(evaluators): signal-check@1 (Kruskal–Wallis + ICC, three Bonferroni looks) and confirm-fisher@1`

### Task 9: Summary scores, the rescore tool, and the newborn runs with their reference tool

**Files:**
- Create: `tools/eval/lives.ts`, `tools/eval/rescore.ts`, `tools/eval/newborn-reference.ts`, `tools/eval/test/eval.test.ts`
- Create: `packages/harness/src/newborn.ts`, `packages/harness/test/newborn.test.ts`
- Modify: `tools/gate/summary.ts` (`buildSummary` gains `evals`), `tools/gate/summary-cli.ts`, `packages/harness/src/log.ts` (`SessionRow.evals`, `parseSessionRow`)
- Test: `tools/gate/test/summary.test.ts`, `packages/harness/test/log.test.ts`

**Interfaces:**
- Produces:
  ```ts
  // tools/eval/lives.ts
  export function collectLives(runsDir: string): LifeInput[]           // every learner dir, files parsed and sorted by seq
  // tools/gate/summary.ts
  export function buildSummary(runsDir: string, scoring?: { evaluator: Evaluator; refs: References }): Summary
  // log.ts
  export interface SessionRow { /* as before */ evals?: Record<string, { eligible: boolean; score: number | null; C: number | null; H: number | null }> }
  // harness/newborn.ts
  export const NEWBORN_SEED_BASE = 1000
  export const NEWBORN_COUNT = 64
  export interface NewbornConfig { learnerId: string; k: number; codeSha: string; knowledgeSha: string; hostSig: LogFileV2["hostSig"]; widths: Widths; settings: LearnerSettings; genome: { schema: number; values: Record<string, number | string>; sha: string } }
  export function runNewborn(cfg: NewbornConfig, deps: { createLearner: SessionDeps["createLearner"]; checkupCanvas(): CanvasLike; idb: IDBFactory; now(): Date; yieldNow?: () => Promise<void> }): Promise<LogFileV2>
  ```
- **What `runNewborn` does** (spec §2.3; "how a newborn runs" was left to the builder):
  - **Its seed and panel.** The learner seed is `NEWBORN_SEED_BASE + k`, and `evalPanel = 1 + (k % 16)`.
  - **Its held-out score.** It builds an untrained learner with its held-out set, then calls `sleep()` once. On an empty buffer that makes no update; it only scores held-out. It records `{loss, baseline}` in `meta.heldOut`.
  - **Its checkups.** It runs `runCheckup` at each of `NEWBORN_LEVELS` with `variantSeed` set to `panelSeed(panel, level, m)`.
  - **Its one log file:** seq 0, `final: true`, cu 0, `role` and `method` both `"newborn"`, queued with `queue.put`.
  - Its checkups and held-out cost go to evalCU.
- **The summary.** `summary-cli.ts` loads `config/evaluation.json` and the references. It scores each learner with the pinned search evaluator and writes `evals["search-composite@1"]`. Without a reference, the rows say `eligible: false`.
- **`rescore.ts <id@version> [runsDir]`** writes `.local/rescore-<id@version>.tsv`, one row per learner: learnerId, role, method, genomeSha, eligible, reason, score and components. It also prints the pinned evaluator's score beside it for comparison.
- **`newborn-reference.ts [runsDir] [--check]`** computes `newborn-baseline@1` over `data/runs` and writes the reference JSON to the configured path. A person reviews it and commits it under a decision. `--check` exits 1 when the committed file differs from the recomputation. This is the **remeasure path**: run `?newborns=0-63` again under new code, rebuild with a new file name, and re-pin by decision.

- [ ] **Step 1: Write the failing tests**
  - newborn: `"runNewborn on the CPU backend at tiny widths writes one file that parseLogFile accepts and validateSessionPr accepts; its checkups' variants are the panel's; the learner's weights are unchanged (readParams before = after)"`.
  - summary:
    - `"rows carry evals['search-composite@1'] when scoring is given; absent otherwise (old summaries)"`;
    - `"parseSessionRow drops a malformed evals entry, not the row"`.
  - eval:
    - `"rescore writes one row per learner with the evaluator's id@version in the header"`;
    - `"newborn-reference --check passes on its own output and fails after one measure changes"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run tools/eval tools/gate/test/summary.test.ts packages/harness/test/newborn.test.ts packages/harness/test/log.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.** In `deploy.yml` the summary is already built by `summary-cli.ts`, which now reads `config/`. That is read-only.
- [ ] **Step 4: Run them; they pass.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(eval): summary rows carry the pinned score; rescore and newborn-reference tools; newborn reference lives`

### Task 10: The gate builds `data/meta` — calibration, the noise check and draw re-verification

**Files:**
- Create: `tools/gate/meta.ts`, `tools/gate/meta-cli.ts`, `tools/gate/test/meta.test.ts`, `tools/gate/test/fixtures/meta/` (runs, the previous state, expected bytes)

**Interfaces:**
- Consumes:
  - `collectLives` (Task 9) and `getEvaluator` (Task 7);
  - `birthDraw`, `parsePosterior`, `genomeSha`, `parseGenomeSpace`, `parseCentre` and `sha256Hex` (Tasks 3–4);
  - `sobolPoints` (Task 1).
- Produces:
  ```ts
  export interface CalibrationState {
    schemaVersion: 1; genomeSchema: number; evaluator: string; signal: string; seed: "calib/1"
    genomes: { genomeSha: string; values: Genome }[]                    // centre first, then 9 Sobol points decoded
    repeats: 3 | 6
    looks: { look: 1 | 2 | 3; lives: number; learnerIds: string[]; output: EvalOutput; sourceCommit: string }[] // append-only
    outcome: "pending" | "signal" | "flat"
  }
  export interface MetaInput {
    runsDir: string; config: EvalConfig; refs: References; space: GenomeSpace; centre: Genome; sourceCommit: string
    previous: { calibration: CalibrationState | null; posteriorText: string | null }
    posteriorsBySha: ReadonlyMap<string, string>   // every posterior.json in main's history, by sha256 of its bytes
  }
  export interface MetaFiles { "data/meta/posterior.json": string; "data/meta/calibration.json": string; indexLine: string | null }
  export function buildMeta(input: MetaInput): Promise<MetaFiles>
  export function posteriorHistory(git: (args: string[]) => string): Map<string, string> // git log --format=%H -- data/meta/posterior.json; git show <c>:data/meta/posterior.json
  ```
- **The calibration design** (spec §4.6). The 10 genomes are the centre plus `fromUnit(space, p, centre)` for `p` in `sobolPoints(searched(space).length, 9, "calib/1")`. They are computed once, when `previous.calibration` is null, and kept from then on.
- **Re-verification** (spec §7.4; 05a does it for `centre`, `calibrate` and `uniform`).
  - **What is recomputed.** For each learner whose first `meta` has role `search`, `birthDraw` is recomputed with `posteriorsBySha.get(posteriorSha)` and the learner's widths class.
  - **Ineligible rows.** These make a row ineligible and are flagged in `diagnostics.verify`:
    - an unknown `posteriorSha`;
    - a mismatched `method` or `evalPanel`;
    - a `calibrate` genome that is not one of the design genomes.
  - **A genome mismatch alone** keeps the row on its logged genome, with the mismatch counted.
- **Rows.** A row is an eligible life under the pinned search evaluator, of methods `calibrate`, `uniform` or `centre`, schema 1, class `gpu`. Calibration rows are those whose method is `calibrate`.
- **Looks.**
  - **When a look runs.** When calibration rows ≥ `SIGNAL_LOOKS[looks.length]` and `outcome` is `"pending"`, run the pinned signal evaluator over all current calibration rows. Append the look if it is eligible; otherwise wait.
  - **The outcomes.**
    - Signal: `outcome: "signal"`, and mode becomes `search`.
    - No signal on looks 1–2: `repeats: 6`, and mode stays `calibrate` (see the spec conflicts below).
    - No signal on look 3: `outcome: "flat"`, and mode becomes `flat`.
  - **Recorded looks are never recomputed.**
- **Need.** `need[i] = max(0, repeats − rows on genome i)`.
- **The posterior** is the §7.2 shape with:
  - `models: {}`, `incumbent: null`, `confirmation: null`;
  - `diagnostics.perGenome` with born, rows, failed and abandoned;
  - `diagnostics.verify`;
  - the `page` counts.

  It is serialised as `JSON.stringify(v, null, 1) + "\n"` with keys in a fixed order. It contains no time.
- **`indexLine`** is `{"posteriorSha","sourceCommit","builtAt"}` when the posterior bytes changed, else null. It is the only line with a time in it, and it goes to `posterior-index.jsonl`.
- **`meta-cli.ts`** reads from the working tree and `git`, writes the files, and prints `changed=true|false`.

- [ ] **Step 1: Write the failing tests** (`meta.test.ts`, fixture runs written with Task 6's session against the CPU backend, or hand-built LogFiles)
  - `"first build: mode calibrate, 10 genomes (centre first), need 3 each, no looks"`.
  - `"buildMeta twice on the same input → identical bytes; a different Map insertion order → identical bytes"` (Review Focus 4).
  - `"30 calibration rows with a planted effect → look 1 appended, outcome signal, mode search; a second build appends nothing"`.
  - `"no signal at 30 → repeats 6, mode calibrate; no signal at 45 and 60 → flat"`.
  - `"re-verification: a life whose logged method or panel was altered → ineligible and flagged; a calibrate genome not in the design → ineligible; a genome off by 1 ulp with a valid sha → kept, mismatch counted"`.
  - `"a newborn, legacy or cpu life is never a row"`.
  - `"unchanged rows → indexLine null and posterior bytes unchanged"`.
  - `"output has no trailing whitespace and ends with a newline"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run tools/gate/test/meta.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them; they pass.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(gate): compute data/meta: calibration design, the noise check's looks, draw re-verification, a byte-stable posterior`

### Task 11: The gate's direct commit of `data/meta/**`, and the deploy trigger

**Files:**
- Create: `tools/gate/commit-meta.ts`, `tools/gate/test/commit-meta.test.ts`
- Modify: `.github/workflows/gate.yml`, `.github/workflows/deploy.yml`

**Interfaces:**
- Produces:
  ```ts
  export const META_PATH_RE = /^data\/meta\/(posterior\.json|calibration\.json|posterior-index\.jsonl)$/
  export function metaPathErrors(staged: string[]): string[]      // every staged path must match; empty means none to commit
  export function commitMeta(deps: { git: (args: string[]) => string; rebuild(): Promise<boolean>; log(m: string): void }, maxTries = 3): Promise<"committed" | "unchanged">
  ```
- **`commitMeta`** works in four steps:
  1. **Stage.** `git add data/meta`, then check that `git diff --cached --name-only` lists only paths `metaPathErrors` accepts. Otherwise it throws, and nothing is committed.
  2. **Check the rest of the tree.** `git status --porcelain` must show no other change. `npm ci` may touch nothing, or the run aborts.
  3. **Commit.** It commits as `github-actions[bot]` with `meta: <mode>, <n> eligible lives, posterior <sha8>`.
  4. **Push.** `git push origin HEAD:main`. On a non-fast-forward it runs `git fetch origin main && git reset --hard origin/main`, then `rebuild()` (Task 10 again, on the new main), and tries again. It never uses `--force`.
- **`gate.yml`.** After the `gate` step:
  - `git pull --ff-only origin main`, so the new main's code, config and merged logs are used;
  - `npx tsx tools/gate/meta-cli.ts`;
  - `npx tsx tools/gate/commit-meta.ts`, with output `meta=committed|unchanged`.

  The deploy job's `if` also fires on `needs.gate.outputs.meta == 'committed'`. The permissions stay `contents: write` and `pull-requests: write`.
- **`deploy.yml`.** After `cp knowledge/*.json`: `if [ -f data/meta/posterior.json ]; then cp data/meta/posterior.json app/dist/knowledge/posterior.json; fi`. It is not in `PRIOR_FILES`, so `knowledgeSha` is unchanged.

- [ ] **Step 1: Probe first (AGENTS.md: verify against real artifacts).**
  - Run a `workflow_dispatch` of a temporary branch workflow that pushes an empty commit to a scratch branch `gate-probe` with `GITHUB_TOKEN` and `contents: write`. Then delete the branch.
  - Check `gh api repos/markwhiting/learner/branches/main/protection`. If main is protected against `GITHUB_TOKEN` pushes, **stop and report**: the user decides between a bypass rule and a bot PR.
  - Record the result in 0022.
- [ ] **Step 2: Write the failing tests** (a local bare repo, as `git-fixture.ts` does)
  - `"only data/meta files staged → committed and pushed"`.
  - `"a staged config/evaluation.json, knowledge/genome.json or data/runs file → throws, nothing committed"` (Review Focus 3).
  - `"another tracked file modified → throws"`.
  - `"a push rejected because main moved → fetch, rebuild, push; at most 3 tries; never --force (the git fake asserts the args)"`.
  - `"nothing changed → unchanged, no commit"`.
  - `tools/walls.test.ts` (or a workflow test): `"gate.yml's deploy condition includes meta == 'committed'; deploy.yml copies data/meta/posterior.json"`. This reads the YAML as text.
- [ ] **Step 3: Run them; they fail.** Run `npx vitest run tools/gate/test/commit-meta.test.ts`. Expected: FAIL.
- [ ] **Step 4: Implement.**
- [ ] **Step 5: Run the full gates.** Expected: PASS.
- [ ] **Step 6: Commit** `feat(gate): commit data/meta/** directly (only gate-computed files, never forced); a meta commit deploys`

### Task 12: The page — knowledge, the draw in the worker, the switch, the recipe and the state lines

**Before starting:** check whether the `layout-fix` branch has merged. If it has, rebase onto main. If not, ask the controller whether to wait. This task edits `app/index.html` and `tests/e2e/loop.spec.ts`, which that work also edits. New e2e tests go in a new file, `tests/e2e/meta.spec.ts`, to keep the overlap small.

**Files:**
- Create: `app/src/birth.ts`, `app/src/recipe.ts`, `app/src/newborns.ts`
- Create: `app/test/birth.test.ts`, `app/test/recipe.test.ts`, `tests/e2e/meta.spec.ts`
- Modify:
  - `packages/harness/src/knowledge.ts`: `Knowledge.meta`, `loadKnowledge`; delete `chooseSettings` and `settingsSpace`.
  - `app/src/main.ts`: `birth()`, `live()`, the switch, newborn mode.
  - `app/src/worker.ts`: the draw at the first run; `shortLives`; the newborn runner.
  - `app/src/protocol.ts`, `app/src/ui.ts`, `app/index.html`, `app/src/updates.ts` (types only).
- Delete: `app/src/settings.ts`, `app/test/settings.test.ts`. Their range checks move to `tools/genome-centre.test.ts` (Task 3).
- Modify: `tests/e2e/loop.spec.ts`. Its `KNOWLEDGE` stub serves `genome-space.json`, `genome.json` and a `posterior.json`. The `replayPerSleep = 16` and `maxStepsPerSec = 10` assertions are replaced by the stubbed genome's values.

**Interfaces:**
- Produces:
  ```ts
  // knowledge.ts (browser-safe, no meta import: raw values; the app parses)
  export interface Knowledge { /* offline, summary, manifest, caregiverPrior as before */ meta: { genomeSpace: unknown; genome: unknown; posteriorText: string | null } | null }
  // app/src/birth.ts
  export interface DrawContext { space: GenomeSpace; centre: Genome; centreSha: string; posterior: Posterior | null; posteriorText: string | null; posteriorSha: string | null }
  export function drawContextOf(k: Knowledge): Promise<DrawContext | null>       // null: no valid space or centre (Review Focus 5)
  export function birthFromDraw(ctx: DrawContext, learnerId: string, widthsClass: "gpu" | "cpu"): Promise<{ meta: MetaBlock; learnerSettings: LearnerSettings; caregiver: CaregiverSettings; maxStepsPerSec: number; sleepInterval: number }>
  export function sessionMetaOf(ctx: DrawContext, birth: MetaBlock): SessionMeta  // legacy() uses legacyGenome + genomeSha
  // app/src/recipe.ts (pure text)
  export function recipeText(meta: MetaBlock, posterior: Posterior | null): string
  export function genomeHighlights(genome: Genome, centre: Genome, space: GenomeSpace, n = 3): string[] // "learning rate ×2.6"
  export function metaStateText(posterior: Posterior | null): string
  export const GENE_LABELS: Readonly<Record<string, string>>                    // lr "learning rate", curiosityWeight "curiosity", …
  // app/src/newborns.ts
  export function readNewborns(search: string): { from: number; to: number } | null // ?newborns=a-b, 0 ≤ a ≤ b < NEWBORN_COUNT; production feature
  // protocol.ts
  //   StartMessage gains  draw?: { spaceJson: unknown; genomeJson: unknown; posteriorText: string | null }; shortLives: boolean; newborns?: { from: number; to: number }
  //   ToWorker gains      { type: "shortLives"; on: boolean }
  //   FromWorker gains    { type: "meta"; meta: MetaBlock }
  ```
- **The draw happens in the worker,** on the first `run`, once the backend is known. It is memoised, so a GPU-to-CPU restore keeps the birth genome. Its decoded values replace the config's learner settings, caregiver, `maxStepsPerSec` and `sleepInterval`. Test knobs still override afterwards, and those lives are dev builds and never rows.
- **`loadKnowledge`** fetches `genome-space.json`, `genome.json` and `posterior.json` (as text, so the sha is of the bytes) alongside the rest. Their failure gives `meta: null`, or `posteriorText: null`, and never makes the knowledge offline.
- **The switch.**
  - It is a checkbox on the LEARNER tab, labelled "Short lives".
  - Its help text: "End each life at its 2,048 CU checkup and start a newborn (about every 30 minutes). The meta-learner only gets results at a useful rate from tabs with this on."
  - It is stored in `localStorage["blanktab:shortLives"]` and defaults to off.
  - A change posts `shortLives` to the worker.
  - Switching it on while the life is past 2,048 CU first asks `confirm("This learner is past its 2,048 CU budget, so it will end at its next sleep and a newborn will start. Go on?")`.
- **The recipe and state lines** (spec §6):
  - `dt` "Recipe" holds `recipeText`, the highlights, and a toggle with the full genome, `genomeSha` and `posteriorSha` (8 characters each).
  - `dt` "Meta-learner" holds `metaStateText`.
  - "Score" appears once the summary row for this learner has `evals`: `"0.41 (search-composite@1)"`.
  - **`metaStateText` copy:**
    - `"Calibrating: 18 of 30 lives measured, to learn whether the score can tell recipes apart."`;
    - after a no-signal look: `"Paused: the score can't yet tell recipes apart; measuring more (34 of 45)."`;
    - flat: `"Flat: after 60 lives the score could not tell recipes apart. Recipes are the current one, with 1 in 5 random."`;
    - search in 05a: `"Signal found: the search can start. (Recipes are the current one or random until it does.)"`;
    - no posterior: `"Not reachable: this learner runs the current recipe."`.
  - **`recipeText` copy:**
    - calibrate: `"a calibration life (recipe 4 of 10)"`;
    - uniform: `"one of the 1-in-5 random explorers"`;
    - centre: `"the current recipe"`;
    - legacy: `"born before the meta-learner (its old settings)"`;
    - newborn: `"a newborn reference (panel 7)"`.
- **Newborn mode.** `?newborns=a-b` runs `runNewborn` for k = a…b in sequence, in place of a life. The page shows `"Newborn reference k of a–b"`. Each file goes to the outbox and is contributed as usual.

- [ ] **Step 1: Write the failing tests**
  - birth:
    - `"birthFromDraw at the committed knowledge with no posterior → method centre, DEFAULT_SETTINGS values, CAREGIVER_DEFAULTS"`;
    - `"birthFromDraw equals birthDraw + decodeGenome for 50 ids (the worker computes what the gate recomputes)"`;
    - `"drawContextOf: malformed space or genome → null; malformed posterior → posterior null, sha null"`.
  - recipe: each copy line above for its input; `genomeHighlights` ranks by the unit-cube distance from the centre and shows ratios v/centre (the horizon ratio for `gamma`).
  - `meta.spec.ts`:
    - `"a calibrate posterior: the page shows 'a calibration life' and 'Calibrating: …'; the outbox's first file has meta.method calibrate and the genome the page shows"`;
    - `"the draw in the page equals birthDraw in Node for the same learnerId and posterior bytes"` (Review Focus 4);
    - `"posterior.json 404 → 'the current recipe', posteriorSha null; genome-space.json garbage → no meta, DEFAULT_SETTINGS"` (Review Focus 5);
    - `"the short-lives switch: off by default; checking it stores blanktab:shortLives = 1, survives a reload, and posts shortLives to the worker"`. 2,048 CU cannot be reached in e2e time, even at tiny widths, so the ending itself is pinned by the session tests (Task 6) and the live check (Task 13).
    - `"?newborns=0-0 writes one newborn file that parses and validates"`.
- [ ] **Step 2: Run them; they fail.** Run `npx vitest run app/test && npm run e2e -- meta`. Expected: FAIL.
- [ ] **Step 3: Implement.** Then run `npm run dev`, open the page, and take screenshots of the LEARNER tab with the switch, recipe and meta-learner lines to `.local/05a-page.png`. Look at them.
- [ ] **Step 4: Run the full gates and `npm run e2e`.** Expected: PASS, `loop.spec.ts` included.
- [ ] **Step 5: Commit** `feat(app): seeded genome births in the worker, the short-lives switch, the recipe and meta-learner lines, newborn reference mode`

### Task 13: Merge, deploy, CI green, live check, write-up

- [ ] **Step 1: Merge and deploy.**
  - Merge to `main` and push.
  - Open the CI run and wait for green. Linux fonts differ from macOS, so a local e2e pass is not enough.
  - Wait for the deploy, then check the deployed `knowledge/manifest.json`: the new `codeSha`, the same `learnerStateVersion` (2), and a new `knowledgeSha` (the genome files joined it).
- [ ] **Step 2: First gate run.**
  - Run `gate.yml` by `workflow_dispatch`.
  - Check that its log shows `meta=committed`.
  - Check that the commit on main touches only `data/meta/**`: `git show --stat <sha>`.
  - Check that the deploy ran and `https://markwhiting.github.io/learner/knowledge/posterior.json` is served with `mode: "calibrate"` and 10 genomes.
- [ ] **Step 3: A forever life still continues.** If a tab holds a pre-05a learner, reload it onto the new code and confirm:
  - it resumes (same learnerId, step count continues);
  - the page says "born before the meta-learner";
  - its next queued file has `meta.role "legacy"`.
- [ ] **Step 4: The short-lives live check.** Wrap the run in `caffeinate -dims` and use Playwright Chromium (`--use-angle=metal`, WebGPU, no token) on the live page.
  - Turn the switch on. Record the drawn recipe line and genome.
  - Run until the 2,048 CU checkup, about 15–30 min. Confirm:
    - the life ends there, and the final file has `meta.end "budget"` and the 2,048 checkup;
    - a newborn starts in the same tab with a new learnerId and a different `drawSeed`;
    - its genome equals `birthDraw` recomputed in Node from its learnerId and the served posterior bytes.
  - Read every queued file from IndexedDB. Every one passes `parseLogFile`, and `await validateSessionPr(files, [], { space })` returns `ok: true`.
  - Take a screenshot of the LEARNER tab showing the genome and the meta-learner line. Look at it.
- [ ] **Step 5: The gate folds it in.** With a token, let the tab's PR reach the gate, or run the gate by hand. Confirm:
  - the PR merged;
  - the next `data/meta/posterior.json` shows `diagnostics.perGenome` counting the life;
  - the meta commit triggered a deploy.
- [ ] **Step 6: Write `docs/research/2026-10-2x-meta-foundations-live.md`.** Keep it short:
  - the checks above, with the screenshots;
  - steps/s before and after (same protocol as 04c, 10 min);
  - the first posterior's sha;
  - caveats: one machine; calibration not yet answered; the newborn reference not yet pinned.

  Set row 05a in `docs/plans/README.md`, replacing row 05 with 05a, 05b and 05c. Update Q14 in `docs/open-questions.md` to cite 0020–0022.
- [ ] **Step 7: Commit, push, and check CI green.** Commit message: `docs: Plan 05a live check — meta-learner foundations`.

**After this plan (spec §10: 05a is "done when the signal check has answered").**
1. Run `?newborns=0-63`, about 4.3 h on one machine.
2. Merge the newborn PRs.
3. Build the reference with `tools/eval/newborn-reference.ts`. The user, or the controller with the user's go-ahead, commits it: it pins an evaluation input.
4. Run short lives until look 1.
5. Write the answer up in a research note, favourable or not.

---

## Spec conflicts and gaps found while planning (resolutions adopted above; each goes into 0022 as an agent's call)
1. **Where the deterministic maths lives.** The spec puts `detmath.ts` in `packages/meta`, and forbids evaluators to import meta, yet requires them to use detmath. Resolution: a leaf package `packages/detmath`, which both import.
2. **Paused or still calibrating.** §4.6 says "no signal → paused; calibration continues to 6 repeats". But §4.1 says `paused` births are 20 % uniform and 80 % centre, which would stop calibration. Resolution: after a no-signal look 1 or 2, `mode` stays `calibrate` with `repeats: 6`, and only the page calls it "Paused". The `paused` mode is left for 05b.
3. **`search` mode before 05b.** It has no GP, so it draws like `paused` (uniform or centre). Those rows stay valid for 05b's fit.
4. **A legacy genome out of bounds.** Today's picker spans (for example, `entropyWeight` 0.001–0.05) exceed the new ranges. The §7.3 bounds rule would make the gate close forever lives' PRs. Resolution: bounds apply to every role but `legacy`.
5. **The newborn held-out score has no field.** A curve point needs finite own-stream losses, which a 0-step learner lacks. Resolution: `meta.heldOut`, for `role: "newborn"` only.
6. **The index's `commit` cannot be known.** A commit cannot contain its own sha. Resolution: the index line is `{posteriorSha, sourceCommit, builtAt}`, and the gate finds the bytes by scanning `data/meta/posterior.json`'s history (`posteriorHistory`).
7. **Re-verification moved earlier.** §10 places it in 05b. It moves to 05a for `centre`, `calibrate` and `uniform`, so the noise check cannot be fed mislabelled calibration rows. 05b adds `ts` and `confirm`.
8. **The posterior source commit in the log.** §4.1 step 5 lists it, but §7.3 does not. Resolution: not logged; `posteriorSha` and the index already identify it.
9. **The switch and a life already past 2,048 CU.** 0019 does not say what happens then. Resolution: it ends at its next clean sleep, after a confirmation prompt. Without the switch no forever life ends (Goal 6).
10. **How newborns are run.** This was left to the builder. Resolution: a production query mode, `?newborns=a-b`, with learner seed 1000 + k, panel 1 + (k mod 16), and a `sleep()` on an empty buffer for the held-out score. Newborn logs arrive as ordinary PRs (64 one-file PRs).
11. **Other genome schemas.** The validator can bound-check only the current `genome-space.json`. Files of another `genomeSchema` are checked for structure only. The evaluator excludes them anyway.

## Self-review
- **Spec coverage, §10 05a:**
  - evaluators and config → Tasks 7–8;
  - genome files, `packages/meta`, decode and `gamma` → Tasks 2–3;
  - the seeded `centre`/`uniform`/`calibrate` births and panels → Tasks 4–5;
  - the newborn mode → Tasks 9 and 12;
  - the `meta` block and filters → Tasks 5–6;
  - validation → Task 5;
  - summary `evals` and `rescore.ts` → Task 9;
  - the newborn reference → Task 9 (tool) and the follow-up (pinning);
  - the gate's noise check → Task 10;
  - the page lines → Task 12;
  - the short-lives switch and `end` reasons → Tasks 6 and 12;
  - the gate's direct commits and the deploy trigger → Task 11;
  - decisions 0020, 0021 and 0022, plus 0016's amended-by line → Tasks 3, 5 and 7;
  - the live check → Task 13.
- **Interfaces 05b consumes:**
  - `Posterior.models`, `incumbent` and `confirmation`;
  - the `birthDraw` rng order (u0 first) and its reserved `ts`/`confirm` branches;
  - `Method`/`Role`, which already include `ts` and `confirm`;
  - `MetaBlock.evalSeed` and `confirmation`;
  - `confirm-fisher@1`;
  - `META_PATH_RE`, to widen with `knowledge/genome.json`;
  - `posteriorHistory`;
  - `buildMeta`'s previous-state input.
- **Names across tasks:**
  - `hashSeed`/`createRng` (detmath) and `sobolPoints`;
  - `GenomeSpace`, `Genome`, `decodeGenome`, `legacyGenome`, `genomeSha` and `checkGenome`;
  - `Posterior`, `birthDraw`, `evalPanelOf` and `UNIFORM_SHARE`;
  - `MetaBlock`, `BUDGET_CU`, `FILTER_LEVELS`, `NEWBORN_LEVELS` and `panelSeed`;
  - `SessionMeta` and `SessionInbox.setShortLives`;
  - `REGISTRY`, `getEvaluator`, `rankAverage`, `NewbornReference`, `EvalConfig` and `loadEvalConfig`;
  - `collectLives`, `runNewborn`, `buildMeta`, `MetaFiles`, `commitMeta` and `META_PATH_RE`;
  - `DrawContext`, `birthFromDraw` (app; named apart from harness `final.ts` `birthOf`), `recipeText` and `metaStateText`.
- **Order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13. Tasks 7–8 depend only on Tasks 1 and 5, so they can run beside Task 6.
- **Proportion:** this plan holds signatures, values and test names. Bodies are given only for algorithm choices the tests do not fix (`dexp`/`dlog` reduction, AS241, Lentz, Sobol digital shift, the ICC formula).
