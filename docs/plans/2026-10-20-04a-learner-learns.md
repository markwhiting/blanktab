# Plan 04a — The learner learns

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## In plain terms

Today the tab's "baby" flails at random. This plan replaces it with a real learner, built from scratch on our own math engine (Plan 02) and living in the four-room world (Plan 03).

**What the learner does, every moment:**
- **Sees** the screen: the low-resolution full view, the sharp patch around its cursor, and where its cursor is.
- **Guesses what it will see next,** given what it is about to do. This is how it learns about its world.
- **Is curious:** it gets an internal reward for *learning*. That reward is how much its guess improved after it updated itself, not how wrong it was. This way, random noise doesn't stay interesting forever (spec §5.4).
- **Picks its next move** to seek out that kind of learning.

**Every so often it sleeps:**
- It stops acting and replays a sample of what it experienced.
- It saves itself to the browser, so a reload or a closed tab doesn't kill it.
- It writes a log of how it's doing.

**How we know it's learning:** the log, the page and the GitHub PR all report how well it predicts the next frame, compared with the lazy guess "the next frame looks like this one". When its guesses are clearly better than that baseline, it has passed the first milestone (M1) in spirit. Plan 04b makes milestones and checkups formal.

**What this plan doesn't do:**
- **No caregiver.** It doesn't move the learner between rooms; a learner mostly stays in the crib unless it uses the taskbar itself.
- **No milestones or checkups.**
- **Only the simplest consolidation condition (A):** one set of weights, plain replay.

Those come in Plan 04b (caregiver, milestones, checkups) and Plan 05 (conditions B and C, and the population search).

**Goal:** the live page runs a cold-start learner that trains continuously on WebGPU, falling back to the CPU. It survives reloads, and logs a prediction-loss curve against a copy-last-frame baseline, measured in compute units. The live run shows whether it learns.

**Architecture:**
- **`packages/learner`** gains:
  - `encode.ts`: pixels to tensors, actions to one-hot.
  - `model.ts`: encoders, GRU core, an action-conditioned predictor, policy and value heads.
  - `learner.ts`: online learning, curiosity reward, actor-critic, FIFO replay buffer, sleep, save/load.
- **`packages/harness`** session v2:
  - lifetime measured in compute units;
  - sleep cycles;
  - checkpoints in IndexedDB;
  - log schema v2 carrying the loss curve;
  - a polite speed cap.
- **The app:** the worker picks a backend, and the page shows the loss curve.

**Tech Stack:** TypeScript, `@blanktab/tensor` (CPU + WebGPU), Vitest, Playwright.

**Spec:** `docs/specs/2026-10-19-blanktab-v1-design.md` §5.1–5.5, §6, §10, and condition A of §5.6. Lifecycle rules: `packages/tensor/README.md`. These rules are mandatory.

## Global Constraints

- **Cold start (AGENTS.md invariant 1).** Every weight is drawn from a seeded initializer; log the seed. No pretrained anything.
- **Learner inputs (invariant 2).** The learner sees only `Observation` (periphery, fovea, cursor, buttonHeld). It never reads `world.state()`, the trace, or room names.
- **Walls.** `learner` may import only `@blanktab/interface` and `@blanktab/tensor`.
- **Tensor lifecycle (packages/tensor/README.md).**
  - Training step: `backward` → `opt.step` → `opt.zeroGrad` → `disposeGraph`.
  - Acting runs inside `noGrad(() => composite(...))`, or every result is disposed.
  - Recurrent state is carried with `detach()`.
  - Any `read()` rejection is fatal: stop and restore from the last checkpoint.
  - `backend.stats.liveBuffers` must not grow in steady state.
- **Compute unit:** 1 CU = 1e9 FLOPs, counted as `backend.stats.flops` deltas on the learner's own backend instance.
- **Default widths** (spec §5.5, confirmed by `docs/research/2026-10-20-webgpu-step-throughput.md`):
  - WebGPU: `{ conv1: 16, conv2: 32, enc: 128, hidden: 256 }`.
  - CPU fallback: half width, `{ conv1: 8, conv2: 16, enc: 64, hidden: 128 }`.
- **Log volume (decision 0011).** At most 16 files per PR. Log files are written only when CU crosses `2^k` (k = 0, 1, 2, …) and at the end of life. Commits stay at one per flush, with flushes on final and every 10 minutes.
- **Speed cap:** the learner takes at most `maxStepsPerSec` steps per second (default 30), to be polite to the visitor's machine.
- **Determinism (CPU):** the same seed and the same observation stream give identical actions and losses, including across save/load.
- **Dependencies:** after any `npm install`, run `npm run lint && npm run typecheck && npm test`. Stop if npm reports "removed N" or "changed N" packages.
- Commit after every task. Don't push until Task 8.

## Review Focus

1. **Memory over a long life:** after 2,000 CPU steps with sleeps, `liveBuffers` equals its value after the first sleep. Test in Task 5.
2. **Reload mid-life:** reload resumes from the last checkpoint. Step count, CU and the loss curve continue, nothing is double-counted, and no log file `seq` is reused. Test in Task 6.
3. **WebGPU missing or lost:** the learner runs on the CPU at half width, which is recorded in `hostSig.webgpu`. If a GPU read is rejected mid-life, the tab restores from its checkpoint and does not freeze. Test in Task 7.
4. **A bad observation:** an all-black frame, or the cursor at a corner. No NaN reaches the weights; the losses stay finite. Test in Task 4.
5. **Tiny or huge settings** at the edges of `settings-space.json`. The learner runs without NaN, and the learning rate is clamped. Test in Task 4.

---

## File structure

```
packages/interface/src/observation.ts   Learner.step becomes async
packages/learner/src/
  encode.ts       obsToInputs, targetsFromObs, actionOneHot, ACTION_DIM
  model.ts        createModel(backend, rng, widths) → Model (params, forward pieces)
  learner.ts      createLearner(opts) → TrainableLearner (step, sleep, save, load, stats, dispose)
  replay.ts       FIFO transition buffer (uint8 frames, actions, detached h)
  flailer.ts      async step
  index.ts
packages/learner/test/{encode,model,learner,replay}.test.ts
packages/harness/src/
  session.ts      runSession v2 (CU lifetime, sleeps, CU-level logs, checkpoints, speed cap)
  checkpoint.ts   IndexedDB store "checkpoints" (latest per learnerId)
  log.ts          LogFile v2 (+ v1 still accepted)
tools/gate/      validator/summary accept v2; summary rows carry lossRatio
knowledge/settings-space.json   learner settings ranges
app/src/{worker,main,ui}.ts     backend selection, checkpoint resume, loss curve
```

---

### Task 1: The learner interface becomes asynchronous

**Files:**
- Modify: `packages/interface/src/observation.ts` (`Learner`), `packages/learner/src/flailer.ts`, `packages/harness/src/session.ts`, and the tests that call `step` synchronously.

**Interfaces:**
- Produces: `export interface Learner { step(o: Observation): Promise<Action> }`. The flailer returns `Promise.resolve(action)`. `runSession` awaits `learner.step`.

The reason: GPU learners must `await read()` the policy logits to sample an action.

- [ ] **Step 1: Write the failing tests.**
  - The flailer's `step` returns a Promise that resolves to a valid action.
  - `runSession` with a learner whose `step` resolves after a `setTimeout(0)` still writes the same files as today (6 files, the 3000/500 case).
- [ ] **Step 2: Run them to see them fail.** Run: `npx vitest run packages/learner packages/harness`. Expected: FAIL (a type or behaviour mismatch).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gate.** Run: `npm run lint && npm run typecheck && npm test && npx playwright test`. Expected: PASS.
- [ ] **Step 5: Commit.** `refactor(interface): Learner.step is async`

---

### Task 2: Encoding observations and actions

**Files:** create `packages/learner/src/encode.ts` and `test/encode.test.ts`.

**Interfaces:**

```ts
export const ACTION_DIM = 25 + 3 + 3 + 45   // 76, one-hot per factor, concatenated
export interface Inputs { periphery: Float32Array /* [3,50,80] CHW, 0..1 */; fovea: Float32Array /* [3,32,32] */; proprio: Float32Array /* [cursor.x, cursor.y, buttonHeld?1:0] */ }
export interface Targets { periphery: Float32Array /* [40*25*3] HWC 0..1, 2×2 box of the 80×50 periphery */; fovea: Float32Array /* [16*16*3], 2×2 box of the 32×32 fovea */ }
export function obsToInputs(o: Observation): Inputs
export function targetsFromObs(o: Observation): Targets
export function actionOneHot(a: Action): Float32Array          // length ACTION_DIM, exactly 4 ones
export function copyLastBaseline(prev: Targets, next: Targets): number   // MSE between consecutive targets (the lazy "nothing changes" guess)
```

- [ ] **Step 1: Write the failing tests.**
  - Shapes and value ranges.
  - Channel order: an all-red pixel becomes `[1,0,0]` in CHW.
  - The 2×2 box average is exact on a known pattern.
  - `actionOneHot(NOOP_ACTION)` has ones at 0, 25, 28 and 31.
  - `copyLastBaseline` of identical targets is 0.
- [ ] **Step 2: Run them to see them fail.** Run: `npx vitest run packages/learner/test/encode.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to see them pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `feat(learner): encode observations and actions`

---

### Task 3: The model

**Files:** create `packages/learner/src/model.ts` and `test/model.test.ts`.

**Architecture.** This follows spec §5.5 and mirrors `packages/tensor/bench/learner-model.ts`, plus action conditioning.

- **Periphery encoder:** conv 3→c1 (k5 s2 p2), relu, conv c1→c2 (k3 s2 p1), relu, linear → enc.
- **Fovea encoder:** conv 3→c1 (k3 s2 p1), relu, conv c1→c2 (k3 s2 p1), relu, linear → enc.
- **Core input:** periphery code + fovea code + linear(proprio 3 → enc).
- **Core:** a GRU, enc → hidden.
- **Policy and value:** from `h`. Policy logits are 76 numbers (25|3|3|45). Value is 1.
- **Predictor:** takes `h` and the chosen action. `z = relu(linear(h) + linear(actionOneHot))` (hidden → enc, 76 → enc), then two outputs:
  - periphery: linear enc → 3000, then sigmoid;
  - fovea: linear enc → 768, then sigmoid.

**Interfaces:**

```ts
export interface Widths { conv1: number; conv2: number; enc: number; hidden: number }
export const GPU_WIDTHS: Widths, CPU_WIDTHS: Widths
export interface Model {
  readonly params: readonly Tensor[]; readonly paramCount: number; readonly widths: Widths
  core(inp: { periphery: Tensor; fovea: Tensor; proprio: Tensor }, h: Tensor): Tensor   // → h' [1,hidden]
  heads(h: Tensor): { logits: Tensor /* [1,76] */; value: Tensor /* [1,1] */ }
  predict(h: Tensor, action: Tensor /* [1,76] */): { periphery: Tensor /* [1,3000] */; fovea: Tensor /* [1,768] */ }
  zeroState(): Tensor                                                                      // [1,hidden] zeros, caller-owned
}
export function createModel(backend: Backend, seed: number, widths: Widths): Model
```

- **Initialisation:** uniform ±1/√fanIn from `createRng(seed)`, with parameters created in a fixed order.

- [ ] **Step 1: Write the failing tests** (CPU backend):
  - every output has the right shape at both widths;
  - `paramCount` at `GPU_WIDTHS` is within 10 % of the benchmark model's 2.62 M;
  - the same seed gives identical parameters, and a different seed differs;
  - `gradCheck` passes on `predict` with tiny widths (`{2,2,4,4}`), with inputs at least 0.2 away from relu kinks;
  - one forward/backward pass leaves `liveBuffers` unchanged after `disposeGraph`.
- [ ] **Step 2: Run them to see them fail.** Run: `npx vitest run packages/learner/test/model.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to see them pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `feat(learner): model with action-conditioned predictor`

---

### Task 4: Online learning, curiosity and acting

**Files:** create `packages/learner/src/{learner,replay}.ts` and `test/{learner,replay}.test.ts`.

**Behaviour.** This is `step(obs_t)`, condition A, with a single set of weights.
1. **Encode `obs_t`.** If a previous step exists, the learner has `h_{t-1}` (detached), `a_{t-1}` and the prediction target `y_t = targetsFromObs(obs_t)`.
2. **Learn from the last transition (if any):**
   - `L_before = mse(predict(h_{t-1}, a_{t-1}), y_t)`, summed over both outputs and weighted by size.
   - Run `backward(L_before + actorCritic)`, then `opt.step()`. The actor-critic part uses the stored log-probability and value from step t−1, with TD target `r_{t-1} + γ·V(h_t)`, where `V(h_t)` is computed without gradient.
   - **Curiosity:** `r_{t-1} = clamp(L_before − L_after, −1, 1)`. `L_after` is `predict(h_{t-1}, a_{t-1})` recomputed with the updated weights, without gradient.
   - **Total reward:** `curiosityWeight · r`.
   - **Loss:** `actorLoss = −logπ(a_{t-1}) · stopgrad(A)`, `criticLoss = A²`, minus `entropyWeight · H(π)`.
   - **Discount:** γ = 0.9.
3. **Act:**
   - Compute `h_t = core(inputs_t, h_{t-1})` and the heads.
   - Read the logits back once per step (76 floats).
   - Sample each factor from its softmax using the learner's seeded RNG, giving `a_t`.
   - Do **not** keep a graph alive across steps. Next step, recompute `h_{t-1} = core(inputs_{t-1}, h_{t-2})` from the stored detached `h_{t-2}` and stored `inputs_{t-1}`, then derive `logπ(a_{t-1})` and `V(h_{t-1})` from it with gradient, for the actor-critic update. This is lifecycle-safe (README rule 4) and gives truncated backpropagation through time of length 1. **[controller ruling]**
4. **Store** the transition `(inputs_{t-1}, h_{t-2} detached, a_{t-1}, y_t)` in the FIFO replay buffer.
   - The buffer keeps raw `Uint8Array` frames, so it can be reconstructed. Its capacity is `bufferSize` transitions.
5. **Carry state:** `h ← detach(h_t)`, and free everything else (README rules 1–4).
6. **Statistics (exposed as `stats()`):**
   - `predLoss`, an exponential moving average of `L_before` with α = 0.01;
   - `baselineLoss`, an exponential moving average of `copyLastBaseline(y_{t-1}, y_t)`;
   - `cu`, from `backend.stats.flops` deltas divided by 1e9;
   - `steps`.
7. **Guards:**
   - Clamp `lr` to [1e-5, 1e-2].
   - If any loss read is non-finite, skip that update. Count it in `stats().skippedUpdates`, do not apply the gradients, and do not throw.

**Interfaces:**

```ts
export interface LearnerSettings { lr: number; entropyWeight: number; curiosityWeight: number; bufferSize: number; replayPerSleep: number }
export const DEFAULT_SETTINGS: LearnerSettings   // { lr: 1e-3, entropyWeight: 0.01, curiosityWeight: 1, bufferSize: 2000, replayPerSleep: 64 }
export interface LearnerStats { steps: number; cu: number; predLoss: number; baselineLoss: number; skippedUpdates: number; sleeps: number }
export interface TrainableLearner extends Learner {
  stats(): LearnerStats
  sleep(): Promise<void>                 // Task 5
  save(): Promise<LearnerState>          // Task 5
  dispose(): void
}
export function createLearner(opts: { backend: Backend; seed: number; widths: Widths; settings: LearnerSettings; state?: LearnerState }): Promise<TrainableLearner>
```

- [ ] **Step 1: Write the failing tests** (CPU backend, tiny widths, a synthetic observation stream — the world package must not be imported, because of the wall):
  - **Learning:** on a stream where the next frame is a fixed function of the action (the cursor moves by the action's dx/dy on a plain background), `predLoss` after 1,500 steps is less than 0.5 × `predLoss` after 100 steps;
  - **Determinism:** two learners with the same seed on the same stream produce identical actions and losses;
  - every returned action is valid (`isValidAction`);
  - `liveBuffers` stays flat between step 100 and step 600;
  - **Review Focus 4:** an all-black frame stream and a cursor in the corner keep losses finite;
  - **Review Focus 5:** `lr` = 1 is clamped and stays finite, and the extremes of `entropyWeight` and `curiosityWeight` in `settings-space.json` run without NaN;
  - the curiosity reward is finite and clamped to [−1, 1];
  - the FIFO buffer evicts the oldest entry first and never exceeds `bufferSize`.
- [ ] **Step 2: Run them to see them fail.** Run: `npx vitest run packages/learner`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to see them pass.** Same command. Expected: PASS. The learning test must pass on several seeds; report how many.
- [ ] **Step 5: Commit.** `feat(learner): online prediction learning with curiosity and actor-critic`

---

### Task 5: Sleep, replay, save and load

**Files:** modify `learner.ts` and `replay.ts`, and add tests.

**Behaviour:**
- **`sleep()`:**
  - Runs `replayPerSleep` minibatch updates.
  - Each update is one uniformly sampled transition, using the learner's RNG.
  - It trains `core` from the stored `h_{t-2}` and `inputs_{t-1}`, then `predict` with `a_{t-1}` against `y_t`, using MSE and Adam.
  - This is condition A: plain uniform replay.
  - It increments `sleeps`.
- **`save()`:** returns a `LearnerState`:
  - parameters, Adam `m` and `v` and step counts;
  - `h`, the replay buffer contents and RNG state;
  - the statistics;
  - widths, settings and seed.

  Everything is plain typed arrays plus JSON, structured-clone friendly for IndexedDB.
- **`createLearner({ state })`:** restores all of it exactly. Optimizer and RNG state live in the learner and are exposed on `Adam` and the RNG. Add getters and setters there if missing: `Adam.state()` / `Adam.setState()` in `@blanktab/tensor`, and an RNG with `getState()` / `setState()`.

- [ ] **Step 1: Write the failing tests** (CPU, tiny widths):
  - **Save/load:** run 300 steps with sleeps every 100 → save → load into a fresh learner → 200 more steps. This must give identical actions and losses to an uninterrupted 500-step run;
  - `sleep()` lowers the average prediction loss on replayed transitions (deterministic seed);
  - **Review Focus 1:** after 2,000 steps with a sleep every 250, `liveBuffers` equals its value after the first sleep;
  - `LearnerState` survives `structuredClone`.
- [ ] **Step 2: Run them to see them fail.** Run: `npx vitest run packages/learner`. Expected: FAIL.
- [ ] **Step 3: Implement,** including the Adam and RNG state accessors in tensor, with their own tests.
- [ ] **Step 4: Run the gate.** Run: `npx vitest run packages/learner packages/tensor && npm run lint && npm run typecheck`. Expected: PASS.
- [ ] **Step 5: Commit.** `feat(learner): sleep with uniform replay; exact save/load`

---

### Task 6: The harness lives a real life

**Files:** modify `packages/harness/src/{session,log}.ts`, create `checkpoint.ts`, modify `tools/gate/{validate,summary}.ts`, and add tests.

**LogFile v2.** Written with `schemaVersion: 2`; v1 files are still accepted by the parser and validator.

It keeps every v1 field plus:

```ts
cu: number                       // finite, ≥ 0, never decreases across seq
condition: "A"
widths: { conv1: number; conv2: number; enc: number; hidden: number }
curve: Array<{ cu: number; steps: number; predLoss: number; baselineLoss: number }>   // ≤ 64 entries, since the previous log file
skippedUpdates: number
```

`settings` now holds the learner settings (lr, entropyWeight, curiosityWeight, bufferSize, replayPerSleep, maxStepsPerSec) plus the seed. They must still match the existing key, value and count rules.

**Session v2.** `runSession(cfg, deps)`:
- **Lifetime:** `cfg.lifetimeCU`, default 1024 = 2^10.
- **Sleep:** every `cfg.sleepInterval` steps (default 250), call `learner.sleep()`, append a `curve` point, then write a checkpoint.
- **Logs:** written when CU crosses the next `2^k` (k = 0, 1, 2, …) and at the end of life. At most 11 + 1 files for 1024 CU, which is under the gate's 16.
- **Speed cap:** `cfg.maxStepsPerSec`, enforced with the existing `pace` hook.
- **Checkpoints** (`checkpoint.ts`):
  - IndexedDB object store `checkpoints`, keyed by learnerId, holding the latest checkpoint only.
  - The record is `{ learnerId, savedAt, steps, cu, seq, curveTail, learnerState }`.
  - `saveCheckpoint`, `loadCheckpoint` and `deleteCheckpoint` are exported.
  - A final learner's checkpoint is deleted after its final log is queued.
- **Resume:** when a checkpoint exists for the tab's learnerId, the session starts from its steps, CU and seq, and the learner is created from `learnerState`.

**Gate.**
- The validator accepts v2. It also checks that `cu` never decreases across seq, that `curve.length` is at most 64, and that every curve number is finite.
- Summary rows gain `cu` and `lossRatio` (final predLoss ÷ baselineLoss, or null if missing).

- [ ] **Step 1: Write the failing tests.** Use `fake-indexeddb`, a flailer-like stub `TrainableLearner` that adds known CU per step, and a real tiny CPU learner with a stub world (a `WorldView` producing a synthetic stream):
  - log files are written exactly at the 2^k CU crossings plus the final one;
  - `cu` and steps only increase;
  - every file passes `parseLogFile` (v2);
  - **Review Focus 2:** interrupt mid-life → resume from the checkpoint. Steps, CU and seq continue with no duplicate seq, and the curve continues;
  - the checkpoint is deleted after the final log;
  - the validator accepts a valid v2 PR and rejects decreasing `cu`, a curve of 65 entries, and a NaN in the curve;
  - v1 files still validate;
  - the speed cap is honoured (with a fake clock).
- [ ] **Step 2: Run them to see them fail.** Run: `npx vitest run packages/harness tools/gate`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gate.** Run: `npm run lint && npm run typecheck && npm test`. Expected: PASS.
- [ ] **Step 5: Commit.** `feat(harness): lives measured in compute, sleeps, checkpoints, log v2`

---

### Task 7: The page runs the real learner

**Files:** modify `app/src/{worker,main,ui,site}.ts`, `app/index.html` and `knowledge/settings-space.json`, and add tests in `tests/e2e/` and `app/test/`.

**Behaviour:**
- **Backend selection** (worker):
  - Try `createWebGpuBackend()`. Use it only if the adapter is hardware (`!isSoftwareAdapter(adapterInfo)`), with `GPU_WIDTHS`.
  - Otherwise use the CPU backend with `CPU_WIDTHS`.
  - `hostSig.webgpu` records whether WebGPU was actually used.
- **Settings:**
  - `knowledge/settings-space.json` gains ranges for `lr` [3e-4, 3e-3], `entropyWeight` [0.001, 0.05], `curiosityWeight` [0.5, 2], `bufferSize` [500, 4000], `replayPerSleep` [16, 128], `maxStepsPerSec` [30, 30], and keeps `moveProbability` unused.
  - The picker stays uniform; Plan 05 replaces it.
  - The integers `bufferSize` and `replayPerSleep` are rounded.
- **Resume:** the worker resumes from a checkpoint when one exists for this tab's learnerId.
- **Fatal GPU read (Review Focus 3):** the worker restores from the last checkpoint once. If that fails again, it falls back to the CPU backend by restoring the checkpoint on CPU at the same widths. A GPU checkpoint at `GPU_WIDTHS` is valid on CPU, only slower. This is logged in the status line.
- **LEARNER tab additions:**
  - CU lived out of `lifetimeCU`;
  - steps per second;
  - backend (WebGPU or CPU);
  - a small sparkline (canvas) of `predLoss` and `baselineLoss` over the curve, labelled in plain words: "how wrong its guesses are" and "the lazy guess: nothing changes";
  - the ratio.
- **HISTORY table:** gains `lossRatio` and `cu` columns.
- **ABOUT text:** changes from "flails at random" to an honest description: it learns to predict what it sees and seeks out what it can learn; it doesn't yet move between rooms by itself.

**E2E:**
- The CPU backend, enforced with `?testBackend=cpu`, which is non-production only.
- Tiny widths via `?testWidths=tiny`, non-production only.
- `?testLifetimeCU=0.05`.

The page runs, the CU and loss curve update, a final v2 log is PR'd (to the mocked GitHub), and a newborn follows. The existing token and host-allowlist assertions keep passing.

- [ ] **Step 1: Write the failing e2e tests and unit tests:**
  - the backend choice logic, with a fake adapter: hardware gives GPU, software gives CPU, none gives CPU;
  - the sparkline data mapping;
  - the e2e flow above;
  - a production build still drops all `?test*` knobs (the existing build test, extended).
- [ ] **Step 2: Run them to see them fail.** Run: `npx playwright test && npx vitest run app`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gate** (`npm run lint && npm run typecheck && npm test && npx playwright test`), plus a manual run of `npm run dev` with a screenshot of the LEARNER tab after 60 s on this machine's GPU. Look at it.
- [ ] **Step 5: Commit.** `feat(app): the page runs the real learner`

---

### Task 8: Live: does it learn?

Everything here is outward-facing and was already authorised by the user ("keep going all the way"). Report what happens.

- [ ] **Step 1: Merge and push.** Merge to `main` and push. Wait for CI and deploy to go green.
- [ ] **Step 2: Watch a live tab.** Open `https://markwhiting.github.io/learner/` in Playwright Chromium with `--use-angle=metal`, so WebGPU is on hardware. Leave it running for 20 minutes. Do not enter a token, so this run contributes nothing. Every 2 minutes, sample `steps`, `cu`, `predLoss`, `baselineLoss`, steps per second and backend.
  - **Expected:** WebGPU is used, there are no page errors, and steps per second is close to the cap of 30.
- [ ] **Step 3: Write the research note** `docs/research/2026-10-20-does-it-learn.md`:
  - **Summary first:** does `predLoss` fall below the copy-last baseline, and below 0.8 × the baseline (the M1 bar)? By how much, after how many CU?
  - The samples table, and a screenshot of the LEARNER tab.
  - **Honest caveats:** one run, one machine, mostly in the crib.
- [ ] **Step 4: Commit and push the note.**

---

## Self-review notes

**Spec coverage:**
- §5.1 cold start: Tasks 3 and 4.
- §5.2–5.3 observation and action: Task 2.
- §5.4 objectives: prediction, curiosity as learning progress, actor-critic with entropy (Task 4).
- §5.5 architecture: Task 3. The fast/slow tiers and sleep split are condition B/C, in Plan 05. Condition A's single tier is here.
- §5.6 condition A: Tasks 4 and 5.
- §6 compute and CU: Tasks 4 and 6.
- §10:
  - lifetime in CU: Task 6;
  - checkpoints and resume: Tasks 6 and 7;
  - GPU fallback: Task 7.

**Deliberately left to Plan 04b:**
- the caregiver;
- milestones M1–M6;
- checkups at 2^k CU with learning frozen;
- the milestone curve in logs.

Log files are already written at 2^k CU, so checkups slot in there.

**Deliberately left to Plan 05:**
- conditions B and C;
- the GP picker;
- knowledge PRs.
