# Plan 04a.1 — Forever lives, scheduled merges, live updates

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## In plain terms
A learner now keeps learning for as long as its tab is open (decision 0014). The site shares new knowledge with running learners. When a new version of the code is deployed, each tab saves its learner, reloads at a random moment, and the learner carries on under the new code. The repo also merges what tabs send every hour, so later tabs see recent progress.

**Goal:** implement decision 0014 end to end.

**Spec:** `docs/specs/2026-10-19-blanktab-v1-design.md` §9, §10, plus decisions 0013 and 0014.

## Global Constraints
- **Every rule of Plan 04a still applies:** walls, tensor lifecycle, the token rule, logs and checkpoints written in one transaction, restores kept on the record, and no NaN in logs.
- **Log schedule:** a log is written each time CU crosses 8, 16, 32, 64, 128, 256 or 512, then each further multiple of 1024, forever. A PR therefore stays well under the gate's 16-file cap between hourly merges.
- **No lifetime limit:** `lifetimeCU` is optional, and when it is absent the life never ends. Test knobs may still set a lifetime. `final: true` is written only on failure, or when a test lifetime ends.
- **Polling:**
  - knowledge every 15 min ± 3 min of jitter;
  - the manifest every 10 min ± 2 min of jitter;
  - before a code reload, a random wait of 0–5 min;
  - nothing polls while the tab is hidden, and a check runs straight away when it becomes visible.
- **Dependencies:** after any `npm install`, run the full gate. Stop if npm reports "removed N" or "changed N" packages.

## Review Focus
1. **A tab that reloads for new code resumes the same learner.** Same learnerId, steps and CU continue, and no seq is reused. Test in Task 3 (e2e with a mocked manifest change).
2. **Incompatible new code never kills a learner.** The tab keeps running and shows "A newer version is available; this learner continues on the version it was born with." Test in Task 3.
3. **A merged PR, then more logs from the same learner.** A new PR opens on a fresh branch, and no file is lost or duplicated. This includes the race where the gate merges while a tab is committing. Test in Task 2.
4. **The gate's 16-file and 1 MB caps hold** for a learner that runs for 10,000 CU between merges. Test in Task 1 (via the log schedule).

---

### Task 1: Forever lives and the log schedule
**Files:** `packages/harness/src/session.ts`, its tests, and `app/src/worker.ts` / `main.ts` (drop the default lifetime).

**Interfaces:**
- `SessionConfig.lifetimeCU?: number`; undefined means no end.
- `export function logThresholds(): Iterable<number>`, which yields 8, 16, …, 512, 1024, 1536, …
- The page shows "Compute lived N CU", with no "of …".

**Tests:**
- the schedule's first 12 values;
- no `final` log without a lifetime;
- a stub learner adding 10,000 CU writes exactly 7 + 19 = 26 files: 8…512 is 7 thresholds, then 1024…10240 in steps of 512 is 19;
- a test lifetime still ends in a final log.

Commit: `feat(harness): learners live as long as their tab`

### Task 2: The gate merges on a schedule
**Files:** `tools/gate/gate.ts` (`decide`), its tests, and contributor tests if needed.

**Behaviour:** `decide` merges every valid session PR whose base is `main`, without waiting for a final log or a stale heartbeat. Invalid PRs still close, and non-log paths still get the `needs-human` label. Merging uses `--match-head-commit` as before, so if a tab pushed after validation the merge fails cleanly and is retried next hour.

**Tests:**
- a valid PR that isn't final is merged;
- a head that moved since validation is not merged (the existing pinning);
- Review Focus 3, against the fake GitHub: the PR merges and its branch is deleted, the tab queues more logs and flushes, a new PR opens, and every file ends up on exactly one PR.

Commit: `feat(gate): merge valid session PRs every run`

### Task 3: Knowledge refresh and code reload
**Files:**
- `tools/gate/manifest.ts`: the manifest gains `learnerStateVersion`, imported from `@blanktab/learner`.
- `packages/harness/src/knowledge.ts`
- `app/src/{main,worker,updates}.ts`
- the knowledge and manifest test files, and `tests/e2e/updates.spec.ts`.

**Behaviour:**
- **`updates.ts`** has a scheduler with injectable timers and visibility. It holds two pollers: knowledge every 15 ± 3 min and manifest every 10 ± 2 min. While the tab is hidden it pauses, and it checks immediately when the tab becomes visible.
- **On new knowledge:** the main thread posts `{ type: "knowledge", knowledge }` to the worker. The session's `knowledgeSha`, as written into later logs, updates. The worker hands it to the learner through an optional `learner.onKnowledge?.(k)`, which is a no-op in 04a.
- **On a new `codeSha` in the manifest:**
  - **compatible** (same `learnerStateVersion`): wait a random 0–5 min, post `{ type: "checkpointNow" }`, wait for `{ type: "checkpointed" }` (the session forces a save, bypassing the throttle, still in one transaction with no log), then `location.reload()`;
  - **incompatible:** show the exact text from Review Focus 2, and stop polling the manifest.

**Tests:**
- the scheduler's jitter bounds, its pause while hidden and its check on becoming visible;
- a forced checkpoint is written even inside the throttle window;
- the new `knowledgeSha` appears in the next log file;
- e2e:
  - the mocked manifest's `codeSha` changes, so the page reloads, and the same learnerId resumes with steps and CU continuing (Review Focus 1);
  - with an incompatible version the page keeps running and shows the message (Review Focus 2).
- Test-only knobs (non-production): `?testPollMs` and `?testReloadJitterMs`.

Commit: `feat(app): knowledge refresh and code reload with resume`

### Task 4: Live: does it learn?
This is Plan 04a's Task 8, now on forever lives:
1. Merge to `main`, push, and wait for CI and the deploy to go green.
2. Run Playwright Chromium (`--use-angle=metal`) against the live site, wrapped in `caffeinate -dims`, for 20 minutes of wall time, sampling every 2 minutes, with no token.
3. Before starting, check the machine's load average is under 10. If it rises above 15 during the run, stop and report rather than drawing conclusions.
4. Write `docs/research/2026-10-21-does-it-learn.md`:
   - summary first: does `predLoss` fall below the copy-last baseline, and below 0.8 × baseline? By how much, at what CU?
   - the table of samples;
   - a screenshot;
   - caveats.
5. Commit and push.
