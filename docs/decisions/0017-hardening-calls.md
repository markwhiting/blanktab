# 0017 — Hardening calls (Plan 04c)

- **Date:** 2026-10-22
- **Status:** accepted (agent's call; not yet confirmed by the user)
- **Decided by:** agent, under the user's "make best choices"; not yet confirmed. The plan controller made the resolutions marked as its own below; those are agent calls too.
- **Amends:** [0016](0016-milestones-and-checkups-v1.md) (carries lost to a restore are now counted); builds on 0014
- **Resolves:** no open question; records the choices Plan 04c makes on the user's behalf
- **Plan:** `docs/plans/2026-10-22-04c-hardening.md` (Task 3 creates this file; Tasks 4–7 add their line)

## Context
Plan 04c fixes problems deferred from Plans 04a–04b: robustness, record-keeping, the gate, and step speed. It changes no learning (the Global Constraints of the plan). Several fixes needed a choice the user has not made. This file lists each one so it can be confirmed or reversed.

## Decision

### Log and checkpoint correctness (Task 3)
- **`lostCarries`.** A log file's `caregiver.lostCarries` counts the caregiver's carries since the previous file that a restore or a failed life threw away. They are not in `carries` or `to`, and not in the `to` sum. It is an integer >= 1 when present. It is omitted at 0, so a file without loss is byte-identical to before. Over a learner's files, the sum of `carries` and `lostCarries` equals every carry the caregiver made.
  - A restore adds `max(0, carries now − the checkpoint's carries)`, plus the loss earlier restores recorded since the last file. The caregiver state keeps that count as `lost` (optional; absent in older states). The restore's own save holds the restored caregiver with it, so a reload right after a restore keeps the count.
  - A failed life's final file adds the carries lived since its checkpoint to the checkpoint's `lost`.
  - The session counts the carries since the last committed save itself. The count is zeroed only once a save commits, so it is not read back from the caregiver. A save or log write that fails after the caregiver's summary was taken cannot hide carries, and neither can a learner too broken to report.
- **A failed life's final log that cannot be written is reported** on `console.error`, with its reason. Writing it stays best effort, and the life's own error is still the one thrown.
  - Carries lost to a reload or a recoverable (GPU) error are not counted: those resume from the checkpoint without the harness seeing the lost steps. *(Agent's call: left as is; the curve and CU of such steps are not on the record either.)*
- **Tails carry their suite version.** A checkpoint stores `checkups.suiteVersion` (the current `MILESTONE_SUITE_VERSION` when written; a resumed tail keeps its own). Absent, it is suite 1. The tail is checked against that suite's ids, and an unknown version fails to load. A resumed tail of another suite goes into the next log file under its own `milestoneSuiteVersion`, and no new checkup runs until that file is written. One file never mixes two suites.
- **A tail needs the learner's evalCU.** A checkpoint whose tail is non-empty must have a learner state with `stats.evalCU` (finite, >= 0). Otherwise it does not load: `"checkups.tail without the learner's evalCU"`. The session checks the same rule before it writes a checkpoint, so it never writes one that would not load. A failed life's final file now carries a non-empty tail always. If evalCU is missing there anyway, writing that file fails explicitly. A tail is no longer dropped silently.
- **`MAX_CHECKUPS_PER_FILE` defers checkups.** While the tail holds 8 results, the next level stays pending (no checkup runs) until a log file takes the tail. A log file with more than 8 is never written: the writer throws.
- **Checkup-score bounds.** A summary row's `checkupScore` is kept only when its level is on the log schedule and its score is an integer from 0 to `MAX_SUITE_SIZE`. `MAX_SUITE_SIZE` is the largest known suite's size (6 now), since summary rows carry no suite version. Anything else reads as null.

### Step speed (Task 2)
- **One GPU round trip fewer a step.** The step reads `lossAfter` with the acting heads in one round, so a step makes 3 serial reads where it made 4. On CPU this is bit-identical to the code before (the golden trajectory, recorded on `5658165`).
- **Measured** (uncapped WebGPU, `GPU_WIDTHS`, Apple metal-3, the same harness, 3 runs each): before (`01d01c5`) 26.75, 26.15, 27.70 steps/s; after (`ba6ce50`) 29.82, 28.74, 30.87 steps/s.
- **Rejected: deferring the actor-critic read to the next step.** It would save another round trip, but it shifts `consecutiveSkips` (which drives the restore after 20 skips), `cu` and `onUpdate` by one step. That is not bit-identical at the harness level, so it is not done.
- **The live speed cap stays 30 steps/s.** A live run can show the gain only up to the cap; the uncapped harness above is the real measure. *(Controller's resolution.)*

### Resolutions the controller made for later tasks (recorded here; those tasks add their own lines)
- **Folder ownership.** A `data/runs/<learnerId>/` folder that only direct pushes have changed is unbound until a PR merges. The author of the first merged PR claims it.
- **Orphaned checkpoints.** A stale or surplus orphan gets a final log file, as a failed life does; its record does not end on a non-final file.
- **The room is restored** with the learner on every restore, the in-life restore after 20 skips included.
- **The gate's fetch path.** The gate probes the `git fetch` path first, with the REST API as the fallback.

## Alternatives considered
- **Counting lost carries in `carries`.** Rejected: `carries` and `to` describe the carries behind the steps on the record. Folding lost ones in would make the counts disagree with the steps and the world the files describe.
- **Labelling a resumed tail with the current suite.** Rejected: its ids would be checked against the wrong suite, and could be refused or silently mean another milestone.
- **Dropping a tail without evalCU from a failed life's final file (the old behaviour).** Rejected: checkup results disappeared from the record without trace (AGENTS.md invariant 7).
- **Running a checkup past 8 and splitting the tail over files.** Rejected: one file would then be written for a reason the log schedule does not have. Deferring is simpler, and the level still runs once.

## Consequences
- Log `schemaVersion` stays 2, `milestoneSuiteVersion` stays 1, and `LEARNER_STATE_VERSION` stays 2. The new fields (`caregiver.lostCarries`, `checkups.suiteVersion`, caregiver state `lost`) are optional and strict when present. Every file under `data/runs/` still parses, and every checkpoint written by `87d115c` code still loads (a non-empty tail there always has a real learner's evalCU).
- A future suite version can ship without stranding checkpoints that hold suite-1 results.
- The parser (`parseLogFile`) is shared by the tab and the gate, so both apply the same `lostCarries` rule.
