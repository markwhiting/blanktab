# 0011 — Contribution volume and gate rules (from the Plan 01 final review)

- **Date:** 2026-10-19
- **Status:** accepted
- **Decided by:** agent, on the user's instruction to make best choices; open to revision
- **Amends:** spec §9.4, §9.5, §10; decision 0006 (rolling submissions)

## Context
A whole-branch review found that one open tab, flushing per file with a separate heartbeat file, would exceed the repo's GitHub Actions API budget (1,000 requests per hour for `GITHUB_TOKEN`) and leave PRs stuck.

## Decision
- **One commit per flush.** A tab writes all of a learner's pending log files in a single Git Data API commit (blobs, tree, commit, ref).
- **No `heartbeat.json`.** Liveness is the newest log file's `writtenAt`. A future-dated `writtenAt` is ignored in favour of the PR's `updatedAt`.
- **Flush cadence:** when a final log is queued, every 10 minutes while anything is pending, and when a token is saved.
- **Placeholder pacing:** 2 learner steps per second, so a 3,000-step life takes about 25 minutes. This is roughly 2–3 PRs per hour per open tab.
- **Gate rules:**
  - a PR may change at most 16 files, checked from the PR's `changed_files` before anything is listed;
  - session PRs open for more than 7 days are decided regardless of liveness (merge if valid, otherwise close);
  - logs with `codeSha: "dev"` are closed ("Development builds don't contribute.");
  - PRs must target `main`.
- **`validate` is not a required status check.** The gate re-validates every PR itself and is the enforcement. `validate` is advisory and cancels stale runs.
- **Fresh branch after merge.** If a learner has no open PR but its branch exists, the branch is reset to upstream `main` before new files are committed.
- **The learner always runs:** a worker error starts a newborn, and a Web Lock on the learner id stops duplicated tabs from sharing a learner.
