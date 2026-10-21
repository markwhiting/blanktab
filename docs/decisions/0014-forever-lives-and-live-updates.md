# 0014 — Learners live forever in their tab; knowledge and code update during a life

- **Date:** 2026-10-21
- **Status:** accepted
- **Decided by:** user (no fixed lifetimes; auto-update; a learner's history may span code versions), agent (mechanics)
- **Amends:** spec §10 lifetime; decisions 0006 (rolling submissions), 0011 (log cadence), 0013 (§8 logs)

## Decision
- **No fixed life window.** A learner lives as long as its tab is open. It continues in another tab through orphan adoption (under 24 hours, per 0013). A "final" log is written only when a life fails.
- **Log schedule.** Logs are written at CU 8, 16, 32, …, 512, then every 1024 CU, forever. Checkups (Plan 04b) use the same schedule.
- **Merges on a schedule.** The hourly gate merges any valid session PR; it no longer waits for a final log or 48 hours of silence. The tab's next flush opens a new PR on a fresh branch.
- **Knowledge refresh without reload.** Every ~15 minutes (with jitter) the tab re-reads the shared knowledge and passes it to the running learner. Each log's `knowledgeSha` is the knowledge in force *when that file was written*, not at birth.
- **Code reload with resume.** Every ~10 minutes the tab checks the deployed `manifest.json`.
  - If `codeSha` changed and the new code's `learnerStateVersion` matches the running learner's, the tab waits a random 0–5 minutes, forces a checkpoint, and reloads. The learner resumes on the new code.
  - If the versions are incompatible, the learner keeps running on the old code, and the page says so.
- **Research consequence (user accepted).** One learner's history can span code versions. Every log records `codeSha`. Comparisons between conditions (Plan 05) use only segments with constant code and knowledge.
- **Boundary logs (final-review ruling, agent).** Every log file has exactly one code version and one knowledge version:
  - the forced checkpoint before a code reload also writes a log file, in the same transaction, so every curve point lived before the reload carries the old `codeSha`;
  - when a knowledge refresh changes `knowledgeSha`, a log file under the old sha is written first (with its checkpoint), then the new sha is in force.
  These are ordinary log files: they count toward the contributor's 16-per-PR cap like any other.
- **`codeSha` names the code (final-review ruling, agent).** It is the last commit that touched a build input (`app/`, `packages/`, `package.json`, `package-lock.json`, `tsconfig.base.json`; `tools/gate/manifest.ts` `BUILD_INPUTS`), not the head of `main`, so merged session logs, docs, knowledge, tests and workflow changes do not reload every tab.
- **Seq gaps and local-only learners (final-review ruling, agent).** A PR the gate closes takes only its own files with it:
  - the gate accepts a learner's log whose `seq` is strictly greater than the highest seq already on `main` (and unique in the PR); gaps are allowed and stay visible in the record;
  - the contributor marks the closed PR's files dropped (never resent) and goes on from where the learner is;
  - after 3 closed-unmerged PRs in a row (a merge resets the count) the learner becomes local-only, kept in IndexedDB with its outbox: it keeps learning but its logs are no longer sent, and the page says "This learner's logs were rejected; it keeps learning here but no longer contributes."
- **Outbox bounds (final-review ruling, agent).** Sent files are pruned except each learner's newest (so numbering resumes, and a finished life is still known to be over); unsent files are capped at 200 per learner, oldest dropped first (gaps are legal).
