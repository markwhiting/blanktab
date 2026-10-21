# 0014 — Learners live forever in their tab; knowledge and code update during a life

- **Date:** 2026-10-21
- **Status:** accepted
- **Decided by:** user (no fixed lifetimes; auto-update; a learner's history may span code versions), agent (mechanics)
- **Amends:** spec §10 lifetime; decisions 0006 (rolling submissions), 0011 (log cadence), 0013 (§8 logs)

## Decision
- **No fixed life window.** A learner lives as long as its tab is open. It continues in another tab through orphan adoption (under 24 hours, per 0013). A "final" log is written only when a life fails.
- **Log schedule.** Logs are written at CU 8, 16, 32, …, 512, then every 512 CU, forever. Checkups (Plan 04b) use the same schedule.
- **Merges on a schedule.** The hourly gate merges any valid session PR; it no longer waits for a final log or 48 hours of silence. The tab's next flush opens a new PR on a fresh branch.
- **Knowledge refresh without reload.** Every ~15 minutes (with jitter) the tab re-reads the shared knowledge and passes it to the running learner. Each log's `knowledgeSha` is the knowledge in force *when that file was written*, not at birth.
- **Code reload with resume.** Every ~10 minutes the tab checks the deployed `manifest.json`.
  - If `codeSha` changed and the new code's `learnerStateVersion` matches the running learner's, the tab waits a random 0–5 minutes, forces a checkpoint, and reloads. The learner resumes on the new code.
  - If the versions are incompatible, the learner keeps running on the old code, and the page says so.
- **Research consequence (user accepted).** One learner's history can span code versions. Every log records `codeSha`. Comparisons between conditions (Plan 05) use only segments with constant code and knowledge.
