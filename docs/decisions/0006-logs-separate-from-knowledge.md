# 0006 — Every run's log is kept; only improvements change shared knowledge

- **Date:** 2026-10-19
- **Status:** accepted (principle); channel and mechanics proposed
- **Decided by:** user (principle)
- **Resolves:** Q7 (partially)

## Decision
"Record this run" and "adopt what this run changed" are separate decisions.
- **Logs** (what was tried, under what conditions, with what milestone results) are kept for every valid run, including failures. They feed future sessions' choices.
- **Knowledge changes** (meta-learner parameters, priors, later architecture) are adopted only when they beat the current best.

## Rolling submissions
Each session keeps one open submission and updates it on a schedule. It is triaged when it beats the current best (merge, and the session continues), or when the session has been silent for X hours.
