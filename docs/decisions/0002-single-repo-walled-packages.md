# 0002 — One repo with walled-off packages

- **Date:** 2026-10-19
- **Status:** accepted
- **Decided by:** agent proposal, accepted by user ("overall sounds good")
- **Resolves:** —

## Context
The user's first instinct was two repos (learner; web playground). The learner/world interface will churn early, and experiments must pin exact learner and world versions together.

## Decision
One repo with separate `learner/`, `world/`, `harness/` packages plus a shared interface. A build check fails if `learner/` imports anything from `world/` or `harness/` beyond that interface (this enforces AGENTS.md invariant 2).

## Alternatives considered
Two repos: cleaner ownership, but every interface change becomes two coordinated PRs and reproducibility needs cross-repo pinning.

## Consequences
Split `world/` into its own repo later if distribution or the extension demands it.
The learner *navigating* the world is exploration and is fine. The learner *modifying* the world is a separate, explicitly recorded experiment and must never touch the evaluation world.
