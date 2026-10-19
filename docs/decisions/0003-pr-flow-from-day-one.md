# 0003 — Session-to-GitHub PR flow from day one; everything built from scratch

- **Date:** 2026-10-19
- **Status:** accepted
- **Decided by:** user
- **Resolves:** Q8 (partially)

## Context
The synthesis proposed reusing existing local harnesses and xot. The user wants the PR-back flow to exist from the very first runs, because early work is essentially a distributed hyperparameter/architecture search whose results should feed back into future searches.

## Decision
- No existing harness or xot integration; all infrastructure is built in this repo.
- From the first runnable version, a browser session can contribute its experiment results back to GitHub as a PR. Automated checks validate the PR, and merged results update shared state that future sessions read.

## Consequences
- The contribution pipeline (session → relay → PR → Actions validation → merge → shared prior) is part of the first milestone, not a later extension.
- See open questions for credentials, conflict avoidance, and validation design.
