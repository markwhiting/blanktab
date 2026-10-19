# 0001 — World staging: in-tab playground first, browser extension second

- **Date:** 2026-10-19
- **Status:** accepted
- **Decided by:** user
- **Resolves:** Q2 (partially), Q1 (partially)

## Context
A plain web page cannot read pixels from, or send input to, other origins. The learner can run in a tab, but the tab can't be its window onto the live web.

## Decision
1. **Stage 1:** the learner runs in a browser tab against a playground we own, rendered in the page and filled with curated training material.
2. **Stage 2:** once stage 1 runs, a browser extension gives access to the real web (tab capture plus synthetic input).

The learner/world interface must let stage 2 plug in without changing the learner.

## Alternatives considered
- Streaming a remote browser into the tab (no install, but hosting cost per session). Kept as a later option.
- Native/local harness first. Rejected as the primary path because distribution by link is the top preference.

## Consequences
- The early competence targets (scrolling, clicking, engaging with text) are developed in the playground.
- The extension is a second world adapter, not a rewrite.
