# 0004 — Two sharing layers: live peer layer and PR inheritance layer

- **Date:** 2026-10-19
- **Status:** accepted
- **Decided by:** user
- **Resolves:** Q6 (partially)

## Decision
- **Live layer:** sessions can exchange state directly and in real time (e.g. a CRDT relay, as in markwhiting/life). This is the medium for social learning: imitating behavior, distilling another session, sharing structure.
- **PR layer:** vetted, durable inheritance through GitHub.

## Constraints
- Distillation and imitation are **made possible by the design, not hard-coded** as behavior. Sessions may come to pursue them; we don't script it. Expected far downstream.
- A learner that has absorbed state from a peer is labelled as such (not a fresh from-scratch individual).
