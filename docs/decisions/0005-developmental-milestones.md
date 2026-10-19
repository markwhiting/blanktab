# 0005 — Progress measured by a developmental milestone ladder

- **Date:** 2026-10-19
- **Status:** accepted (direction); milestone format still to be specified
- **Decided by:** user (idea), agent (first draft of milestones)
- **Resolves:** Q5, Q10, Q11 (partially)

## Decision
Progress is measured against an ordered set of developmental milestones, analogous to a child's (crawls, walks, talks). A run's score is which milestones it reached and the experience and compute spent to reach each one. Measuring time-to-milestone makes short and long sessions comparable.

Milestones are detected by the harness using privileged world state; the learner never sees the detectors. They must be formalized as checkable specs and versioned, and new milestones are added as sessions routinely pass the existing ones.

## First draft (agent's guess, not yet confirmed)
1. Predicts the next frame better than "nothing changes."
2. Moves the cursor purposefully rather than jittering.
3. Clicks responsive elements more often than unresponsive ones.
4. Scrolls to reveal hidden content.
5. Returns to a previously visited place.
6. Treats text differently from images.
7. Reaches a target page by following links.
