# 0018 — One meta-learner over a learning genome

- **Date:** 2026-10-22
- **Status:** accepted
- **Decided by:** user (direction, 2026-10-22: "Overall i think this sounds good to me. LFG"); details marked *agent's call* await confirmation
- **Resolves:** reshapes Q14 ("what knowledge holds and how a session uses it"), Plan 05 and the learner v2 design

## Context

The learner v2 design (`docs/specs/2026-10-22-learner-v2-capabilities-design.md`) proposed a hand-designed redesign with an offline comparison. The user rejected the separate-comparison model: improvements "should not be theorized or conceptualized by us, but be natural explorations of the underlying design space", sampled by one meta-learner with adaptive sampling, as part of normal running. They want the learning pipeline as "all data pipeline": one statistical representation from the meta-learner down to an individual's finest behaviour. "How it learns" is the shared core, inherited through GitHub. "What it learns in a run" is local. Learned material that many lives acquire should strategically move up the hierarchy. A coding agent in the loop may come later; this version comes first.

The literature behind this is in `docs/research/2026-10-22-meta-learner-synthesis.md` and its four verified notes.

## Decision

**User's:**
- **One meta-learner.** All improvements, including alternative learning strategies, go through it: a population meta-learner that samples variants adaptively, in proportion to how much better they look.
- **The learner is one point in a parameterized learning genome.** The meta-learner's state and the promoted genome are the shared, inherited layer. Within-life weights stay local.
- **Gaussian-process-style adaptive sampling is the default outer loop.** Draws should get better over time, with no coding agent in the loop at first.
- **Learned material that generalises moves up the hierarchy** over time.
- **Scoring:** the agent's proposed score is acceptable. The search uses a continuous score made of the checkup measures against newborns plus the held-out prediction ratio. Confirmation uses milestone passes against newborn luck. **Every evaluation function must be compartmentalised so it can be switched later.**

**Agent's calls (from the synthesis; not yet confirmed):**
- **Sampling:** Thompson sampling, not UCB.
- **Genome v1:** about 17 continuous genes plus 1–2 switches. Today's learner is the centre, and mechanisms the learner doesn't have yet enter as genes whose centre is "off".
- **What moves up first:** the first artifact is first-layer filters, gated by a convergence probe and always run beside fresh and matched-control conditions.
- **Seeds:** search and confirmation use disjoint seeds.

## Alternatives considered

- **The hand-designed v2 redesign plus an offline 108-life comparison.** Rejected by the user: it is a break-off learning cycle, and the improvement is theorised rather than found.
- **Lamarckian sharing (copying learned weights).** Kept only as a labelled condition (invariant 1). The literature favours inheriting how a learner learns when the task distribution is broad (Fernando et al. 2018).
- **A coding agent proposing changes.** Deferred by the user.

## Consequences

- **Plan 05 becomes the meta-learner.** It covers the genome, swappable evaluators, the Thompson-sampling picker, a deterministic refit in the gate, and protected promotion. The learner v2 mechanisms are implemented later as genes, not as a fixed redesign.
- **What success means lives in versioned evaluator modules.** Changing an evaluator is a recorded decision with a version bump (invariant 7).
- **Progress is paced by tab-lives.** That is about 110–340 per machine per week (estimate).
