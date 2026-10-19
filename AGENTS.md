# AGENTS.md

Instructions for any coding or research agent working in this repository (Claude Code, Codex, Cursor, Gemini, etc.). `CLAUDE.md` and any other agent-specific files should point here rather than duplicate it.

## What this project is

A continuously learning system that starts with **no pretrained world knowledge**, experiences a computer-like world through **pixels in, primitive actions out**, and selectively carries learning into more persistent and more widely shared forms (working state → episodic memory → fast plastic state → slow parameters → learning policies → shared inheritance).

The central research question: **does learning how to retain, consolidate, and reuse experience make later learning cheaper, on a small compute budget (~1–10 ordinary computers)?**

Full background: `docs/research/2026-10-16-research-synthesis-v0.1.md`. Read it before proposing designs.

## Status

Pre-implementation. The project is **blanktab**: a browser-tab learner hosted at `markwhiting.github.io/blanktab`. The v1 design is `docs/specs/2026-10-19-blanktab-v1-design.md`, approved 2026-10-19. Settled choices are in `docs/decisions/`; don't contradict them without a new decision.

## Repository layout

| Path | Purpose |
|---|---|
| `AGENTS.md` | This file. Shared rules for all agents. |
| `docs/research/` | Background syntheses and literature notes. Dated, treated as historical once superseded. |
| `docs/open-questions.md` | Live list of unresolved design questions and their status. |
| `docs/decisions/` | One file per settled decision (`NNNN-short-title.md`). Append-only; supersede rather than edit. |
| `docs/specs/` | Design documents an implementer can plan from (`YYYY-MM-DD-topic-design.md`). |
| `docs/plans/` | Implementation plans derived from approved specs. |

Code directories will be added once the first runtime decision is made.

## Invariants — do not violate without an explicit, recorded decision

These are the corrections that previous discussions had to make repeatedly. Treat them as hard constraints.

1. **Cold start means cold start.** Architectural priors (sensors, memory machinery, objectives, optimizers) are allowed. Pretrained visual or linguistic weights, embeddings, tokenizers, or other learned competence are not — unless the experiment is explicitly labelled as a pretrained comparison track. An inherited checkpoint is a different condition from a fresh individual; label it.
2. **Learner inputs vs. privileged instrumentation.** The learner sees declared observations (default: pixels) and emits primitive actions. DOM, accessibility trees, transcripts, object IDs, URLs, and environment internals may be used by the *harness* for verification, never silently passed to the learner. Any added modality or reward channel is an explicit, recorded experimental choice. Curriculum selection is a prior — record it.
3. **Autonomous interaction is required.** Passive observation (watching video, watching a person browse) can contribute experience but does not replace the learner acting. Measure passive prediction and interactive control separately.
4. **Learner vs. environment vs. harness are separate components.** Keep their interfaces explicit.
5. **Research-agent improvements ≠ learner-discovered improvements.** When an agent (including you) changes the implementation, that is design improvement, not within-lifetime learning. Keep the source of every change and its compute cost visible.
6. **GitHub is the population inheritance layer, not the learning loop.** Ordinary weight updates stay local. Only selected, evaluated artifacts become shared contributions.
7. **Evaluation is protected.** Candidates (code, policies, or agents) must not weaken their own tests, redefine success, or access credentials. Hold implementation and budgets fixed within a run; report interaction and compute costs separately; keep unfavorable results.
8. **No claims without evidence.** Novelty and feasibility are hypotheses. Lower loss, bigger memory, more commits, or more replay do not by themselves demonstrate improved learning efficiency.
9. **Simplify the world, not the architecture's ceiling.** Start with a small environment, but don't hard-code limits on the learner's eventual action vocabulary or capability.

## Working conventions

- **Notes are Markdown**, in `docs/`. Keep them concise and cite sources (`[S1]`, `[R3]`, file paths, URLs).
- **Decisions:** when the user settles a question, write `docs/decisions/NNNN-title.md` using `docs/decisions/0000-template.md`, and update its row in `docs/open-questions.md`. Distinguish what the user decided from what an agent proposed.
- **Mark guesses as guesses.** If you fill a gap with a default, say so in the doc ("agent's call, not yet confirmed").
- **Verify against real artifacts.** For any environment/runtime claim (e.g. "this browser API gives readable pixels"), demonstrate it with a minimal working probe before building on it. Reading docs or code is not proof.
- **Session logs** live in `data/runs/` and are committed through the PR gate. Local scratch output goes in `.local/` (gitignored).
- **Commits:** small, descriptive; don't commit or push unless asked.
