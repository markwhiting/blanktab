# 0019 — A short-lives switch, and the gate commits meta-learner files

- **Date:** 2026-10-23
- **Status:** accepted
- **Decided by:** user (the direction and the gate permission); details marked *agent's call* await confirmation
- **Resolves:** the two user questions from `docs/specs/2026-10-22-meta-learner-v1-design.md` (search throughput; how the gate writes meta-learner state)

## Context

The meta-learner (0018) scores one result per life. Lives never end (0014), so with forever lives the search would see only a handful of results a day. Running unattended, the gate also has to write the meta-learner's state and, when a confirmed winner exists, the promoted genome. Today the gate only merges PRs.

## Decision

**User's:**
- **Lives stay forever by default.** The user's words: "I kind of prefer a world where lives don't end".
- **The page gets a switch for short lives.** In the user's words, "a UI switch where we can just say this session is going to auto die every 30 minutes or something just so that we get some signal, but we can also turn that off and just let it live forever".
- **Gate permission.** The gate may commit directly to `main` with the repo's Actions token, but only for the meta-learner data (`data/meta/**`) and the promoted genome (`knowledge/genome.json`). The user said: "I'm fine with your recommendation".

**Agent's calls:**
- **When a short life ends.** It ends when its evaluation-budget checkup (2,048 CU) has been logged, which is roughly 15–30 minutes. Then a newborn starts at once in the same tab.
- **Where the switch lives.** It belongs to the browser (stored in localStorage), defaults to off, and shows on the LEARNER tab.
- **How short lives are recorded.** A life ended by the switch writes a final log marked as ended by budget. Lives with the switch on or off are both valid search rows.
- **Deploys.** A gate commit of either path triggers the deploy, like a merge does.
- **The gate's commits can't come from a PR.** Its commit only ever contains files the gate computed itself. PR content can never reach these paths (invariant 7).

## Alternatives considered

- **Every life ends at its budget.** This would reverse 0014, and the user prefers forever lives.
- **Keep forever lives only and accept a slow search.** The search would be too slow to show any signal.
- **The gate opens a self-merging PR for each update.** This is slower and noisier, and it is not needed given the user's approval.

## Consequences

- **The search depends on the switch.** It runs at a useful rate only on tabs where the switch is on. The page should say this plainly.
- **Ending a life on purpose is a new reason a life can end.** The log format and the gate must accept it.
- **The gate becomes a writer to `main` for two paths.** Its permissions and validation must keep those paths free of PR-supplied content.
