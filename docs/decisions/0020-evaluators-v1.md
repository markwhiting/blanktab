# 0020 — Evaluators v1: swappable, versioned, pinned

- **Date:** 2026-10-23
- **Status:** proposed
- **Decided by:** the requirement is the user's ("make sure that all our evaluation functions are easily compartmentalized so we can switch them later"; spec `docs/specs/2026-10-22-meta-learner-v1-design.md` §2). Everything below is an agent proposal awaiting confirmation (agent's call, not yet confirmed).
- **Resolves:** spec §2 (swappable evaluators) for plan 05a (`docs/plans/2026-10-23-05a-meta-learner-foundations.md`, Task 7)

## Context

The meta-learner (0018) ranks genomes by a score, decides when to promote by a test, and checks for repeat noise. Each of these is an evaluation function. The user asked that they be easy to swap later. Evaluation is also protected (AGENTS.md invariant 7): a contribution must not be able to move its own score or the reference it is scored against.

## Decision

**Agent's calls (not yet confirmed):**

- **One package, one interface.** `packages/evaluators` (`@blanktab/evaluators`) holds every evaluator behind the spec §2.1 interface (`Evaluator`, `LifeInput`, `EvalOutput`, `References`). It may import only `@blanktab/harness/log` and `@blanktab/detmath`, and it is pure: no IO, no clock, no `Math.random`, and no engine transcendentals or `**` (the lint purity rules of plan 05a). The same input bytes give the same output bytes.
- **Versioned, and old versions stay.** `REGISTRY` is keyed `id@version` (`search-composite@1`, `newborn-baseline@1`; `signal-check@1` and `confirm-fisher@1` join in Task 8). `getEvaluator` throws on an unknown key. A registered version is never edited or removed, so any merged life can be rescored under any version.
- **The version-bump rule.** Any change to what an evaluator computes, including its constants, means a new version in `REGISTRY` and a recorded decision.
- **Swapping an evaluator touches only a new module, one `REGISTRY` line and `config/evaluation.json`** (the user's requirement). To make that hold:
  - **The budget lives on the evaluator.** A life evaluator declares its own `budget` (`{ budgetCU, levels }`). `search-composite@1`'s is 2,048 CU at 512, 1,024 and 2,048, so it cannot move silently with harness constants. `tools/eval/config.test.ts` checks the config's `budgetCU` and `levels` against `REGISTRY[config.search].budget`, whichever evaluator is pinned, and against the harness's `BUDGET_CU` and `FILTER_LEVELS`.
  - **References are generic.** `config.references` maps a reference name to a file directly under `config/references/`. Each evaluator's `needs` is the source of truth. `loadEvalConfig` fails, naming the evaluator and the reference, when a pinned evaluator needs a reference that has no registered parser (`REFERENCE_PARSERS`) or no pinned path. It also fails when the config pins a reference that no pinned evaluator needs, or pins an unknown evaluator (other than the pending Task 8 pins in `PENDING_PINS`). `loadReferences` loads exactly the needed references, each with its parser. `newborn` is the only reference in v1.
- **The pinned config.** `config/evaluation.json` is spec §2.2, with `decision` pointing here. It is not in `knowledgeSha`, because learners never read it. `tools/eval/config.ts` reads it strictly (`loadEvalConfig`) and loads its references (`loadReferences`). An absent reference gives `{}`, so scores are ineligible ("no newborn reference"). A malformed one throws.
- **The gate never writes `config/`.** A change to a pin is an edit to `config/evaluation.json` by a person or a coding agent, under a recorded decision. The gate labels any PR touching it `needs-human`, as for every non-log path.
- **Rows never mix evaluator versions.** A posterior or a table is built under one `evaluator@version`, and says which. Changing the search evaluator starts a fresh fit, and old lives are rescored rather than mixed. This is enforced in code: `targets` and `rankAverage` throw on any output whose `evaluator` or `version` is not their own, whether or not it is eligible.
- **`search-composite@1`** is spec §2.4. These are the details the spec left open:
  - **The budget window.** It reads the files up to and including the first one that holds the 2,048 checkup. If no file holds it, the window ends at the first file past 2,048 CU, or else it is all the files. Within the window it reads only curve points at cu ≤ 2,048 and the checkups at 512, 1,024 and 2,048 of suite 1. Later files never change a score.
  - **The eligibility order.** The reasons are checked in this order:
    1. "not a search life": `meta` has role `search`, genome schema 1 and inheritance `none`;
    2. "no meta block before the checkup";
    3. "not gpu widths": the widths class only, so a GPU life that fell back to the CPU still counts;
    4. "no newborn reference";
    5. "wrong milestone suite": a window file's checkups are not of the reference's suite (1);
    6. then the life's end, when the window holds no 2,048 checkup and its last file is final:
       - with `end: "failed"`, the life is a row with C = H = −2 at any cu. A life that failed past 2,048 CU before its 2,048 checkup ran is ranked worst, not dropped;
       - otherwise, below 2,048 CU, it is "abandoned", including a final file with no `end` from older code;
    7. "no 2048 checkup";
    8. "fewer than 2 of 3 levels";
    9. "fewer than 3 held-out points in (1024, 2048]".

    "Abandoned" comes before the checkup checks because such a life can never hold the 2,048 checkup. A life whose last file is not final and is below 2,048 CU may still be running, so it reads "no 2048 checkup", not "abandoned". So does an orphan past 2,048 CU without the checkup.
  - **Held-out points** with a zero baseline are skipped.
  - **Measures.** A measure that would make (x + κ) ≤ 0 gives q = −2. M1's diagnostic q uses κ = 0.08 (0.1 × its 0.8 threshold), and a null M1 stays null. M1 is not part of C.
  - **The GP targets** rank a_i through the exact rank sums rank(C) + rank(H). These are half-integers, so floating-point rounding cannot split a tie. `rankAverage` returns (rank(C) + rank(H) − 1)/(2n), which is (u_C + u_H)/2.
- **`newborn-baseline@1`** is spec §2.3, over the group `newborn`.
  - **Which lives count.** A life counts if its newborn file has genome schema 1, suite 1, GPU widths and checkups at exactly 512, 1,024 and 2,048. Duplicates and other lives are ignored, and the number ignored is reported.
  - **Medians and pass rates** are taken over all of the counted checkups, pooled across levels, with a null measure read as 0. The sustained pass rate is over lives (≥ 2 of 3 levels).
  - **The held-out median** is the median of `meta.heldOut.loss / baseline`, skipping zero baselines.
  - **Its output.** `diagnostics.reference` is the `NewbornReference` with sorted learnerIds, and `score` is null.
  - **What it cannot check.** It cannot see whether a newborn ran the centre genome, because evaluators may not import meta. The gate's newborn checks (0022) cover that.

## Alternatives considered

- **Evaluators inside the gate.** Then they could not be rescored or tested in isolation, and swapping one would mean editing the gate.
- **Reading `BUDGET_CU` and the levels from the harness inside the evaluator.** Then a harness change would silently change `search-composite@1` without a version bump.
- **Ranking a_i as floats.** Equal rank sums can round to different floats (0.1 + 0.7 is not 0.3 + 0.5), which would split real ties.
- **Mixing versions in one fit.** Then the scores in one posterior would mean different things.

## Consequences

- The gate (Task 10) and the tools (Task 9) import `@blanktab/evaluators` and read `config/evaluation.json`. The page shows numbers the gate computed.
- `config/references/newborn-baseline@1-gpu-suite1-genome1.json` does not exist yet. Until a person pins it (Task 9's newborn runs), every search life is ineligible, and that is shown, not hidden.
- Until Task 8 registers them, `confirm-fisher@1` and `signal-check@1` are pinned but not in `REGISTRY`. `tools/eval/config.test.ts` lists them as pending, and it fails once they are registered, so the list is removed then.
- A new evaluator, or a new version of one, is a new registry entry plus a decision. Re-pinning is a `config/` edit plus a decision.
