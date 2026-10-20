# Plans roadmap — blanktab v1

Spec: `../specs/2026-10-19-blanktab-v1-design.md`.

**The organising idea:** blanktab is a page served by GitHub Pages from `markwhiting/blanktab`. Each tab is a learner, and it writes what it learned back to that same repo. Plan 01 builds that whole loop first, with placeholders: a randomly flailing "baby" in a one-room world. Every later plan swaps a placeholder for the real thing without changing the loop. Plans are written one at a time, once the previous plan has landed.

| # | Plan | What becomes real | Status |
|---|---|---|---|
| 01 | [The whole loop, end to end](2026-10-19-01-end-to-end-loop.md) | Page on Pages → tab runs a session → PR to the same repo → gate merges → summary → next tab reads it. Includes spike 1 (PR from a page) as the live test. | **done, live 2026-10-20** ([result](../research/2026-10-20-loop-live.md)) |
| 02 | [Tensor library](2026-10-19-02-tensor-library.md) | Our own in-browser neural-network math (CPU + WebGPU), including spike 2 (speed). | **done 2026-10-20** — ~108 learner steps/s on an M4 GPU ([result](../research/2026-10-20-webgpu-step-throughput.md)) |
| 03 | [World](2026-10-20-03-world.md) | The desktop shell and the four rooms (crib, toys, hide-and-seek, places), plus spike 3 (rendering speed). | **done 2026-10-20** — ~900–2,300 render steps/s per room ([result](../research/2026-10-20-world-render-speed.md)) |
| 04 | The real learner + harness | Replaces the flailer: prediction, curiosity, fast/slow memory, sleep, checkpoints and resume (spike 4), caregiver, milestones, checkups. | not written |
| 05 | Getting smarter as a population | GP picker over past sessions, knowledge PRs and `best.json`, consolidation conditions B and C, the first A/B/C comparison. | not written |
