# Plan 01 — The whole loop, end to end, in its simplest form

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## In plain terms

blanktab is a web page served by GitHub Pages from the repo `markwhiting/blanktab`. Opening it starts a learner in that tab. The learner lives, and the tab writes what happened back to **the same repo** as a pull request. A GitHub Action in that repo merges the PR and updates a summary file. The page is redeployed, and the next tab that opens reads that summary before it starts.

```
 markwhiting/blanktab (repo)
   │  GitHub Pages serves the page + knowledge/summary.json
   ▼
 a browser tab ── learner lives in a tiny world ──► log file
   │  PR via the contributor's own GitHub token
   ▼
 markwhiting/blanktab: Action validates → merges → rebuilds summary.json → redeploys Pages
   │
   └──► the next tab reads the new summary.json ... (loop)
```

This plan builds that **entire loop** with placeholders where the clever parts will go:
- **The learner is a "flailing baby":** it acts randomly and doesn't learn yet.
- **The world is one room:** a crib where shapes react to the cursor.
- **"What do I do given what I know" is trivial:** the tab reads how many sessions came before it and picks one placeholder setting at random.

Later plans replace each placeholder without changing the loop. When this plan is done, you can open `https://markwhiting.github.io/blanktab`, paste a token, watch the baby flail, see its PR appear in the repo, see it merged, and see the next tab report that one more session has happened.

**Goal:** a live, deployed blanktab page whose sessions round-trip through the repo it is served from.

**Architecture:**
- **Packages:** npm workspaces with walls (spec §4).
- **The page:** a Vite app in `app/`. It runs world + learner + harness in one Web Worker and talks to GitHub's REST API directly from the page, using a token stored in `localStorage`.
- **Repo automation:** the Action scripts in `tools/gate/` read PR files as data only, merge them, rebuild `knowledge/summary.json`, and deploy Pages in the same job.

**Tech Stack:** TypeScript (strict), Node 22, npm workspaces, Vite, Vitest, Playwright, ESLint flat config, GitHub Actions, GitHub Pages, GitHub REST API via `fetch`.

**Spec:** `docs/specs/2026-10-19-blanktab-v1-design.md` (§3, §4, §5.2–5.3, §9.3–9.5, §10, §12). Read it and `AGENTS.md` first.

## Global Constraints

- **The repo is `markwhiting/blanktab`, public; the page is `https://markwhiting.github.io/blanktab/`.** Vite `base: "/blanktab/"`.
- **Creating the GitHub repo, pushing, enabling Pages and branch protection are outward-facing. Ask the user before each one.**
- **Walls:** `learner` may import only `@blanktab/interface` and `@blanktab/tensor`. `world` may import only `@blanktab/interface`. `tensor` and `interface` import no workspace package. `harness` and `app` may import anything. A violation fails `npm run lint`.
- **Token rule (user, decision 0010):** the GitHub token lives only in `localStorage['blanktab:githubToken']`. It is never written to IndexedDB, logs, checkpoints, telemetry, the repo, or sent anywhere except `https://api.github.com`. All storage keys are prefixed `blanktab:`.
- **The learner always runs.** Without a token, logs queue locally and are submitted once a token is added.
- **The gate never executes PR code.** `validate.yml` declares `permissions: contents: read` and runs `main`'s scripts on the PR's files as data.
- **Code and other non-log paths in a session PR are never merged automatically;** label them `needs-human`.
- `SCHEMA_VERSION = 1`.
- **Dependencies:** after any `npm install`, run `npm run lint && npm run typecheck && npm test`. If npm reports "removed N" or "changed N" packages, stop and investigate.
- Commit after every task. Push only with the user's go-ahead.

## Review Focus

1. **Token leakage:** after a full session with a token set, the token string appears in no IndexedDB record, no submitted file content, no request body, and no request to any host other than `api.github.com`. Test in Task 6.
2. **Two tabs on one machine:** each gets its own `learnerId`, branch and PR, and they never write the same file. Test in Task 5.
3. **Reload mid-session:** the same tab keeps its `learnerId` and continues its `seq` numbering, without overwriting an earlier log file. Test in Task 5.
4. **GitHub failures:** a bad token (401), rate limiting (403/429), or being offline leaves the queue intact, retries with backoff, and shows a plain status message on the page. Test in Task 6.
5. **No token at all:** the session runs to completion, its logs queue, and they are all submitted once a token is pasted, in `seq` order. Test in Task 6.

---

## File structure

```
package.json  tsconfig.base.json  eslint.config.js  vitest.config.ts  playwright.config.ts
.github/workflows/ci.yml          lint, typecheck, unit tests, e2e smoke
.github/workflows/validate.yml    on pull_request: read-only validation of session PRs
.github/workflows/gate.yml        hourly + manual: merge, rebuild summary, deploy (calls deploy.yml)
.github/workflows/deploy.yml      workflow_call + push to main: build app, publish Pages
tools/walls.test.ts
tools/gate/{validate,validate-cli,gate,summary,github}.ts  + tools/gate/test/
knowledge/summary.json            starts as {"schemaVersion":1,"sessions":[]}
knowledge/settings-space.json     placeholder space (see Task 7)
data/runs/.gitkeep
packages/interface/src/...        observation/action contract
packages/world/src/{index,world,crib}.ts            tiny one-room world
packages/learner/src/{index,flailer}.ts            placeholder random learner
packages/harness/src/{index,log,session,loop,queue,identity,contributor,token,github,knowledge}.ts
packages/tensor/src/index.ts      `export {}` (filled by Plan 02)
app/{index.html,vite.config.ts}  app/src/{main,worker,ui}.ts
```

---

### Task 1: Repo scaffold, walls and CI

**Files:**
- Create: `package.json`, `tsconfig.base.json`, `eslint.config.js`, `vitest.config.ts`, `.github/workflows/ci.yml`, `tools/walls.test.ts`, and for each of `interface`, `tensor`, `learner`, `world`, `harness`: `packages/<p>/package.json` (name `@blanktab/<p>`, `"type": "module"`, `"main": "src/index.ts"`), `packages/<p>/tsconfig.json`, `packages/<p>/src/index.ts` containing `export {}`.

**Interfaces:**
- Produces: root scripts `lint`, `typecheck`, `test` (`vitest run`). Package names `@blanktab/interface|tensor|learner|world|harness`.

- [ ] **Step 1: Write the failing wall test** in `tools/walls.test.ts`

```ts
import { ESLint } from "eslint"
import { describe, it, expect } from "vitest"
const eslint = new ESLint()
async function errors(code: string, filePath: string) {
  const [r] = await eslint.lintText(code, { filePath })
  return r.messages.filter(m => m.ruleId === "no-restricted-imports")
}
describe("package walls", () => {
  it("learner cannot import world or harness", async () => {
    expect(await errors(`import "@blanktab/world"`, "packages/learner/src/x.ts")).toHaveLength(1)
    expect(await errors(`import "@blanktab/harness"`, "packages/learner/src/x.ts")).toHaveLength(1)
    expect(await errors(`import "../../world/src/index"`, "packages/learner/src/x.ts")).toHaveLength(1)
  })
  it("world cannot import learner, tensor or harness", async () => {
    for (const p of ["learner", "tensor", "harness"])
      expect(await errors(`import "@blanktab/${p}"`, "packages/world/src/x.ts")).toHaveLength(1)
  })
  it("tensor and interface import no workspace package", async () => {
    expect(await errors(`import "@blanktab/interface"`, "packages/tensor/src/x.ts")).toHaveLength(1)
    expect(await errors(`import "@blanktab/tensor"`, "packages/interface/src/x.ts")).toHaveLength(1)
  })
  it("allowed imports pass", async () => {
    expect(await errors(`import "@blanktab/tensor"; import "@blanktab/interface"`, "packages/learner/src/x.ts")).toHaveLength(0)
    expect(await errors(`import "@blanktab/world"; import "@blanktab/learner"`, "packages/harness/src/x.ts")).toHaveLength(0)
  })
})
```

- [ ] **Step 2: Run it to verify it fails.** Run `npm install` if needed, then `npx vitest run tools/walls.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement the configs.** In `eslint.config.js`, use per-package `files` globs with `no-restricted-imports` `patterns`. The patterns block `@blanktab/<p>` and relative `**/<p>/**` paths for each forbidden package. Use `typescript-eslint` to parse TS. `ci.yml` runs on push and PR with Node 22: `npm ci`, `npm run lint`, `npm run typecheck`, `npm test`.
- [ ] **Step 4: Verify.** Run: `npm run lint && npm run typecheck && npm test`. Expected: all pass.
- [ ] **Step 5: Commit.** This is the repo's first commit, so include the existing `docs/`, `AGENTS.md`, `CLAUDE.md`, `README.md` and `.gitignore`.

```bash
git add -A && git commit -m "chore: scaffold monorepo with enforced package walls and CI"
```

---

### Task 2: The learner↔world contract (`@blanktab/interface`)

**Files:**
- Create: `packages/interface/src/{constants,action,observation,index}.ts`; `packages/interface/test/interface.test.ts`

**Interfaces:**
- Produces (exact):

```ts
export const SCHEMA_VERSION = 1
export const WORLD = { width: 320, height: 200 } as const
export const PERIPHERY = { width: 80, height: 50, channels: 3 } as const   // RGB, HWC, uint8
export const FOVEA = { width: 32, height: 32, channels: 3 } as const
export const PRED_PERIPHERY = { width: 40, height: 25, channels: 3 } as const
export const PRED_FOVEA = { width: 16, height: 16, channels: 3 } as const
export const TICKS_PER_WORLD_SECOND = 10
export const MOVE_MAGNITUDES = [1, 4, 16] as const
export const MOVE_OPTIONS = 25      // 0 none; 1..24 = 8 directions × 3 magnitudes
export const BUTTON_OPTIONS = 3     // 0 none, 1 press, 2 release
export const SCROLL_OPTIONS = 3     // 0 none, 1 up, 2 down
export const KEYS: readonly string[]  // 45: "", "a".."z", "0".."9", " ", "Enter", "Backspace", "Tab", "ArrowLeft", "ArrowUp", "ArrowRight", "ArrowDown"
export interface Action { move: number; button: number; scroll: number; key: number }
export const NOOP_ACTION: Action
export function isValidAction(a: Action): boolean
export function decodeMove(move: number): { dx: number; dy: number }
  // directions E, NE, N, NW, W, SW, S, SE (screen y grows downward: N is dy<0); index = 1 + dir*3 + magIndex
export interface Observation {
  periphery: Uint8Array; fovea: Uint8Array
  cursor: { x: number; y: number }   // normalized 0..1
  buttonHeld: boolean
}
export function emptyObservation(): Observation
// The learner-facing world surface. Nothing privileged crosses it.
export interface WorldView { observe(): Observation; act(a: Action): void }
// The learner surface the harness drives.
export interface Learner { step(o: Observation): Action }
```

- [ ] **Step 1: Write the failing tests**

```ts
it("action space sizes", () => { expect(KEYS).toHaveLength(45); expect(KEYS[0]).toBe(""); expect(MOVE_OPTIONS).toBe(25) })
it("decodeMove", () => {
  expect(decodeMove(0)).toEqual({ dx: 0, dy: 0 })
  expect(decodeMove(1)).toEqual({ dx: 1, dy: 0 })
  expect(decodeMove(3)).toEqual({ dx: 16, dy: 0 })
  expect(decodeMove(1 + 2*3 + 1)).toEqual({ dx: 0, dy: -4 })
  expect(decodeMove(1 + 1*3 + 0)).toEqual({ dx: 1, dy: -1 })
})
it("isValidAction", () => {
  expect(isValidAction(NOOP_ACTION)).toBe(true)
  expect(isValidAction({ move: 25, button: 0, scroll: 0, key: 0 })).toBe(false)
  expect(isValidAction({ move: 1.5, button: 0, scroll: 0, key: 0 })).toBe(false)
  expect(isValidAction({ move: 0, button: 0, scroll: 0, key: 44 })).toBe(true)
})
it("observation carries pixels only, no clock", () => {
  const o = emptyObservation()
  expect(o.periphery.length).toBe(80*50*3); expect(o.fovea.length).toBe(32*32*3); expect("tick" in o).toBe(false)
})
```

- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run packages/interface`. Expected: FAIL.
- [ ] **Step 3: Implement** the exports above.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `npx vitest run packages/interface && npm run lint`. Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(interface): learner-world contract"`

---

### Task 3: A one-room world and a flailing baby

**Files:**
- Create: `packages/world/src/{world,crib,index}.ts`, `packages/world/test/world.test.ts`; `packages/learner/src/{flailer,index}.ts`, `packages/learner/test/flailer.test.ts`

**Interfaces:**
- Consumes: `@blanktab/interface` (Task 2).
- Produces:

```ts
// world
export interface TraceEvent { tick: number; kind: "hover" | "click" | "respond"; elementId: string }
export interface World extends WorldView {
  readonly tick: number
  drainTrace(): TraceEvent[]          // privileged; harness only
}
export function createWorld(opts: { seed: number; canvas: OffscreenCanvas }): World
// learner
export function createFlailer(opts: { seed: number; moveProbability: number }): Learner
```

- **The crib:** 6–10 seeded shapes (circles and squares, random colours) on a plain background, plus a 16 px taskbar strip at the bottom with one icon. A shape under the cursor is drawn brighter. Pressing on a shape flips its colour (a "respond" event).
- **The cursor** is drawn into the frame as a small arrow. It starts at the centre and is clamped to the screen.
- **Each `act` advances `tick` by 1. Every `observe()` call:** renders with Canvas2D on the `OffscreenCanvas`, reads it back with `getImageData`, box-downsamples 4× to the periphery, and crops 32×32 around the cursor (edges padded black) for the fovea.
- **`createFlailer`:** on each step it moves with probability `moveProbability`, choosing a uniform random move, and presses or releases with probability 0.05. It never scrolls or types. It ignores its observation; it is a placeholder.
- **Tests use the `canvas` npm package as a stand-in for `OffscreenCanvas` in Node,** or run in Vitest browser mode, whichever the installed versions support. Record which in the test file header.

- [ ] **Step 1: Write the failing tests:**
  - `createWorld({seed:1})` twice gives identical first observations, and seed 2 differs;
  - after `act({move: <E,16>})`, `observe().cursor.x` increases by `16/320`;
  - moving the cursor onto a known shape and pressing produces a `respond` event in `drainTrace()`, and the shape's pixels change in the next periphery frame;
  - `drainTrace()` empties after reading;
  - the flailer is deterministic for a seed, always returns `isValidAction` actions, and with `moveProbability: 0` returns only `move: 0`.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run packages/world packages/learner`. Expected: FAIL.
- [ ] **Step 3: Implement** the world, the crib and the flailer.
- [ ] **Step 4: Run the tests to verify they pass.** Run: `npx vitest run packages/world packages/learner && npm run lint`. Expected: PASS. The walls hold.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat: one-room crib world and placeholder flailing learner"`

---

### Task 4: The session log format and the gate's validator

The log format is shared by the tab, which writes it, and the Action, which checks it. Defining both together keeps them consistent.

**Files:**
- Create: `packages/harness/src/log.ts` (format + parser; browser-safe, no Node imports), `tools/gate/validate.ts` (imports `parseLogFile` from `@blanktab/harness`), `tools/gate/test/validate.test.ts`

**Interfaces:**
- Produces:

```ts
export interface LogFile {
  schemaVersion: 1
  learnerId: string          // "lrn_" + 16 hex chars
  seq: number                // 0,1,2,... per learner
  codeSha: string            // commit the page was built from
  knowledgeSha: string       // sha256 hex of the prior files the session loaded (spec §5.1)
  writtenAt: string          // ISO time
  stepsTotal: number         // cumulative, never decreases across seq
  settings: Record<string, number | string>
  hostSig: { cores: number; deviceMemoryGb: number | null; webgpu: boolean; tabsOnMachine: number }
  sessionsSeenAtBirth: number
  final: boolean
}
export function parseLogFile(json: unknown): { ok: true; value: LogFile } | { ok: false; error: string }
export interface PrFile { path: string; content: string }
export function validateSessionPr(files: PrFile[], previous: LogFile[]): { ok: boolean; errors: string[]; kind: "log" | "needs-human" }
  // previous = this learner's files already on main
```

- **Rules (spec §9.5):**
  - every path matches `^data/runs/(lrn_[0-9a-f]{16})/(\d+\.json|heartbeat\.json)$`, all for the same `learnerId`, and that id matches each file's `learnerId`;
  - any other path → `kind: "needs-human"`;
  - each file is ≤ 1 MB;
  - `seq` values are unique and continue from `previous`;
  - `stepsTotal` never decreases;
  - all numbers are finite;
  - at most one file has `final: true`, and it has the highest `seq`.
- `heartbeat.json` is `{ learnerId, writtenAt }`.

- [ ] **Step 1: Write the failing tests:** one passing PR with seq 0–2; and one rejected case per rule above, each asserting that the specific error text is present. One case: a PR adding `app/src/main.ts` gives `kind: "needs-human"`.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run tools/gate`. Expected: FAIL.
- [ ] **Step 3: Implement** `parseLogFile` (hand-written checks, no schema library) and `validateSessionPr`.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(gate): session log format and validator"`

---

### Task 5: The harness — session, identity, loop, log queue

**Files:**
- Create: `packages/harness/src/{identity,session,loop,queue,index}.ts` (`log.ts` exists from Task 4), `packages/harness/test/{identity,session,queue}.test.ts`

**Interfaces:**
- Consumes: `createWorld`, `createFlailer`, `LogFile`, `parseLogFile`.
- Produces:

```ts
export function getOrCreateLearnerId(session: Storage, rng?: () => number): string   // sessionStorage key "blanktab:learnerId"
export interface Queue {                   // IndexedDB database "blanktab", store "outbox"
  put(f: LogFile): Promise<void>; pending(): Promise<LogFile[]>   // ordered by learnerId, seq
  markSent(learnerId: string, seq: number): Promise<void>
}
export function openQueue(idb: IDBFactory): Promise<Queue>
export interface SessionConfig { learnerId: string; codeSha: string; knowledgeSha: string; sessionsSeenAtBirth: number; settings: Record<string, number | string>; hostSig: LogFile["hostSig"]; lifetimeSteps: number; logEverySteps: number; startSeq: number; startSteps: number }
export function runSession(cfg: SessionConfig, deps: { world: World; learner: Learner; queue: Queue; now: () => Date; onTick?: (s: { steps: number; obs: Observation }) => void }): Promise<void>
  // loops world/learner; every logEverySteps writes the next LogFile (seq++) to the queue; at lifetimeSteps writes a final:true file and resolves
export function resumePoint(queue: Queue, learnerId: string): Promise<{ startSeq: number; startSteps: number }>
```

- **Placeholder values:** `lifetimeSteps = 3000`, `logEverySteps = 500`. The logger yields to the event loop every 50 steps so the UI stays live.

- [ ] **Step 1: Write the failing tests.** Use `fake-indexeddb` and a stub `Storage`:
  - a full session writes exactly 6 files, one every 500 steps: seq 0–4, plus seq 5 at step 3000 with `final: true` and `stepsTotal: 3000`;
  - **Review Focus 3 (reload):** the same storage returns the same `learnerId`. After interrupting at step 1200 and calling `resumePoint`, a resumed session continues from seq 2 with no duplicate seq;
  - **Review Focus 2 (two tabs):** two separate `Storage` stubs give two different `learnerId`s, and running both sessions against one queue yields no file with the same `(learnerId, seq)`;
  - every queued file passes `parseLogFile`.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run packages/harness`. Expected: FAIL.
- [ ] **Step 3: Implement** the identity, queue, session and resume functions.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(harness): session loop, learner identity, IndexedDB log queue"`

---

### Task 6: Writing back to the repo — token and contributor

**Files:**
- Create: `packages/harness/src/{token,contributor,github}.ts`, `packages/harness/test/{token,contributor}.test.ts`

**Interfaces:**
- Produces:

```ts
export const REPO = { owner: "markwhiting", name: "blanktab" } as const
export function getToken(ls: Storage): string | null; export function setToken(ls: Storage, t: string): void; export function clearToken(ls: Storage): void
  // key "blanktab:githubToken"
export function checkToken(token: string, fetchFn: typeof fetch): Promise<{ ok: true; login: string; canPush: boolean } | { ok: false; reason: string }>
  // GET /user, then GET /repos/markwhiting/blanktab → permissions.push
export interface ContributorStatus { state: "no-token" | "idle" | "sending" | "error"; message: string; pending: number }
export function createContributor(deps: { queue: Queue; ls: Storage; fetchFn: typeof fetch; now: () => Date; onStatus: (s: ContributorStatus) => void }): { flush(): Promise<void> }
```

**`flush()` behaviour:**
1. **Pick the destination:** if the user `canPush`, use upstream `markwhiting/blanktab`. Otherwise use their fork; create it with `POST /repos/markwhiting/blanktab/forks` if missing, then poll until it exists.
2. **Create the branch** `run/<learnerId>` from upstream `main`'s SHA if it doesn't exist.
3. **Send each pending file in `seq` order.** `PUT /repos/{dest}/contents/data/runs/<learnerId>/<seq>.json` on that branch. Then update `heartbeat.json`. Then call `markSent`.
4. **Open one draft PR** titled `run: <learnerId>` to upstream `main`, if none exists yet.

**On errors:**
- 401 → state `error`, message "Your GitHub token was rejected — paste a new one."
- 403/429 → back off, honouring `Retry-After` or `X-RateLimit-Reset`.
- Network failure → retry with exponential backoff (5 s doubling to 10 min).
- In every case, nothing is removed from the queue unless it was sent.

All requests go to `https://api.github.com` with `Authorization: Bearer <token>` and `X-GitHub-Api-Version: 2022-11-28`.

- [ ] **Step 1: Write the failing tests.** Use a fake `fetchFn` that records every request and simulates GitHub's responses:
  - the push-access path creates the branch, puts 6 files plus the heartbeat, and opens 1 draft PR;
  - the no-push path creates a fork first;
  - a second `flush` with nothing pending makes no requests;
  - **Review Focus 4:** a 401 leaves `pending` unchanged and shows the rejection message; a 403 with `Retry-After: 30` schedules a retry ≥ 30 s; offline (fetch throws) retries;
  - **Review Focus 5:** queue 6 files with no token, so the state is `no-token`. Call `setToken`, then `flush`, and all 6 are sent in seq order;
  - **Review Focus 1:** across all recorded requests, every URL host is `api.github.com`, and no request body contains the token string. After a full session, no IndexedDB record contains the token string.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run packages/harness/test/token.test.ts packages/harness/test/contributor.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the token functions, `checkToken` and `createContributor`.
- [ ] **Step 4: Run the tests to verify they pass.** Same command. Expected: PASS.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(harness): token onboarding helpers and repo contributor"`

---

### Task 7: The repo side — gate, summary, Pages deploy

**Files:**
- Create: `tools/gate/{github,gate,summary,validate-cli}.ts`, `tools/gate/test/{gate,summary}.test.ts`, `.github/workflows/{validate,gate,deploy}.yml`, `knowledge/summary.json`, `knowledge/settings-space.json`, `data/runs/.gitkeep`

**Interfaces:**
- Consumes: `validateSessionPr`, `parseLogFile` (Task 4).
- Produces:

```ts
// summary.ts
export interface SessionRow { learnerId: string; settings: LogFile["settings"]; hostSig: LogFile["hostSig"]; stepsTotal: number; final: boolean; codeSha: string; knowledgeSha: string; sessionsSeenAtBirth: number }
export function buildSummary(runsDir: string): { schemaVersion: 1; sessions: SessionRow[] }   // one row per learner, from its highest-seq file
// gate.ts
export interface OpenPr { number: number; headRef: string; files: PrFile[]; previous: LogFile[]; heartbeatAt: Date | null }
export type GateDecision = { pr: number; action: "merge" | "wait" | "close" | "label-needs-human"; reason: string }
export function decide(prs: OpenPr[], now: Date): GateDecision[]
  // invalid → close (reason = validator errors); needs-human → label; valid and (has final file OR heartbeat older than 48 h) → merge; otherwise wait
```

- **`knowledge/settings-space.json`:** `{"schemaVersion":1,"settings":[{"name":"moveProbability","min":0.1,"max":1}]}`, the placeholder setting the flailer uses.
- **`knowledgeSha`:** the sha256 of `settings-space.json`, plus `consolidation-prior.json` and `caregiver-prior.json` once they exist (spec §5.1). Computed at build time and published as `knowledge/manifest.json` `{ knowledgeSha, codeSha }`.

**Workflows:**
- **`validate.yml`** (`on: pull_request`, `permissions: contents: read`):
  1. Check out `main`.
  2. Fetch the PR's changed files through the API as data.
  3. Run `npx tsx tools/gate/validate-cli.ts`, which fails the check with the errors.
- **`gate.yml`** (`schedule: hourly` + `workflow_dispatch`; `permissions: contents: write, pull-requests: write, pages: write, id-token: write`):
  1. Run `tools/gate/gate.ts` to list open PRs titled `run: *`, decide, and act. Merging uses `gh pr merge --squash`.
  2. Run `buildSummary`.
  3. Commit `knowledge/summary.json` if it changed.
  4. Call `deploy.yml` through `workflow_call`. Merges made with `GITHUB_TOKEN` don't trigger other workflows, so the gate must call deploy itself.
- **`deploy.yml`** (`on: push to main` + `workflow_call`):
  1. Build `app/` with `CODE_SHA=${{ github.sha }}`.
  2. Copy `knowledge/` into `dist/knowledge/`.
  3. Publish with `actions/upload-pages-artifact` + `actions/deploy-pages`.

- [ ] **Step 1: Write the failing tests** for `decide`:
  - valid + final → merge;
  - valid, no final, heartbeat 47 h old → wait;
  - heartbeat 49 h old → merge;
  - invalid → close, with the validator error in the reason;
  - code path → `label-needs-human`.

  For `buildSummary`, use a temp dir with two learners, one with seq 0–5 and final, one with seq 0–1. Expect 2 rows with the right `stepsTotal` and `final`.
- [ ] **Step 2: Run the tests to verify they fail.** Run: `npx vitest run tools/gate`. Expected: FAIL.
- [ ] **Step 3: Implement** `summary.ts`, `gate.ts`, `github.ts` (thin `gh api` wrappers), `validate-cli.ts` and the three workflows.
- [ ] **Step 4: Verify.** Run: `npx vitest run tools/gate && npx actionlint` (install actionlint if missing). Expected: PASS, and actionlint reports no errors.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(gate): merge rules, summary builder, validate/gate/deploy workflows"`

---

### Task 8: The page — and closing the loop

**Files:**
- Create: `app/index.html`, `app/vite.config.ts`, `app/src/{main,worker,ui}.ts`, `packages/harness/src/knowledge.ts`, `playwright.config.ts`, `tests/e2e/loop.spec.ts`

**Interfaces:**
- Consumes: everything above.
- Produces:

```ts
// knowledge.ts
export interface Knowledge { summary: { schemaVersion: 1; sessions: SessionRow[] }; manifest: { knowledgeSha: string; codeSha: string }; settingsSpace: { settings: { name: string; min: number; max: number }[] } }
export function loadKnowledge(baseUrl: string, fetchFn: typeof fetch): Promise<Knowledge>   // fetches `${baseUrl}knowledge/*.json`
export function chooseSettings(k: Knowledge, rng: () => number): Record<string, number>   // placeholder: uniform within each range
```

**Page behaviour:**
1. **Load knowledge.** If it's unreachable, fall back to an empty summary and show "offline".
2. **Get the learner identity**, resuming if this tab already has one.
3. **Start the worker.** The worker owns the `OffscreenCanvas`, world, learner and session.
4. **Show:**
   - **what the learner sees:** the periphery scaled up, and the fovea beside it;
   - **steps lived** out of 3000;
   - **sessions before this one:** `summary.sessions.length`, the visible proof the loop is closed;
   - **the chosen setting;**
   - **contribution status** from `ContributorStatus`;
   - **if there's no token:** a short how-to with a link to `https://github.com/settings/tokens/new?scopes=public_repo&description=blanktab`, a paste field, and a "Check & save" button. The button calls `checkToken` before `setToken`.
5. **Flush the contributor** after each log file is queued and every 60 s.
6. **When the session goes final:** show "This learner's life is complete", then start a newborn (new `learnerId`).

`codeSha` comes from Vite `define: { __CODE_SHA__: JSON.stringify(process.env.CODE_SHA ?? "dev") }`.

- [ ] **Step 1: Write the failing e2e test** in `tests/e2e/loop.spec.ts`, run against `vite preview` with the GitHub API mocked through Playwright `page.route("https://api.github.com/**")` and a fixture `knowledge/summary.json` holding 2 sessions:
  - the page shows "2 sessions before this one";
  - the steps counter increases;
  - with no token, the token panel is visible. After pasting `test-token` (where the mock `/user` returns 200 with push access), the panel hides;
  - running with a test-only `lifetimeSteps=600` (query param `?testLifetime=600`, honoured only in non-production builds) produces requests creating `data/runs/<id>/0.json` and `data/runs/<id>/1.json` (the final file) and one `POST /repos/markwhiting/blanktab/pulls`;
  - no request goes to any host other than `api.github.com` and the page's own origin (Review Focus 1).
- [ ] **Step 2: Run it to verify it fails.** Run: `npx playwright test tests/e2e/loop.spec.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the page, the worker wiring and `knowledge.ts`. Add `npx playwright test` to `ci.yml`.
- [ ] **Step 4: Verify locally.** Run: `npm run lint && npm run typecheck && npm test && npx playwright test`. Expected: all pass. Then run `npm run dev` and look at the page yourself: the baby's view updates, the cursor visibly flails, and shapes flash when clicked.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(app): blanktab page; loop closed end to end against mocked GitHub"`

---

### Task 9: Go live and verify the real round trip

Every step here is outward-facing. **Get the user's go-ahead before step 1, and report to them after step 5.**

- [ ] **Step 1: Create the repo and push.** Run `gh repo create markwhiting/blanktab --public --source . --push`.
- [ ] **Step 2: Enable Pages.** Settings → Pages → Source: "GitHub Actions". Add branch protection on `main` requiring the `validate` check. Allow the gate's `GITHUB_TOKEN` to merge (Settings → Actions → Workflow permissions: read and write; allow GitHub Actions to create and approve pull requests).
- [ ] **Step 3: Confirm the deploy.** Wait for `deploy.yml` to finish, then open `https://markwhiting.github.io/blanktab/`. Expected: the page loads and shows "0 sessions before this one".
- [ ] **Step 4: Run a real session.** Paste a real token with `public_repo` scope and let one session run to completion (3000 steps). Expected: a draft PR `run: lrn_…` appears in `markwhiting/blanktab` with 6 files plus `heartbeat.json`, and `validate` passes.
- [ ] **Step 5: Close the loop.** Trigger `gate.yml` manually. Expected: the PR is merged, `knowledge/summary.json` on `main` has 1 session, Pages redeploys, and reloading the page shows **"1 session before this one"**.
- [ ] **Step 6: Record the result** in `docs/research/<date>-loop-live.md`: what happened, timings, and any surprises, such as token type or fork behaviour (this is spike 1's answer). Commit and push with the user's go-ahead.

---

## Self-review notes

- **Spec coverage:**
  - §3 overview and §4 layout and walls: Tasks 1 and 3.
  - §5.2–5.3 observation and action: Tasks 2 and 3.
  - §9.3 onboarding and the token rule: Tasks 6 and 8.
  - §9.4 log stream: Tasks 5 and 6.
  - §9.5 gate (log PRs) and deploy: Task 7.
  - §10 identity, multi-tab and lifetime (simplified to a fixed step count): Task 5.
  - §12 CI: Task 1.
- **Deliberately left to later plans:**
  - the real learner, the tensor library, rooms beyond the crib, sleep, checkpoints and Web Lock adoption, the caregiver, milestones, checkups, knowledge PRs and `best.json`, and the GP picker.
  - Each of these replaces a placeholder named in "In plain terms" without changing the loop.
