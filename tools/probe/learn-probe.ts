// Offline diagnostic probe: the real learner (CPU backend) in the real default four-room world (Node canvas).
// Research instrumentation only (docs/research/2026-10-21-why-not-learning.md); not part of the product.
//
//   npx tsx tools/probe/learn-probe.ts --steps 3000 --seed 1 --widths cpu --settings live --log 250
//
// Per step it measures, at full precision and split by output, the prediction loss *before* the online update
// (on a shadow copy of the learner's pre-step weights, so the learner itself is untouched) against copy-last,
// the gradient norm reaching the residual output layers, how often the target changed at all, and the action
// distribution. The shadow total is checked against the learner's own UpdateInfo.lossBefore every step.
import { createCanvas } from "@napi-rs/canvas"
import { PRED_FOVEA, PRED_PERIPHERY, decodeMove, type Action, type Observation } from "@blanktab/interface"
import {
  CPU_WIDTHS, DEFAULT_SETTINGS, GPU_WIDTHS, actionOneHot, createLearner, createModel, obsToInputs, targetsFromObs,
  type LearnerSettings, type Targets, type UpdateInfo, type Widths,
} from "@blanktab/learner"
import { add, backward, composite, createCpuBackend, disposeGraph, linear, mse, noGrad, scale, tensor } from "@blanktab/tensor"
import { createWorld } from "@blanktab/world"

const args = new Map<string, string>()
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i]!.replace(/^--/, ""), process.argv[i + 1] ?? "")
const num = (k: string, d: number) => (args.has(k) ? Number(args.get(k)) : d)
const STEPS = num("steps", 3000)
const SEED = num("seed", 1)
const LOG = num("log", 250)
const SLEEP_EVERY = num("sleepEvery", 250) // harness DEFAULT_SLEEP_INTERVAL
const NO_SLEEP = args.get("sleep") === "off"
const SHADOW_EVERY = num("shadowEvery", 1) // shadow split measurement every k steps (1 = every step)
const TINY: Widths = { conv1: 4, conv2: 4, enc: 16, hidden: 16 }
const widths: Widths = { cpu: CPU_WIDTHS, gpu: GPU_WIDTHS, tiny: TINY }[args.get("widths") ?? "cpu"]!
const LIVE: LearnerSettings = { lr: 0.002, headLrScale: 0.086, entropyWeight: 0.025, curiosityWeight: 1.32, bufferSize: 1491, replayPerSleep: 48 }
const settings: LearnerSettings = { ...(args.get("settings") === "default" ? DEFAULT_SETTINGS : LIVE) }
for (const k of Object.keys(settings) as (keyof LearnerSettings)[]) if (args.has(k)) settings[k] = Number(args.get(k))
const JSON_OUT = args.get("json")

const PP = PRED_PERIPHERY.width * PRED_PERIPHERY.height * PRED_PERIPHERY.channels
const PF = PRED_FOVEA.width * PRED_FOVEA.height * PRED_FOVEA.channels
const W_P = PP / (PP + PF), W_F = PF / (PP + PF)

const sq = (a: Float32Array, b: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i]! - b[i]!) ** 2; return s / a.length }
const changed = (a: Float32Array, b: Float32Array) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return true; return false }
const norm = (a: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]! * a[i]!; return Math.sqrt(s) }

/**
 * Oracle (computed by the probe from what the learner itself sees: the two native fovea frames' predecessor and
 * the cursor move): the previous native 32x32 fovea shifted by the cursor's move, with the cursor sprite's box
 * (fixed at the fovea centre) kept in place, then 2x box-downsampled like the target. Uncovered pixels fall back
 * to copy-last. It knows "the fovea is a window centred on the cursor" and nothing else: a measure of how much
 * of the copy-last error is pure fovea translation.
 */
function shiftedFovea(prevNative: Uint8Array, dxPx: number, dyPx: number): Float32Array {
  const N = 32, C = 3
  const src = prevNative
  const sh = new Uint8Array(src.length)
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const inCursor = x >= 16 && x <= 23 && y >= 16 && y <= 27
    const xs = inCursor ? x : x + dxPx, ys = inCursor ? y : y + dyPx
    const ok = xs >= 0 && ys >= 0 && xs < N && ys < N
    for (let c = 0; c < C; c++) sh[(y * N + x) * C + c] = ok ? src[(ys * N + xs) * C + c]! : src[(y * N + x) * C + c]!
  }
  const out = new Float32Array(16 * 16 * C)
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) for (let c = 0; c < C; c++) {
    const at = (yy: number, xx: number) => sh[(yy * N + xx) * C + c]!
    out[(y * 16 + x) * C + c] = (at(2 * y, 2 * x) + at(2 * y, 2 * x + 1) + at(2 * y + 1, 2 * x) + at(2 * y + 1, 2 * x + 1)) / (4 * 255)
  }
  return out
}

class Acc {
  n = 0; s: Record<string, number> = {}
  add(k: string, v: number) { this.s[k] = (this.s[k] ?? 0) + v }
  mean(k: string) { return (this.s[k] ?? 0) / Math.max(1, this.n) }
}

async function main() {
  const backend = createCpuBackend()
  const shadowBackend = createCpuBackend() // shadow work is not billed to the learner's CU
  const canvas = createCanvas(320, 200)
  const world = createWorld({ seed: SEED, canvas })
  const got: { info: UpdateInfo | null } = { info: null }
  const lastInfo = (): UpdateInfo | null => got.info
  const learner = await createLearner({ backend, seed: SEED, widths, settings, onUpdate: (u) => { got.info = u } })

  let win = new Acc()
  const all = new Acc()
  const rows: Record<string, number>[] = []
  const moveHist = new Map<string, number>()
  let maxCheckErr = 0
  let prevObs: Observation | null = null
  let prevAction: Action | null = null
  let prevTargets: Targets | null = null
  const t0 = Date.now()

  for (let step = 1; step <= STEPS; step++) {
    const obs = world.observe()
    const y = targetsFromObs(obs)
    const doShadow = prevObs !== null && step % SHADOW_EVERY === 0
    const pre = doShadow ? await learner.readParams() : null
    got.info = null
    const action = await learner.step(obs)
    world.act(action)

    if (prevObs && prevTargets && prevAction) {
      const tr = learner.replay.at(learner.replay.size - 1) // (obs_{t-1}, h_{t-2}, a_{t-1}, obs_t), just learned from
      if (tr.next.fovea !== obs.fovea && !(tr.next.fovea.every((v, i) => v === obs.fovea[i]))) throw new Error("replay misaligned")
      const clP = sq(prevTargets.periphery, y.periphery), clF = sq(prevTargets.fovea, y.fovea)
      const { dx, dy } = decodeMove(prevAction.move)
      const realDx = Math.round((obs.cursor.x - prevObs.cursor.x) * 320), realDy = Math.round((obs.cursor.y - prevObs.cursor.y) * 200)
      const shF = sq(shiftedFovea(prevObs.fovea, realDx, realDy), y.fovea)
      win.n++; all.n++
      for (const a of [win, all]) {
        a.add("clP", clP); a.add("clF", clF); a.add("cl", W_P * clP + W_F * clF)
        a.add("shF", shF); a.add("shiftOracle", W_P * clP + W_F * shF)
        a.add("chP", changed(prevTargets.periphery, y.periphery) ? 1 : 0)
        a.add("chF", changed(prevTargets.fovea, y.fovea) ? 1 : 0)
        a.add("moved", realDx !== 0 || realDy !== 0 ? 1 : 0)
        a.add("move0", prevAction.move === 0 ? 1 : 0)
        a.add("btn", prevAction.button !== 0 ? 1 : 0)
        a.add("key", prevAction.key !== 0 ? 1 : 0)
        a.add("scroll", prevAction.scroll !== 0 ? 1 : 0)
        a.add("lb", lastInfo()?.lossBefore ?? NaN)
      }
      const mag = dx === 0 && dy === 0 ? "0" : String(Math.max(Math.abs(dx), Math.abs(dy)))
      moveHist.set(mag, (moveHist.get(mag) ?? 0) + 1)

      if (pre) {
        // Shadow: the learner's pre-step weights, on the transition it just learned from.
        const m = createModel(shadowBackend, 0, widths, pre)
        const T = (d: Float32Array, s: number[]) => tensor(d, s, { backend: shadowBackend })
        const inp = obsToInputs(tr.obs)
        const it = { periphery: T(inp.periphery, [3, 50, 80]), fovea: T(inp.fovea, [3, 32, 32]), proprio: T(inp.proprio, [1, 3]) }
        const base = targetsFromObs(tr.obs)
        const bT = { periphery: T(base.periphery, [1, PP]), fovea: T(base.fovea, [1, PF]) }
        const yT = { periphery: T(y.periphery, [1, PP]), fovea: T(y.fovea, [1, PF]) }
        const hT = T(tr.hIn, [1, widths.hidden])
        const aT = T(actionOneHot(tr.action), [1, 76])
        const hNow = m.core(it, hT)
        const p = m.predict(hNow, aT, bT)
        // The predictor's hidden layer z = relu(linear(h) + linear(a)): how many units are alive on this transition.
        const q = m.params.length
        const zPre = noGrad(() => composite(() => add(linear(hNow, m.params[q - 8]!, m.params[q - 7]!), linear(aT, m.params[q - 6]!, m.params[q - 5]!))))
        const zv = await zPre.read()
        zPre.dispose()
        let alive = 0
        for (const v of zv) if (v > 0) alive++
        const lP = mse(p.periphery, yT.periphery), lF = mse(p.fovea, yT.fovea)
        const loss = add(scale(lP, W_P), scale(lF, W_F))
        const [vP, vF, vL] = await Promise.all([lP.read(), lF.read(), loss.read()])
        backward(loss)
        const [gPw, , gFw] = await Promise.all(m.outputParams.map((q) => (q.grad ? q.grad.read() : Promise.resolve(new Float32Array(1)))))
        const predHw = m.params[m.params.length - 8]!
        const gH = predHw.grad ? await predHw.grad.read() : new Float32Array(1)
        const lb = lastInfo()?.lossBefore ?? NaN
        maxCheckErr = Math.max(maxCheckErr, Math.abs(vL[0]! - lb) / Math.max(1e-12, lb))
        const nP = m.params.length
        for (const a of [win, all]) {
          a.add("sn", 1)
          a.add("lP", vP[0]!); a.add("lF", vF[0]!); a.add("l", vL[0]!)
          a.add("clPs", clP); a.add("clFs", clF)
          a.add("zAlive", alive / zv.length)
          a.add("gHeadP", norm(gPw!)); a.add("gHeadF", norm(gFw!)); a.add("gPredH", norm(gH))
        }
        win.s.wHeadP = norm(pre[nP - 4]!); win.s.wHeadF = norm(pre[nP - 2]!)
        win.s.bHeadP = norm(pre[nP - 3]!); win.s.bHeadF = norm(pre[nP - 1]!)
        disposeGraph(loss, [])
        for (const x of [it.periphery, it.fovea, it.proprio, bT.periphery, bT.fovea, yT.periphery, yT.fovea, hT, aT]) x.dispose()
        m.dispose()
      }
    }
    prevObs = obs; prevTargets = y; prevAction = action

    if (!NO_SLEEP && step % SLEEP_EVERY === 0) await learner.sleep()
    if (step % LOG === 0) {
      const st = learner.stats()
      const sn = Math.max(1, win.s.sn ?? 0)
      const r = (k: string) => (win.s[k] ?? 0) / sn
      const row = {
        step, cu: st.cu, emaRatio: st.predLoss / st.baselineLoss,
        winRatio: win.mean("lb") / win.mean("cl"),
        lP: r("lP"), clP: r("clPs"), ratioP: r("lP") / r("clPs"),
        lF: r("lF"), clF: r("clFs"), ratioF: r("lF") / r("clFs"),
        shiftOracleRatio: win.mean("shiftOracle") / win.mean("cl"),
        chP: win.mean("chP"), chF: win.mean("chF"), moved: win.mean("moved"), move0: win.mean("move0"), btn: win.mean("btn"), key: win.mean("key"),
        zAlive: r("zAlive"), gHeadP: r("gHeadP"), gHeadF: r("gHeadF"), gPredH: r("gPredH"),
        wHeadP: win.s.wHeadP ?? NaN, wHeadF: win.s.wHeadF ?? NaN, bHeadP: win.s.bHeadP ?? NaN, bHeadF: win.s.bHeadF ?? NaN,
        sps: step / ((Date.now() - t0) / 1000),
      }
      rows.push(row)
      const f = (v: number, d = 4) => (Number.isFinite(v) ? (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(2) : v.toFixed(d)) : "nan")
      console.log(
        `step ${step} cu ${f(row.cu, 1)} | ema ${f(row.emaRatio, 5)} win ${f(row.winRatio, 5)} | P ${f(row.lP, 6)}/${f(row.clP, 6)}=${f(row.ratioP, 5)}`
        + ` F ${f(row.lF, 6)}/${f(row.clF, 6)}=${f(row.ratioF, 5)} | shiftOracle ${f(row.shiftOracleRatio, 4)}`
        + ` | chP ${f(row.chP, 2)} chF ${f(row.chF, 2)} moved ${f(row.moved, 2)} m0 ${f(row.move0, 2)} btn ${f(row.btn, 2)} key ${f(row.key, 2)}`
        + ` | zAlive ${f(row.zAlive, 3)} | |g| P ${f(row.gHeadP)} F ${f(row.gHeadF)} predH ${f(row.gPredH)} | |W| P ${f(row.wHeadP)} F ${f(row.wHeadF)} b ${f(row.bHeadP)}/${f(row.bHeadF)} | ${f(row.sps, 1)} st/s`,
      )
      win = new Acc()
    }
  }
  const sn = Math.max(1, all.s.sn ?? 0)
  const summary = {
    seed: SEED, steps: STEPS, widths, settings, sleep: !NO_SLEEP,
    finalEmaRatio: learner.stats().predLoss / learner.stats().baselineLoss,
    overallRatio: all.mean("lb") / all.mean("cl"),
    lastQuarterRatio: rows.slice(-Math.max(1, Math.floor(rows.length / 4))).reduce((a, r) => a + (r.winRatio ?? NaN), 0) / Math.max(1, Math.floor(rows.length / 4)),
    ratioP: (all.s.lP ?? 0) / (all.s.clPs ?? 1), ratioF: (all.s.lF ?? 0) / (all.s.clFs ?? 1),
    shareOfCopyLastFromFovea: (W_F * all.mean("clF")) / all.mean("cl"),
    shiftOracleRatio: all.mean("shiftOracle") / all.mean("cl"),
    chP: all.mean("chP"), chF: all.mean("chF"), moved: all.mean("moved"),
    moveMagnitudes: Object.fromEntries(moveHist), maxShadowCheckRelErr: maxCheckErr, shadowSamples: sn,
  }
  console.log("SUMMARY " + JSON.stringify(summary))
  if (JSON_OUT) (await import("node:fs")).writeFileSync(JSON_OUT, JSON.stringify({ summary, rows }, null, 1))
  learner.dispose()
}

void main()
