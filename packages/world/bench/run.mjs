// Usage: node bench/run.mjs [--shots <dir>] [--port <n>] [--runs <n>]  (served with the Vite dev server; prints per-run lines when runs > 1, then medians)
import { createServer } from "vite"
import { chromium } from "@playwright/test"
import { mkdirSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"

const arg = (name, dflt) => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : dflt }
const shots = arg("--shots", tmpdir())
const runs = Math.max(1, Number(arg("--runs", "1")))
const port = Number(arg("--port", "5273"))
mkdirSync(shots, { recursive: true })

const server = await createServer({
  root: dirname(fileURLToPath(import.meta.url)),
  configFile: false,
  server: { port, strictPort: true },
  logLevel: "error",
})
await server.listen()
const browser = await chromium.launch()
try {
  const all = []
  let last
  for (let run = 1; run <= runs; run++) {
    const page = await browser.newPage()
    page.on("pageerror", e => console.error("pageerror", e))
    await page.goto(`http://localhost:${port}/render.html`)
    await page.waitForFunction(() => window.__benchResult || window.__benchError, null, { timeout: 120_000 })
    const err = await page.evaluate(() => window.__benchError)
    if (err) throw new Error(`worker failed: ${err}`)
    last = await page.evaluate(() => window.__benchResult)
    await page.close()
    all.push(last)
    if (runs > 1) for (const r of last) console.log(JSON.stringify({ run, room: r.room, msPerStep: +r.msPerStep.toFixed(3), stepsPerSec: +r.stepsPerSec.toFixed(1) }))
  }
  const median = xs => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]
  for (const [i, { room, fresh, after }] of last.entries()) {
    writeFileSync(join(shots, `${room}.png`), Buffer.from(fresh, "base64"))
    writeFileSync(join(shots, `${room}-after.png`), Buffer.from(after, "base64"))
    const ms = median(all.map(r => r[i].msPerStep))
    console.log(JSON.stringify({ room, runs, msPerStep: +ms.toFixed(3), stepsPerSec: +(1000 / ms).toFixed(1) }))
  }
  console.log(JSON.stringify({ browser: browser.version(), shots }))
} finally {
  await browser.close()
  await server.close()
}
