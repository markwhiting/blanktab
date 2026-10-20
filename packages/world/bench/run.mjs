// Usage: node bench/run.mjs [--shots <dir>] [--port <n>]
import { createServer } from "vite"
import { chromium } from "@playwright/test"
import { mkdirSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"

const arg = (name, dflt) => { const i = process.argv.indexOf(name); return i >= 0 ? process.argv[i + 1] : dflt }
const shots = arg("--shots", tmpdir())
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
  const page = await browser.newPage()
  page.on("pageerror", e => console.error("pageerror", e))
  await page.goto(`http://localhost:${port}/render.html`)
  await page.waitForFunction(() => window.__benchResult || window.__benchError, null, { timeout: 120_000 })
  const err = await page.evaluate(() => window.__benchError)
  if (err) throw new Error(`worker failed: ${err}`)
  const rows = await page.evaluate(() => window.__benchResult)
  for (const { room, msPerStep, stepsPerSec, fresh, after } of rows) {
    writeFileSync(join(shots, `${room}.png`), Buffer.from(fresh, "base64"))
    writeFileSync(join(shots, `${room}-after.png`), Buffer.from(after, "base64"))
    console.log(JSON.stringify({ room, msPerStep: +msPerStep.toFixed(3), stepsPerSec: +stepsPerSec.toFixed(1) }))
  }
  console.log(JSON.stringify({ browser: browser.version(), shots }))
} finally {
  await browser.close()
  await server.close()
}
