// The deploy contract: the production build ignores ?testLifetime, ships no knowledge/, and stamps CODE_SHA.
import { describe, it, expect, beforeAll, afterAll } from "vitest"
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { build } from "vite"

const APP = fileURLToPath(new URL("..", import.meta.url))
const SHA = "0123456789abcdef0123456789abcdef01234567"
const files = (dir: string): string[] => readdirSync(dir).flatMap(n => {
  const p = join(dir, n)
  return statSync(p).isDirectory() ? files(p) : [p]
})

describe("production build", { timeout: 120_000 }, () => {
  let out: string
  let js: string
  beforeAll(async () => {
    out = mkdtempSync(join(tmpdir(), "blanktab-build-"))
    const prev = process.env.CODE_SHA
    process.env.CODE_SHA = SHA
    try {
      await build({ root: APP, configFile: join(APP, "vite.config.ts"), mode: "production", logLevel: "silent", build: { outDir: out, emptyOutDir: true } })
    } finally {
      if (prev === undefined) delete process.env.CODE_SHA
      else process.env.CODE_SHA = prev
    }
    js = files(out).filter(f => f.endsWith(".js")).map(f => readFileSync(f, "utf8")).join("\n")
  })
  afterAll(() => rmSync(out, { recursive: true, force: true }))

  it("drops the test-only knobs", () => {
    expect(js.length).toBeGreaterThan(0)
    expect(js).not.toContain("testLifetime")
    expect(js).not.toContain("testStepsPerSecond")
  })
  it("ships no knowledge/ of its own and builds the worker", () => {
    expect(files(out).some(f => f.includes(`${join(out, "knowledge")}`))).toBe(false)
    expect(files(out).filter(f => /worker.*\.js$/.test(f))).toHaveLength(1)
  })
  it("stamps CODE_SHA", () => {
    expect(js).toContain(SHA)
  })
})
