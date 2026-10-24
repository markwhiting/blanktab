// The gate's direct commit of data/meta/** (decision 0019; invariant 7): only the gate's own files, never forced.
import { execFileSync } from "node:child_process"
import { createHash } from "node:crypto"
import { chmodSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { afterEach, describe, expect, it } from "vitest"
import { commitMeta, META_PATH_RE, metaPathErrors } from "../commit-meta.ts"
import { makeFixture, type Fixture } from "./git-fixture.ts"

const REPO = fileURLToPath(new URL("../../..", import.meta.url))
// No author or committer names in the environment: they would override the bot identity commitMeta sets.
const ENV: NodeJS.ProcessEnv = { ...process.env, GIT_CONFIG_GLOBAL: "/dev/null", GIT_CONFIG_NOSYSTEM: "1" }
for (const k of ["GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"]) delete ENV[k]

const posterior = (mode: string, eligibleLives: number, tag = "") => JSON.stringify({ mode, page: { eligibleLives }, tag }, null, 1) + "\n"
const sha8 = (text: string) => createHash("sha256").update(text, "utf8").digest("hex").slice(0, 8)
const META = (tag: string) => ({
  "data/meta/posterior.json": posterior("calibrate", 3, tag),
  "data/meta/calibration.json": `{"c":"${tag}"}\n`,
  "data/meta/diagnostics.json": `{"d":"${tag}"}\n`,
})

/** A forced push: --force, --force-with-lease, -f (alone or in a flag group) or a +refspec. */
const isForce = (a: string) => a.startsWith("--force") || /^-[a-z]*f/.test(a) || a.startsWith("+")

let fx: Fixture | null = null
afterEach(() => { fx?.cleanup(); fx = null })

function setup() {
  fx = makeFixture()
  const work = fx.work
  const calls: string[][] = []
  const git = (args: string[]): string => {
    calls.push(args)
    try {
      return execFileSync("git", args, { cwd: work, encoding: "utf8", env: ENV, stdio: ["ignore", "pipe", "pipe"] })
    } catch (e) {
      const err = e as { stderr?: string; message: string }
      throw new Error(`git ${args.join(" ")} failed: ${err.stderr ?? err.message}`, { cause: e })
    }
  }
  const write = (files: Record<string, string>) => {
    for (const [p, t] of Object.entries(files)) { mkdirSync(dirname(join(work, p)), { recursive: true }); writeFileSync(join(work, p), t) }
  }
  // git in the bare "origin": what main really is after a push.
  const origin = (...args: string[]) => execFileSync("git", ["--git-dir", join(work, "../origin.git"), ...args], { encoding: "utf8", env: ENV }).trim()
  // The lockfile npm ci "installed" from: main's, as the workflow's LOCK_BLOB records it.
  fx.commitMain({ "package-lock.json": "{\"lockfileVersion\":3}\n" })
  const installedLock = git(["rev-parse", "HEAD:package-lock.json"]).trim()
  calls.length = 0
  const logs: string[] = []
  return { fx, work, git, calls, write, origin, logs, installedLock, log: (m: string) => { logs.push(m) } }
}

describe("metaPathErrors", () => {
  it("accepts exactly the gate's four data/meta files", () => {
    expect(metaPathErrors(["data/meta/posterior.json", "data/meta/calibration.json", "data/meta/diagnostics.json", "data/meta/posterior-index.jsonl"])).toEqual([])
    expect(metaPathErrors([])).toEqual([])
    for (const p of ["data/meta/genome.json", "data/meta/x/posterior.json", "data/meta/posterior.jsonx", "data/meta/../config/evaluation.json",
      "config/evaluation.json", "knowledge/genome.json", "data/runs/a/0.json", "data/meta", "xdata/meta/posterior.json"]) {
      expect(META_PATH_RE.test(p)).toBe(false)
      expect(metaPathErrors(["data/meta/posterior.json", p])).toHaveLength(1)
    }
  })
})

describe("commitMeta", () => {
  it("only data/meta files staged → committed and pushed", async () => {
    const t = setup()
    const before = t.origin("rev-parse", "main")
    t.write({ ...META("a"), "data/meta/posterior-index.jsonl": `{"i":1}\n` })
    const rebuild = async () => { throw new Error("no rebuild without a rejected push") }
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild, log: t.log })).toBe("committed")
    const head = t.origin("rev-parse", "main")
    expect(head).not.toBe(before)
    expect(t.origin("rev-parse", "main^")).toBe(before)
    expect(t.origin("log", "-1", "--format=%an <%ae>|%cn <%ce>|%s", "main")).toBe(
      `github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>|github-actions[bot] <41898282+github-actions[bot]@users.noreply.github.com>|meta: calibrate, 3 eligible lives, posterior ${sha8(META("a")["data/meta/posterior.json"])}`)
    expect(t.origin("diff-tree", "--no-commit-id", "--name-only", "-r", "main").split("\n").sort()).toEqual(
      ["data/meta/calibration.json", "data/meta/diagnostics.json", "data/meta/posterior-index.jsonl", "data/meta/posterior.json"])
    expect(t.origin("show", "main:data/meta/posterior.json") + "\n").toBe(META("a")["data/meta/posterior.json"])
  })

  for (const other of ["config/evaluation.json", "knowledge/genome.json", "data/runs/abc/0.json", "data/meta/genome.json"]) {
    it(`a staged ${other} → throws, nothing committed`, async () => {
      const t = setup()
      const before = t.origin("rev-parse", "main")
      const localBefore = t.git(["rev-parse", "HEAD"]).trim()
      t.write({ ...META("a"), [other]: "{}\n" })
      t.git(["add", "--", other])
      await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(other)
      expect(t.origin("rev-parse", "main")).toBe(before)
      expect(t.git(["rev-parse", "HEAD"]).trim()).toBe(localBefore)
      expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
    })
  }

  it("another tracked file modified → throws", async () => {
    const t = setup()
    const before = t.origin("rev-parse", "main")
    t.write({ ...META("a"), "README.md": "changed by npm ci\n" })
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow("README.md")
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })

  it("an untracked file outside data/meta → throws", async () => {
    const t = setup()
    t.write({ ...META("a"), "stray.txt": "x\n" })
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow("stray.txt")
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })

  it("a local commit not on origin/main → throws, nothing pushed", async () => {
    const t = setup()
    const before = t.origin("rev-parse", "main")
    t.write({ "README.md": "local\n" })
    t.git(["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "local"])
    t.write(META("a"))
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/origin\/main/)
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "push")).toBe(false)
  })

  it("a push rejected because main moved → fetch, rebuild on the new main, push; never --force", async () => {
    const t = setup()
    // The first build creates the index (untracked): it must not survive into the rebuilt commit.
    t.write({ ...META("stale"), "data/meta/posterior-index.jsonl": `{"stale":1}\n` })
    const moved = t.fx.commitMain({ "data/runs/abc/0.json": "{}\n" }, [], false)
    let rebuilds = 0
    const rebuild = async () => {
      rebuilds++
      // Task 10 again, on the new main: it must see the moved commit's files.
      expect(t.git(["rev-parse", "HEAD"]).trim()).toBe(moved)
      expect(readFileSync(join(t.work, "data/runs/abc/0.json"), "utf8")).toBe("{}\n")
      t.write(META("fresh"))
      return true
    }
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild, log: t.log })).toBe("committed")
    expect(rebuilds).toBe(1)
    expect(t.origin("rev-parse", "main^")).toBe(moved)
    expect(t.origin("show", "main:data/meta/calibration.json")).toBe(`{"c":"fresh"}`)
    expect(t.origin("ls-tree", "--name-only", "main", "data/meta/").split("\n").sort()).toEqual(
      ["data/meta/calibration.json", "data/meta/diagnostics.json", "data/meta/posterior.json"])
    expect(t.origin("log", "-1", "--format=%s", "main")).toBe(`meta: calibrate, 3 eligible lives, posterior ${sha8(META("fresh")["data/meta/posterior.json"])}`)
    const pushes = t.calls.filter(a => a[0] === "push")
    expect(pushes).toEqual([["push", "origin", "HEAD:main"], ["push", "origin", "HEAD:main"]])
    expect(t.calls.filter(a => a[0] === "push").flat().some(isForce)).toBe(false)
  })

  it("main moving on every try → at most 3 pushes, then throws; never --force", async () => {
    const t = setup()
    t.write(META("a"))
    let n = 0
    t.fx.commitMain({ [`data/runs/abc/${n}.json`]: "{}\n" }, [], false)
    let before = ""
    const rebuild = async () => {
      n++
      t.fx.commitMain({ [`data/runs/abc/${n}.json`]: "{}\n" }, [], false) // main moves again before this push
      t.write(META(`r${n}`))
      before = t.origin("rev-parse", "main")
      return true
    }
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild, log: t.log })).rejects.toThrow(/3 tries/)
    expect(t.calls.filter(a => a[0] === "push")).toHaveLength(3)
    expect(n).toBe(2)
    expect(t.origin("log", "--format=%s", "main").split("\n").some(s => s.startsWith("meta:"))).toBe(false)
    expect(t.origin("rev-list", "--count", `${before}..main`)).toBe("0") // only the fixture's commits landed
    expect(t.calls.filter(a => a[0] === "push").flat().some(isForce)).toBe(false)
  })

  it("a push refused for another reason → throws at once, no retry", async () => {
    const t = setup()
    t.write(META("a"))
    execFileSync("git", ["-C", t.work, "remote", "set-url", "--push", "origin", join(t.work, "../nowhere.git")], { env: ENV })
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/push/)
    expect(t.calls.filter(a => a[0] === "push")).toHaveLength(1)
  })

  it("a rebuild on the new main that changes nothing → unchanged", async () => {
    const t = setup()
    t.write(META("a"))
    t.fx.commitMain(META("a"), [], false) // someone else already committed the same files
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => false, log: t.log })).toBe("unchanged")
    expect(t.origin("log", "-1", "--format=%s", "main")).toBe("c")
  })

  it("main's package-lock.json changed since npm ci (pulled in before the build) → throws, nothing committed", async () => {
    const t = setup()
    t.fx.commitMain({ "package-lock.json": "{\"lockfileVersion\":3,\"new\":1}\n" }) // the workflow's git pull brought it
    const before = t.origin("rev-parse", "main")
    t.write(META("a"))
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/package-lock\.json/)
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })

  it("main's package-lock.json changed by the commit that rejected the push → throws before the rebuild, nothing pushed", async () => {
    const t = setup()
    t.write(META("a"))
    const moved = t.fx.commitMain({ "package-lock.json": "{\"lockfileVersion\":3,\"new\":1}\n" }, [], false)
    let rebuilds = 0
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => { rebuilds++; return true }, log: t.log })).rejects.toThrow(/package-lock\.json/)
    expect(rebuilds).toBe(0)
    expect(t.origin("rev-parse", "main")).toBe(moved)
    expect(t.calls.filter(a => a[0] === "push")).toHaveLength(1)
  })

  it("a tracked data/meta file deleted → throws, nothing committed", async () => {
    const t = setup()
    t.fx.commitMain(META("a"))
    const before = t.origin("rev-parse", "main")
    t.write({ "data/meta/calibration.json": `{"c":"b"}\n` })
    rmSync(join(t.work, "data/meta/diagnostics.json"))
    await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/diagnostics\.json" is D/)
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })

  for (const [what, make] of [
    ["a tracked data/meta file turned into a symlink", (w: string) => { rmSync(join(w, "data/meta/posterior.json")); symlinkSync("../../README.md", join(w, "data/meta/posterior.json")) }],
    ["a new data/meta file that is a symlink", (w: string) => { symlinkSync("../../README.md", join(w, "data/meta/posterior-index.jsonl")) }],
    ["a data/meta file made executable", (w: string) => { chmodSync(join(w, "data/meta/calibration.json"), 0o755) }],
  ] as const) {
    it(`${what} → throws, nothing committed`, async () => {
      const t = setup()
      t.fx.commitMain(META("a"))
      const before = t.origin("rev-parse", "main")
      t.git(["config", "core.fileMode", "true"])
      make(t.work)
      await expect(commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).rejects.toThrow(/not a regular file added or modified/)
      expect(t.origin("rev-parse", "main")).toBe(before)
      expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
    })
  }

  it("nothing changed → unchanged, no commit", async () => {
    const t = setup()
    t.fx.commitMain(META("a"))
    const before = t.origin("rev-parse", "main")
    t.write(META("a"))
    expect(await commitMeta({ git: t.git, installedLock: t.installedLock, rebuild: async () => true, log: t.log })).toBe("unchanged")
    expect(t.origin("rev-parse", "main")).toBe(before)
    expect(t.calls.some(a => a[0] === "commit" || a[0] === "push")).toBe(false)
  })
})

describe("workflows", () => {
  const gate = readFileSync(join(REPO, ".github/workflows/gate.yml"), "utf8")
  const deploy = readFileSync(join(REPO, ".github/workflows/deploy.yml"), "utf8")
  it("gate.yml's deploy condition includes meta == 'committed'; deploy.yml copies data/meta/posterior.json", () => {
    const deployJob = gate.slice(gate.indexOf("\n  deploy:\n"))
    expect(deployJob).toMatch(/^ {4}needs: gate\n {4}if: \$\{\{ !cancelled\(\) && \(\(needs\.gate\.outputs\.merged != '' && needs\.gate\.outputs\.merged != '0'\) \|\| needs\.gate\.outputs\.meta == 'committed'\) \}\}\n {4}uses: \.\/\.github\/workflows\/deploy\.yml$/m)
    expect(gate).toMatch(/meta: \$\{\{ steps\.meta\.outputs\.meta \}\}/)
    expect(deploy).toContain("if [ -f data/meta/posterior.json ]; then cp data/meta/posterior.json app/dist/knowledge/posterior.json; fi")
  })
  it("gate.yml pulls main, builds data/meta and commits it after the gate step, with only contents and pull-requests write", () => {
    const at = (s: string) => { const i = gate.indexOf(s); expect(i, s).toBeGreaterThan(-1); return i }
    const order = [at("run: npm ci"), at(`LOCK_BLOB=$(git hash-object package-lock.json)`), at("npx tsx tools/gate/gate.ts"), at("git pull --ff-only origin main"),
      at(`"$(git rev-parse HEAD:package-lock.json)" = "$LOCK_BLOB"`), at("npx tsx tools/gate/meta-cli.ts"), at("npx tsx tools/gate/commit-meta.ts")]
    expect([...order].sort((a, b) => a - b)).toEqual(order)
    const gateJob = gate.slice(gate.indexOf("  gate:"), gate.indexOf("  deploy:"))
    expect(gateJob).toMatch(/permissions:\n\s+contents: write\n\s+pull-requests: write\n\s+outputs:/)
    expect(gate).not.toMatch(/--force|push -f/)
  })
})
