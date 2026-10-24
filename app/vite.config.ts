import { existsSync, readFileSync } from "node:fs"
import { join } from "node:path"
import { fileURLToPath } from "node:url"
import { defineConfig, type Plugin } from "vite"
import { buildSummary, pinnedScoring } from "../tools/gate/summary.ts"
import { buildManifest } from "../tools/gate/manifest.ts"

const REPO_ROOT = fileURLToPath(new URL("..", import.meta.url))
const BASE = "/learner/"
const APP_DIR = fileURLToPath(new URL(".", import.meta.url))

/** The code the page is built from (deploy sets CODE_SHA: tools/gate/manifest.ts codeShaOf), or "dev". Anything else is a build error. */
function codeSha(): string {
  const sha = process.env.CODE_SHA ?? "dev"
  if (!/^([0-9a-f]{40}|dev)$/.test(sha)) throw new Error(`CODE_SHA must be 40 lowercase hex chars or "dev", got ${JSON.stringify(sha)}`)
  return sha
}

/**
 * Dev server only: serves /learner/knowledge/*.json the way the deploy workflow assembles it
 * (summary from data/runs, scored as deploy scores it: pinnedScoring; manifest with codeSha "dev"; posterior.json
 * as data/meta/posterior.json's exact bytes, as deploy copies it, so a dev tab's posteriorSha is the gate's; the rest
 * from knowledge/). Never part of a build.
 */
function devKnowledge(): Plugin {
  return {
    name: "blanktab-dev-knowledge",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use(`${BASE}knowledge`, (req, res, next) => {
        const name = (req.url ?? "").split("?")[0]!.replace(/^\//, "")
        let body: unknown
        if (name === "posterior.json") {
          const file = join(REPO_ROOT, "data/meta/posterior.json")
          if (!existsSync(file)) { res.statusCode = 404; return res.end() } // no gate run yet: tabs draw the centre
          res.setHeader("content-type", "application/json")
          res.setHeader("cache-control", "no-store")
          return res.end(readFileSync(file))
        }
        if (name === "summary.json") body = buildSummary(join(REPO_ROOT, "data/runs"), pinnedScoring(REPO_ROOT))
        else if (name === "manifest.json") body = buildManifest(join(REPO_ROOT, "knowledge"), "dev")
        else if (/^[a-z0-9-]+\.json$/.test(name) && existsSync(join(REPO_ROOT, "knowledge", name))) body = JSON.parse(readFileSync(join(REPO_ROOT, "knowledge", name), "utf8"))
        else return next()
        res.setHeader("content-type", "application/json")
        res.setHeader("cache-control", "no-store")
        res.end(JSON.stringify(body))
      })
    },
  }
}

export default defineConfig(({ mode }) => ({
  base: BASE,
  define: { __CODE_SHA__: JSON.stringify(codeSha()), __BUILD_TIME__: JSON.stringify(new Date().toISOString()) },
  plugins: [devKnowledge()],
  worker: { format: "es" },
  build: {
    outDir: mode === "e2e" ? "dist-e2e" : "dist", emptyOutDir: true, target: "es2022",
    // The e2e build also has the detmath probe page (tests/e2e/detmath.spec.ts); the deploy build never does.
    ...(mode === "e2e" ? { rollupOptions: { input: { main: join(APP_DIR, "index.html"), detmath: join(APP_DIR, "detmath-probe.html") } } } : {}),
  },
}))
