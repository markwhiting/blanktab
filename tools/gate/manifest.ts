// Computes knowledgeSha / codeSha and writes <out> as { knowledgeSha, codeSha, learnerStateVersion }.
// A running tab reads learnerStateVersion to tell whether newly deployed code can resume its learner (decision 0014).
// Usage: tsx tools/gate/manifest.ts <knowledgeDir> <outFile>
//        tsx tools/gate/manifest.ts --code-sha      (prints codeShaOf the current directory: the deploy's CODE_SHA)
import { execFileSync } from "node:child_process"
import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { pathToFileURL } from "node:url"
import { LEARNER_STATE_VERSION } from "@blanktab/learner/version"

const PRIOR_FILES = ["caregiver-prior.json", "consolidation-prior.json", "genome-space.json", "genome.json", "settings-space.json"] // sorted

export function computeKnowledgeSha(dir: string): string {
  const h = createHash("sha256")
  for (const name of PRIOR_FILES) {
    const p = join(dir, name)
    if (!existsSync(p)) continue
    h.update(`${name}\n${readFileSync(p, "utf8")}\n`)
  }
  return h.digest("hex")
}

/**
 * What the deployed page is built from (deploy.yml: `npm ci`, then `vite build` in app/): the app, the packages it
 * bundles, the root package files npm installs from, the compiler settings app/ and packages/ extend, and
 * config/evaluation.json (the harness bundles its evaluation budget, decision 0020; only that file: the rest of config/,
 * such as its references, is read by the gate and tools only). Nothing else is code: session logs, docs, knowledge (it
 * reaches learners by refresh, tracked by knowledgeSha), README and AGENTS, tests, tools, workflows, and the root
 * tsconfig.json (it covers tools and tests only).
 */
export const BUILD_INPUTS = ["app", "packages", "package.json", "package-lock.json", "tsconfig.base.json", "config/evaluation.json"] as const

/**
 * The code a build is made of: the last commit under `repoDir` that changed a BUILD_INPUTS path, or "dev" without one
 * (no git, no commits, none touching code). The page bakes it in, the manifest carries it and every log file records
 * it, so a merged batch of session logs (or a docs, knowledge, test or workflow edit) is not new code: tabs do not
 * reload for it, and Plan 05's constant-code segments stay whole. An include list, so a new kind of non-code file is
 * not code by default. Needs the history back to that commit (deploy.yml checks out with fetch-depth 0).
 */
export function codeShaOf(repoDir: string): string {
  try {
    const sha = execFileSync("git", ["log", "-1", "--format=%H", "--", ...BUILD_INPUTS.map(p => `:(top,literal)${p}`)], {
      cwd: repoDir, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"],
    }).trim()
    return /^[0-9a-f]{40}$/.test(sha) ? sha : "dev"
  } catch {
    return "dev"
  }
}

/** CODE_SHA as the deploy passes it (computed by codeShaOf), else codeShaOf(repoDir). */
export function resolveCodeSha(codeSha: string | undefined, repoDir: string): string {
  return codeSha !== undefined && /^[0-9a-f]{40}$/.test(codeSha) ? codeSha : codeShaOf(repoDir)
}

export interface Manifest { knowledgeSha: string; codeSha: string; learnerStateVersion: number }

/** The deployed manifest: the knowledge, the code, and the checkpoint format (LearnerState.version) that code reads and writes. */
export function buildManifest(knowledgeDir: string, codeSha: string): Manifest {
  return { knowledgeSha: computeKnowledgeSha(knowledgeDir), codeSha, learnerStateVersion: LEARNER_STATE_VERSION }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  if (process.argv[2] === "--code-sha") {
    console.log(codeShaOf(process.cwd()))
    process.exit(0)
  }
  const [dir, out] = process.argv.slice(2)
  if (!dir || !out) { console.error("usage: manifest.ts <knowledgeDir> <outFile>"); process.exit(2) }
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(out, JSON.stringify(buildManifest(dir, resolveCodeSha(process.env.CODE_SHA, process.cwd()))) + "\n")
  console.log(`wrote ${out}`)
}
