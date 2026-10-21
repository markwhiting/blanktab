// Computes knowledgeSha / codeSha and writes <out> as { knowledgeSha, codeSha, learnerStateVersion }.
// A running tab reads learnerStateVersion to tell whether newly deployed code can resume its learner (decision 0014).
// Usage: tsx tools/gate/manifest.ts <knowledgeDir> <outFile>
import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { pathToFileURL } from "node:url"
import { LEARNER_STATE_VERSION } from "@blanktab/learner/version"

const PRIOR_FILES = ["caregiver-prior.json", "consolidation-prior.json", "settings-space.json"] // sorted

export function computeKnowledgeSha(dir: string): string {
  const h = createHash("sha256")
  for (const name of PRIOR_FILES) {
    const p = join(dir, name)
    if (!existsSync(p)) continue
    h.update(`${name}\n${readFileSync(p, "utf8")}\n`)
  }
  return h.digest("hex")
}

export function resolveCodeSha(githubSha: string | undefined): string {
  return githubSha !== undefined && /^[0-9a-f]{40}$/.test(githubSha) ? githubSha : "dev"
}

export interface Manifest { knowledgeSha: string; codeSha: string; learnerStateVersion: number }

/** The deployed manifest: the knowledge, the code, and the checkpoint format (LearnerState.version) that code reads and writes. */
export function buildManifest(knowledgeDir: string, codeSha: string): Manifest {
  return { knowledgeSha: computeKnowledgeSha(knowledgeDir), codeSha, learnerStateVersion: LEARNER_STATE_VERSION }
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const [dir, out] = process.argv.slice(2)
  if (!dir || !out) { console.error("usage: manifest.ts <knowledgeDir> <outFile>"); process.exit(2) }
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(out, JSON.stringify(buildManifest(dir, resolveCodeSha(process.env.CODE_SHA ?? process.env.GITHUB_SHA))) + "\n")
  console.log(`wrote ${out}`)
}
