// Computes knowledgeSha / codeSha and writes <out> as { knowledgeSha, codeSha }.
// Usage: tsx tools/gate/manifest.ts <knowledgeDir> <outFile>
import { createHash } from "node:crypto"
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs"
import { dirname, join } from "node:path"
import { pathToFileURL } from "node:url"

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

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  const [dir, out] = process.argv.slice(2)
  if (!dir || !out) { console.error("usage: manifest.ts <knowledgeDir> <outFile>"); process.exit(2) }
  mkdirSync(dirname(out), { recursive: true })
  writeFileSync(out, JSON.stringify({ knowledgeSha: computeKnowledgeSha(dir), codeSha: resolveCodeSha(process.env.CODE_SHA ?? process.env.GITHUB_SHA) }) + "\n")
  console.log(`wrote ${out}`)
}
