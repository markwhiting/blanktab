// sha256 through Web Crypto (in the browser and in Node 22), and the genome's integrity hash (spec §3.1).
import type { Genome } from "./genome"

/** Lowercase hex sha256 of `bytes`. */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  // The one use of the crypto global this package's src may make: eslint.config.js exempts this file only (0021).
  const digest = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>)
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, "0")).join("")
}

/** sha256 of the canonical JSON `{ schemaVersion, values }`, with `values` keys sorted, so key order never matters. */
export function genomeSha(schemaVersion: number, g: Genome): Promise<string> {
  const values: Genome = {}
  for (const k of Object.keys(g).sort()) values[k] = g[k]!
  return sha256Hex(new TextEncoder().encode(JSON.stringify({ schemaVersion, values })))
}
