// The e2e build's detmath probe page: the golden fixture's values computed by Chromium, on the main thread and in a
// module worker (where the page's birth draws run), for tests/e2e/detmath.spec.ts to compare with golden.json bit for bit.
import { compute, type Golden } from "../../../packages/detmath/test/golden-inputs"

declare global { interface Window { __detmath?: { main: Golden; worker: Golden } | { error: string } } }

const worker = new Worker(new URL("./detmath-probe.worker.ts", import.meta.url), { type: "module" })
worker.onmessage = (e: MessageEvent<Golden>) => {
  window.__detmath = { main: compute(), worker: e.data }
  document.getElementById("out")!.textContent = "done"
}
worker.onerror = e => { window.__detmath = { error: e.message } }
