// Page side: start the worker, hand its results to Playwright via window.__benchResult.
const worker = new Worker(new URL("./render-worker.ts", import.meta.url), { type: "module" })
worker.onmessage = e => { (window as unknown as { __benchResult: unknown }).__benchResult = e.data }
worker.onerror = e => { (window as unknown as { __benchError: string }).__benchError = String(e.message) }
worker.postMessage("go")
