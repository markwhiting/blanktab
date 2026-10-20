# @blanktab/tensor

Our own tensor and reverse-mode autodiff library: a CPU reference backend and a WebGPU backend
behind one `Backend` interface. No ML libraries and no pretrained weights (AGENTS.md invariant 1).

## Memory and lifecycle rules

Tensors hold backend storage (GPU buffers on WebGPU) that is freed explicitly, not by GC. These
are the rules a training loop such as Plan 04's continual learner must follow. `backend.stats.liveBuffers`
is the leak detector: in a steady-state loop it must not grow.

1. **Training step: `backward(loss)` → `opt.step()` → `opt.zeroGrad()` → `disposeGraph(loss, keep)`.**
   `disposeGraph` frees every tensor with a graph node under `loss` (and their grads). Leaves
   (params, inputs, constants) are never freed by it.
2. **Tensors without a graph node are caller-owned.** That includes every op run under `noGrad`
   and every op whose inputs are all constants (no `requiresGrad`). `disposeGraph` never frees
   them. Create constants once, outside the loop, and dispose them yourself.
3. **Acting / inference: wrap it in `composite()` (inside `noGrad`) or dispose every result.**
   `composite(fn)` supports exactly one output, the return value; it frees every other node-less
   tensor `fn` created, even one stashed in an outer variable. So never let a tensor escape a
   composite except as its result, and dispose that result when done.
4. **Carry recurrent state with `detach()`.** `detach(h)` copies `h` out of the graph (no node,
   no grad, its own storage), so no gradient flows through time and the old graph can be freed:

   ```ts
   const next = detach(h2)
   disposeGraph(loss, [])
   h.dispose()
   h = next
   ```

   `keep` in `disposeGraph(loss, keep)` is for reading a value (e.g. the loss) after freeing the
   graph; every kept tensor must be reachable from `loss`, otherwise it throws `ResourceError`
   ("kept tensor not part of this graph — use detach()").
5. **A GPU error is fatal for the run.** WebGPU work is batched (up to 64 passes per submit). A GPU
   error makes the *next* `read()` reject with `ResourceError`, and the failing batch was skipped
   entirely — possibly including `opt.step()` Adam updates, so parameters and optimizer moments
   may be partially updated. Treat any read rejection as fatal: stop, and restore from a checkpoint.
6. **FLOP counting is per backend instance.** `backend.stats.flops` counts every primitive on that
   backend. Measure deltas (`after - before`) around the work you are costing, and run evaluation
   on a separate backend instance so it is not billed to training.
