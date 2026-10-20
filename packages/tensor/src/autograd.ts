import { ShapeError } from "./backend"
import type { Storage } from "./backend"
import { Tensor, noGrad } from "./tensor"

/** Every tensor reachable from `root` through graph nodes, parents before children. */
function topoOrder(root: Tensor): Tensor[] {
  const order: Tensor[] = []
  const seen = new Set<Tensor>([root])
  // Iterative post-order DFS (deep graphs must not overflow the call stack).
  const stack: [Tensor, number][] = [[root, 0]]
  while (stack.length > 0) {
    const top = stack[stack.length - 1]!
    const [t, i] = top
    const parents = t._node?.parents ?? []
    if (i < parents.length) {
      top[1] = i + 1
      const p = parents[i]!
      if (!seen.has(p)) {
        seen.add(p)
        stack.push([p, 0])
      }
    } else {
      stack.pop()
      order.push(t)
    }
  }
  return order
}

/**
 * Reverse-mode autodiff from a scalar `loss`: seeds d(loss)/d(loss) = 1, walks the graph in reverse
 * topological order, and accumulates (adds) into `.grad` of every reachable leaf with requiresGrad.
 *
 * Intermediate gradients are freed as soon as they have been propagated. Because some rules return
 * a gradient that shares storage with their input gradient (add, addScalar, reshape), storages are
 * reference-counted here and freed only when the last gradient using them is released. Each leaf's
 * `.grad` always owns its storage exclusively, so `zeroGrad`/accumulation can dispose it safely.
 */
export function backward(loss: Tensor): void {
  if (loss.shape.length !== 0) throw new ShapeError(`backward: loss must be a scalar (shape []), got [${loss.shape.join(",")}]`)
  if (!loss.requiresGrad) throw new Error("backward: loss does not require grad (no parameter reaches it)")
  const be = loss.backend
  const order = topoOrder(loss)

  noGrad(() => {
    const grads = new Map<Tensor, Tensor>()
    const refs = new Map<Storage, number>()
    const acquire = (g: Tensor) => refs.set(g.storage, (refs.get(g.storage) ?? 0) + 1)
    const release = (g: Tensor) => {
      const n = refs.get(g.storage)! - 1
      if (n > 0) refs.set(g.storage, n)
      else {
        refs.delete(g.storage)
        be.dispose(g.storage)
      }
    }
    const accumulate = (t: Tensor, g: Tensor) => {
      if (g.size !== t.size) throw new ShapeError(`backward: gradient [${g.shape.join(",")}] does not match tensor [${t.shape.join(",")}]`)
      acquire(g)
      const prev = grads.get(t)
      if (!prev) {
        grads.set(t, g)
        return
      }
      const sum = new Tensor(be.binary("add", prev.storage, [prev.size], g.storage, [g.size], [prev.size]), t.shape)
      acquire(sum)
      release(prev)
      release(g)
      grads.set(t, sum)
    }

    try {
      accumulate(loss, new Tensor(be.upload(new Float32Array([1])), []))
      for (let i = order.length - 1; i >= 0; i--) {
        const t = order[i]!
        const g = grads.get(t)
        if (!g) continue
        grads.delete(t)
        const node = t._node
        if (node) {
          const needs = node.parents.map((p) => p.requiresGrad)
          try {
            const pg = node.backward(g, needs)
            node.parents.forEach((p, j) => {
              const gp = pg[j]
              if (needs[j] && gp) accumulate(p, gp)
              else if (gp && gp.storage !== g.storage && !refs.has(gp.storage)) be.dispose(gp.storage)
            })
          } finally {
            release(g)
          }
        } else if (t.requiresGrad) {
          assignLeafGrad(t, g)
        } else {
          release(g)
        }
      }
    } finally {
      // Entries remain only if a rule threw (the throwing node's own g is released above); free them
      // so a failed backward leaks nothing. Leaf grads already assigned in this pass are kept.
      for (const g of grads.values()) release(g)
    }

    function assignLeafGrad(t: Tensor, g: Tensor) {
      if (t.grad) {
        const sum = be.binary("add", t.grad.storage, [t.size], g.storage, [t.size], [t.size])
        t.grad.dispose()
        t.grad = new Tensor(sum, t.shape)
        release(g)
      } else if (refs.get(g.storage) === 1) {
        refs.delete(g.storage) // transfer ownership to the leaf
        t.grad = new Tensor(g.storage, t.shape)
      } else {
        // Still shared with another pending gradient: give the leaf its own copy.
        t.grad = new Tensor(be.scalar("mul", g.storage, 1), t.shape)
        release(g)
      }
    }
  })
}

/**
 * Frees the graph under `loss`: every non-leaf tensor reachable from it (including `loss`) and any
 * gradient stored on them, except tensors in `keep`. Leaves (params, inputs, constants) are never
 * freed, nor is any storage they share with a reshaped view. Each storage is freed at most once.
 *
 * "Non-leaf" means "has a graph node". A tensor computed only from tensors that do not require grad
 * (or under noGrad) has no node, so it is treated as a leaf and is NOT freed here — create such
 * constants once, outside training loops, and dispose them yourself.
 *
 * Kept tensors are detached (PyTorch `.detach()` semantics): their node is removed and
 * `requiresGrad` set to false, so reusing one in a later graph (e.g. carried recurrent state)
 * propagates no gradient into it and never accrues an orphan `.grad`.
 */
export function disposeGraph(loss: Tensor, keep: Tensor[]): void {
  const order = topoOrder(loss)
  const keepSet = new Set(keep)
  const protectedStorages = new Set<Storage>()
  for (const t of order) {
    if (!t._node || keepSet.has(t)) protectedStorages.add(t.storage)
    if (t.grad && (!t._node || keepSet.has(t))) protectedStorages.add(t.grad.storage)
  }
  const freed = new Set<Storage>()
  const free = (s: Storage) => {
    if (protectedStorages.has(s) || freed.has(s)) return
    freed.add(s)
    s.backend.dispose(s)
  }
  for (const t of order) {
    if (!t._node) continue
    delete t._node
    if (keepSet.has(t)) {
      t.requiresGrad = false
      continue
    }
    free(t.storage)
    if (t.grad) {
      free(t.grad.storage)
      t.grad = null
    }
  }
}
