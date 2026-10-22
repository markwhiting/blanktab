/**
 * Awaits every promise; once all have settled, resolves to their values in order, or rejects with the first rejection
 * (in time). Unlike Promise.all, it never returns while one is still pending, so a caller may then free what they
 * read: nothing is freed under a pending read.
 */
export async function settleAll<T>(ps: readonly Promise<T>[]): Promise<T[]> {
  let first: { e: unknown } | null = null
  const rs = await Promise.allSettled(ps.map((p) => p.catch((e: unknown) => { first ??= { e }; throw e })))
  if (first) throw (first as { e: unknown }).e
  return rs.map((r) => (r as PromiseFulfilledResult<T>).value)
}
