import { describe, it, expect } from "vitest"
import { NEWBORN_COUNT } from "@blanktab/harness/newborn"
import { readNewborns } from "../src/newborns"

describe("readNewborns", () => {
  it("?newborns=a-b with 0 ≤ a ≤ b < NEWBORN_COUNT", () => {
    expect(readNewborns("?newborns=0-0")).toEqual({ from: 0, to: 0 })
    expect(readNewborns("?x=1&newborns=3-17")).toEqual({ from: 3, to: 17 })
    expect(readNewborns(`?newborns=0-${NEWBORN_COUNT - 1}`)).toEqual({ from: 0, to: NEWBORN_COUNT - 1 })
  })
  it("anything else is not newborn mode", () => {
    for (const s of ["", "?newborns", "?newborns=", "?newborns=5", "?newborns=5-4", `?newborns=0-${NEWBORN_COUNT}`, "?newborns=-1-2", "?newborns=1.5-2", "?newborns=01-2", "?newborns=a-b", "?newborns=1-2-3"])
      expect(readNewborns(s), s).toBeNull()
  })
})
