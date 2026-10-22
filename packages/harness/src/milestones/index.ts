import { M1 } from "./m1"
import { M2 } from "./m2"
import { M3 } from "./m3"
import { M4 } from "./m4"
import { M5 } from "./m5"
import { M6 } from "./m6"
import type { Milestone } from "./types"

export type { CheckupTrace, Milestone, MilestoneId, Policy } from "./types"
export { YIELD_EVERY, recordRun } from "./record"
export { M1, M2, M3, M4, M5, M6 }
export { M2_MIN_IN_VIEW } from "./m2"
export { M5_MIN_PAGES, M5_REFERENCE_CLICKERS } from "./m5"

/** Milestone suite v1 (decision 0016), in suite order: MILESTONE_IDS[1]. */
export const MILESTONES_V1: readonly Milestone[] = Object.freeze([M1, M2, M3, M4, M5, M6])
