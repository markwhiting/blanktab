import { ROOM_KINDS, createWorld } from "@blanktab/world"
import type { Milestone } from "./types"

/**
 * M6 uses the taskbar: room switches the learner makes itself (focus events with source "learner"). Carries by the
 * caregiver and clicks on the active room's own icon are not switches.
 */
export const M6: Milestone = {
  id: "M6", version: 1,
  description: "Uses the taskbar: ≥ 3 self-initiated room switches through the taskbar within budget",
  room: "desktop", prerequisites: ["M3"], budgetSteps: 500, measuresLoss: false,
  makeVariant: (seed, canvas) =>
    createWorld({ seed, canvas, rooms: ROOM_KINDS, start: ROOM_KINDS[(seed >>> 0) % ROOM_KINDS.length]! }),
  measure: trace => trace.events.filter(e => e.kind === "focus" && e.source === "learner").length,
  passes: (trace, stats) => M6.measure(trace, stats) >= 3,
}
