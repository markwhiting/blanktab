export const SCHEMA_VERSION = 1
export const WORLD = { width: 320, height: 200 } as const
export const PERIPHERY = { width: 80, height: 50, channels: 3 } as const // RGB, HWC, uint8
export const FOVEA = { width: 32, height: 32, channels: 3 } as const
export const PRED_PERIPHERY = { width: 40, height: 25, channels: 3 } as const
export const PRED_FOVEA = { width: 16, height: 16, channels: 3 } as const
export const TICKS_PER_WORLD_SECOND = 10
export const MOVE_MAGNITUDES = [1, 4, 16] as const
export const MOVE_OPTIONS = 25 // 0 none; 1..24 = 8 directions x 3 magnitudes
export const BUTTON_OPTIONS = 3 // 0 none, 1 press, 2 release
export const SCROLL_OPTIONS = 3 // 0 none, 1 up, 2 down

const LETTERS = "abcdefghijklmnopqrstuvwxyz".split("")
const DIGITS = "0123456789".split("")
export const KEYS: readonly string[] = [
  "", ...LETTERS, ...DIGITS, " ", "Enter", "Backspace", "Tab",
  "ArrowLeft", "ArrowUp", "ArrowRight", "ArrowDown",
]
