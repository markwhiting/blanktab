import { BUTTON_OPTIONS, KEYS, MOVE_MAGNITUDES, MOVE_OPTIONS, SCROLL_OPTIONS } from "./constants"

export interface Action { move: number; button: number; scroll: number; key: number }

export const NOOP_ACTION: Action = Object.freeze({ move: 0, button: 0, scroll: 0, key: 0 })

const inRange = (v: number, n: number) => Number.isInteger(v) && v >= 0 && v < n

export function isValidAction(a: Action): boolean {
  return inRange(a.move, MOVE_OPTIONS) && inRange(a.button, BUTTON_OPTIONS)
    && inRange(a.scroll, SCROLL_OPTIONS) && inRange(a.key, KEYS.length)
}

// Directions E, NE, N, NW, W, SW, S, SE. Screen y grows downward, so N is dy < 0.
const DIRS: readonly (readonly [number, number])[] = [
  [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1],
]

export function decodeMove(move: number): { dx: number; dy: number } {
  if (!inRange(move, MOVE_OPTIONS)) throw new RangeError(`invalid move ${move}`)
  if (move === 0) return { dx: 0, dy: 0 }
  const i = move - 1
  const [ux, uy] = DIRS[Math.floor(i / 3)]!
  const mag = MOVE_MAGNITUDES[i % 3]!
  return { dx: ux * mag, dy: uy * mag }
}
