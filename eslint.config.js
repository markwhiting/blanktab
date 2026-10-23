import js from "@eslint/js"
import tseslint from "typescript-eslint"

const PKGS = ["interface", "tensor", "learner", "world", "harness", "detmath", "meta", "evaluators"]

// Walls: which workspace packages each package may NOT import.
const FORBIDDEN = {
  interface: PKGS.filter(p => p !== "interface"),
  tensor: PKGS.filter(p => p !== "tensor"),
  learner: ["world", "harness", "detmath", "meta", "evaluators"],
  world: ["learner", "tensor", "harness", "detmath", "meta", "evaluators"],
  harness: [],
  detmath: PKGS.filter(p => p !== "detmath"),
  meta: PKGS.filter(p => !["meta", "detmath"].includes(p)),
  evaluators: PKGS.filter(p => !["evaluators", "detmath"].includes(p)),
}
// Subpath exports a package may import from a package that is otherwise behind its wall.
const ALLOWED_SUBPATHS = { evaluators: { harness: ["log"] } }
const allowed = (pkg, p) => ALLOWED_SUBPATHS[pkg]?.[p] ?? []

// Two patterns per banned package: its "@blanktab/<p>" specifiers except allowed subpaths, and relative or
// absolute paths into its directory (skipping "@blanktab/..." so one import reports once). Regexes, not
// gitignore-style groups: a group cannot re-include "@blanktab/<p>/log" once "@blanktab/<p>" is excluded.
const except = subs => subs.length ? `(?!/(${subs.join("|")})$)` : ""
const wallPatterns = pkg => FORBIDDEN[pkg].flatMap(p => [
  { regex: `^@blanktab/${p}${except(allowed(pkg, p))}(/.*)?$`, message: `${pkg} must not import ${p} (package wall).` },
  { regex: `^(?!@blanktab/)((.*/)?${p}/.+|(.*/)?packages/${p})$`, message: `${pkg} must not import ${p} (package wall).` },
])
const wall = pkg => ({
  files: [`packages/${pkg}/**/*.{ts,tsx,js}`],
  rules: { "no-restricted-imports": ["error", { patterns: wallPatterns(pkg) }] },
})

// Dynamic import() and require() are invisible to no-restricted-imports; ban them by syntax.
const specifier = (p, subs = []) =>
  `/^(@blanktab\\/${p}${except(subs).replaceAll("/", "\\/")}(\\/.*)?|(?!@blanktab\\/).*\\/${p}\\/.*|.*\\/packages\\/${p})$/`
const dynamicSelectors = pkg => FORBIDDEN[pkg].flatMap(p => [
  { selector: `ImportExpression > Literal.source[value=${specifier(p, allowed(pkg, p))}]`, message: `${pkg} must not import ${p} (package wall).` },
  { selector: `CallExpression[callee.name='require'] > Literal.arguments:first-child[value=${specifier(p, allowed(pkg, p))}]`, message: `${pkg} must not require ${p} (package wall).` },
])
const dynamicWall = pkg => ({
  files: [`packages/${pkg}/**/*.{ts,tsx,js}`],
  rules: { "no-restricted-syntax": ["error", ...dynamicSelectors(pkg)] },
})

// Determinism and text-as-pixels for the world (constraints: no Math.random, Date or performance; text only via
// font.ts). Separate rules from the walls above, so no-restricted-syntax/imports for these files stay as they are.
const NONDETERMINISTIC = "is not allowed in packages/world/src: the world must be a pure function of seed and actions."
const GLOBAL_HOSTS = ["globalThis", "window", "self"]
const worldPurity = {
  files: ["packages/world/src/**/*.{ts,tsx,js}"],
  rules: {
    "no-restricted-globals": ["error",
      { name: "Date", message: `Date ${NONDETERMINISTIC}` },
      { name: "performance", message: `performance ${NONDETERMINISTIC}` },
    ],
    "no-restricted-properties": ["error",
      { object: "Math", property: "random", message: `Math.random ${NONDETERMINISTIC} Use rng.ts (mulberry32).` },
      ...GLOBAL_HOSTS.flatMap(object => ["Date", "performance"].map(property => ({ object, property, message: `${object}.${property} ${NONDETERMINISTIC}` }))),
      { property: "fillText", message: "Text in the world is pixels from font.ts (Gfx.text), never fillText." },
      { property: "strokeText", message: "Text in the world is pixels from font.ts (Gfx.text), never strokeText." },
    ],
  },
}

// Purity for detmath, meta and evaluators (plan 05a): pure and deterministic, so no clock, no Math.random and
// no IO. The same globals and properties as the world's ban, plus Node built-ins and bare fs/path. A later
// config's rule replaces an earlier one's options, so these blocks restate the package wall alongside the new bans.
const PURE = ["detmath", "meta", "evaluators"]
const IMPURE = "is not allowed in packages/{detmath,meta,evaluators}/src: they must be pure and deterministic."
const NODE_MODULES = ["fs", "fs/*", "path", "path/*"]
const impureSpecifier = "/^(node:.*|fs|fs\\/.*|path|path\\/.*)$/"
const purity = pkg => ({
  files: [`packages/${pkg}/src/**/*.{ts,tsx,js}`],
  rules: {
    "no-restricted-globals": ["error",
      { name: "Date", message: `Date ${IMPURE}` },
      { name: "performance", message: `performance ${IMPURE}` },
      { name: "process", message: `process ${IMPURE}` }, // detmath's tests have Node types; its src must not use them
    ],
    "no-restricted-properties": ["error",
      { object: "Math", property: "random", message: `Math.random ${IMPURE} Use createRng from @blanktab/detmath.` },
      ...GLOBAL_HOSTS.flatMap(object => ["Date", "performance"].map(property => ({ object, property, message: `${object}.${property} ${IMPURE}` }))),
    ],
    "no-restricted-imports": ["error", { patterns: [
      ...wallPatterns(pkg),
      { group: ["node:*", ...NODE_MODULES], message: `Node built-ins (IO) ${IMPURE}` },
    ] }],
    "no-restricted-syntax": ["error",
      ...dynamicSelectors(pkg),
      { selector: `ImportExpression > Literal.source[value=${impureSpecifier}]`, message: `Node built-ins (IO) ${IMPURE}` },
      { selector: `CallExpression[callee.name='require'] > Literal.arguments:first-child[value=${impureSpecifier}]`, message: `Node built-ins (IO) ${IMPURE}` },
    ],
  },
})

export default tseslint.config(
  { ignores: ["node_modules/**", "dist/**", "**/dist/**", "app/dist-e2e/**", "test-results/**", "playwright-report/**", ".local/**", ".superpowers/**", ".worktrees/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  { // bench harness scripts run in Node and evaluate snippets in the page
    files: ["packages/*/bench/**/*.mjs"],
    languageOptions: { globals: { console: "readonly", process: "readonly", Buffer: "readonly", window: "readonly" } },
  },
  ...Object.entries(FORBIDDEN).filter(([, b]) => b.length).flatMap(([p]) => [wall(p), dynamicWall(p)]),
  worldPurity,
  ...PURE.map(purity),
)
