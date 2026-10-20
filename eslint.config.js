import js from "@eslint/js"
import tseslint from "typescript-eslint"

const PKGS = ["interface", "tensor", "learner", "world", "harness"]

// Walls: which workspace packages each package may NOT import.
const FORBIDDEN = {
  interface: PKGS.filter(p => p !== "interface"),
  tensor: PKGS.filter(p => p !== "tensor"),
  learner: ["world", "harness"],
  world: ["learner", "tensor", "harness"],
  harness: [],
}

const wall = (pkg, banned) => ({
  files: [`packages/${pkg}/**/*.{ts,tsx,js}`],
  rules: {
    "no-restricted-imports": ["error", {
      patterns: banned.flatMap(p => [
        { group: [`@blanktab/${p}`, `@blanktab/${p}/*`], message: `${pkg} must not import ${p} (package wall).` },
        { group: [`**/${p}/**`, `**/packages/${p}`], message: `${pkg} must not import ${p} (package wall).` },
      ]),
    }],
  },
})

// Dynamic import() and require() are invisible to no-restricted-imports; ban them by syntax.
const specifier = p => `/^(@blanktab\\/${p}(\\/.*)?|.*\\/${p}\\/.*|.*\\/packages\\/${p})$/`
const dynamicWall = (pkg, banned) => ({
  files: [`packages/${pkg}/**/*.{ts,tsx,js}`],
  rules: {
    "no-restricted-syntax": ["error", ...banned.flatMap(p => [
      { selector: `ImportExpression > Literal.source[value=${specifier(p)}]`, message: `${pkg} must not import ${p} (package wall).` },
      { selector: `CallExpression[callee.name='require'] > Literal.arguments:first-child[value=${specifier(p)}]`, message: `${pkg} must not require ${p} (package wall).` },
    ])],
  },
})

export default tseslint.config(
  { ignores: ["node_modules/**", "dist/**", "**/dist/**", "app/dist-e2e/**", "test-results/**", "playwright-report/**", ".local/**", ".superpowers/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  { // bench harness scripts run in Node and evaluate snippets in the page
    files: ["packages/*/bench/**/*.mjs"],
    languageOptions: { globals: { console: "readonly", process: "readonly", Buffer: "readonly", window: "readonly" } },
  },
  ...Object.entries(FORBIDDEN).filter(([, b]) => b.length).flatMap(([p, b]) => [wall(p, b), dynamicWall(p, b)]),
)
