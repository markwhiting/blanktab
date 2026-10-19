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

export default tseslint.config(
  { ignores: ["node_modules/**", "dist/**", "**/dist/**", ".superpowers/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...Object.entries(FORBIDDEN).filter(([, b]) => b.length).map(([p, b]) => wall(p, b)),
)
