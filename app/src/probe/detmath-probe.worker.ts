// The detmath probe's worker (see detmath-probe.ts): the golden fixture's values computed in a module worker.
import { compute } from "../../../packages/detmath/test/golden-inputs"

postMessage(compute())
