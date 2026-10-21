// A leaf module (no imports), so build tools that run under plain Node (tools/gate/manifest.ts, loaded by the Vite
// config) can read the checkpoint format version without loading the learner.

/** The checkpoint format version; bump it whenever LearnerState or Model.params order changes. */
export const LEARNER_STATE_VERSION = 2
