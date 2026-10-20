import { createCpuBackend } from "../src/cpu"
import { describeBackendContract } from "./backend-contract"

describeBackendContract("cpu", () => Promise.resolve(createCpuBackend()))
