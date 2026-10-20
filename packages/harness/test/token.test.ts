import { describe, it, expect } from "vitest"
import { getToken, setToken, clearToken, checkToken, REPO } from "../src/token"
import { stubStorage } from "./helpers"
import { fakeGithub } from "./fakeGithub"

describe("token storage", () => {
  it("round-trips under blanktab:githubToken and clears", () => {
    const ls = stubStorage()
    expect(getToken(ls)).toBeNull()
    setToken(ls, "  ghp_abc \n")
    expect(ls.getItem("blanktab:githubToken")).toBe("ghp_abc")
    expect(getToken(ls)).toBe("ghp_abc")
    clearToken(ls)
    expect(getToken(ls)).toBeNull()
  })
  it("treats an empty/blank token as no token", () => {
    const ls = stubStorage()
    setToken(ls, "   ")
    expect(getToken(ls)).toBeNull()
  })
  it("names the repo", () => expect(REPO).toEqual({ owner: "markwhiting", name: "blanktab" }))
})

describe("checkToken", () => {
  it("reports login and push access", async () => {
    const gh = fakeGithub({ token: "t", login: "ann", canPush: true })
    expect(await checkToken("t", gh.fetchFn)).toEqual({ ok: true, login: "ann", canPush: true })
    const gh2 = fakeGithub({ token: "t", login: "bob", canPush: false })
    expect(await checkToken("t", gh2.fetchFn)).toEqual({ ok: true, login: "bob", canPush: false })
  })
  it("rejects a bad token with a plain reason", async () => {
    const gh = fakeGithub({ token: "t", login: "ann", canPush: true })
    const r = await checkToken("wrong", gh.fetchFn)
    expect(r.ok).toBe(false)
    if (!r.ok) expect(r.reason).toMatch(/rejected/i)
  })
  it("reports a network failure as not ok", async () => {
    const r = await checkToken("t", (() => Promise.reject(new TypeError("offline"))) as typeof fetch)
    expect(r.ok).toBe(false)
  })
  it("classic token: X-OAuth-Scopes must include public_repo or repo", async () => {
    for (const scopes of ["public_repo", "repo", "gist, public_repo, read:org", "repo,workflow"]) {
      const gh = fakeGithub({ token: "t", login: "ann", canPush: true, scopes })
      expect(await checkToken("t", gh.fetchFn)).toEqual({ ok: true, login: "ann", canPush: true })
    }
    for (const scopes of ["", "gist", "read:repo_hook, public_repo_x", "read:user"]) {
      const gh = fakeGithub({ token: "t", login: "ann", canPush: true, scopes })
      const r = await checkToken("t", gh.fetchFn)
      expect(r.ok).toBe(false)
      if (!r.ok) expect(r.reason).toMatch(/public_repo/)
    }
  })
  it("no X-OAuth-Scopes header (fine-grained token): no scope check", async () => {
    const gh = fakeGithub({ token: "t", login: "ann", canPush: true })
    expect((await checkToken("t", gh.fetchFn)).ok).toBe(true)
  })
})
