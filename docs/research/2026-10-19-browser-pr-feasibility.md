# Can a browser tab open a PR using only in-page JS?

**Date:** 2026-10-19. **Method:** curl probes with an `Origin:` header, imitating a cross-origin page.

## Findings

| Endpoint | Purpose | CORS for browser JS? | Evidence |
|---|---|---|---|
| `api.github.com` (REST) | fork, branch, commit, open PR | **Yes.** `access-control-allow-origin: *`; preflight allows `Authorization`, methods GET/POST/PATCH/PUT/DELETE | OPTIONS preflight returned 204 with those headers |
| `github.com/login/device/code` | start device login | **No.** It returns 200 with a user code but sends no `Access-Control-Allow-Origin`, so the page can't read the response | real POST using GitHub CLI's public client id; preflight 404 |
| `github.com/login/oauth/access_token` | exchange code/device code for token | **No.** Same: 200, no CORS headers; preflight 404 | probe |

## Conclusion

- **With a token, a tab can do the whole PR flow in pure JS** (public repo, REST API).
- **A tab cannot *obtain* a token on its own** through GitHub's login flows, because their responses aren't readable cross-origin.
- Ways to get a token into the tab:
  1. The contributor pastes a fine-grained personal access token once, stored locally. No server, but it needs a GitHub account and a short setup.
  2. A tiny relay (one serverless function) does the OAuth exchange, or holds a GitHub App and opens PRs itself.
  3. Embed a shared token in the public page. **Rejected:** anyone could extract and abuse it.

## Not yet verified
An end-to-end PR opened from a real page. This needs a test repo and a real token, and is a good first spike.
