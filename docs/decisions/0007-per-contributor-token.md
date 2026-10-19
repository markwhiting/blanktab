# 0007 — Each contributor supplies their own GitHub token, kept in local storage

- **Date:** 2026-10-19
- **Status:** accepted
- **Decided by:** user
- **Resolves:** Q9
- **Supersedes:** the shared bot-token / inbox proposals discussed earlier

## Decision
- A visitor without a stored token sees instructions for creating a GitHub token, ideally via a pre-filled creation link, plus an input field. The token is saved to `localStorage`.
- All tabs on that machine and browser profile share the token. Another machine or browser needs its own token.
- The tab opens PRs directly through the GitHub REST API from the contributor's own fork. There are no servers.

## Agent's calls (not yet confirmed)
- Check the token on entry (e.g. `GET /user` plus a permission check) before saving it, and show a clear error if it's insufficient.
- A tab without a token can still learn locally and queue its results, submitting them once a token is added.
- Each tab gets its own session id (`sessionStorage` or generated), so multiple tabs on one machine are separate trials. Logs record how many sessions shared the machine, because they compete for the GPU.
- Host the page on its own origin. All of an account's `*.github.io` project sites share one origin and therefore one `localStorage`, so any other Pages project on that account could read the token. Use a custom domain or a dedicated account or org.
- Merge safety stays in the repo's rules (decision 0006 paths; never auto-merge code; never run PR code with secrets).

## To verify with a spike
- Which token type (fine-grained vs. classic `public_repo`) can fork and open a PR into the upstream repo with the least access.
- Whether a pre-filled token-creation URL works for that type.
