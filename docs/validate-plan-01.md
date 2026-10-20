# How to check that Plan 01 works (about 35 minutes, mostly waiting)

Plan 01's goal: a page served from GitHub Pages runs a learner, sends what happened back to the same repo as a pull request, the repo merges it, and the next tab sees it.

## 1. The page runs (2 min)
1. Open https://markwhiting.github.io/blanktab/ in Chrome.
2. You should see:
   - **Two pictures:** the learner's whole screen (shapes on a light background with a dark bar at the bottom), and a zoomed square around its cursor.
   - **A cursor that moves on its own,** and shapes that sometimes change colour when it clicks them.
   - **"Steps lived" counting up** by about 2 per second, out of 3000.
   - **"History: N sessions before this one".**
   - **A "Share this learner's logs" panel,** if you haven't added a token yet.

## 2. Adding a token (3 min)
1. Click **"Create a classic token with the public_repo scope"**. GitHub opens with the scope already ticked. Create the token and copy it.
2. Paste it into the box and click **Check & save**. The panel should disappear. "Contribution" should then show a sharing status and a small **change token** link.
3. Optional check that a bad token is handled: click *change token*, paste `nonsense`, and click **Check & save**. You should get a plain error, and the panel should stay open.

## 3. A real round trip (about 30 min, mostly waiting)
1. Leave the tab open (it can be in the background) for about 25 minutes, until it says **"This learner's life is complete"**.
2. Open https://github.com/markwhiting/blanktab/pulls. Expect a draft PR titled **`run: lrn_…`** that adds about 6 small `.json` files under `data/runs/lrn_…/`.
3. Merge it. Either wait for the hourly gate, or go to **Actions → gate → Run workflow**.
4. Expect:
   - the PR is **merged**;
   - **Actions → deploy** runs green;
   - https://markwhiting.github.io/blanktab/knowledge/summary.json lists one more session;
   - reloading the page shows **"History: N+1 sessions before this one"**.

## 4. The safety rules hold (5 min, optional)
- **Token stays put:** in Chrome DevTools → Network, filter by your token's first few characters. It should appear only in the `Authorization` header of requests to `api.github.com`.
- **Two tabs:** open two tabs at once. Each should show a different **Learner** id.
- **Wrong-branch PR:** a PR that targets a branch other than `main`, or adds anything outside `data/runs/`, is never merged automatically. The gate closes it, or labels it `needs-human`.

## If something looks wrong
- **Actions logs:** https://github.com/markwhiting/blanktab/actions show what happened to each PR.
- **Local tests:** `npm test` and `npx playwright test` run the full checks.
