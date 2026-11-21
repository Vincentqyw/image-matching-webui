# imcui telemetry worker

Counts how many times each `matcher_zoo` entry is run in the production
HuggingFace Space. After every completed match the Space POSTs
`{"key": "<matcher_zoo key>", "name": "<matcher_zoo display name>"}`; this
Worker atomically increments a counter in D1 and keeps the display name in
sync with `matcher_zoo[key]["info"]["name"]`.

Only the matcher key and its label are sent — no images, IPs, or user
identifiers.

## Deploy

```bash
cd infra/telemetry-worker
npx wrangler login
npx wrangler d1 create imcui-counts          # copy database_id into wrangler.toml
npx wrangler d1 execute imcui-counts --remote \
  --command "CREATE TABLE counts (day TEXT NOT NULL, key TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (day, key))"
npx wrangler d1 execute imcui-counts --remote \
  --command "CREATE TABLE names (key TEXT PRIMARY KEY, name TEXT NOT NULL)"
npx wrangler secret put TELEMETRY_TOKEN      # any long random string
npx wrangler deploy
```

Then add two **secrets** to the Space settings
(https://huggingface.co/spaces/Realcat/image-matching-webui/settings):

| Secret                  | Value                          |
| ----------------------- | ------------------------------ |
| `MATCH_TELEMETRY_URL`   | `https://imcui-telemetry.<subdomain>.workers.dev/track` |
| `MATCH_TELEMETRY_TOKEN` | the same string as above       |

Restart the Space after adding the secrets.

## Read the counts

`GET /stats` is public (aggregates only) and sends
`Access-Control-Allow-Origin: *` so the dashboard can read it from GitHub Pages:

```bash
curl https://imcui-telemetry.<subdomain>.workers.dev/stats
```

```json
{
  "daily": [{ "day": "2026-11-21", "key": "disk+lightglue", "n": 12 }],
  "totals": [
    { "key": "disk+lightglue", "name": "DISK + LightGlue", "n": 1234 }
  ],
  "updated": "2026-11-21T12:00:00.000Z"
}
```

`day` is the UTC date the run was recorded.

## Dashboard

`docs/` is a static page served by GitHub Pages:

1. Enable **Settings → Pages → Source: Deploy from a branch → `main` / `/docs`**.
2. Replace `STATS_URL` at the top of `docs/stats.js` with the deployed Worker's
   `/stats` URL.
3. Open `https://<user>.github.io/image-matching-webui/`.

The page fetches `/stats` at load: KPI cards, a stacked runs-per-day chart
(7d / 30d / all), and a ranked leaderboard using the `matcher_zoo` display
names.

## Notes

- If `MATCH_TELEMETRY_URL` is unset (local dev, CI), `imcui.ui.telemetry.record`
  is a no-op.
- Cloudflare's edge rejects Python's default `Python-urllib/*` User-Agent with
  `error code: 1010`, so `telemetry.py` sends a Mozilla-compatible UA.
- The Space runs the released `imcui` wheel, so run this with a version that
  includes `imcui/ui/telemetry.py` (>= the release that added it).
- D1 free tier (5M reads / 100k writes per day) is far above what this needs.
