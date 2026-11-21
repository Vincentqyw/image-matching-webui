# imcui telemetry worker

Counts how many times each `matcher_zoo` entry is run in the production
HuggingFace Space. The Space POSTs `{"key": "<matcher_zoo key>"}` after every
completed match; this Worker atomically increments a counter in D1.

Only the matcher key is sent — no images, IPs, or user identifiers.

## Deploy

```bash
cd telemetry-worker
npx wrangler login
npx wrangler d1 create imcui-counts          # copy database_id into wrangler.toml
npx wrangler d1 execute imcui-counts --remote \
  --command "CREATE TABLE counts (day TEXT NOT NULL, key TEXT NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (day, key))"
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
  "totals": [{ "key": "disk+lightglue", "n": 1234 }],
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

The page fetches `/stats` at load: runs-per-day chart, all-time bar chart, and a
breakdown table (total, last 7 days, share).

## Notes

- If `MATCH_TELEMETRY_URL` is unset (local dev, CI), `imcui.ui.telemetry.record`
  is a no-op.
- The Space runs the released `imcui` wheel, so run this with a version that
  includes `imcui/ui/telemetry.py` (>= the release that added it).
- D1 free tier (5M reads / 100k writes per day) is far above what this needs.
