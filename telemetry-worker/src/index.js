// Anonymous per-matcher usage counter for the imcui HuggingFace Space.
// Deploy steps: see README.md.

const CORS = { "Access-Control-Allow-Origin": "*" };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "POST" && url.pathname === "/track") {
      if (request.headers.get("Authorization") !== `Bearer ${env.TELEMETRY_TOKEN}`) {
        return new Response("forbidden", { status: 403 });
      }
      const { key } = await request.json();
      if (typeof key !== "string" || key.length === 0) {
        return new Response("bad request", { status: 400 });
      }
      const day = new Date().toISOString().slice(0, 10); // UTC
      await env.DB.prepare(
        "INSERT INTO counts (day, key, n) VALUES (?1, ?2, 1) ON CONFLICT(day, key) DO UPDATE SET n = n + 1",
      )
        .bind(day, key)
        .run();
      return new Response("ok");
    }

    if (request.method === "GET" && url.pathname === "/stats") {
      const [daily, totals] = await env.DB.batch([
        env.DB.prepare("SELECT day, key, n FROM counts ORDER BY day, key"),
        env.DB.prepare(
          "SELECT key, SUM(n) AS n FROM counts GROUP BY key ORDER BY n DESC",
        ),
      ]);
      return Response.json(
        {
          daily: daily.results,
          totals: totals.results,
          updated: new Date().toISOString(),
        },
        { headers: CORS },
      );
    }

    return new Response("not found", { status: 404 });
  },
};
