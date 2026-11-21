// Replace with the URL printed by `wrangler deploy` (README.md).
const STATS_URL = "https://imcui-telemetry.realcat.workers.dev/stats";

const TOP_VISIBLE = 10; // models shown by default before toggling the legend

const prettyKey = (key) => key.split("+").join(" + ");

const escapeHtml = (text) =>
  text.replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );

function colorFor(text) {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash = Math.imul(hash ^ text.charCodeAt(i), 16777619);
  }
  return `hsl(${(hash >>> 0) % 360}, 62%, 52%)`;
}

function showError(message) {
  const box = document.getElementById("error");
  box.textContent = message;
  box.hidden = false;
}

const plotlyConfig = { responsive: true, displaylogo: false };

function render(data) {
  const { daily, totals, updated } = data;
  if (daily.length === 0) {
    showError("No match runs recorded yet.");
    return;
  }

  const days = [...new Set(daily.map((row) => row.day))].sort();
  const perKey = new Map();
  for (const { day, key, n } of daily) {
    if (!perKey.has(key)) perKey.set(key, new Map());
    perKey.get(key).set(day, n);
  }
  const ordered = [...totals].sort((a, b) => b.n - a.n);
  const grandTotal = ordered.reduce((sum, row) => sum + row.n, 0);

  document.getElementById("meta").textContent =
    `${grandTotal.toLocaleString()} runs · ${ordered.length} matchers · ` +
    `data fetched ${new Date(updated).toLocaleString()}`;

  const template = window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "plotly_dark"
    : "plotly_white";

  Plotly.newPlot(
    "daily",
    ordered.map(({ key }, index) => ({
      x: days,
      y: days.map((day) => perKey.get(key)?.get(day) ?? 0),
      name: prettyKey(key),
      mode: "lines+markers",
      line: { color: colorFor(key), width: 2 },
      marker: { size: 5 },
      visible: index < TOP_VISIBLE ? true : "legendonly",
      hovertemplate: `<b>${escapeHtml(prettyKey(key))}</b><br>%{x}<br>runs: %{y}<extra></extra>`,
    })),
    {
      template,
      margin: { t: 10, r: 16, b: 80, l: 56 },
      xaxis: { type: "date" },
      yaxis: { rangemode: "tozero", title: "runs" },
      legend: { orientation: "h", y: -0.35 },
    },
    plotlyConfig,
  );

  const barRows = ordered.slice().reverse();
  Plotly.newPlot(
    "totals",
    [
      {
        type: "bar",
        orientation: "h",
        x: barRows.map((row) => row.n),
        y: barRows.map((row) => prettyKey(row.key)),
        marker: { color: barRows.map((row) => colorFor(row.key)) },
        hovertemplate: "%{y}<br>%{x:,} runs<extra></extra>",
      },
    ],
    {
      template,
      height: Math.max(320, barRows.length * 20),
      margin: { t: 10, r: 24, b: 40, l: 190 },
      xaxis: { title: "total runs" },
    },
    plotlyConfig,
  );

  const lastDay = Date.parse(`${days[days.length - 1]}T00:00:00Z`);
  const weekStart = new Date(lastDay - 6 * 86400000).toISOString().slice(0, 10);

  const rows = ordered
    .map(({ key, n }) => {
      const week = [...perKey.get(key)]
        .filter(([day]) => day >= weekStart)
        .reduce((sum, [, count]) => sum + count, 0);
      return (
        `<tr><td>${escapeHtml(prettyKey(key))}</td>` +
        `<td>${n.toLocaleString()}</td><td>${week.toLocaleString()}</td>` +
        `<td>${((n / grandTotal) * 100).toFixed(1)}%</td></tr>`
      );
    })
    .join("");

  document.getElementById("table").innerHTML =
    "<table><thead><tr><th>Matcher</th><th>Total</th>" +
    `<th>Last 7 days</th><th>Share</th></tr></thead><tbody>${rows}</tbody></table>`;
}

fetch(STATS_URL)
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(render)
  .catch((error) =>
    showError(`Could not load stats from ${STATS_URL}\n${error.message}`),
  );
