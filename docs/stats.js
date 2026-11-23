// Replace with the URL printed by `wrangler deploy` (infra/telemetry-worker/README.md).
const STATS_URL = "https://imcui-telemetry.realcat.workers.dev/stats";

const PALETTE = [
  "#667eea",
  "#f97316",
  "#10b981",
  "#ec4899",
  "#0ea5e9",
  "#8b5cf6",
  "#eab308",
  "#14b8a6",
  "#ef4444",
  "#64748b",
];
const LEADERBOARD_PREVIEW = 12;
const DAY = 86400000;

const el = (id) => document.getElementById(id);
const prefersDark = () =>
  window.matchMedia("(prefers-color-scheme: dark)").matches;
const colorAt = (index) =>
  PALETTE[index] ?? `hsl(${(index * 137.5) % 360}, 60%, 55%)`;

const escapeHtml = (text) =>
  text.replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );

const dayMs = (day) => Date.parse(`${day}T00:00:00Z`);
const fromMs = (ms) => new Date(ms).toISOString().slice(0, 10);
const shortDay = (day) =>
  new Date(`${day}T00:00:00Z`).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  });

function continuousDays(startDay, endDay) {
  const days = [];
  for (let time = dayMs(startDay); time <= dayMs(endDay); time += DAY) {
    days.push(fromMs(time));
  }
  return days;
}

function processRaw({ daily, totals, updated }) {
  const days = [...new Set(daily.map((row) => row.day))].sort();
  const perKey = new Map();
  const dayTotals = new Map();
  for (const { day, key, n } of daily) {
    if (!perKey.has(key)) perKey.set(key, new Map());
    perKey.get(key).set(day, n);
    dayTotals.set(day, (dayTotals.get(day) ?? 0) + n);
  }
  const ordered = [...totals].sort((a, b) => b.n - a.n);
  const rank = new Map(ordered.map((row, index) => [row.key, index]));
  const names = new Map(ordered.map((row) => [row.key, row.name]));
  return { days, perKey, dayTotals, totals: ordered, rank, names, updated };
}

let model = null;
let range = 30;
let expanded = false;

function renderKpis(m) {
  const total = m.totals.reduce((sum, row) => sum + row.n, 0);
  const lastDay = m.days[m.days.length - 1];
  const weekStart = fromMs(dayMs(lastDay) - 6 * DAY);
  let week = 0;
  let peakDay = lastDay;
  let peak = 0;
  for (const [day, n] of m.dayTotals) {
    if (day >= weekStart) week += n;
    if (n > peak) {
      peak = n;
      peakDay = day;
    }
  }

  el("kpi-total").textContent = total.toLocaleString();
  el("kpi-matchers").textContent = m.totals.length.toLocaleString();
  el("kpi-week").textContent = week.toLocaleString();
  el("kpi-peak").textContent = peak.toLocaleString();
  el("kpi-peak-sub").textContent = shortDay(peakDay);
  el("meta").textContent = `${total.toLocaleString()} runs · data fetched ${new Date(
    m.updated,
  ).toLocaleString("en-US")}`;
}

function emptyChart() {
  el("daily").innerHTML = `
    <div class="empty">
      <span class="empty-icon">⌛</span>
      <strong>No runs recorded yet</strong>
      <span>Runs appear here as people use the Space.</span>
    </div>`;
}

function renderChart(m) {
  if (!m.totals.length) {
    emptyChart();
    return;
  }

  const end = m.days[m.days.length - 1];
  const rangeStart = range
    ? fromMs(dayMs(end) - (range - 1) * DAY)
    : m.days[0];
  const start = rangeStart > m.days[0] ? rangeStart : m.days[0];
  const days = continuousDays(start, end);

  const keys = m.totals
    .map((row) => row.key)
    .filter((key) => days.some((day) => (m.perKey.get(key)?.get(day) ?? 0) > 0));

  // One smoothed line per matcher, so every point sits at its own count on
  // the axis.  A single day has no segment to draw, so show a marker instead.
  const line = (name, color, y) => ({
    x: days,
    y,
    customdata: days.map(shortDay),
    name,
    type: "scatter",
    mode: days.length > 1 ? "lines" : "markers",
    line: { width: 1.5, color, shape: "spline", smoothing: 1 },
    marker: { size: 5, color },
    hovertemplate:
      "<b>%{fullData.name}</b><br>%{customdata} · %{y:,.0f}<extra></extra>",
  });
  // Label about a week's worth of days; a category axis labels every one.
  const step = Math.ceil(days.length / 7);
  const tickDays = days.filter((_, index) => index % step === 0);
  const traces = keys.map((key) =>
    line(
      m.names.get(key),
      colorAt(m.rank.get(key)),
      days.map((day) => m.perKey.get(key)?.get(day) ?? 0),
    ),
  );

  const dark = prefersDark();
  Plotly.react(
    "daily",
    traces,
    {
      template: dark ? "plotly_dark" : "plotly_white",
      height: 360,
      margin: { t: 8, r: 6, b: 34, l: 52 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      hovermode: "closest",
      hoverlabel: dark
        ? {
            bgcolor: "#1c2433",
            bordercolor: "#33415c",
            font: { color: "#e8edf4", size: 12 },
          }
        : {
            bgcolor: "#ffffff",
            bordercolor: "#dfe4ec",
            font: { color: "#15181e", size: 12 },
          },
      xaxis: {
        type: "category",
        showgrid: false,
        fixedrange: true,
        tickvals: tickDays,
        ticktext: tickDays.map(shortDay),
        tickfont: { size: 11 },
      },
      yaxis: {
        rangemode: "tozero",
        zeroline: false,
        fixedrange: true,
        gridcolor: dark ? "#232c3b" : "#e8ecf2",
        tickfont: { size: 11 },
        title: { text: "runs", font: { size: 11 } },
      },
      showlegend: false,
    },
    { responsive: true, displayModeBar: false },
  );

  renderLegend(traces);
  el("chart-note").textContent = `${keys.length} matchers · ${
    range ? `last ${days.length} day${days.length === 1 ? "" : "s"}` : "all time"
  }`;
}

function renderLegend(traces) {
  const legend = el("legend");
  legend.innerHTML = "";
  traces.forEach((trace, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<i style="background:${trace.line.color}"></i>${escapeHtml(
      trace.name,
    )}`;
    button.addEventListener("click", () => {
      const hidden = button.classList.toggle("off");
      Plotly.restyle("daily", { visible: hidden ? "legendonly" : true }, [
        index,
      ]);
    });
    legend.append(button);
  });
}

function renderBoard(m) {
  if (!m.totals.length) {
    el("board").innerHTML = '<li class="empty">No runs recorded yet.</li>';
    el("show-all").hidden = true;
    return;
  }

  const rows = expanded ? m.totals : m.totals.slice(0, LEADERBOARD_PREVIEW);
  const max = m.totals[0].n;
  el("board").innerHTML = rows
    .map(
      (row, index) => `
      <li class="row">
        <span class="rank">${String(index + 1).padStart(2, "0")}</span>
        <span class="dot" style="background:${colorAt(index)}"></span>
        <span class="name" title="${escapeHtml(row.key)}">${escapeHtml(
          row.name,
        )}</span>
        <span class="track"><span style="width:${(
          (row.n / max) *
          100
        ).toFixed(1)}%;background:${colorAt(index)}"></span></span>
        <span class="count">${row.n.toLocaleString()}</span>
      </li>`,
    )
    .join("");

  const button = el("show-all");
  button.hidden = m.totals.length <= LEADERBOARD_PREVIEW;
  button.textContent = expanded
    ? "Show less"
    : `Show all ${m.totals.length} matchers`;
}

function renderEmpty(updated) {
  for (const id of ["kpi-total", "kpi-matchers", "kpi-week", "kpi-peak"]) {
    el(id).textContent = "0";
  }
  el("kpi-peak").textContent = "—";
  el("kpi-peak-sub").textContent = "no data";
  el("meta").textContent = `No runs recorded · data fetched ${new Date(
    updated,
  ).toLocaleString("en-US")}`;
  emptyChart();
  renderBoard({ totals: [] });
}

function render(m) {
  if (!m.days.length) {
    renderEmpty(m.updated);
    return;
  }
  renderKpis(m);
  renderChart(m);
  renderBoard(m);
}

document.querySelectorAll(".seg button").forEach((button) => {
  button.addEventListener("click", () => {
    range = Number(button.dataset.range);
    document.querySelectorAll(".seg button").forEach((other) => {
      const active = other === button;
      other.classList.toggle("active", active);
      other.setAttribute("aria-pressed", String(active));
    });
    if (model) renderChart(model);
  });
});

el("show-all").addEventListener("click", () => {
  expanded = !expanded;
  if (model) renderBoard(model);
});

window
  .matchMedia("(prefers-color-scheme: dark)")
  .addEventListener("change", () => {
    if (model?.days.length) renderChart(model);
  });

fetch(STATS_URL)
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((data) => {
    model = processRaw(data);
    render(model);
  })
  .catch((error) => {
    el("error").hidden = false;
    el("error").textContent = `Could not load stats from ${STATS_URL}\n${error.message}`;
    el("meta").textContent = "Stats unavailable";
    emptyChart();
  });
