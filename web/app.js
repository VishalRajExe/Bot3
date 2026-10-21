/* BTC research & paper-trading dashboard.
 * Plain ES2020, no build step.  All rendered values come from the API; nothing
 * is fabricated client-side.  Paper vs backtest vs replay is always labelled.
 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  view: "dashboard",
  tf: "1d",
  market: null,
  paper: null,
  runs: [],
  runDetail: null,
  selectedRunId: null,
  journalMode: "paper",
  journal: null,
  config: null,
  trade: null,
  chart: null,
  candles: null,
  markersApi: null,
  priceLines: [],
  trades: [],
  timer: null,
};

/* ------------------------------------------------------------------ utils */
const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const fmt = {
  money: (v) => (v == null || Number.isNaN(Number(v)) ? "—" : "$" + Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })),
  num: (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d })),
  pct: (v, d = 2) => (v == null || Number.isNaN(Number(v)) ? "—" : (Number(v) >= 0 ? "+" : "") + Number(v).toFixed(d) + "%"),
  int: (v) => (v == null ? "—" : Number(v).toLocaleString()),
  price: (v) => (v == null ? "—" : Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })),
  dt: (ts) => (ts == null ? "—" : new Date(Number(ts) * 1000).toISOString().replace("T", " ").slice(0, 16) + " UTC"),
  date: (ts) => (ts == null ? "—" : new Date(Number(ts) * 1000).toISOString().slice(0, 10)),
  ago: (s) => (s == null ? "—" : s < 90 ? `${Math.round(s)}s` : s < 5400 ? `${Math.round(s / 60)}m` : `${(s / 3600).toFixed(1)}h`),
  cls: (v) => (v == null ? "" : Number(v) > 0 ? "pos" : Number(v) < 0 ? "neg" : ""),
};

function toast(msg, isError = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.className = "toast show" + (isError ? " error" : "");
  clearTimeout(el._t);
  el._t = setTimeout(() => (el.className = "toast"), 4200);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { const j = await res.json(); detail = j.detail || detail; } catch (_) {}
    throw new Error(`${path}: ${detail}`);
  }
  return res.json();
}

const post = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

function card(label, value, sub, cls = "") {
  return `<div class="card"><div class="label">${esc(label)}</div>
    <div class="value ${cls}">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function pill(verdict) {
  const v = String(verdict || "").replace(/[^a-z_]/gi, "");
  return `<span class="pill ${esc(v)}">${esc(verdict)}</span>`;
}

/* --------------------------------------------------------------- SVG chart */
function drawSeries(el, series, opts = {}) {
  const width = 900;
  const height = opts.height || 200;
  const pad = { l: 8, r: 8, t: 10, b: 18 };
  const all = series.flatMap((s) => s.values.filter((v) => v != null && !Number.isNaN(v)));
  if (!all.length) {
    el.innerHTML = `<div class="muted small" style="padding:20px 0">No data to plot.</div>`;
    return;
  }
  let min = Math.min(...all, opts.zero ? 0 : Infinity);
  let max = Math.max(...all, opts.zero ? 0 : -Infinity);
  if (min === max) { min -= 1; max += 1; }
  const n = series[0].values.length;
  const x = (i) => pad.l + (i / Math.max(n - 1, 1)) * (width - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - min) / (max - min)) * (height - pad.t - pad.b);
  let svg = `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${esc(opts.label || "chart")}">`;
  svg += `<line x1="0" y1="${y(0)}" x2="${width}" y2="${y(0)}" stroke="#1e2b3a" stroke-dasharray="4 4"/>`;
  for (const s of series) {
    const pts = s.values.map((v, i) => (v == null || Number.isNaN(v) ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`)).filter(Boolean);
    if (!pts.length) continue;
    if (s.fill) {
      const area = `M${x(0)},${y(Math.max(min, 0))} L${pts.join(" L")} L${x(n - 1)},${y(Math.max(min, 0))} Z`;
      svg += `<path d="${area}" fill="${s.color}" opacity="0.10"/>`;
    }
    svg += `<polyline points="${pts.join(" ")}" fill="none" stroke="${s.color}" stroke-width="${s.width || 1.6}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
  }
  svg += `<text x="6" y="${height - 4}" fill="#8b9bb0" font-size="11">${esc(fmt.num(min, 0))}</text>`;
  svg += `<text x="${width - 6}" y="${height - 4}" fill="#8b9bb0" font-size="11" text-anchor="end">${esc(fmt.num(max, 0))}</text>`;
  svg += `</svg>`;
  el.innerHTML = svg;
}

/* -------------------------------------------------------------------- chart */
function initChart() {
  if (state.chart || !window.LightweightCharts) return;
  const el = $("#chart");
  el.innerHTML = "";
  const chart = LightweightCharts.createChart(el, {
    autoSize: true,
    layout: { background: { color: "#0e1621" }, textColor: "#9fb3c8", fontSize: 11 },
    grid: { vertLines: { color: "#182331" }, horzLines: { color: "#182331" } },
    rightPriceScale: { borderColor: "#1e2b3a" },
    timeScale: { borderColor: "#1e2b3a", rightOffset: 4 },
    crosshair: { mode: 1 },
  });
  const candles = chart.addSeries(LightweightCharts.CandlestickSeries, {
    upColor: "#22c55e", downColor: "#f05252", borderVisible: false,
    wickUpColor: "#22c55e", wickDownColor: "#f05252",
  });
  const volume = chart.addSeries(LightweightCharts.HistogramSeries, {
    priceFormat: { type: "volume" }, priceScaleId: "vol", lastValueVisible: false, priceLineVisible: false,
  });
  chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
  const ema = chart.addSeries(LightweightCharts.LineSeries, { color: "#38bdf8", lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
  const b1u = chart.addSeries(LightweightCharts.LineSeries, { color: "rgba(56,189,248,0.45)", lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
  const b1l = chart.addSeries(LightweightCharts.LineSeries, { color: "rgba(56,189,248,0.45)", lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
  const b2u = chart.addSeries(LightweightCharts.LineSeries, { color: "rgba(56,189,248,0.25)", lineWidth: 1, lineStyle: 3, priceLineVisible: false, lastValueVisible: false });
  const b2l = chart.addSeries(LightweightCharts.LineSeries, { color: "rgba(56,189,248,0.25)", lineWidth: 1, lineStyle: 3, priceLineVisible: false, lastValueVisible: false });
  state.chart = chart;
  state.candles = { candles, volume, ema, b1u, b1l, b2u, b2l };

  chart.subscribeClick((param) => {
    if (!param.time || !state.windowTrades || !state.windowTrades.length) return;
    const t = Number(param.time);
    let best = null;
    for (const tr of state.windowTrades) {
      for (const key of ["entry_ts", "exit_ts"]) {
        if (!tr[key]) continue;
        const d = Math.abs(tr[key] - t);
        if (d <= (best ? best.d : Infinity)) best = { d, tr };
      }
    }
    if (best && best.d <= 86400 * 1.5) showTrade(best.tr);
  });
}

function setChartData(market, trades) {
  initChart();
  if (!state.chart || !market) return;
  const c = market.candles;
  const rows = c.time.map((t, i) => ({ time: t, open: c.open[i], high: c.high[i], low: c.low[i], close: c.close[i] }));
  state.candles.candles.setData(rows);
  state.candles.volume.setData(c.time.map((t, i) => ({ time: t, value: c.volume[i], color: c.close[i] >= c.open[i] ? "rgba(34,197,94,0.35)" : "rgba(240,82,82,0.35)" })));
  const line = (key) => c.time.map((t, i) => ({ time: t, value: c[key] ? c[key][i] : null })).filter((d) => d.value != null);
  state.candles.ema.setData(line("ema65"));
  state.candles.b1u.setData(line("band_1u"));
  state.candles.b1l.setData(line("band_1l"));
  state.candles.b2u.setData(line("band_2u"));
  state.candles.b2l.setData(line("band_2l"));

  state.trades = trades || [];
  // Only draw markers whose timestamp falls inside the loaded candle window.
  // Lightweight Charts clamps out-of-range markers onto the edge bar, which
  // would otherwise stack every historical trade in one column.
  const windowStart = c.time[0];
  const windowEnd = c.time[c.time.length - 1];
  const inWindow = (ts) => ts && ts >= windowStart && ts <= windowEnd;
  const markers = [];
  for (const tr of state.trades) {
    if (inWindow(tr.entry_ts)) markers.push({ time: tr.entry_ts, position: "belowBar", color: "#22c55e", shape: "arrowUp", text: "BUY" });
    if (inWindow(tr.exit_ts)) markers.push({ time: tr.exit_ts, position: "aboveBar", color: "#f05252", shape: "arrowDown", text: "SELL" });
  }
  // The open paper position is drawn separately: it is exposure, not a fill.
  const openPos = state.paper && state.paper.position;
  if (openPos && inWindow(openPos.entry_bar_ts)) {
    markers.push({
      time: openPos.entry_bar_ts, position: "belowBar", color: "#f59e0b",
      shape: "circle", text: "OPEN LONG", size: 2,
    });
  }
  state.visibleMarkers = markers.length;
  state.windowTrades = state.trades.filter((tr) => inWindow(tr.entry_ts));
  markers.sort((a, b) => a.time - b.time);
  if (state.markersApi) state.markersApi.setMarkers(markers);
  else state.markersApi = LightweightCharts.createSeriesMarkers(state.candles.candles, markers);

  for (const pl of state.priceLines) { try { state.candles.candles.removePriceLine(pl); } catch (_) {} }
  state.priceLines = [];
  const pos = state.paper && state.paper.position;
  if (pos && pos.stop_price) {
    state.priceLines.push(state.candles.candles.createPriceLine({
      price: pos.stop_price, color: "#f59e0b", lineWidth: 1, lineStyle: 2,
      axisLabelVisible: true, title: "stop",
    }));
  }
  state.chart.timeScale().fitContent();
}

/* --------------------------------------------------------------- dashboard */
function renderMarketCards(market, paper) {
  const c = market.candles;
  const n = c.close.length;
  const change = n > 1 ? ((c.close[n - 1] / c.close[n - 2] - 1) * 100) : null;
  const spot = market.spot || {};
  const live = spot.price != null;
  const latest = market.latest_signal;
  const pos = paper && paper.position;
  const risk = paper && paper.risk;
  const html = [
    card("BTC spot", live ? fmt.money(spot.price) : "—",
      live ? `${esc(spot.source)} quote · ${fmt.ago(spot.age_seconds)} old` : `unavailable: ${esc(spot.error || "")}`,
      ""),
    card(`Change (${esc(state.tf)})`, fmt.pct(change), `bar close ${fmt.date(c.time[n - 1])}`, fmt.cls(change)),
    card("Market regime", esc(market.regime || "unknown"),
      `last closed ${fmt.dt(new Date(market.last_closed_bar).getTime() / 1000)}`),
    card("Latest decision", latest ? esc(latest.action) : "—",
      latest ? `${esc(latest.strategy)} · ${esc(String(latest.reason).slice(0, 64))}` : "no stored signal yet"),
    card("Paper equity", paper && paper.session ? fmt.money(paper.equity) : "no session",
      paper && paper.session ? `start ${fmt.money(paper.session.initial_equity)} · ${esc(paper.session.status)}` : "start one in Controls"),
    card("Open position", pos ? `${fmt.num(pos.qty, 5)} BTC` : "flat",
      pos ? `entry ${fmt.price(pos.entry_price)} · stop ${fmt.price(pos.stop_price)}` : "no exposure"),
    card("Risk", risk ? `DD ${fmt.num(risk.drawdown_pct, 2)}%` : "—",
      risk ? `daily ${fmt.num(risk.daily_loss_pct, 2)}% · loss streak ${risk.consecutive_losses}` +
        (risk.drawdown_halted ? ' · <span class="neg">HALTED</span>' : "") : "no session"),
  ].join("");
  $("#dashCards").innerHTML = html;
}

function renderSignals(signals) {
  const rows = signals.signals || [];
  if (!rows.length) {
    $("#signalList").innerHTML = `<div class="muted small">No decisions recorded yet. Signals are written when the paper loop evaluates a closed candle.</div>`;
    return;
  }
  $("#signalList").innerHTML = rows.slice(0, 12).map((s) => `
    <div class="row">
      <div>
        <div class="title"><strong>${esc(s.action.toUpperCase())}</strong> · ${esc(s.strategy)} v${esc(s.strategy_version)}</div>
        <div class="meta">${esc(s.reason)}</div>
        <div class="meta">${fmt.dt(s.bar_ts)} · close ${fmt.price(s.price)}</div>
      </div>
      <div class="num ${s.action === "enter_long" ? "pos" : s.action === "exit_long" ? "neg" : "muted"}">${esc(s.action)}</div>
    </div>`).join("");
}

function showTrade(t) {
  state.trade = t;
  const win = (t.net_pnl ?? 0) > 0;
  $("#tradeDetailMode").textContent = t.mode ? `${t.mode}` : "";
  $("#tradeDetail").className = "detail";
  $("#tradeDetail").innerHTML = `
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px">
      <span class="pill ${win ? "win" : "loss"}">${win ? "WIN" : "LOSS"}</span>
      <strong>${esc(t.strategy)} v${esc(t.strategy_version)}</strong>
      <span class="muted small">${esc(t.mode || "")}</span>
    </div>
    <dl>
      <dt>Entry</dt><dd>${fmt.dt(t.entry_ts)} @ ${fmt.price(t.entry_price)}</dd>
      <dt>Exit</dt><dd>${fmt.dt(t.exit_ts)} @ ${fmt.price(t.exit_price)}</dd>
      <dt>Size</dt><dd>${fmt.num(t.qty, 6)} BTC</dd>
      <dt>Stop distance</dt><dd>${fmt.num(t.stop_distance_pct, 2)}%</dd>
      <dt>Net P&amp;L</dt><dd class="${fmt.cls(t.net_pnl)}">${fmt.money(t.net_pnl)} (${fmt.pct(t.return_pct)})</dd>
      <dt>Fees / slippage</dt><dd>${fmt.money(t.fees)} / ${fmt.money(t.slippage_cost)}</dd>
      <dt>R multiple</dt><dd>${t.r_multiple == null ? "—" : fmt.num(t.r_multiple, 2)}</dd>
      <dt>MFE / MAE</dt><dd>${fmt.pct(t.mfe_pct)} / ${fmt.pct(t.mae_pct)}</dd>
      <dt>Holding</dt><dd>${fmt.num((t.holding_hours ?? 0) / 24, 1)} days</dd>
      <dt>Regime</dt><dd>${esc(t.regime || "unknown")}</dd>
      <dt>Entry reason</dt><dd>${esc(t.entry_reason)}</dd>
      <dt>Exit reason</dt><dd>${esc(t.exit_reason)}</dd>
    </dl>
    ${t.anomalies && t.anomalies.length ? `<div class="note">Anomalies: ${esc(t.anomalies.join("; "))}</div>` : ""}`;
}

async function renderPerf() {
  let runId = null;
  let source = "";
  if (state.paper && state.paper.session) {
    runId = state.paper.session.id;
    source = `paper session (${state.paper.session.mode}) — simulated`;
  } else if (state.runs.length) {
    runId = state.runs[0].id;
    source = `backtest run ${state.runs[0].id} — historical simulation`;
  }
  if (!runId) {
    $("#perfCards").innerHTML = "";
    $("#perfChart").innerHTML = `<div class="muted small">No results yet. Run a backtest or start a paper session.</div>`;
    $("#perfSource").textContent = "none";
    return;
  }
  $("#perfSource").textContent = source;
  try {
    const perf = await api(`/api/performance?run_id=${encodeURIComponent(runId)}&mode=${state.paper && state.paper.session ? "paper" : "backtest"}`);
    const m = perf.metrics || {};
    $("#perfCards").innerHTML = [
      card("Net return", fmt.pct(m.net_return_pct), `${m.trades ?? 0} trades`, fmt.cls(m.net_return_pct)),
      card("Max drawdown", fmt.pct(-Math.abs(m.max_drawdown_pct ?? 0)), `${fmt.num(m.drawdown_duration_days, 0)}d longest`, "neg"),
      card("Sharpe / Sortino", `${fmt.num(m.sharpe, 2)} / ${fmt.num(m.sortino, 2)}`, `vol ${fmt.pct(m.volatility_pct)}`),
      card("Win rate", m.win_rate_pct == null ? "—" : fmt.pct(m.win_rate_pct, 1), `PF ${m.profit_factor == null ? "—" : fmt.num(m.profit_factor, 2)}`),
      card("Expectancy", fmt.money(m.expectancy_per_trade), `avg R ${fmt.num(m.avg_r_multiple, 2)}`),
      card("Exposure", fmt.pct(m.exposure_pct, 1), `turnover ${fmt.num(m.turnover_x, 1)}x`),
      card("Fees", fmt.money((m.total_fees || 0) + (m.open_position_fees || 0)),
        m.open_position_fees ? `incl. ${fmt.money(m.open_position_fees)} on the open position` : "realized fees only"),
      card("Period", `${fmt.num(m.period_years, 1)}y`, `${m.trades ?? 0} round trips · ${esc(m.timeframe || "")}`),
    ].join("");
    const eq = perf.equity.map((r) => r.equity);
    drawSeries($("#perfChart"), [
      { values: eq, color: "#38bdf8", fill: true },
    ], { height: 190, label: "equity curve" });
    $("#perfChart").insertAdjacentHTML("beforeend",
      `<div class="muted small">${esc(m.period_start || "")} → ${esc(m.period_end || "")} · ${esc(m.timeframe || "")} · initial ${fmt.money(m.initial_equity)}</div>`);
  } catch (e) {
    $("#perfCards").innerHTML = "";
    $("#perfChart").innerHTML = `<div class="muted small">${esc(e.message)}</div>`;
  }
}

async function loadDashboard() {
  try {
    const [market, paper, signals] = await Promise.all([
      api(`/api/market?timeframe=${state.tf}&limit=420`),
      api("/api/paper"),
      api("/api/signals?limit=15"),
    ]);
    state.market = market;
    state.paper = paper;
    $("#spotPrice").textContent = market.spot && market.spot.price != null ? fmt.money(market.spot.price) : "—";
    $("#spotMeta").textContent = market.spot && market.spot.price != null
      ? `${market.spot.source} · ${fmt.ago(market.spot.age_seconds)} · ${market.last_bar_age_seconds ? "bar " + fmt.ago(market.last_bar_age_seconds) + " old" : ""}`
      : (market.spot && market.spot.error) || "price unavailable";
    renderMarketCards(market, paper);
    renderSignals(signals);
    const replayBadge = $("#replayBadge");
    if (replayBadge) replayBadge.hidden = !(paper.session && paper.session.replay_seeded);

    let trades = [];
    let note = "no paper trades yet";
    try {
      const j = await api("/api/trades?mode=paper&limit=100");
      trades = j.trades.slice().reverse();
      if (trades.length) note = `${trades.length} paper trades`;
    } catch (_) {}
    if (!trades.length && paper && paper.position) note = "open paper position (no closed paper trades yet)";
    if (!trades.length && state.runs.length && !(paper && paper.position)) {
      try {
        const d = await api(`/api/runs/${state.runs[0].id}`);
        trades = d.trades || [];
        note = `backtest markers from ${state.runs[0].strategy} ${state.runs[0].timeframe} (no paper trades yet)`;
      } catch (_) {}
    }
    setChartData(market, trades);
    if (state.visibleMarkers != null) {
      note += ` · ${state.visibleMarkers} marker(s) inside the ${market.candles.time.length}-bar window`;
    }
    if (paper && paper.position && paper.position.stop_price) note += " · dashed line = protective stop";
    $("#chartTfLabel").textContent = state.tf;
    $("#chartNote").innerHTML = `Closed candles only · ${esc(market.data_summary.bars)} cached bars · ` +
      `quality <strong>${esc(market.quality?.status || "?")}</strong> · markers: ${esc(note)}` +
      (market.stale ? ' · <span class="warn">data may be stale</span>' : "");
    await renderPerf();
  } catch (e) {
    toast(e.message, true);
  }
}

/* ----------------------------------------------------------------- research */
async function loadRuns() {
  try {
    const data = await api("/api/runs?limit=80");
    state.runs = data.runs;
    const rows = data.runs.map((r) => `
      <tr data-id="${esc(r.id)}" class="${r.id === state.selectedRunId ? "selected" : ""}">
        <td>${esc(r.strategy)} <span class="muted">v${esc(r.strategy_version)}</span></td>
        <td>${esc(r.timeframe)}</td>
        <td>${fmt.date(new Date(r.start).getTime() / 1000)} → ${fmt.date(new Date(r.end).getTime() / 1000)}</td>
        <td class="${fmt.cls(r.net_return_pct)}">${fmt.pct(r.net_return_pct)}</td>
        <td>${fmt.pct(r.cagr_pct)}</td>
        <td class="neg">${fmt.pct(-Math.abs(r.max_drawdown_pct ?? 0))}</td>
        <td>${fmt.num(r.sharpe, 2)}</td>
        <td>${fmt.int(r.trades)}</td>
        <td>${fmt.pct(r.win_rate_pct, 1)}</td>
        <td>${r.profit_factor == null ? "—" : fmt.num(r.profit_factor, 2)}</td>
        <td class="muted">${fmt.pct(r.buy_hold_return_pct)}</td>
      </tr>`).join("");
    $("#runsTable").innerHTML = `<thead><tr>
        <th>Strategy</th><th>TF</th><th>Period</th><th>Net</th><th>CAGR</th><th>MaxDD</th>
        <th>Sharpe</th><th>Trades</th><th>Win</th><th>PF</th><th>B&amp;H</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="11" class="muted">No backtest runs yet. Run one from Controls.</td></tr>`}</tbody>`;
    $$("#runsTable tbody tr[data-id]").forEach((tr) =>
      tr.addEventListener("click", () => selectRun(tr.dataset.id)));
    if (state.selectedRunId) selectRun(state.selectedRunId, false);
  } catch (e) { toast(e.message, true); }
}

async function selectRun(id, scroll = true) {
  state.selectedRunId = id;
  $$("#runsTable tbody tr").forEach((tr) => tr.classList.toggle("selected", tr.dataset.id === id));
  try {
    const d = await api(`/api/runs/${id}`);
    state.runDetail = d;
    $("#runDetailPanel").hidden = false;
    const m = d.metrics;
    $("#runTitle").textContent = `${d.run.strategy} v${d.run.strategy_version} · ${d.run.timeframe}`;
    $("#runPeriod").textContent = `${m.period_start} → ${m.period_end} (${m.period_years}y) · ${d.run.id}`;
    const bh = d.benchmark.buy_hold || {};
    $("#runCards").innerHTML = [
      card("Net return", fmt.pct(m.net_return_pct), `${m.trades} trades`, fmt.cls(m.net_return_pct)),
      card("CAGR", fmt.pct(m.cagr_pct), `Sharpe ${fmt.num(m.sharpe, 2)}`),
      card("Max drawdown", fmt.pct(-Math.abs(m.max_drawdown_pct ?? 0)), `Calmar ${fmt.num(m.calmar, 2)}`, "neg"),
      card("Win rate", fmt.pct(m.win_rate_pct, 1), `PF ${m.profit_factor == null ? "—" : fmt.num(m.profit_factor, 2)}`),
      card("Exposure", fmt.pct(m.exposure_pct, 1), `fees ${fmt.money(m.total_fees)}`),
      card("Buy & hold", fmt.pct(bh.net_return_pct), `B&amp;H MaxDD ${fmt.pct(-Math.abs(bh.max_drawdown_pct ?? 0))}`),
      card("Excess vs B&H", fmt.pct(m.excess_return_vs_buy_hold_pct), "absolute-return gap", fmt.cls(m.excess_return_vs_buy_hold_pct)),
    ].join("");
    drawSeries($("#runEquity"), [
      { values: d.equity.map((r) => r.equity), color: "#38bdf8", fill: true },
      { values: d.equity.map((r) => r.drawdown), color: "#f05252", width: 1 },
    ], { height: 220, label: "equity and drawdown" });
    $("#runBench").innerHTML = `<dl class="detail">
      <dt>Strategy</dt><dd class="${fmt.cls(m.net_return_pct)}">${fmt.pct(m.net_return_pct)}</dd>
      <dt>Buy &amp; hold</dt><dd class="${fmt.cls(bh.net_return_pct)}">${fmt.pct(bh.net_return_pct)}</dd>
      <dt>Cash</dt><dd>0.00%</dd>
      <dt>Strategy MaxDD</dt><dd>${fmt.pct(-Math.abs(m.max_drawdown_pct ?? 0))}</dd>
      <dt>B&amp;H MaxDD</dt><dd>${fmt.pct(-Math.abs(bh.max_drawdown_pct ?? 0))}</dd>
      <dt>Entry cost model</dt><dd>${esc(JSON.stringify(bh.assumptions || {}))}</dd>
      <dt>Dataset hash</dt><dd>${esc(d.run.dataset_hash || "")}</dd>
      <dt>Costs</dt><dd>${esc(d.run.costs_json)}</dd>
      </dl>`;
    $("#runTradeCount").textContent = `(${d.trades.length})`;
    $("#runTrades").innerHTML = `<thead><tr><th>Entry</th><th>Exit</th><th>Entry px</th><th>Exit px</th>
      <th>Return</th><th>Net</th><th>R</th><th>Regime</th><th>Exit reason</th></tr></thead><tbody>` +
      (d.trades.map((t) => `<tr data-trade='${esc(JSON.stringify(t))}'>
        <td>${fmt.dt(t.entry_ts)}</td><td>${fmt.dt(t.exit_ts)}</td>
        <td>${fmt.price(t.entry_price)}</td><td>${fmt.price(t.exit_price)}</td>
        <td class="${fmt.cls(t.return_pct)}">${fmt.pct(t.return_pct)}</td>
        <td class="${fmt.cls(t.net_pnl)}">${fmt.money(t.net_pnl)}</td>
        <td>${t.r_multiple == null ? "—" : fmt.num(t.r_multiple, 2)}</td>
        <td>${esc(t.regime || "unknown")}</td><td>${esc(String(t.exit_reason).slice(0, 44))}</td></tr>`).join("")
        || `<tr><td colspan="9" class="muted">No trades in this run.</td></tr>`) + `</tbody>`;
    $$("#runTrades tbody tr[data-trade]").forEach((tr) =>
      tr.addEventListener("click", () => { state.trades = d.trades; showTrade(JSON.parse(tr.dataset.trade)); switchView("dashboard"); }));
    $("#runParams").textContent = JSON.stringify({ params: d.params, metrics: {
      sharpe: m.sharpe, sortino: m.sortino, expectancy: m.expectancy_per_trade,
      regime_breakdown: m.regime_breakdown, monthly_returns: m.monthly_returns?.slice(-6) } }, null, 2);
    if (scroll) $("#runDetailPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { toast(e.message, true); }
}

/* ------------------------------------------------------------------ journal */
async function loadJournal() {
  try {
    const j = await api(`/api/journal?mode=${state.journalMode}&limit=250&review=true`);
    state.journal = j;
    const s = j.summary || {};
    $("#journalCards").innerHTML = [
      card("Trades", fmt.int(s.trades), `mode: ${state.journalMode}`),
      card("Win rate", s.win_rate_pct == null ? "—" : fmt.pct(s.win_rate_pct, 1), `${s.wins || 0}W / ${s.losses || 0}L`),
      card("Net P&L", fmt.money(s.net_pnl), "after fees", fmt.cls(s.net_pnl)),
      card("Avg win / loss", `${fmt.money(s.avg_win)} / ${fmt.money(s.avg_loss)}`, ""),
      card("Fees paid", fmt.money(s.total_fees), ""),
      card("Exit reasons", Object.entries(s.exit_reasons || {}).map(([k, v]) => `${esc(k)}:${v}`).join(" · ") || "—", ""),
    ].join("");
    $("#journalCount").textContent = `${j.trades.length} most recent`;
    $("#journalNote").innerHTML = state.journalMode === "paper"
      ? "Paper records only. Includes replay-seeded sessions, which are historical simulations — not live performance."
      : "Backtest records. Simulated with next-bar-open fills, fees, spread and slippage.";
    $("#journalTable").innerHTML = `<thead><tr><th>Entry</th><th>Strategy</th><th>Entry px</th><th>Exit px</th>
      <th>Return</th><th>Net</th><th>R</th><th>Regime</th><th>Exit</th></tr></thead><tbody>` +
      (j.trades.map((t) => `<tr data-trade='${esc(JSON.stringify(t))}'>
        <td>${fmt.dt(t.entry_ts)}</td>
        <td>${esc(t.strategy)} <span class="muted">v${esc(t.strategy_version)}</span></td>
        <td>${fmt.price(t.entry_price)}</td><td>${fmt.price(t.exit_price)}</td>
        <td class="${fmt.cls(t.return_pct)}">${fmt.pct(t.return_pct)}</td>
        <td class="${fmt.cls(t.net_pnl)}">${fmt.money(t.net_pnl)}</td>
        <td>${t.r_multiple == null ? "—" : fmt.num(t.r_multiple, 2)}</td>
        <td>${esc(t.regime || "unknown")}</td>
        <td>${esc(String(t.exit_reason).slice(0, 40))}</td></tr>`).join("")
        || `<tr><td colspan="9" class="muted">No trades recorded for this mode.</td></tr>`) + `</tbody>`;
    $$("#journalTable tbody tr[data-trade]").forEach((tr) =>
      tr.addEventListener("click", () => showTrade(JSON.parse(tr.dataset.trade))));

    $("#findings").innerHTML = (j.findings || []).map((f) => `
      <div class="row"><div><div class="title">${esc(f.finding)}</div>
      <div class="meta">category: ${esc(f.category)} · evidence: ${esc(JSON.stringify(f.evidence))}</div></div></div>`).join("")
      || `<div class="muted small">No findings triggered by the current trades.</div>`;
    $("#riskEvents").innerHTML = (j.risk_events || []).map((e) => `
      <div class="row"><div><div class="title">${esc(e.kind)} → ${esc(e.action)}</div>
      <div class="meta">${fmt.dt(e.ts)} · ${esc(e.detail)}</div></div></div>`).join("")
      || `<div class="muted small">No risk events recorded.</div>`;
  } catch (e) { toast(e.message, true); }
}

/* ----------------------------------------------------------------- learning */
async function loadLearning() {
  try {
    const d = await api("/api/learning");
    $("#expTable").innerHTML = `<thead><tr><th>Experiment</th><th>Verdict</th><th>Hypothesis</th><th>Notes</th><th>When</th></tr></thead><tbody>` +
      (d.experiments.map((e) => `<tr><td>${esc(e.name)}</td><td>${pill(e.verdict)}</td>
        <td>${esc(String(e.hypothesis).slice(0, 90))}</td>
        <td>${esc(String(e.notes).slice(0, 140))}</td><td>${esc(String(e.created_at).slice(0, 16))}</td></tr>`).join("")
        || `<tr><td colspan="5" class="muted">No experiments recorded yet.</td></tr>`) + `</tbody>`;
    $("#modelTable").innerHTML = `<thead><tr><th>Version</th><th>Status</th><th>AUC</th><th>Predictions</th><th>Feature count</th><th>Created</th></tr></thead><tbody>` +
      (d.models.map((m) => `<tr><td>${esc(m.version)}</td><td>${pill(m.status)}</td>
        <td>${fmt.num(m.metrics?.auc, 4)}</td><td>${fmt.int(m.metrics?.predictions)}</td>
        <td>${(m.features || []).length}</td><td>${esc(String(m.created_at).slice(0, 16))}</td></tr>`).join("")
        || `<tr><td colspan="6" class="muted">No models trained yet.</td></tr>`) + `</tbody>`;
    const p = d.promotion_policy || {};
    $("#promotionNote").textContent = `Promotion gate: OOF AUC ≥ ${p.min_oof_auc}, ≥ ${p.min_predictions} predictions, monotonic calibration, and the strategy must beat the baseline after costs. ${p.note || ""}`;
    $("#reviews").innerHTML = (d.reviews || []).map((r) => `
      <div class="row"><div><div class="title">${esc(r.category)}</div>
      <div class="meta">${esc(r.finding)}</div></div></div>`).join("")
      || `<div class="muted small">No stored reviews.</div>`;
    $("#errors").innerHTML = (d.errors || []).map((e) => `
      <div class="row"><div><div class="title">${esc(e.level)} · ${esc(e.message)}</div>
      <div class="meta">${esc(String(e.ts).slice(0, 19))} ${esc(e.logger || "")}</div></div></div>`).join("")
      || `<div class="muted small">No errors logged.</div>`;
  } catch (e) { toast(e.message, true); }
}

/* ---------------------------------------------------------------- sentiment */
async function loadSentiment() {
  try {
    const d = await api("/api/sentiment");
    const fg = d.fear_greed;
    $("#sentCards").innerHTML = [
      card("Fear & Greed", fg ? fmt.num(fg.value, 0) : "—", fg ? `${esc(fg.label)} · ${esc(fg.date)}` : "not fetched"),
      card("Sources", Object.keys(d.coverage || {}).length, Object.entries(d.coverage || {}).map(([k, v]) => `${esc(k)}:${v.count}`).join(" · ")),
      card("Trading use", "disabled", "experimental only"),
    ].join("");
    $("#sentNote").innerHTML = esc(d.note) + (fg ? ` <span class="muted">(${esc(fg.attribution)})</span>` : "");
    $("#headlines").innerHTML = (d.headlines || []).map((h) => `
      <div class="row"><div><div class="title">${esc(h.title)}</div>
      <div class="meta">${esc(h.source)} · ${esc(h.published.slice(0, 16))} · score ${fmt.num(h.score, 2)}</div></div>
      ${h.link ? `<a class="small" href="${esc(h.link)}" target="_blank" rel="noopener noreferrer">source</a>` : ""}</div>`).join("")
      || `<div class="muted small">No headlines stored. Use “Refresh feeds”.</div>`;
  } catch (e) { toast(e.message, true); }
}

/* ----------------------------------------------------------------- controls */
async function loadControls() {
  try {
    const [cfg, strategies, paper] = await Promise.all([
      api("/api/config"), api("/api/strategies"), api("/api/paper"),
    ]);
    state.config = cfg;
    const r = cfg.risk;
    $("#configCards").innerHTML = [
      card("Risk / trade", `${fmt.num(r.risk_per_trade_pct, 2)}%`, "of current equity"),
      card("Max position", `${fmt.num(r.max_position_pct, 0)}%`, "of equity"),
      card("Max positions", r.max_positions, "spot, long only"),
      card("Daily loss halt", `${fmt.num(r.daily_loss_limit_pct, 1)}%`, "blocks new entries"),
      card("Drawdown halt", `${fmt.num(r.max_drawdown_pct, 1)}%`, "manual reset required"),
      card("Loss cooldown", `${r.consecutive_losses_limit} losses`, `${fmt.num(r.cooldown_hours, 0)}h pause`),
      card("Leverage", r.leverage, "cannot be changed"),
      card("Live orders", cfg.trading.enable_live_orders ? "ENABLED" : "disabled", "paper only", cfg.trading.enable_live_orders ? "neg" : "pos"),
    ].join("");

    const sel = $("#btStrategy");
    sel.innerHTML = strategies.strategies.map((s) =>
      `<option value="${esc(s.name)}">${esc(s.name)} v${esc(s.version)} (default ${esc(s.timeframe)})</option>`).join("");
    const tfSel = $("#btTimeframe");
    tfSel.innerHTML = cfg.timeframes.map((t) => `<option value="${esc(t)}">${esc(t)}</option>`).join("");
    sel.addEventListener("change", () => {
      const s = strategies.strategies.find((x) => x.name === sel.value);
      if (s) {
        tfSel.value = s.timeframe;
        $("#btParams").value = JSON.stringify(s.params, null, 2);
      }
    });
    sel.dispatchEvent(new Event("change"));

    const st = paper.session;
    $("#controlStatus").innerHTML = st
      ? `<strong>Session ${esc(st.id)}</strong> · ${esc(st.mode)} · ${esc(st.status)} · ${esc(st.strategy)} ${esc(st.timeframe)}<br>
         last closed bar ${esc(st.last_processed_bar || "—")} · loop ${paper.loop_running ? "running" : "stopped"} ·
         equity ${fmt.money(paper.equity)} · trades ${paper.trades}`
      : "No paper session yet. Starting one begins with a clean simulated balance.";
  } catch (e) { toast(e.message, true); }
}

async function act(fn, okMsg) {
  try { const r = await fn(); toast(okMsg || "ok"); await loadControls(); await loadDashboard(); return r; }
  catch (e) { toast(e.message, true); }
}

/* --------------------------------------------------------------------- nav */
function switchView(view) {
  state.view = view;
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $$(".view").forEach((v) => (v.hidden = v.id !== `view-${view}`));
  const loaders = { dashboard: loadDashboard, research: loadRuns, journal: loadJournal, learning: loadLearning, sentiment: loadSentiment, controls: loadControls };
  if (loaders[view]) loaders[view]();
  if (view === "dashboard" && state.chart) setTimeout(() => state.chart.timeScale().fitContent(), 60);
}

function wire() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
  $$("#tfButtons button").forEach((b) => b.addEventListener("click", async () => {
    state.tf = b.dataset.tf;
    $$("#tfButtons button").forEach((x) => x.classList.toggle("active", x === b));
    await loadDashboard();
  }));
  $$("#journalMode button").forEach((b) => b.addEventListener("click", () => {
    state.journalMode = b.dataset.mode;
    $$("#journalMode button").forEach((x) => x.classList.toggle("active", x === b));
    loadJournal();
  }));
  $("#refreshRuns").addEventListener("click", loadRuns);
  $("#refreshSentiment").addEventListener("click", () => act(() => post("/api/sentiment/refresh"), "Sentiment refreshed"));
  $("#btnStart").addEventListener("click", () => act(() => post("/api/paper/start", { replay_bars: Number($("#replayBars").value) || 0 }), "Paper session started (simulated only)"));
  $("#btnPause").addEventListener("click", () => act(() => post("/api/paper/pause"), "Entries paused; stops still monitored"));
  $("#btnResume").addEventListener("click", () => act(() => post("/api/paper/resume"), "Entries resumed"));
  $("#btnStop").addEventListener("click", () => act(() => post("/api/paper/stop"), "Paper session stopped"));
  $("#btnTick").addEventListener("click", () => act(() => post("/api/paper/tick"), "Poll complete"));
  $("#btnResetDd").addEventListener("click", () => act(() => post("/api/paper/reset-drawdown"), "Drawdown breaker reset"));
  $("#btnBacktest").addEventListener("click", async () => {
    let params = {};
    try { params = JSON.parse($("#btParams").value || "{}"); } catch (e) { toast("Params must be valid JSON", true); return; }
    $("#btResult").textContent = "running…";
    try {
      const r = await post("/api/backtest", {
        strategy: $("#btStrategy").value,
        timeframe: $("#btTimeframe").value,
        params,
        start: $("#btStart").value || null,
        end: $("#btEnd").value || null,
        notes: "requested from dashboard",
      });
      $("#btResult").innerHTML = `<strong>${esc(r.run_id)}</strong> · net ${fmt.pct(r.metrics.net_return_pct)} ·
        maxDD ${fmt.pct(-Math.abs(r.metrics.max_drawdown_pct ?? 0))} · ${r.trades} trades ·
        B&amp;H ${fmt.pct(r.benchmark.buy_hold.net_return_pct)}`;
      toast("Backtest completed");
      await loadRuns();
    } catch (e) {
      $("#btResult").textContent = e.message;
      toast(e.message, true);
    }
  });
}

async function boot() {
  try {
    const h = await api("/api/health");
    $("#healthBadge").textContent = `v${h.version} · ${h.symbol}`;
    $("#healthBadge").className = "badge badge-ok";
    $("#loopBadge").textContent = `loop: ${h.paper_loop_running ? "running" : "stopped"}`;
  } catch (e) {
    $("#healthBadge").textContent = "offline";
    $("#healthBadge").className = "badge badge-bad";
    toast("Backend unreachable: " + e.message, true);
  }
  $("#tfButtons").innerHTML = ["1d", "4h"]
    .map((t) => `<button data-tf="${t}" class="${t === state.tf ? "active" : ""}">${t}</button>`).join("");
  wire();
  await loadRuns();
  await loadDashboard();
  state.timer = setInterval(() => {
    if (state.view === "dashboard") loadDashboard();
    api("/api/health").then((h) => {
      $("#loopBadge").textContent = `loop: ${h.paper_loop_running ? "running" : "stopped"}`;
    }).catch(() => {});
  }, 20000);
}

document.addEventListener("DOMContentLoaded", boot);
