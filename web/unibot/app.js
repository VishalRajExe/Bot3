/**
 * UniBot Dashboard — app.js
 * Polls /api/* endpoints every 30s; builds live candlestick charts, market board,
 * position cards, AI model status, trade journal, and research panels.
 */
"use strict";

// ====================================================================== CONFIG
const API = "";           // same-origin
const POLL_MS = 30_000;
const MAX_FEED_ROWS = 60;
const SYMBOLS = [
  "BTC/USD","ETH/USD","SOL/USD","AVAX/USD","LINK/USD",
  "UNI/USD","AAVE/USD","XRP/USD","LTC/USD","DOGE/USD"
];
const SYM_SHORT = {
  "BTC/USD":"BTC","ETH/USD":"ETH","SOL/USD":"SOL","AVAX/USD":"AVAX",
  "LINK/USD":"LINK","UNI/USD":"UNI","AAVE/USD":"AAVE","XRP/USD":"XRP",
  "LTC/USD":"LTC","DOGE/USD":"DOGE"
};
const ACTION_ICON = {
  enter_long:"🟢", exit_long:"🔴", hold:"⚪",
  buy_signal:"🟡", buy_filled:"✅", exit_filled:"🔴",
  blocked:"🚫", stop_filled:"🛑"
};
const REGIME_COLOR = { bull:"#22c55e", bear:"#ef4444", chop:"#f59e0b", unknown:"#94a3b8" };

// ====================================================================== STATE
let state = {
  overview: null, coverage: null, models: null,
  decisions: [], trades: [], runs: [],
  selectedSymbol: "BTC/USD",
  selectedTf: "4H",
  selectedRun: null,
  charts: {},
  feedRows: 0,
};

// ====================================================================== UTILS
const $ = id => document.getElementById(id);
const fmt = (n, d=2) => n == null ? "—" : Number(n).toFixed(d);
const fmtMoney = n => n == null ? "—" : (Math.abs(n)>=1 ? `$${Number(n).toLocaleString("en",{minimumFractionDigits:2,maximumFractionDigits:2})}` : `$${Number(n).toFixed(4)}`);
const fmtPct = (n,d=2) => n == null ? "—" : `${Number(n)>=0?"+":""}${Number(n).toFixed(d)}%`;
const fmtDate = ts => {
  if (!ts) return "—";
  const d = typeof ts === "number" ? new Date(ts * 1000) : new Date(ts);
  return isNaN(d.getTime()) ? "—" : d.toLocaleString();
};
const fmtRelTime = ts => {
  if (!ts) return "—";
  const epochSec = typeof ts === "number" ? ts : Math.floor(new Date(ts).getTime() / 1000);
  if (isNaN(epochSec)) return "—";
  const s = Math.floor(Date.now() / 1000 - epochSec);
  if (s < 0) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s/60)}m ago`;
  if (s < 86400) return `${Math.floor(s/3600)}h ago`;
  return `${Math.floor(s/86400)}d ago`;
};
const colorPnl = n => Number(n) >= 0 ? "td-pos" : "td-neg";
const signPnl  = n => Number(n) >= 0 ? "positive" : "negative";

async function apiFetch(path) {
  const r = await fetch(API + path, {cache:"no-store"});
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

// ====================================================================== INIT
document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initCharts();
  buildSymbolChips();
  buildTfButtons();
  initControls();
  pollAll();
  setInterval(pollAll, POLL_MS);
  setInterval(updateFooterTime, 1000);
});

// ====================================================================== TABS
function initTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
      document.querySelectorAll(".view").forEach(v=>v.classList.remove("active"));
      btn.classList.add("active");
      const view = document.getElementById("view-" + btn.dataset.view);
      if (view) view.classList.add("active");
      // refresh charts on tab show after paint tick so dimensions are non-zero
      if (btn.dataset.view === "overview" && state.overview) {
        setTimeout(() => renderEquityMini(state.overview), 50);
      }
      if (btn.dataset.view === "charts") {
        setTimeout(refreshPriceChart, 50);
      }
      if (btn.dataset.view === "trades") {
        setTimeout(loadTrades, 50);
      }
      if (btn.dataset.view === "models") {
        setTimeout(loadModels, 50);
      }
      if (btn.dataset.view === "backtests") {
        setTimeout(loadRuns, 50);
      }
      if (btn.dataset.view === "risk") {
        setTimeout(loadRiskData, 50);
      }
    });
  });
}

// ====================================================================== CHIPS
function buildSymbolChips() {
  const container = $("symbol-chips");
  if (!container) return;
  container.innerHTML = "";
  SYMBOLS.forEach(sym => {
    const chip = document.createElement("button");
    chip.className = "chip" + (sym === state.selectedSymbol ? " active" : "");
    chip.textContent = SYM_SHORT[sym] || sym;
    chip.dataset.sym = sym;
    chip.addEventListener("click", () => {
      state.selectedSymbol = sym;
      document.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      refreshPriceChart();
    });
    container.appendChild(chip);
  });
}

function buildTfButtons() {
  document.querySelectorAll("#tf-chips .seg").forEach(btn => {
    btn.addEventListener("click", () => {
      state.selectedTf = btn.dataset.tf;
      document.querySelectorAll("#tf-chips .seg").forEach(b=>b.classList.remove("active"));
      btn.classList.add("active");
      refreshPriceChart();
    });
  });
}

// ====================================================================== POLL
async function pollAll() {
  await Promise.allSettled([
    pollOverview(),
    pollDecisions(),
    pollCoverage(),
  ]);
  updateHealth();
}

async function pollOverview() {
  try {
    const d = await apiFetch("/api/overview");
    state.overview = d;
    renderHeader(d);
    renderKpis(d);
    renderPositions(d);
    renderRegime(d);
    renderRiskState(d);
    renderBoard(d);
    renderEquityMini(d);
  } catch(e) { console.warn("overview:", e.message); }
}

async function pollDecisions() {
  try {
    const d = await apiFetch("/api/decisions?limit=80");
    state.decisions = d.decisions || [];
    renderFeed();
  } catch(e) { console.warn("decisions:", e.message); }
}

async function pollCoverage() {
  try {
    const d = await apiFetch("/api/coverage");
    state.coverage = d;
    renderDataTable(d);
  } catch(e) { console.warn("coverage:", e.message); }
}

// ====================================================================== HEALTH
function updateHealth() {
  apiFetch("/api/health").then(d => {
    const dot = $("status-dot");
    const txt = $("status-text");
    const pill = $("pill-loop");
    const alive = d.loop_alive;
    if (dot) {
      dot.className = "status-indicator " + (alive ? "connected" : "error");
      if (txt) txt.textContent = alive ? "live" : "offline";
    }
    if (pill) {
      const ago = d.last_cycle_seconds_ago;
      pill.textContent = alive ? `⚡ cycle ${Math.round(ago)}s ago` : "⚠ loop stopped";
      pill.style.background = alive ? "rgba(34,197,94,.15)" : "rgba(239,68,68,.15)";
      pill.style.color = alive ? "#22c55e" : "#ef4444";
    }
  }).catch(() => {});
}

// ====================================================================== HEADER
function renderHeader(d) {
  const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  const equity = d.equity || 0;
  const open   = d.unrealized_pnl || 0;
  const real   = d.realized_pnl || 0;
  set("hdr-equity",   fmtMoney(equity));
  set("hdr-open",     fmtPct(open / Math.max(equity,1) * 100));
  set("hdr-realized", fmtMoney(real));
  set("hdr-pos",      `${(d.positions||[]).length}/${d.risk_config?.max_positions||2}`);
  const regime = d.regime || {};
  const rEl = $("hdr-regime");
  if (rEl) {
    rEl.textContent = regime.label || "—";
    rEl.style.background = `${REGIME_COLOR[regime.label]||"#94a3b8"}22`;
    rEl.style.color = REGIME_COLOR[regime.label] || "#94a3b8";
  }
}

// ====================================================================== KPIS
function renderKpis(d) {
  const equity = d.equity || 0;
  const start  = d.capital || 100;
  const kpis = [
    { label:"Equity", value: fmtMoney(equity), sub: fmtPct((equity-start)/start*100)+" vs start", cls: signPnl(equity-start) },
    { label:"Realized P&L", value: fmtMoney(d.realized_pnl), cls: signPnl(d.realized_pnl) },
    { label:"Total fees", value: fmtMoney(d.total_fees), cls:"" },
    { label:"Trades closed", value: d.n_trades ?? "—", cls:"accent" },
    { label:"Win rate", value: d.win_rate_pct != null ? fmtPct(d.win_rate_pct,1) : "—", cls:"" },
    { label:"Max drawdown", value: d.max_drawdown_pct != null ? fmtPct(-d.max_drawdown_pct,2) : "—", cls: d.max_drawdown_pct>5?"negative":"" },
    { label:"Regime", value: (d.regime||{}).label || "—", cls:"accent" },
    { label:"Open positions", value: `${(d.positions||[]).length}`, sub:"of max 2", cls:"" },
  ];
  const el = $("kpis");
  if (!el) return;
  el.innerHTML = kpis.map(k => `
    <div class="kpi-card ${k.cls||""}">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      ${k.sub ? `<div class="kpi-sub">${k.sub}</div>` : ""}
    </div>`).join("");
}

// ====================================================================== BOARD
function renderBoard(d) {
  const container = $("board");
  if (!container) return;
  const positions = {};
  (d.positions || []).forEach(p => positions[p.symbol] = p);
  const decisions = {};
  (state.decisions || []).slice(0, SYMBOLS.length * 3).forEach(dec => {
    if (!decisions[dec.symbol]) decisions[dec.symbol] = dec;
  });
  container.innerHTML = "";
  SYMBOLS.forEach(sym => {
    const pos = positions[sym];
    const dec = decisions[sym];
    const price = pos ? pos.current_price : (dec ? dec.price : null);
    const action = dec ? dec.action : "hold";
    let sigCls = "hold", sigText = "WATCH";
    if (action === "enter_long" || action === "buy_signal") { sigCls="enter"; sigText="ENTRY"; }
    else if (action === "exit_long" || action === "exit_filled") { sigCls="exit"; sigText="EXIT"; }
    else if (action === "buy_filled") { sigCls="enter"; sigText="LONG"; }
    const card = document.createElement("div");
    card.className = `board-card${sym===state.selectedSymbol?" selected":""}${pos?" has-position":""}`;
    card.dataset.sym = sym;
    const pnlPct = pos ? ((pos.current_price - pos.entry_price) / pos.entry_price * 100) : null;
    card.innerHTML = `
      <div class="bc-sym">${SYM_SHORT[sym]||sym}${pos?" <span style='color:var(--green);font-size:.65rem'>●</span>":""}</div>
      <div class="bc-price">${price ? (price < 1 ? price.toFixed(5) : price.toLocaleString("en",{maximumFractionDigits:2})) : "—"}</div>
      ${pnlPct != null ? `<div class="bc-change ${pnlPct>=0?"up":"dn"}">${fmtPct(pnlPct,2)}</div>` : ""}
      <div class="bc-row">
        <span class="bc-sig ${sigCls}">${sigText}</span>
        ${dec?.prob != null ? `<span class="bc-prob">AI:${(dec.prob*100).toFixed(0)}%</span>` : ""}
      </div>`;
    card.addEventListener("click", () => {
      state.selectedSymbol = sym;
      document.querySelectorAll(".chip").forEach(c => {
        c.classList.toggle("active", c.dataset.sym === sym);
      });
      document.querySelectorAll(".board-card").forEach(c => {
        c.classList.toggle("selected", c.dataset.sym === sym);
      });
      refreshPriceChart();
    });
    container.appendChild(card);
  });
  const upd = $("board-updated");
  if (upd) upd.textContent = new Date().toLocaleTimeString();
}

// ====================================================================== POSITIONS
function renderPositions(d) {
  const container = $("positions");
  const noPos = $("no-positions");
  const count = $("pos-count");
  const positions = d.positions || [];
  if (count) count.textContent = `${positions.length} / ${d.risk_config?.max_positions||2} slots`;
  if (!container) return;
  if (!positions.length) {
    container.innerHTML = "";
    if (noPos) noPos.classList.add("visible");
    return;
  }
  if (noPos) noPos.classList.remove("visible");
  container.innerHTML = positions.map(p => {
    const pnlPct = p.pnl_pct != null ? p.pnl_pct : 0;
    const pnlCls = pnlPct >= 0 ? "pos" : "neg";
    return `
      <div class="pos-card">
        <div class="pc-head">
          <span class="pc-sym">${p.symbol}</span>
          <span class="pc-pnl ${pnlCls}">${fmtPct(pnlPct)}</span>
        </div>
        <div class="pc-grid">
          <div class="pc-item">Entry <span>${fmtMoney(p.entry_price)}</span></div>
          <div class="pc-item">Price <span>${fmtMoney(p.current_price)}</span></div>
          <div class="pc-item">Stop  <span>${fmtMoney(p.stop_price)}</span></div>
          <div class="pc-item">Size  <span>${fmtMoney(p.notional)}</span></div>
          <div class="pc-item">Qty   <span>${Number(p.qty).toPrecision(5)}</span></div>
          <div class="pc-item">Strat <span>${p.strategy||"—"}</span></div>
        </div>
      </div>`;
  }).join("");
}

// ====================================================================== REGIME
function renderRegime(d) {
  const container = $("regime-body");
  if (!container) return;
  const r = d.regime || {};
  const items = [
    { label:"Label",   value: r.label || "—" },
    { label:"Score",   value: r.score != null ? fmt(r.score,3) : "—" },
    { label:"BTC trend",  value: r.btc_trend != null ? fmtPct(r.btc_trend*100,1) : "—" },
    { label:"Breadth",    value: r.breadth != null ? fmtPct(r.breadth*100,1) : "—" },
    { label:"Vol %ile",   value: r.vol_percentile != null ? fmtPct(r.vol_percentile,0) : "—" },
    { label:"Updated",    value: d.regime_ts ? fmtRelTime(d.regime_ts) : "—" },
  ];
  container.innerHTML = items.map(i => `
    <div class="regime-item">
      <div class="ri-label">${i.label}</div>
      <div class="ri-value" style="color:${i.label==='Label'?(REGIME_COLOR[r.label]||'#e2e8f0'):'inherit'}">${i.value}</div>
    </div>`).join("");
  const at = $("regime-at");
  if (at) at.textContent = d.regime_ts ? fmtRelTime(d.regime_ts) : "";
}

// ====================================================================== RISK STATE
function renderRiskState(d) {
  const container = $("risk-body");
  if (!container) return;
  const b = d.breakers || {};
  const items = [
    { label:"Daily loss limit", halted: b.daily_halted, value: b.daily_halted ? "HALTED":"OK" },
    { label:"Drawdown limit",   halted: b.drawdown_halted, value: b.drawdown_halted ? "LATCHED":"OK" },
    { label:"Consec losses",    halted: false, value: `${d.max_consec_losses||0} losses` },
    { label:"Cash reserve",     halted: false, value: fmtPct(d.cash_pct||0,1) },
  ];
  container.innerHTML = items.map(i => {
    const cls = i.halted ? "halted" : "ok";
    return `<div class="risk-item ${cls}">
      <div class="ri-label">${i.label}</div>
      <div class="ri-value">${i.value}</div>
    </div>`;
  }).join("");
}

// ====================================================================== EQUITY MINI
let equityChart = null;
function renderEquityMini(d) {
  const canvas = $("equity-mini");
  if (!canvas) return;
  const eq = d.equity_history || [];
  const footer = $("equity-footer");
  if (!eq.length) {
    if (footer) footer.innerHTML = `<span>Capital: $${(d.capital||100).toFixed(2)}</span><span>Waiting for cycles…</span>`;
    return;
  }
  const pts = eq.map(r => ({ x: r.ts * 1000, y: Number(r.equity) }));
  if (equityChart) {
    try { equityChart.destroy(); } catch (_) {}
    equityChart = null;
  }
  const start = pts[0].y;
  const end = pts[pts.length - 1].y;
  const up = end >= start;
  const ctx = canvas.getContext("2d");
  equityChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [{
        label: "Equity",
        data: pts,
        borderColor: up ? "#22c55e" : "#ef4444",
        borderWidth: 2,
        fill: true,
        backgroundColor: up ? "rgba(34,197,94,0.08)" : "rgba(239,68,68,0.08)",
        pointRadius: 0,
        tension: 0.2,
        spanGaps: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: ctx => ctx[0]?.raw?.x ? new Date(ctx[0].raw.x).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : "",
            label: ctx => `Equity: $${Number(ctx.raw?.y || ctx.raw || 0).toFixed(2)}`
          }
        }
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "hour", displayFormats: { hour: "HH:mm", day: "MMM d" } },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#64748b", maxTicksLimit: 5 }
        },
        y: {
          position: "right",
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#64748b", callback: v => `$${Number(v).toFixed(0)}` }
        }
      }
    }
  });
  if (footer) {
    const dd = d.max_drawdown_pct || 0;
    footer.innerHTML = `<span>Start: $${(d.capital||100).toFixed(2)}</span>
      <span style="color:${up ? "#22c55e" : "#ef4444"};font-weight:600">Now: ${fmtMoney(end)}</span>
      <span>Max DD: ${fmtPct(-dd)}</span>`;
  }
}

// ====================================================================== FEED
function renderFeed() {
  const container = $("decisions");
  if (!container) return;
  const rows = (state.decisions || []).slice(0, MAX_FEED_ROWS);
  container.innerHTML = rows.map(d => {
    const icon = ACTION_ICON[d.action] || "ℹ️";
    const actCls = d.action.includes("long") || d.action.includes("buy") ? "buy"
                 : d.action.includes("exit") || d.action.includes("stop") ? "sell"
                 : "hold";
    return `<div class="feed-row anim-in">
      <span class="fr-icon">${icon}</span>
      <span class="fr-sym">${SYM_SHORT[d.symbol]||d.symbol}</span>
      <span class="fr-act ${actCls}">${d.action.replace("_"," ")}</span>
      <span class="fr-reason" title="${(d.reason||"").replace(/"/g,"'")}">
        ${(d.reason||"").slice(0,90)}
      </span>
      <span class="fr-time">${fmtRelTime(d.ts)}</span>
    </div>`;
  }).join("");
}

// ====================================================================== PRICE CHART
let priceChart = null, rsiChart = null, atrChart = null;

function initCharts() {
  // Charts initialized on demand
}

async function refreshPriceChart() {
  const sym = state.selectedSymbol;
  const meta = $("chart-meta");
  if (meta) meta.textContent = `loading ${sym}…`;
  try {
    const tf = state.selectedTf === "4H" ? "4h" : "1d";
    const d = await apiFetch(`/api/candles/${encodeURIComponent(sym)}?timeframe=${tf}&limit=300`);
    const bars = d.candles || [];
    if (!bars.length) {
      if (meta) meta.textContent = "no data";
      return;
    }
    const signals = d.signals || [];
    buildPriceChart(bars, signals, sym);
    buildRsiChart(bars);
    buildAtrChart(bars);
    if (meta) meta.textContent = `${sym} · ${tf} · ${bars.length} bars · last ${fmtRelTime(d.last_ts)}`;
    renderSnapshot(d.snapshot || {});
  } catch(e) {
    if (meta) meta.textContent = "error: " + e.message;
  }
}

function buildPriceChart(bars, signals, sym) {
  const canvas = $("price-chart");
  if (!canvas) return;
  if (priceChart) {
    try { priceChart.destroy(); } catch (_) {}
    priceChart = null;
  }

  const ohlc = bars.map(b => ({
    x: b.ts * 1000,
    o: Number(b.open),
    h: Number(b.high),
    l: Number(b.low),
    c: Number(b.close)
  }));
  const ema50 = bars.map(b => ({ x: b.ts * 1000, y: b.ema50 != null ? Number(b.ema50) : null }));
  const ema200 = bars.map(b => ({ x: b.ts * 1000, y: b.ema200 != null ? Number(b.ema200) : null }));
  const bbUpper = bars.map(b => ({ x: b.ts * 1000, y: b.bb_upper != null ? Number(b.bb_upper) : null }));
  const bbLower = bars.map(b => ({ x: b.ts * 1000, y: b.bb_lower != null ? Number(b.bb_lower) : null }));

  const entryPts = [], exitPts = [];
  (signals || []).forEach(s => {
    const pt = { x: s.ts * 1000, y: Number(s.price), reason: s.reason, action: s.action };
    if (s.action === "enter_long" || s.action === "buy_filled" || s.action === "buy_signal") {
      entryPts.push(pt);
    } else if (s.action === "exit_long" || s.action === "exit_filled" || s.action === "stop_filled") {
      exitPts.push(pt);
    }
  });

  const ctx = canvas.getContext("2d");
  priceChart = new Chart(ctx, {
    type: "candlestick",
    data: {
      datasets: [
        {
          type: "candlestick",
          label: sym,
          data: ohlc,
          color: { up: "#22c55e", down: "#ef4444", unchanged: "#94a3b8" },
          borderColor: { up: "#22c55e", down: "#ef4444", unchanged: "#94a3b8" },
        },
        {
          type: "line",
          label: "EMA 50",
          data: ema50,
          borderColor: "#f59e0b",
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0,
          spanGaps: true,
        },
        {
          type: "line",
          label: "EMA 200",
          data: ema200,
          borderColor: "#06b6d4",
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0,
          spanGaps: true,
        },
        {
          type: "line",
          label: "BB Upper",
          data: bbUpper,
          borderColor: "rgba(168,85,247,0.5)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          borderDash: [3, 3],
          spanGaps: true,
        },
        {
          type: "line",
          label: "BB Lower",
          data: bbLower,
          borderColor: "rgba(168,85,247,0.5)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          borderDash: [3, 3],
          spanGaps: true,
        },
        {
          type: "scatter",
          label: "Entry Signal",
          data: entryPts,
          backgroundColor: "#22c55e",
          borderColor: "#ffffff",
          borderWidth: 1.5,
          pointRadius: 6,
          pointStyle: "triangle",
        },
        {
          type: "scatter",
          label: "Exit Signal",
          data: exitPts,
          backgroundColor: "#ef4444",
          borderColor: "#ffffff",
          borderWidth: 1.5,
          pointRadius: 6,
          pointStyle: "triangle",
          rotation: 180,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: {
          display: true,
          position: "top",
          labels: { color: "#94a3b8", font: { size: 11 }, boxWidth: 12, padding: 12 },
        },
        tooltip: {
          callbacks: {
            title: ctx => {
              const p = ctx[0];
              if (!p) return "";
              const t = p.raw?.x || p.parsed?.x;
              return t ? new Date(t).toLocaleString([], { dateStyle: "short", timeStyle: "short" }) : "";
            },
            label: ctx => {
              const d = ctx.raw;
              if (!d) return "";
              if (d.o != null) {
                return [
                  `Open:   $${fmt(d.o, 2)}`,
                  `High:   $${fmt(d.h, 2)}`,
                  `Low:    $${fmt(d.l, 2)}`,
                  `Close:  $${fmt(d.c, 2)}`
                ];
              }
              if (d.action) {
                return `${d.action.toUpperCase()}: $${fmtMoney(d.y)} (${d.reason || "signal"})`;
              }
              if (d.y != null) {
                return `${ctx.dataset.label}: $${fmt(d.y, 2)}`;
              }
              return `${ctx.dataset.label}: ${fmt(d, 2)}`;
            }
          }
        }
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "day", displayFormats: { day: "MMM d" } },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#64748b", maxTicksLimit: 8 }
        },
        y: {
          position: "right",
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: {
            color: "#64748b",
            callback: v => v >= 1 ? `$${Number(v).toLocaleString("en", { maximumFractionDigits: 2 })}` : `$${Number(v).toFixed(4)}`
          }
        }
      }
    }
  });
}

function buildRsiChart(bars) {
  const canvas = $("rsi-chart");
  if (!canvas) return;
  if (rsiChart) {
    try { rsiChart.destroy(); } catch (_) {}
    rsiChart = null;
  }
  const rsi = bars.map(b => ({ x: b.ts * 1000, y: b.rsi14 != null ? Number(b.rsi14) : null }));
  const ob  = bars.map(b => ({ x: b.ts * 1000, y: 70 }));
  const os  = bars.map(b => ({ x: b.ts * 1000, y: 30 }));
  const ctx = canvas.getContext("2d");
  rsiChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "RSI (14)",
          data: rsi,
          borderColor: "#a855f7",
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
          tension: 0.1,
          spanGaps: true,
        },
        {
          label: "Overbought (70)",
          data: ob,
          borderColor: "rgba(239,68,68,0.45)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          borderDash: [4, 4],
        },
        {
          label: "Oversold (30)",
          data: os,
          borderColor: "rgba(34,197,94,0.45)",
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
          borderDash: [4, 4],
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: ctx => ctx[0]?.raw?.x ? new Date(ctx[0].raw.x).toLocaleDateString([], { month: "short", day: "numeric" }) : "",
            label: ctx => ctx.raw?.y != null ? `RSI: ${Number(ctx.raw.y).toFixed(1)}` : ""
          }
        }
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "day" },
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#64748b", maxTicksLimit: 5 }
        },
        y: {
          min: 0,
          max: 100,
          position: "right",
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#64748b", stepSize: 25 }
        }
      }
    }
  });
}

function buildAtrChart(bars) {
  const canvas = $("atr-chart");
  if (!canvas) return;
  if (atrChart) {
    try { atrChart.destroy(); } catch (_) {}
    atrChart = null;
  }
  const atr = bars.map(b => ({ x: b.ts * 1000, y: b.atr_pct != null ? Number(b.atr_pct) : null }));
  const ctx = canvas.getContext("2d");
  atrChart = new Chart(ctx, {
    type: "bar",
    data: {
      datasets: [{
        label: "ATR %",
        data: atr,
        backgroundColor: "rgba(99,102,241,0.5)",
        borderColor: "transparent",
        barThickness: "flex",
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: ctx => ctx[0]?.raw?.x ? new Date(ctx[0].raw.x).toLocaleDateString([], { month: "short", day: "numeric" }) : "",
            label: ctx => ctx.raw?.y != null ? `ATR: ${Number(ctx.raw.y).toFixed(2)}%` : ""
          }
        }
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "day" },
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#64748b", maxTicksLimit: 5 }
        },
        y: {
          position: "right",
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#64748b", callback: v => `${Number(v).toFixed(1)}%` }
        }
      }
    }
  });
}

function renderSnapshot(snap) {
  const container = $("symbol-snapshot");
  const sym = $("snap-sym");
  if (!container) return;
  if (sym) sym.textContent = state.selectedSymbol;
  const items = [
    { label:"Price",    value: snap.price != null ? fmtMoney(snap.price) : "—" },
    { label:"RSI 14",   value: snap.rsi14 != null ? fmt(snap.rsi14) : "—" },
    { label:"ATR %",    value: snap.atr_pct != null ? fmtPct(snap.atr_pct,2) : "—" },
    { label:"EMA 50",   value: snap.ema50 != null ? fmtMoney(snap.ema50) : "—" },
    { label:"EMA 200",  value: snap.ema200 != null ? fmtMoney(snap.ema200) : "—" },
    { label:"Trend",    value: snap.trend || "—" },
    { label:"Signal",   value: snap.signal || "—" },
    { label:"AI prob",  value: snap.prob != null ? fmtPct(snap.prob*100,1) : "—" },
    { label:"Regime",   value: snap.regime || "—" },
    { label:"Bars",     value: snap.n_bars != null ? snap.n_bars : "—" },
  ];
  container.innerHTML = items.map(i => `
    <div class="snap-item">
      <div class="si-label">${i.label}</div>
      <div class="si-value">${i.value}</div>
    </div>`).join("");
}

// ====================================================================== TRADES TAB
async function loadTrades() {
  try {
    const mode = ($("trade-mode")||{}).value || "live";
    const sym  = ($("trade-sym-filter")||{}).value || "";
    const url  = `/api/trades?mode=${mode}&limit=200${sym?`&symbol=${encodeURIComponent(sym)}`:""}&sort=desc`;
    const d    = await apiFetch(url);
    state.trades = d.trades || [];
    renderTradesTable(state.trades);
    renderTradeKpis(d);
    renderExitChart(state.trades);
    renderPnlHist(state.trades);
    renderTradeEquity(state.trades);
  } catch(e) { console.warn("trades:", e.message); }
}

function renderTradesTable(trades) {
  const container = $("trades-table");
  const count = $("trade-count");
  if (count) count.textContent = `${trades.length} trades`;
  if (!container) return;
  if (!trades.length) { container.innerHTML = `<div class="note">No trades found.</div>`; return; }
  const rows = trades.map(t => {
    const win = t.net_pnl > 0;
    const stop = t.exit_signal === "stop" || t.exit_reason?.includes("stop");
    const badge = stop ? "stop" : (win ? "win" : "loss");
    return `<tr>
      <td class="td-sym">${SYM_SHORT[t.symbol]||t.symbol}</td>
      <td>${t.strategy||"—"}</td>
      <td class="td-mono">${fmtDate(t.entry_ts)}</td>
      <td class="td-mono">${fmtMoney(t.entry_price)}</td>
      <td class="td-mono">${fmtMoney(t.exit_price)}</td>
      <td class="td-mono ${colorPnl(t.net_pnl)}">${fmtPct(t.return_pct)}</td>
      <td class="td-mono ${colorPnl(t.net_pnl)}">${fmtMoney(t.net_pnl)}</td>
      <td class="td-mono">${fmtMoney(t.fees)}</td>
      <td><span class="td-badge ${badge}">${t.exit_reason||t.exit_signal||"—"}</span></td>
      <td class="td-mono muted">${t.holding_hours ? fmt(t.holding_hours,1)+"h" : "—"}</td>
    </tr>`;
  }).join("");
  container.innerHTML = `<table>
    <thead><tr><th>Symbol</th><th>Strategy</th><th>Entry time</th><th>Entry $</th><th>Exit $</th><th>Return</th><th>Net P&L</th><th>Fees</th><th>Exit reason</th><th>Held</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderTradeKpis(d) {
  const container = $("trade-kpis");
  if (!container) return;
  const trades = d.trades || [];
  const wins = trades.filter(t => t.net_pnl > 0);
  const losses = trades.filter(t => t.net_pnl <= 0);
  const totalPnl = trades.reduce((s,t) => s + (t.net_pnl||0), 0);
  const wr = trades.length ? wins.length/trades.length*100 : 0;
  const avgWin = wins.length ? wins.reduce((s,t)=>s+t.net_pnl,0)/wins.length : 0;
  const avgLoss = losses.length ? losses.reduce((s,t)=>s+t.net_pnl,0)/losses.length : 0;
  const grossProfit = wins.reduce((s,t)=>s+t.net_pnl,0);
  const grossLoss   = Math.abs(losses.reduce((s,t)=>s+t.net_pnl,0));
  const pf = grossLoss > 0 ? grossProfit/grossLoss : 0;
  const kpis = [
    { label:"Trades",       value: trades.length, cls:"" },
    { label:"Win rate",     value: fmtPct(wr,1), cls: wr>=50?"positive":"negative" },
    { label:"Net P&L",      value: fmtMoney(totalPnl), cls: signPnl(totalPnl) },
    { label:"Avg win",      value: fmtMoney(avgWin), cls:"positive" },
    { label:"Avg loss",     value: fmtMoney(avgLoss), cls:"negative" },
    { label:"Profit factor",value: fmt(pf,2), cls: pf>=1?"positive":"negative" },
  ];
  container.innerHTML = kpis.map(k => `
    <div class="kpi-card ${k.cls}">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
    </div>`).join("");
}

let exitChart = null, pnlHist = null, tradeEquityChart = null;

function renderExitChart(trades) {
  const wrap = $("exit-wrap");
  if (exitChart) {
    try { exitChart.destroy(); } catch (_) {}
    exitChart = null;
  }
  if (!wrap) return;
  if (!trades.length) {
    wrap.innerHTML = `<canvas id="exit-chart" style="display:none"></canvas><div class="chart-empty-placeholder">No closed trades recorded yet in live paper session.</div>`;
    return;
  }
  if (!wrap.querySelector("#exit-chart")) {
    wrap.innerHTML = `<canvas id="exit-chart"></canvas>`;
  }
  const canvas = $("exit-chart");
  if (!canvas) return;
  canvas.style.display = "block";
  const reasons = {};
  trades.forEach(t => {
    const r = t.exit_signal || t.exit_reason || "other";
    const key = r.includes("stop") ? "Stop loss" : (r === "exit_long" ? "Strategy exit" : "Take profit / other");
    reasons[key] = (reasons[key] || 0) + 1;
  });
  exitChart = new Chart(canvas.getContext("2d"), {
    type: "doughnut",
    data: {
      labels: Object.keys(reasons),
      datasets: [{
        data: Object.values(reasons),
        backgroundColor: ["#ef4444", "#22c55e", "#f59e0b", "#6366f1"],
        borderColor: "rgba(0,0,0,0.2)",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right", labels: { color: "#94a3b8", font: { size: 11 }, padding: 10 } }
      }
    }
  });
}

function renderPnlHist(trades) {
  const wrap = $("pnl-wrap");
  if (pnlHist) {
    try { pnlHist.destroy(); } catch (_) {}
    pnlHist = null;
  }
  if (!wrap) return;
  if (!trades.length) {
    wrap.innerHTML = `<canvas id="pnl-hist" style="display:none"></canvas><div class="chart-empty-placeholder">No closed trades recorded yet in live paper session.</div>`;
    return;
  }
  if (!wrap.querySelector("#pnl-hist")) {
    wrap.innerHTML = `<canvas id="pnl-hist"></canvas>`;
  }
  const canvas = $("pnl-hist");
  if (!canvas) return;
  canvas.style.display = "block";
  const pnls = trades.map(t => Number(t.net_pnl || 0));
  const colors = pnls.map(p => p >= 0 ? "rgba(34,197,94,0.7)" : "rgba(239,68,68,0.7)");
  pnlHist = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: pnls.map((_, i) => `#${i + 1}`),
      datasets: [{
        label: "Net P&L",
        data: pnls,
        backgroundColor: colors,
        borderColor: "transparent"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: c => `P&L: ${fmtMoney(c.raw)}` } }
      },
      scales: {
        x: { ticks: { display: false }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b", callback: v => fmtMoney(v) } }
      }
    }
  });
}

function renderTradeEquity(trades) {
  const wrap = $("trade-equity-wrap");
  if (tradeEquityChart) {
    try { tradeEquityChart.destroy(); } catch (_) {}
    tradeEquityChart = null;
  }
  if (!wrap) return;
  if (!trades.length) {
    wrap.innerHTML = `<canvas id="trade-equity" style="display:none"></canvas><div class="chart-empty-placeholder">No closed trades recorded yet to calculate cumulative curve.</div>`;
    return;
  }
  if (!wrap.querySelector("#trade-equity")) {
    wrap.innerHTML = `<canvas id="trade-equity"></canvas>`;
  }
  const canvas = $("trade-equity");
  if (!canvas) return;
  canvas.style.display = "block";
  let running = 100.0;
  const pts = [{ x: (trades[0].entry_ts - 3600) * 1000, y: 100.0 }];
  trades.forEach(t => {
    running += Number(t.net_pnl || 0);
    pts.push({ x: (t.exit_ts || t.entry_ts) * 1000, y: running });
  });

  tradeEquityChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      datasets: [{
        label: "Realized Equity",
        data: pts,
        borderColor: "#6366f1",
        backgroundColor: "rgba(99,102,241,0.08)",
        borderWidth: 2,
        fill: true,
        pointRadius: 2,
        tension: 0.1,
        stepped: "after"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: ctx => ctx[0]?.raw?.x ? new Date(ctx[0].raw.x).toLocaleString() : "",
            label: ctx => `Equity: $${Number(ctx.raw?.y || 0).toFixed(2)}`
          }
        }
      },
      scales: {
        x: {
          type: "time",
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#64748b", maxTicksLimit: 6 }
        },
        y: {
          grid: { color: "rgba(255,255,255,0.04)" },
          ticks: { color: "#64748b", callback: v => `$${Number(v).toFixed(2)}` }
        }
      }
    }
  });
}

// ====================================================================== MODELS TAB
async function loadModels() {
  try {
    const d = await apiFetch("/api/models");
    renderModelsTable(d.models||[]);
    renderLearningRules(d.learning_state||{});
    renderLessons(d.lessons||[]);
    renderCohorts(d.cohorts||[]);
    renderExperiments(d.experiments||[]);
    const lc = $("lesson-count");
    if (lc) lc.textContent = `${(d.lessons||[]).length} recorded`;
  } catch(e) { console.warn("models:", e.message); }
}

function renderModelsTable(models) {
  const container = $("models-table");
  if (!container) return;
  if (!models.length) { container.innerHTML=`<div class="note">No models trained yet. The bot trains after ${25} closed trades.</div>`; return; }
  const rows = models.map(m => `<tr>
    <td class="td-sym">${SYM_SHORT[m.symbol]||m.symbol}</td>
    <td>${m.strategy||"—"}</td>
    <td class="td-mono">${m.version||"—"}</td>
    <td class="td-mono ${m.status==="active"?"td-pos":"td-neg"}">${m.status||"—"}</td>
    <td class="td-mono">${m.auc != null ? fmt(m.auc,3) : "—"}</td>
    <td class="td-mono">${m.threshold != null ? fmt(m.threshold,2) : "—"}</td>
    <td class="td-mono">${m.test_samples||"—"}</td>
    <td class="muted small">${fmtDate(m.trained_at)}</td>
  </tr>`).join("");
  container.innerHTML = `<table>
    <thead><tr><th>Symbol</th><th>Strategy</th><th>Version</th><th>Status</th><th>OOS AUC</th><th>Gate</th><th>Test N</th><th>Trained</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderLearningRules(ls) {
  const container = $("learning-rules");
  if (!container) return;
  const risk_mult = ls.risk_multipliers || {};
  const cooldowns = ls.cooldowns || {};
  const items = [
    { title:"Symbol risk multipliers", desc:Object.entries(risk_mult).map(([s,v])=>`${s}: ${(v*100).toFixed(0)}%`).join(" · ")||"all at 100%", icon:"⚖️" },
    { title:"Active cooldowns", desc:Object.keys(cooldowns).join(", ")||"none", icon:"⏱️" },
    { title:"Model gates active", desc:`${ls.n_gates||0} symbols using AI gate`, icon:"🧠" },
    { title:"Lessons stored", desc:`${ls.n_lessons||0} heuristic reviews from closed trades`, icon:"📚" },
  ];
  container.innerHTML = items.map(i => `
    <div class="rule-item">
      <div class="ri-title">${i.icon} ${i.title}</div>
      <div class="ri-desc">${i.desc||"—"}</div>
    </div>`).join("");
}

function renderLessons(lessons) {
  const container = $("lessons");
  if (!container) return;
  container.innerHTML = lessons.slice(0,20).map(l => `
    <div class="feed-row">
      <span class="fr-icon">${l.net_pnl>0?"💰":"📉"}</span>
      <span class="fr-sym">${SYM_SHORT[l.symbol]||l.symbol}</span>
      <span class="fr-reason">${l.lesson||l.finding||"—"}</span>
      <span class="fr-time">${fmtRelTime(l.ts)}</span>
    </div>`).join("") || `<div class="note">No lessons yet.</div>`;
}

function renderCohorts(cohorts) {
  const container = $("cohorts");
  if (!container || !cohorts.length) { if(container) container.innerHTML=`<div class="note">No cohort data yet.</div>`; return; }
  const rows = cohorts.map(c => `<tr>
    <td class="td-sym">${SYM_SHORT[c.symbol]||c.symbol}</td>
    <td>${c.strategy||"—"}</td>
    <td class="td-mono">${c.n_trades}</td>
    <td class="td-mono ${c.win_rate_pct>=50?"td-pos":"td-neg"}">${fmtPct(c.win_rate_pct,1)}</td>
    <td class="td-mono ${colorPnl(c.net_pnl)}">${fmtMoney(c.net_pnl)}</td>
    <td class="td-mono">${c.profit_factor != null ? fmt(c.profit_factor,2) : "—"}</td>
    <td class="td-mono">${c.risk_multiplier != null ? fmtPct((c.risk_multiplier-1)*100,0) : "—"}</td>
  </tr>`).join("");
  container.innerHTML = `<table>
    <thead><tr><th>Symbol</th><th>Strategy</th><th>Trades</th><th>Win rate</th><th>Net P&L</th><th>PF</th><th>Risk adj</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderExperiments(exps) {
  const container = $("experiments");
  if (!container || !exps.length) { if(container) container.innerHTML=`<div class="note">No experiments run yet.</div>`; return; }
  const rows = exps.map(e => `<tr>
    <td class="td-sym">${e.symbol||"portfolio"}</td>
    <td>${e.kind||"—"}</td>
    <td class="td-mono ${e.status==="accepted"?"td-pos":"td-neg"}">${e.status||"—"}</td>
    <td class="td-mono">${fmtPct(e.base_return_pct)}</td>
    <td class="td-mono">${fmtPct(e.variant_return_pct)}</td>
    <td class="muted small">${(e.hypothesis||"").slice(0,80)}</td>
    <td class="muted small">${fmtDate(e.created_at)}</td>
  </tr>`).join("");
  container.innerHTML = `<table>
    <thead><tr><th>Symbol</th><th>Kind</th><th>Status</th><th>Base</th><th>Variant</th><th>Hypothesis</th><th>Date</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

// ====================================================================== RESEARCH TAB
async function loadRuns() {
  try {
    const d = await apiFetch("/api/runs?limit=30");
    state.runs = d.runs || [];
    buildRunSelect(state.runs);
  } catch(e) { console.warn("runs:", e.message); }
}

function buildRunSelect(runs) {
  const sel = $("run-select");
  if (!sel) return;
  sel.innerHTML = runs.map(r => `<option value="${r.run_id}">${r.label || r.run_id} (${fmtDate(r.end_ts)})</option>`).join("");
  if (runs.length) {
    state.selectedRun = runs[0].run_id;
    loadRunDetail(runs[0].run_id);
  }
  sel.addEventListener("change", () => {
    state.selectedRun = sel.value;
    loadRunDetail(sel.value);
  });
}

let runEquityChart = null;
async function loadRunDetail(runId) {
  try {
    const d = await apiFetch(`/api/runs/${encodeURIComponent(runId)}`);
    const m = d.metrics || {};
    renderRunKpis(m, d);
    renderRunEquity(d.equity_curve||[]);
    renderRunBench(m.benchmarks||{});
    renderRunSymbols(m.per_symbol||{});
    renderRunStrategies(m.per_strategy||{});
    renderRunTrades(d.trades||[]);
    const per = $("run-period");
    if (per && d.start_ts && d.end_ts) per.textContent = `${fmtDate(d.start_ts)} → ${fmtDate(d.end_ts)}`;
  } catch(e) { console.warn("run detail:", e.message); }
}

function renderRunKpis(m, d) {
  const container = $("run-summary");
  if (!container) return;
  const kpis = [
    { label:"Net return",   value: fmtPct(m.net_return_pct,2), cls: signPnl(m.net_return_pct) },
    { label:"Ann. return",  value: fmtPct(m.annualized_return_pct,2), cls: signPnl(m.annualized_return_pct) },
    { label:"Max drawdown", value: fmtPct(-(m.max_drawdown_pct||0),2), cls:"negative" },
    { label:"Sharpe",       value: fmt(m.sharpe,3), cls: (m.sharpe||0)>=0?"positive":"negative" },
    { label:"Sortino",      value: fmt(m.sortino,3), cls:"" },
    { label:"Win rate",     value: fmtPct(m.win_rate_pct,1), cls: (m.win_rate_pct||0)>=50?"positive":"negative" },
    { label:"Profit factor",value: fmt(m.profit_factor,2), cls: (m.profit_factor||0)>=1?"positive":"negative" },
    { label:"Trades",       value: m.n_trades||0, cls:"" },
    { label:"Total fees",   value: fmtMoney(m.total_fees), cls:"" },
    { label:"Exposure",     value: fmtPct(m.exposure_pct,1), cls:"" },
  ];
  container.innerHTML = kpis.map(k => `
    <div class="kpi-card ${k.cls||""}">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
    </div>`).join("");
}

function renderRunEquity(curve) {
  const canvas = $("run-equity");
  if (!canvas) return;
  if (runEquityChart) {
    try { runEquityChart.destroy(); } catch (_) {}
    runEquityChart = null;
  }
  if (!curve.length) return;
  const eq = curve.map(r => ({ x: r.ts * 1000, y: Number(r.equity) }));
  const dd = curve.map(r => ({ x: r.ts * 1000, y: Number(r.drawdown) }));
  const ctx = canvas.getContext("2d");
  runEquityChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          type: "line",
          label: "Equity ($)",
          data: eq,
          borderColor: "#6366f1",
          fill: true,
          backgroundColor: "rgba(99,102,241,0.08)",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          yAxisID: "y",
          spanGaps: true,
        },
        {
          type: "line",
          label: "Drawdown (%)",
          data: dd,
          borderColor: "rgba(239,68,68,0.7)",
          fill: false,
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0,
          yAxisID: "y2",
          borderDash: [4, 2],
          spanGaps: true,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 250 },
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: { labels: { color: "#94a3b8", font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: ctx => ctx[0]?.raw?.x ? new Date(ctx[0].raw.x).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" }) : "",
            label: ctx => `${ctx.dataset.label}: ${ctx.dataset.yAxisID === "y2" ? `-${Number(ctx.raw?.y || 0).toFixed(1)}%` : `$${Number(ctx.raw?.y || 0).toFixed(2)}`}`
          }
        }
      },
      scales: {
        x: {
          type: "time",
          time: { unit: "month" },
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#64748b", maxTicksLimit: 6 }
        },
        y: {
          position: "left",
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#64748b", callback: v => `$${Number(v).toFixed(0)}` }
        },
        y2: {
          position: "right",
          grid: { display: false },
          ticks: { color: "rgba(239,68,68,0.7)", callback: v => `-${Number(v).toFixed(0)}%` }
        }
      }
    }
  });
}

function renderRunBench(bench) {
  const container = $("run-bench");
  if (!container) return;
  if (!Object.keys(bench).length) { container.innerHTML=`<div class="note">No benchmark data.</div>`; return; }
  container.innerHTML = Object.values(bench).map(b => {
    const pos = b.net_return_pct >= 0;
    return `<div class="bench-card">
      <div>
        <div class="bn-name">${b.name}</div>
        <div class="bn-dd">Max DD: ${fmtPct(-(b.max_drawdown_pct||0))}</div>
      </div>
      <div class="bn-ret ${pos?"pos":"neg"}">${fmtPct(b.net_return_pct)}</div>
    </div>`;
  }).join("");
}

function renderRunSymbols(ps) {
  const container = $("run-symbols");
  if (!container) return;
  const rows = Object.entries(ps).filter(([,s])=>s.n_trades>0).map(([sym,s]) => `<tr>
    <td class="td-sym">${SYM_SHORT[sym]||sym}</td>
    <td class="td-mono">${s.n_trades}</td>
    <td class="td-mono ${colorPnl(s.net_pnl)}">${fmtMoney(s.net_pnl)}</td>
    <td class="td-mono ${(s.win_rate_pct||0)>=50?"td-pos":"td-neg"}">${fmtPct(s.win_rate_pct,1)}</td>
    <td class="td-mono">${fmt(s.profit_factor,2)}</td>
    <td class="td-mono">${fmt(s.avg_win,4)}</td>
    <td class="td-mono">${fmt(s.avg_loss,4)}</td>
  </tr>`).join("");
  container.innerHTML = rows.length ? `<table>
    <thead><tr><th>Symbol</th><th>Trades</th><th>Net P&L</th><th>Win rate</th><th>PF</th><th>Avg win</th><th>Avg loss</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>` : `<div class="note">No trades in this run.</div>`;
}

function renderRunStrategies(ps) {
  const container = $("run-strategies");
  if (!container) return;
  const rows = Object.entries(ps).filter(([,s])=>s.n_trades>0).map(([strat,s]) => `<tr>
    <td class="td-sym">${strat}</td>
    <td class="td-mono">${s.n_trades}</td>
    <td class="td-mono ${colorPnl(s.net_pnl)}">${fmtMoney(s.net_pnl)}</td>
    <td class="td-mono ${(s.win_rate_pct||0)>=50?"td-pos":"td-neg"}">${fmtPct(s.win_rate_pct,1)}</td>
    <td class="td-mono">${fmt(s.profit_factor,2)}</td>
  </tr>`).join("");
  container.innerHTML = rows.length ? `<table>
    <thead><tr><th>Strategy</th><th>Trades</th><th>Net P&L</th><th>Win rate</th><th>PF</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>` : `<div class="note">No strategy breakdown.</div>`;
}

function renderRunTrades(trades) {
  const container = $("run-trades");
  const cnt = $("run-trade-count");
  if (cnt) cnt.textContent = `${trades.length} trades`;
  if (!container) return;
  if (!trades.length) { container.innerHTML=`<div class="note">No trades.</div>`; return; }
  const rows = trades.slice(0,200).map(t => `<tr>
    <td class="td-sym">${SYM_SHORT[t.symbol]||t.symbol}</td>
    <td>${t.strategy||"—"}</td>
    <td class="td-mono muted">${fmtDate(t.entry_ts)}</td>
    <td class="td-mono">${fmtMoney(t.entry_price)}</td>
    <td class="td-mono">${fmtMoney(t.exit_price)}</td>
    <td class="td-mono ${colorPnl(t.net_pnl)}">${fmtPct(t.return_pct)}</td>
    <td class="td-mono ${colorPnl(t.net_pnl)}">${fmtMoney(t.net_pnl)}</td>
    <td class="td-mono muted">${t.exit_signal||"—"}</td>
  </tr>`).join("");
  container.innerHTML = `<table>
    <thead><tr><th>Symbol</th><th>Strategy</th><th>Entry</th><th>Entry $</th><th>Exit $</th><th>Return</th><th>Net P&L</th><th>Exit</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

async function loadRiskData() {
  try {
    const d = await apiFetch("/api/risk");
    renderBreakers(d.breakers||{});
    renderRecon(d.reconcile||{});
    renderCosts(d.costs||{});
    renderRiskLimits(d.risk||{});
    renderAlerts(d.alerts||[]);
    loadSystemLogs();
  } catch(e) { console.warn("risk:", e.message); }
}

function renderDataTable(d) {
  const container = $("data-table");
  if (!container) return;
  const rows = (d.coverage||d||[]).map(r => {
    const h1 = r["1h"]||{}, d1 = r["1d"]||{};
    return `<tr>
      <td class="td-sym">${SYM_SHORT[r.symbol]||r.symbol}</td>
      <td>${r.strategy||"—"}</td>
      <td class="td-mono">${h1.bars||0}</td>
      <td class="td-mono">${h1.first||"—"}</td>
      <td class="td-mono ${h1.status==="ok"?"td-pos":h1.status==="warn"?"":"td-neg"}">${h1.status||"—"}</td>
      <td class="td-mono">${h1.missing_bars||0}</td>
      <td class="td-mono">${d1.bars||0}</td>
      <td class="td-mono ${d1.status==="ok"?"td-pos":d1.status==="warn"?"":"td-neg"}">${d1.status||"—"}</td>
    </tr>`;
  }).join("");
  container.innerHTML = `<table>
    <thead><tr><th>Symbol</th><th>Strategy</th><th>1h bars</th><th>1h start</th><th>1h status</th><th>1h gaps</th><th>1d bars</th><th>1d status</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function renderBreakers(b) {
  const container = $("breaker-body");
  if (!container) return;
  const items = [
    { name:"Daily loss limit",  halted: b.daily_halted,    value: b.daily_halted?"⛔ HALTED":"✅ OK" },
    { name:"Drawdown circuit",  halted: b.drawdown_halted, value: b.drawdown_halted?"🔒 LATCHED":"✅ OK" },
    { name:"Day start equity",  halted: false, value: fmtMoney(b.equity) },
    { name:"High-water mark",   halted: false, value: fmtMoney(b.hwm) },
  ];
  container.innerHTML = items.map(i => `
    <div class="breaker-item ${i.halted?"halted":"ok"}">
      <span class="bi-name">${i.name}</span>
      <span class="bi-val">${i.value}</span>
    </div>`).join("");
}

function renderRecon(r) {
  const container = $("recon-body");
  const status = $("recon-status");
  if (!container) return;
  if (status) status.textContent = r.status || "unknown";
  container.textContent = (r.issues||[]).join("\n") || "positions match";
}

function renderCosts(c) {
  const container = $("cost-body");
  if (!container) return;
  const items = [
    { label:"Taker fee",   value: `${c.taker_fee_bps||25} bps` },
    { label:"Maker fee",   value: `${c.maker_fee_bps||15} bps` },
    { label:"Spread",      value: `${c.spread_bps||4} bps` },
    { label:"Slippage",    value: `${c.slippage_bps||5} bps` },
    { label:"Round trip",  value: `${c.round_trip_bps||(2*(25+2+5))} bps` },
    { label:"Min notional",value: `$${c.min_notional||1}` },
  ];
  container.innerHTML = items.map(i => `
    <div class="cost-item"><div class="ci-label">${i.label}</div><div class="ci-value">${i.value}</div></div>`).join("");
}

function renderRiskLimits(r) {
  const container = $("risk-limits");
  if (!container) return;
  const items = [
    { label:"Risk / trade",   value: `${r.risk_per_trade_pct||1}%` },
    { label:"Max positions",  value: r.max_positions||2 },
    { label:"Max exposure",   value: `${r.max_position_pct||35}%` },
    { label:"Daily loss",     value: `${r.daily_loss_limit_pct||3}%` },
    { label:"Max drawdown",   value: `${r.max_drawdown_pct||12}%` },
    { label:"Consec losses",  value: `${r.consecutive_losses_limit||3}` },
    { label:"Cooldown",       value: `${r.cooldown_hours||12}h` },
    { label:"Cash reserve",   value: `${r.cash_reserve_pct||25}%` },
  ];
  container.innerHTML = items.map(i => `
    <div class="cost-item"><div class="ci-label">${i.label}</div><div class="ci-value">${i.value}</div></div>`).join("");
}

async function loadSystemLogs() {
  const container = $("system-logs");
  if (!container) return;
  try {
    const res = await apiFetch("/api/logs?limit=80");
    if (res.logs && res.logs.length) {
      container.innerHTML = res.logs.map(line => {
        let cls = "";
        if (line.includes("ERROR") || line.includes("CRITICAL")) cls = "log-err";
        else if (line.includes("WARNING")) cls = "log-warn";
        else if (line.includes("INFO")) cls = "log-info";
        const escaped = line.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        return `<div class="${cls}">${escaped}</div>`;
      }).join("");
      container.scrollTop = container.scrollHeight;
    } else {
      container.textContent = "No log entries found.";
    }
  } catch(e) {
    container.textContent = "Unable to fetch logs: " + e.message;
  }
}

function renderAlerts(alerts) {
  const container = $("alerts");
  if (!container) return;
  container.innerHTML = alerts.slice(0, 50).map(a => {
    const statusIcon = a.delivered ? "✅" : (a.error ? "⚠️" : "⏳");
    const escaped = (a.text || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    return `
    <div class="feed-row" style="align-items:flex-start">
      <span class="fr-icon" title="${a.delivered ? 'Delivered to Telegram' : (a.error || 'Pending')}">${statusIcon}</span>
      <span class="fr-sym" style="min-width:110px">${a.kind || "alert"}</span>
      <span class="fr-reason" style="white-space:pre-wrap;line-height:1.4">${escaped}</span>
      <span class="fr-time" style="margin-left:auto">${fmtRelTime(a.ts)}</span>
    </div>`;
  }).join("") || `<div class="note">No alerts logged yet.</div>`;
}

// ====================================================================== CONTROLS
function initControls() {
  // trade mode filter
  const tm = $("trade-mode");
  if (tm) tm.addEventListener("change", loadTrades);
  const tsf = $("trade-sym-filter");
  if (tsf) {
    tsf.innerHTML = `<option value="">all symbols</option>` +
      SYMBOLS.map(s=>`<option value="${s}">${SYM_SHORT[s]||s}</option>`).join("");
    tsf.addEventListener("change", loadTrades);
  }

  // cycle button
  const btnCycle = $("btn-cycle");
  if (btnCycle) btnCycle.addEventListener("click", async () => {
    btnCycle.disabled = true;
    try {
      const d = await fetch("/api/cycle", {method:"POST"}).then(r=>r.json());
      const st = $("ctrl-status");
      if (st) { st.textContent = `Cycle completed. Decisions: ${d.decisions?.length||0}`; st.className="note success"; }
    } catch(e) {
      const st = $("ctrl-status");
      if (st) { st.textContent = `Error: ${e.message}`; st.className="note error"; }
    }
    btnCycle.disabled = false;
    setTimeout(pollAll, 2000);
  });

  // pause
  const btnPause = $("btn-pause");
  if (btnPause) btnPause.addEventListener("click", () => {
    fetch("/api/pause", {method:"POST"}).then(() => {
      const st = $("ctrl-status");
      if (st) { st.textContent = "New entries paused."; st.className="note"; }
    });
  });

  // resume
  const btnResume = $("btn-resume");
  if (btnResume) btnResume.addEventListener("click", () => {
    fetch("/api/resume", {method:"POST"}).then(() => {
      const st = $("ctrl-status");
      if (st) { st.textContent = "Resumed."; st.className="note success"; }
    });
  });

  // reset drawdown breaker
  const btnReset = $("btn-reset-dd");
  if (btnReset) btnReset.addEventListener("click", async () => {
    if (!confirm("Reset the latched drawdown circuit breaker? This allows new entries again.")) return;
    const d = await fetch("/api/reset_drawdown", {method:"POST"}).then(r=>r.json());
    const st = $("ctrl-status");
    if (st) { st.textContent = d.message || "Reset done."; st.className="note success"; }
  });

  // backtest form
  const btnBt = $("btn-backtest");
  if (btnBt) btnBt.addEventListener("click", async () => {
    btnBt.disabled = true;
    const st = $("bt-status");
    if (st) { st.textContent = "Running backtest…"; st.className="note"; }
    try {
      const start = ($("bt-start")||{}).value || null;
      const end   = ($("bt-end")||{}).value || null;
      const eq    = parseFloat(($("bt-equity")||{}).value) || 100;
      const body  = { start_date: start, end_date: end, initial_equity: eq };
      const d = await fetch("/api/backtest", {method:"POST", headers:{"content-type":"application/json"}, body:JSON.stringify(body)}).then(r=>r.json());
      if (st) { st.textContent = `Done: ${d.run_id}. Trades: ${d.n_trades}. Return: ${fmtPct(d.net_return_pct)}`; st.className="note success"; }
      loadRuns();
    } catch(e) {
      if (st) { st.textContent = "Error: " + e.message; st.className="note error"; }
    }
    btnBt.disabled = false;
  });

  // refresh runs
  const btnRefresh = $("refresh-runs");
  if (btnRefresh) btnRefresh.addEventListener("click", loadRuns);

  // close trade detail
  const cd = $("close-detail");
  if (cd) cd.addEventListener("click", () => {
    const card = $("trade-detail-card");
    if (card) card.style.display = "none";
  });

  // lazy-load tab data
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      const v = btn.dataset.view;
      if (v === "trades") loadTrades();
      if (v === "models") loadModels();
      if (v === "risk")   loadRiskData();
    });
  });
}

// ====================================================================== FOOTER
function updateFooterTime() {
  const el = $("footer-time");
  if (el) el.textContent = new Date().toLocaleString();
}
