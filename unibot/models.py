"""Per-symbol meta-labeling models.

The primary strategy proposes a trade; a second, symbol-specific model answers
one question: *given this setup, what is the probability the trade survives its
barriers net of costs?*  The model cannot invent trades, cannot change direction
and cannot resize beyond the risk manager's bounds.

Training follows the evidence on leakage:

* labels come from triple-barrier outcomes using ATR at the signal bar;
* evaluation is **chronological walk-forward** (train on the past, test on the
  next unseen block), with a purge gap of ``purge_bars`` between them;
* the probability threshold is chosen on the training block only;
* a model is promoted to ``active`` only if out-of-sample AUC and net trade
  economics both improve, otherwise it is stored as ``rejected`` and **not
  used for trading**.  Rejected models stay in the registry for review.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from btcbot.data.timeframes import tf_seconds

from .config import ModelCfg, Settings
from .features import triple_barrier_labels
from .strategies import build as build_strategy

log = logging.getLogger(__name__)

MODEL_FEATURES = [
    "rsi14", "adx14", "atr_pct", "bb_pos", "dist_ema50_pct", "dist_ema200_pct",
    "ema_slope_pct", "vol_ratio", "range_pos_20", "donchian_pos",
    "ret_3", "ret_6", "ret_12", "ret_24", "ret_72", "realized_vol",
    "body_pct", "upper_wick_pct", "lower_wick_pct",
    "market_score", "breadth", "btc_trend",
]
CANDIDATE_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def _make_classifier(cfg: ModelCfg, seed: int = 7):
    if cfg.algorithm == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=12,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )
    else:
        clf = GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.04, max_depth=2,
            subsample=0.8, min_samples_leaf=15, random_state=seed,
        )
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


@dataclass
class ModelArtifact:
    symbol: str
    strategy: str
    timeframe: str
    version: str
    threshold: float
    features: list[str]
    pipeline: Pipeline
    trained_at: str
    auc: float | None = None

    def predict(self, row: pd.Series | dict) -> float:
        values = []
        for name in self.features:
            try:
                v = row[name]
            except (KeyError, IndexError):
                v = np.nan
            values.append(float(v) if v is not None and np.isfinite(float(v)) else 0.0)
        x = pd.DataFrame([values], columns=self.features)
        return float(self.pipeline.predict_proba(x)[0, 1])

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "symbol": self.symbol, "strategy": self.strategy, "timeframe": self.timeframe,
            "version": self.version, "threshold": self.threshold,
            "features": self.features, "pipeline": self.pipeline,
            "trained_at": self.trained_at, "auc": self.auc,
        }, p)
        return str(p)

    @classmethod
    def load(cls, path: str | Path) -> "ModelArtifact":
        data = joblib.load(path)
        return cls(**data)


@dataclass
class Gate:
    """Thin adapter used by the backtest and live engine."""
    artifact: ModelArtifact

    @property
    def threshold(self) -> float:
        return self.artifact.threshold

    def predict(self, row) -> float:
        return self.artifact.predict(row)


def build_dataset(df: pd.DataFrame, symbol_cfg, settings: Settings) -> pd.DataFrame:
    """Every primary-strategy entry in history, with features and a barrier label."""
    strategy = build_strategy(symbol_cfg)
    labels = triple_barrier_labels(
        df, horizon=settings.models.horizon_bars,
        upper_atr=settings.models.u_barrier_atr,
        lower_atr=settings.models.l_barrier_atr,
        fee_pct=settings.costs.round_trip_pct,
    )
    feats = [c for c in MODEL_FEATURES if c in df.columns]
    rows = []
    for i in range(strategy.warmup_bars, len(df) - 1):
        try:
            sig = strategy.signal(df, i, None)
        except Exception:  # a single bad row must not poison the dataset
            continue
        if sig.action != "enter_long":
            continue
        label = labels["label"].iloc[i]
        if not np.isfinite(label):
            continue
        row = df.iloc[i]
        record = {name: float(row[name]) if np.isfinite(row[name]) else 0.0 for name in feats}
        record.update({
            "label": int(label),
            "ret_pct": float(labels["ret"].iloc[i]),
            "bar_ts": int(row["bar_ts"]),
            "entry_price": float(row["close"]),
            "stop": float(sig.stop_price) if sig.stop_price else np.nan,
            "outcome": str(labels["outcome"].iloc[i]),
        })
        rows.append(record)
    if not rows:
        return pd.DataFrame(columns=feats + ["label", "ret_pct", "bar_ts", "entry_price",
                                             "stop", "outcome"])
    return pd.DataFrame(rows).sort_values("bar_ts").reset_index(drop=True)


def _month_floor(ts: int) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return int(datetime(dt.year, dt.month, 1, tzinfo=timezone.utc).timestamp())


def _add_months(ts: int, months: int) -> int:
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def walk_forward_splits(data: pd.DataFrame, cfg: ModelCfg) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if data.empty:
        return []
    bars = data["bar_ts"].to_numpy("int64")
    start = _month_floor(int(bars[0]))
    end = _month_floor(int(bars[-1]))
    splits = []
    cursor = _add_months(start, cfg.train_months)
    while cursor < end:
        test_end = _add_months(cursor, cfg.test_months)
        train_mask = bars < cursor
        test_mask = (bars >= cursor) & (bars < test_end)
        train = data[train_mask]
        test = data[test_mask]
        if len(train) >= 40 and len(test) >= 5:
            splits.append((train, test))
        cursor = _add_months(cursor, cfg.step_months)
    return splits


def _choose_threshold(y: np.ndarray, proba: np.ndarray, ret: np.ndarray,
                      min_trades: int = 12) -> tuple[float, dict]:
    best, best_stats = 0.50, {"trades": 0, "mean_ret": 0.0}
    for th in CANDIDATE_THRESHOLDS:
        mask = proba >= th
        n = int(mask.sum())
        if n < min_trades:
            continue
        mean_ret = float(ret[mask].mean())
        stats = {"trades": n, "mean_ret": round(mean_ret, 4),
                 "win_rate": round(float(y[mask].mean()) * 100, 2)}
        if best_stats.get("mean_ret", -1e9) < mean_ret:
            best, best_stats = th, stats
    return best, best_stats


def train_symbol(store, settings: Settings, symbol: str, df: pd.DataFrame, *,
                 predict_only: bool = False) -> dict:
    """Train, validate and (maybe) promote one symbol's model."""
    cfg = settings.models
    sym_cfg = settings.symbol(symbol)
    strategy = build_strategy(sym_cfg)
    version = f"{symbol.replace('/', '')}_{strategy.name}_{settings.timeframe}_{datetime.now(timezone.utc):%Y%m%d%H%M}"
    data = build_dataset(df, sym_cfg, settings)
    out = {"symbol": symbol, "version": version, "strategy": strategy.name,
           "status": "insufficient", "samples": len(data)}

    if len(data) < cfg.min_samples:
        out["reason"] = f"{len(data)} labelled setups < min_samples {cfg.min_samples}"
        store.save_model({**out, "timeframe": settings.timeframe, "algorithm": cfg.algorithm,
                          "trained_at": datetime.now(timezone.utc).isoformat(),
                          "n_train": len(data), "n_test": 0, "metrics_json": json.dumps(out)})
        return out

    feats = [c for c in MODEL_FEATURES if c in data.columns]
    splits = walk_forward_splits(data, cfg)
    if not splits:
        out["reason"] = "not enough history for a walk-forward split"
        return out

    purge = cfg.purge_bars
    oos_probs, oos_y, oos_ret, oos_ts = [], [], [], []
    fold_metrics = []
    for fold, (train, test) in enumerate(splits):
        # purge the tail of the training block so labels cannot overlap the test block
        cutoff = int(test["bar_ts"].min())
        train = train[train["bar_ts"] < cutoff - purge * tf_seconds(settings.timeframe)]
        if len(train) < 40 or train["label"].nunique() < 2:
            continue
        model = _make_classifier(cfg, seed=7 + fold)
        model.fit(train[feats], train["label"])
        proba = model.predict_proba(test[feats])[:, 1]
        oos_probs.extend(proba.tolist())
        oos_y.extend(test["label"].tolist())
        oos_ret.extend(test["ret_pct"].tolist())
        oos_ts.extend(test["bar_ts"].tolist())
        fold_metrics.append({"fold": fold, "train": len(train), "test": len(test),
                             "test_start": int(test["bar_ts"].min())})

    if len(oos_probs) < cfg.min_test_samples:
        out["reason"] = f"only {len(oos_probs)} out-of-sample setups"
        return out

    y = np.array(oos_y); proba = np.array(oos_probs); ret = np.array(oos_ret, dtype=float)
    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    brier = float(brier_score_loss(y, proba)) if len(np.unique(y)) > 1 else float("nan")
    threshold, train_stats = _choose_threshold(y, proba, ret)
    mask = proba >= threshold
    filtered = ret[mask]
    all_mean = float(ret.mean())
    filtered_mean = float(filtered.mean()) if len(filtered) else 0.0
    improvement = filtered_mean - all_mean
    net_pnl_filtered = float(filtered.sum())  # % of a fixed 1-unit stake per trade

    # Final model: refit on the most recent block for live predictions.
    cutoff = int(data["bar_ts"].max())
    final_train = data[data["bar_ts"] < cutoff]
    final_model = _make_classifier(cfg, seed=99)
    final_model.fit(final_train[feats], final_train["label"])
    artifact = ModelArtifact(
        symbol=symbol, strategy=strategy.name, timeframe=settings.timeframe,
        version=version, threshold=round(float(threshold), 3), features=feats,
        pipeline=final_model,
        trained_at=datetime.now(timezone.utc).isoformat(), auc=round(auc, 4),
    )
    path = Path(settings.paths.models) / f"{version}.joblib"
    artifact.save(path)

    passed_auc = np.isfinite(auc) and auc >= cfg.min_auc
    passed_econ = improvement > cfg.min_oos_improvement_pct and len(filtered) >= 20
    status = "active" if (passed_auc and passed_econ) else "rejected"

    metrics = {
        "auc": round(auc, 4) if np.isfinite(auc) else None,
        "brier": round(brier, 4) if np.isfinite(brier) else None,
        "base_rate": round(float(y.mean()), 4),
        "threshold": artifact.threshold,
        "threshold_train_stats": train_stats,
        "oos_setups": int(len(ret)),
        "oos_filtered": int(mask.sum()),
        "oos_mean_ret_all": round(all_mean, 4),
        "oos_mean_ret_filtered": round(filtered_mean, 4),
        "oos_improvement_pct_points": round(improvement, 4),
        "oos_net_pnl_units": round(net_pnl_filtered, 4),
        "folds": fold_metrics,
    }
    out.update({
        "status": status, "auc": metrics["auc"], "threshold": artifact.threshold,
        "oos_trades": int(mask.sum()), "oos_improvement": round(improvement, 4),
        "reason": ("promoted: out-of-sample AUC and trade economics improved"
                   if status == "active" else
                   f"rejected: AUC {metrics['auc']} (min {cfg.min_auc}), improvement "
                   f"{round(improvement, 3)}pp"),
        "artifact": str(path), "metrics": metrics,
    })
    store.save_model({
        "version": version, "symbol": symbol, "strategy": strategy.name,
        "timeframe": settings.timeframe, "algorithm": cfg.algorithm,
        "trained_at": artifact.trained_at,
        "train_start": int(data["bar_ts"].min()), "train_end": int(data["bar_ts"].max()),
        "test_start": int(min(oos_ts)), "test_end": int(max(oos_ts)),
        "n_train": len(data) - len(ret), "n_test": len(ret),
        "auc": metrics["auc"], "brier": metrics["brier"],
        "base_rate": metrics["base_rate"], "threshold": artifact.threshold,
        "oos_trades": metrics["oos_filtered"], "oos_net_pnl": metrics["oos_net_pnl_units"],
        "oos_return_pct": round(filtered_mean, 4),
        "oos_improvement_pct": round(improvement, 4),
        "status": status, "artifact_path": str(path),
        "metrics_json": json.dumps(metrics, default=str),
    })
    return out


def train_all(store, settings: Settings, frames: dict[str, pd.DataFrame]) -> list[dict]:
    results = []
    for sym, df in frames.items():
        if df is None or df.empty:
            results.append({"symbol": sym, "status": "no_data"})
            continue
        try:
            results.append(train_symbol(store, settings, sym, df))
        except Exception as exc:  # recorded, visible, never silently ignored
            log.exception("model training failed for %s", sym)
            results.append({"symbol": sym, "status": "error", "error": str(exc)[:300]})
    return results


def load_active_gates(store, settings: Settings) -> dict[str, Gate]:
    gates: dict[str, Gate] = {}
    base = Path(settings.paths.models)
    for sym in settings.symbols:
        row = store.active_model(sym)
        if not row or not row.get("artifact_path"):
            continue
        path = Path(row["artifact_path"])
        if not path.is_absolute():
            path = base / path.name
        if not path.exists():
            log.warning("active model for %s is missing on disk: %s", sym, path)
            continue
        try:
            gates[sym] = Gate(ModelArtifact.load(path))
        except Exception:
            log.exception("could not load model for %s", sym)
    return gates


def model_report(store, settings: Settings) -> list[dict]:
    """Latest model per symbol plus its attempt history (not every run)."""
    rows = store.models()
    latest: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    history: dict[str, list[str]] = {}
    for r in rows:
        sym = r["symbol"]
        attempts[sym] = attempts.get(sym, 0) + 1
        history.setdefault(sym, []).append(r["status"])
        if sym not in latest:
            try:
                r["metrics"] = json.loads(r.get("metrics_json") or "{}")
            except json.JSONDecodeError:
                r["metrics"] = {}
            latest[sym] = r
    for sym, r in latest.items():
        r["attempts"] = attempts[sym]
        r["history"] = history[sym][:6]
    order = {s: i for i, s in enumerate(settings.symbols)}
    return sorted(latest.values(), key=lambda r: order.get(r["symbol"], 99))
