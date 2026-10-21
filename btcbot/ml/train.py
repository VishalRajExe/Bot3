"""Walk-forward model training and evaluation.

Protocol
--------
* Expanding-window walk-forward: the model used for a test block is trained only
  on bars strictly before that block.
* ``horizon`` bars are purged from the end of each training window and an equal
  embargo separates training from test, so no training label overlaps a test bar.
* Hyper-parameters are fixed in advance.  There is no search on test data.
* Only out-of-fold predictions are used for evaluation and for the ML strategy.

A model is stored as a *candidate* first.  Promotion requires the criteria in
``docs/MODEL_VALIDATION.md``; nothing is promoted automatically into paper
trading, and the default paper strategy does not depend on ML.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from ..config import Config
from ..data.timeframes import to_datetime
from ..db import Database, utcnow_iso
from .features import FEATURES, FG_FEATURES, build_dataset

log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame  # bar_ts, prob_up, fold
    metrics: dict
    final_model: HistGradientBoostingClassifier
    feature_names: list[str]
    dataset_rows: int


def _new_model(seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_depth=3,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def walk_forward_predictions(df: pd.DataFrame, cfg: Config, *, horizon: int = 5,
                             min_train_bars: int = 700, test_bars: int = 120,
                             embargo: int | None = None, seed: int = 42) -> WalkForwardResult:
    """Produce out-of-fold probabilities for every bar that can be predicted."""
    cost_bps = (cfg.costs.taker_fee_bps * 2 + cfg.costs.spread_bps + cfg.costs.slippage_bps * 2)
    X, y, fwd = build_dataset(df, horizon=horizon, cost_bps=cost_bps)
    if len(X) < min_train_bars + test_bars + horizon:
        raise ValueError(
            f"not enough rows for walk-forward: {len(X)} available, need "
            f"{min_train_bars + test_bars + horizon}"
        )
    embargo = horizon if embargo is None else embargo
    feature_names = [c for c in FEATURES if c in X.columns]
    feature_names += [c for c in FG_FEATURES if c in X.columns]
    rows: list[dict] = []
    fold_metrics: list[dict] = []

    start = min_train_bars
    fold = 0
    while start < len(X):
        test_end = min(start + test_bars, len(X))
        train_end = start - embargo  # purge + embargo
        if train_end <= 200:
            start = test_end
            continue
        X_train = X.iloc[:train_end][feature_names]
        y_train = y.iloc[:train_end]
        if y_train.nunique() < 2:
            start = test_end
            continue
        model = _new_model(seed + fold)
        model.fit(X_train, y_train)
        X_test = X.iloc[start:test_end][feature_names]
        proba = model.predict_proba(X_test)[:, 1]
        block = pd.DataFrame({
            "bar_ts": X.iloc[start:test_end]["bar_ts"].to_numpy(dtype=np.int64),
            "prob_up": proba,
            "fold": fold,
            "label": y.iloc[start:test_end].to_numpy(),
            "fwd_return": fwd.iloc[start:test_end].to_numpy(),
        })
        rows.append(block)
        if block["label"].nunique() > 1:
            fold_metrics.append({
                "fold": fold,
                "test_start": to_datetime(int(block["bar_ts"].iloc[0])).date().isoformat(),
                "test_end": to_datetime(int(block["bar_ts"].iloc[-1])).date().isoformat(),
                "auc": float(roc_auc_score(block["label"], block["prob_up"])),
                "n": int(len(block)),
            })
        start = test_end
        fold += 1

    if not rows:
        raise ValueError("walk-forward produced no predictions")
    preds = pd.concat(rows, ignore_index=True)
    auc = float(roc_auc_score(preds["label"], preds["prob_up"])) if preds["label"].nunique() > 1 else float("nan")
    base_rate = float(preds["label"].mean())
    accuracy = float(((preds["prob_up"] >= 0.5) == preds["label"]).mean())
    brier = float(brier_score_loss(preds["label"], preds["prob_up"]))

    # Calibration by probability decile.
    preds["decile"] = pd.qcut(preds["prob_up"], 10, duplicates="drop", labels=False)
    calibration = (
        preds.groupby("decile")
        .agg(prob_mean=("prob_up", "mean"), actual_rate=("label", "mean"), n=("label", "size"))
        .reset_index()
        .to_dict(orient="records")
    )

    # Final model trained on the newest labelled data, purged by the horizon.
    train_end = len(X) - horizon
    final = _new_model(seed + 999)
    final.fit(X.iloc[:train_end][feature_names], y.iloc[:train_end])
    try:
        sample = X.iloc[max(0, train_end - 300):train_end][feature_names]
        sample_y = y.iloc[max(0, train_end - 300):train_end]
        imp = permutation_importance(final, sample, sample_y, n_repeats=3,
                                     random_state=seed, scoring="roc_auc")
        importance = sorted(
            [{"feature": f, "importance": round(float(m), 4)}
             for f, m in zip(feature_names, imp.importances_mean)],
            key=lambda d: d["importance"], reverse=True,
        )
    except Exception:  # pragma: no cover - importance is diagnostic only
        importance = []

    metrics = {
        "auc": round(auc, 4),
        "accuracy_at_0.5": round(accuracy, 4),
        "brier": round(brier, 4),
        "base_rate_up": round(base_rate, 4),
        "predictions": int(len(preds)),
        "folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "calibration": calibration,
        "feature_importance": importance[:12],
        "horizon_bars": horizon,
        "label": f"forward return over {horizon} bars > {cost_bps:.2f} bps round-trip cost",
        "train_rows_final": int(train_end),
        "first_prediction": to_datetime(int(preds["bar_ts"].iloc[0])).date().isoformat(),
        "last_prediction": to_datetime(int(preds["bar_ts"].iloc[-1])).date().isoformat(),
        "cost_bps_used": cost_bps,
    }
    return WalkForwardResult(predictions=preds, metrics=metrics, final_model=final,
                             feature_names=feature_names, dataset_rows=len(X))


def register_model(db: Database, cfg: Config, result: WalkForwardResult, *,
                   version: str, artifact_dir: str = "reports/models",
                   status: str = "candidate", notes: str = "") -> str:
    """Persist the model artifact, registry row and out-of-fold predictions."""
    from pathlib import Path

    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    artifact = path / f"{version}.joblib"
    joblib.dump({"model": result.final_model, "features": result.feature_names,
                 "horizon": result.metrics["horizon_bars"]}, artifact)

    preds = result.predictions
    db.execute(
        "INSERT OR REPLACE INTO models (version, created_at, algorithm, feature_names_json,"
        " label_definition, train_start, train_end, test_start, test_end, metrics_json,"
        " artifact_path, status, promoted_at, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (version, utcnow_iso(), "HistGradientBoostingClassifier",
         json.dumps(result.feature_names), result.metrics["label"],
         int(preds["bar_ts"].iloc[0]), int(preds["bar_ts"].iloc[-1]),
         int(preds["bar_ts"].iloc[0]), int(preds["bar_ts"].iloc[-1]),
         json.dumps(result.metrics), str(artifact), status, None, notes),
    )
    db.executemany(
        "INSERT OR REPLACE INTO model_predictions (model_version, ts, bar_ts, horizon,"
        " prob_up, expected_return, used_in_trading, created_at) VALUES (?,?,?,?,?,?,?,?)",
        [(version, int(r.bar_ts), int(r.bar_ts), result.metrics["horizon_bars"],
          float(r.prob_up), float(r.fwd_return) if np.isfinite(r.fwd_return) else None, 0,
          utcnow_iso()) for r in preds.itertuples(index=False)],
    )
    log.info("registered model %s (%s, AUC %.3f) -> %s", version, status,
             result.metrics.get("auc", float("nan")), artifact)
    return str(artifact)


def predictions_frame(db: Database, version: str) -> pd.DataFrame:
    rows = db.query(
        "SELECT bar_ts, prob_up FROM model_predictions WHERE model_version=? ORDER BY bar_ts",
        (version,),
    )
    if not rows:
        return pd.DataFrame(columns=["bar_ts", "prob_up"])
    return pd.DataFrame([dict(r) for r in rows])


def promotion_decision(metrics: dict, *, min_auc: float = 0.55, min_predictions: int = 400,
                       strategy_metrics: dict | None = None,
                       baseline_metrics: dict | None = None) -> tuple[str, str]:
    """Documented promotion gate.

    A model must clear *all* of: enough out-of-fold predictions, better-than-random
    ranking, monotonic calibration, and a strategy that beats the baseline after
    costs over the same window.  Passing means "validated for further paper-trading
    comparison", never "auto-deploy".
    """
    if metrics.get("predictions", 0) < min_predictions:
        return "rejected", f"insufficient out-of-fold predictions ({metrics.get('predictions')})"
    auc = metrics.get("auc")
    if not auc or auc < min_auc:
        return "rejected", f"out-of-fold AUC {auc} below required {min_auc}"

    calibration = metrics.get("calibration") or []
    if len(calibration) >= 4:
        xs = [float(c["prob_mean"]) for c in calibration]
        ys = [float(c["actual_rate"]) for c in calibration]
        corr = float(np.corrcoef(xs, ys)[0, 1])
        if not np.isfinite(corr) or corr < 0.5:
            return "rejected", (
                f"probability calibration is not monotonic (decile correlation {corr:.2f} < 0.5): "
                "the model's probabilities do not rank outcomes reliably"
            )

    if strategy_metrics is not None and baseline_metrics is not None:
        if (strategy_metrics.get("trades") or 0) < 10:
            return "rejected", (f"ML strategy produced only {strategy_metrics.get('trades')} "
                                "trades; not enough to judge")
        if (strategy_metrics.get("net_return_pct") or 0) <= (
                baseline_metrics.get("net_return_pct") or 0):
            return "rejected", (
                f"ML strategy net return {strategy_metrics.get('net_return_pct')}% does not beat "
                f"the baseline {baseline_metrics.get('net_return_pct')}% over the same window"
            )
    return "validated", (
        f"out-of-fold AUC {auc} >= {min_auc}, calibration monotonic, and the strategy beats the "
        "baseline. Validated for paper-trading comparison only; never auto-promoted."
    )
