from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcbot.config import Config
from btcbot.ml.features import FEATURES, build_dataset, build_features
from btcbot.ml.train import promotion_decision, walk_forward_predictions
from btcbot.strategies.ml import MLGated
from tests.helpers import make_candles


def test_features_are_causal():
    df = make_candles(500, seed=51, vol=0.02)
    full = build_features(df)
    truncated = build_features(df.iloc[:350])
    pd.testing.assert_frame_equal(full.iloc[:350], truncated, check_exact=False, rtol=1e-10)


def test_dataset_labels_match_forward_returns():
    df = make_candles(400, seed=52)
    X, y, fwd = build_dataset(df, horizon=5, cost_bps=24.0)
    assert list(X.columns)[: len(FEATURES)] == FEATURES
    assert not X.isna().any().any()
    idx = X.index[10]
    expected = df["close"].iloc[idx + 5] / df["close"].iloc[idx] - 1.0
    assert fwd.loc[idx] == pytest.approx(expected)
    assert int(y.loc[idx]) == int(expected > 0.0024)


def test_walk_forward_predictions_are_out_of_sample():
    cfg = Config()
    df = make_candles(1400, seed=53, vol=0.03)
    result = walk_forward_predictions(df, cfg, horizon=5, min_train_bars=700, test_bars=120)
    preds = result.predictions
    assert len(preds) > 300
    assert preds["bar_ts"].is_monotonic_increasing
    assert preds["prob_up"].between(0, 1).all()
    # First prediction cannot come before the minimum training window.
    first_x_ts = int(preds["bar_ts"].iloc[0])
    assert first_x_ts >= int(df["bar_ts"].iloc[700 - 10])
    assert 0.0 <= result.metrics["auc"] <= 1.0
    assert result.metrics["folds"] >= 3


def test_promotion_requires_auc_and_sample():
    status, reason = promotion_decision({"auc": 0.50, "predictions": 1000})
    assert status == "rejected" and "AUC" in reason
    status, reason = promotion_decision({"auc": 0.60, "predictions": 100})
    assert status == "rejected" and "insufficient" in reason
    status, _ = promotion_decision({"auc": 0.60, "predictions": 1000})
    assert status == "validated"


def test_ml_strategy_without_predictions_never_trades():
    df = make_candles(500, seed=54)
    strat = MLGated()
    prepared = strat.prepare(df)
    for i in range(strat.warmup_bars, len(prepared)):
        assert strat.signal(prepared, i, None).action == "hold"


def test_ml_strategy_enters_on_high_probability():
    df = make_candles(500, seed=55, drift=0.001)
    probs = pd.DataFrame({"bar_ts": df["bar_ts"], "prob_up": 0.70})
    strat = MLGated(predictions=probs)
    prepared = strat.prepare(df)
    entries = [i for i in range(strat.warmup_bars, len(prepared))
               if strat.signal(prepared, i, None).action == "enter_long"]
    assert entries
    sig = strat.signal(prepared, entries[0], None)
    assert sig.stop_price is not None and sig.stop_price < prepared["close"].iloc[entries[0]]
