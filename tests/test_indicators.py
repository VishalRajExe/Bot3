from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcbot.indicators import (adx, atr, bollinger, donchian, ema, ema_zscore, realized_vol,
                               rolling_slope, rolling_std, rsi, sma, zscore)
from tests.helpers import make_candles


def test_ema_matches_recursive_definition():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(s, 3)
    # alpha = 2/(3+1) = 0.5; first two values are NaN due to min_periods
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.25)
    assert out.iloc[3] == pytest.approx(3.125)
    assert out.iloc[4] == pytest.approx(4.0625)


def test_rolling_std_uses_population_std():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = rolling_std(s, 3, ddof=0)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(np.std([1, 2, 3], ddof=0))
    assert out.iloc[3] == pytest.approx(np.std([2, 3, 4], ddof=0))


def test_sma_and_zscore():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(s, 3).iloc[-1] == pytest.approx(4.0)
    z = zscore(s, 3)
    assert z.iloc[-1] == pytest.approx((5.0 - 4.0) / np.std([3, 4, 5], ddof=0))


def test_ema_zscore_is_undefined_when_std_is_zero():
    # A perfectly flat series has zero dispersion: z is undefined (NaN), not zero.
    s = pd.Series([100.0] * 80)
    z = ema_zscore(s, 65)
    assert z.isna().all()


def test_donchian_excludes_current_bar():
    high = pd.Series(np.arange(1, 11, dtype=float))
    low = high - 0.5
    ch = donchian(high, low, 3)
    # At index 3 the channel must cover bars 0..2 only: max(1,2,3) = 3
    assert ch["upper"].iloc[3] == pytest.approx(3.0)
    assert ch["lower"].iloc[3] == pytest.approx(0.5)
    assert np.isnan(ch["upper"].iloc[2])


def test_rsi_all_gains_is_100_and_all_losses_is_0():
    up = pd.Series(np.arange(1, 60, dtype=float))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_atr_positive_and_causal():
    df = make_candles(200)
    a = atr(df["high"], df["low"], df["close"], 14)
    assert (a.dropna() > 0).all()
    short = atr(df["high"].iloc[:150], df["low"].iloc[:150], df["close"].iloc[:150], 14)
    pd.testing.assert_series_equal(a.iloc[:150], short)


def test_indicators_do_not_change_when_future_bars_are_appended():
    full = make_candles(500, seed=3)
    df = full.iloc[:400].copy()  # exact prefix, extra future rows in `full`
    pairs = [
        (sma(df["close"], 20), sma(full["close"], 20)),
        (ema(df["close"], 65), ema(full["close"], 65)),
        (rolling_std(df["close"], 65), rolling_std(full["close"], 65)),
        (rsi(df["close"], 14), rsi(full["close"], 14)),
        (atr(df["high"], df["low"], df["close"], 14), atr(full["high"], full["low"], full["close"], 14)),
        (adx(df["high"], df["low"], df["close"], 14), adx(full["high"], full["low"], full["close"], 14)),
        (realized_vol(df["close"], 20), realized_vol(full["close"], 20)),
    ]
    for a, b in pairs:
        np.testing.assert_allclose(a.to_numpy(), b.iloc[:400].to_numpy(), equal_nan=True)


def test_bollinger_ordering_and_slope_sign():
    df = make_candles(300, seed=5)
    bb = bollinger(df["close"], 20, 2.0)
    valid = bb.dropna()
    assert (valid["upper"] >= valid["mid"]).all()
    assert (valid["mid"] >= valid["lower"]).all()
    rising = pd.Series(np.linspace(100, 200, 200))
    assert rolling_slope(rising, 10).iloc[-1] > 0
    falling = pd.Series(np.linspace(200, 100, 200))
    assert rolling_slope(falling, 10).iloc[-1] < 0
