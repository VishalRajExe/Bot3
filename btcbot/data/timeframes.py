"""Timeframe helpers.  Every timestamp in this system is UTC epoch seconds."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

TIMEFRAME_SECONDS: dict[str, int] = {
    "1h": 3_600,
    "2h": 7_200,
    "4h": 14_400,
    "6h": 21_600,
    "12h": 43_200,
    "1d": 86_400,
}


def tf_seconds(timeframe: str) -> int:
    try:
        return TIMEFRAME_SECONDS[timeframe]
    except KeyError as exc:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported timeframe: {timeframe}") from exc


def bars_per_year(timeframe: str) -> float:
    return 365.25 * 86_400 / tf_seconds(timeframe)


def to_epoch(value: str | datetime | int) -> int:
    """Convert ISO date/datetime or epoch to UTC epoch seconds."""
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        dt = value
    else:
        txt = str(value).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def to_datetime(epoch: int) -> datetime:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc)


def floor_to_bar(epoch: int, timeframe: str) -> int:
    step = tf_seconds(timeframe)
    return int(epoch) - (int(epoch) % step)


def expected_bar_times(start: int, end: int, timeframe: str) -> list[int]:
    """All bar-open epochs on the UTC grid in ``[start, end)``."""
    step = tf_seconds(timeframe)
    first = floor_to_bar(start, timeframe)
    last = floor_to_bar(end, timeframe)
    return list(range(first, last, step))


def bar_close_time(open_time: int, timeframe: str) -> int:
    return int(open_time) + tf_seconds(timeframe)


def add_bars(epoch: int, timeframe: str, n: int) -> int:
    return int(epoch) + n * tf_seconds(timeframe)


def human_duration(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.0f}m"
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def parse_iso_date(value: str) -> int:
    dt = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


__all__ = [
    "TIMEFRAME_SECONDS",
    "tf_seconds",
    "bars_per_year",
    "to_epoch",
    "to_datetime",
    "floor_to_bar",
    "expected_bar_times",
    "bar_close_time",
    "add_bars",
    "human_duration",
    "parse_iso_date",
    "timedelta",
]
