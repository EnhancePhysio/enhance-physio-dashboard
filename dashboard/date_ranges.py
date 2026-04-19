"""Date-range presets, all interpreted in the configured timezone."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Literal

import pytz

from dashboard.config import timezone_name, load_settings


PresetKey = Literal[
    "last_7", "last_30", "last_90",
    "last_month", "last_quarter", "ytd", "au_fy", "custom",
]

PRESETS: list[tuple[PresetKey, str]] = [
    ("last_7", "Last 7 days"),
    ("last_30", "Last 30 days"),
    ("last_90", "Last 90 days"),
    ("last_month", "Last month"),
    ("last_quarter", "Last quarter"),
    ("ytd", "Year to date"),
    ("au_fy", "This financial year (AU)"),
    ("custom", "Custom"),
]


@dataclass(frozen=True)
class DateRange:
    """Start (inclusive) and end (exclusive) in local time, plus UTC ISO strings."""
    start_local: datetime
    end_local: datetime   # exclusive upper bound

    @property
    def start_iso_utc(self) -> str:
        return self.start_local.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def end_iso_utc(self) -> str:
        return self.end_local.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @property
    def start_date(self) -> date:
        return self.start_local.date()

    @property
    def end_date_inclusive(self) -> date:
        # UI "through" date = end-exclusive minus 1 day
        return (self.end_local - timedelta(seconds=1)).date()

    def label(self) -> str:
        if self.start_date == self.end_date_inclusive:
            return self.start_date.isoformat()
        return f"{self.start_date} → {self.end_date_inclusive}"


def _now_local() -> datetime:
    return datetime.now(pytz.timezone(timezone_name()))


def _start_of_day(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_next_day(dt: datetime) -> datetime:
    return _start_of_day(dt) + timedelta(days=1)


def _au_fy_start(dt: datetime) -> datetime:
    start_month = load_settings().get("financial_year_start_month", 7)
    year = dt.year if dt.month >= start_month else dt.year - 1
    return dt.replace(year=year, month=start_month, day=1,
                      hour=0, minute=0, second=0, microsecond=0)


def resolve_preset(key: PresetKey,
                   custom_start: date | None = None,
                   custom_end: date | None = None) -> DateRange:
    tz = pytz.timezone(timezone_name())
    now = _now_local()
    today_start = _start_of_day(now)
    tomorrow_start = _start_of_next_day(now)

    # "Last N days" = the N most recent complete-or-current days (today back N-1).
    if key == "last_7":
        return DateRange(today_start - timedelta(days=6), tomorrow_start)
    if key == "last_30":
        return DateRange(today_start - timedelta(days=29), tomorrow_start)
    if key == "last_90":
        return DateRange(today_start - timedelta(days=89), tomorrow_start)
    if key == "last_month":
        first_this = today_start.replace(day=1)
        last_month_end = first_this
        last_month_start = (first_this - timedelta(days=1)).replace(day=1)
        return DateRange(last_month_start, last_month_end)
    if key == "last_quarter":
        month = today_start.month
        # Find current quarter start month (1, 4, 7, 10)
        q_start_month = ((month - 1) // 3) * 3 + 1
        current_q_start = today_start.replace(month=q_start_month, day=1)
        # Previous quarter end = current quarter start
        prev_q_end = current_q_start
        # Previous quarter start = 3 months earlier
        year = prev_q_end.year
        month = prev_q_end.month - 3
        if month <= 0:
            month += 12
            year -= 1
        prev_q_start = prev_q_end.replace(year=year, month=month, day=1)
        return DateRange(prev_q_start, prev_q_end)
    if key == "ytd":
        jan1 = today_start.replace(month=1, day=1)
        return DateRange(jan1, tomorrow_start)
    if key == "au_fy":
        fy_start = _au_fy_start(now)
        return DateRange(fy_start, tomorrow_start)
    if key == "custom":
        if not custom_start or not custom_end:
            raise ValueError("custom preset requires custom_start and custom_end")
        start_dt = tz.localize(datetime.combine(custom_start, datetime.min.time()))
        end_dt = tz.localize(datetime.combine(custom_end, datetime.min.time())) + timedelta(days=1)
        return DateRange(start_dt, end_dt)
    raise ValueError(f"Unknown preset key: {key}")


def working_days_in_range(dr: DateRange) -> int:
    """Inclusive day count across the range (for averages over 'days in period').
    Actual 'days worked' per practitioner is computed in metrics.py from availability."""
    delta = (dr.end_local - dr.start_local).days
    return max(1, delta)
