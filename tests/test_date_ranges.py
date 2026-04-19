"""Smoke tests for date-range logic."""
from datetime import date, timedelta

from dashboard.date_ranges import resolve_preset, working_days_in_range


def test_last_7_produces_a_week():
    dr = resolve_preset("last_7")
    assert (dr.end_local - dr.start_local).days == 7   # 6-days-ago → tomorrow exclusive


def test_last_30():
    dr = resolve_preset("last_30")
    assert (dr.end_local - dr.start_local).days == 30


def test_au_fy_start_is_jul_1():
    dr = resolve_preset("au_fy")
    assert dr.start_local.month == 7
    assert dr.start_local.day == 1


def test_ytd_starts_jan_1():
    dr = resolve_preset("ytd")
    assert dr.start_local.month == 1
    assert dr.start_local.day == 1


def test_custom_range():
    start = date(2026, 1, 1)
    end = date(2026, 1, 31)
    dr = resolve_preset("custom", start, end)
    assert dr.start_local.date() == start
    # End is exclusive — so end_local is Feb 1 (start of day)
    assert (dr.end_local.date() - timedelta(days=1)) == end


def test_working_days():
    dr = resolve_preset("last_7")
    # 6-days-ago → tomorrow exclusive = 7 days
    assert working_days_in_range(dr) == 7
