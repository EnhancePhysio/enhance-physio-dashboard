"""Per-practitioner 6-month trend charts.

Reads from:
  * ``dashboard.snapshots`` — monthly snapshots of Cliniko-derived KPIs
    (service hours, consults, utilisation, days worked, PVA, cx/dna rates,
    notes completion). Captured by ``create_snapshot(year, month)`` and
    saved to ``data/snapshots/<year>_<month>.json`` + GitHub.
  * ``data/punctuality/`` CSVs synced from the admin tool — aggregated
    weekly → monthly "within 10 min %" per practitioner.
  * ``data/nps/`` CSVs synced from the admin tool — already monthly.
  * ``data/kpi_sync/audits.csv`` — monthly audit scores from the admin tool.

If a month hasn't been snapshotted yet (common when the trends tab is first
opened), its row is left blank — the chart just shows a gap. A manager can
click "Snapshot missing months now" to backfill.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from dashboard import kpi_sync
from dashboard.manual import load_nps, load_punctuality
from dashboard.snapshots import create_snapshot, load_snapshot


# -------------------------------------------------------------------
# Date helpers
# -------------------------------------------------------------------
def _last_n_months(n: int, today: date | None = None) -> list[tuple[int, int]]:
    """Return the last `n` completed months, oldest-first.
    E.g. called in Oct 2026 with n=6 → [(2026,4), (2026,5), ..., (2026,9)]."""
    today = today or date.today()
    y, m = today.year, today.month
    out: list[tuple[int, int]] = []
    for _ in range(n):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        out.append((y, m))
    out.reverse()
    return out


def _month_label(y: int, m: int) -> str:
    return f"{y}-{m:02d}"


# -------------------------------------------------------------------
# Per-metric extractors — pull one practitioner's value from a snapshot
# -------------------------------------------------------------------
def _snap_metric_sum(snap: dict[str, Any], pid: int,
                      df_name: str, col: str) -> float | None:
    """Sum this practitioner's value for `col` in `df_name` across every
    org in the snapshot. Returns None if the practitioner doesn't appear
    in any org for this month."""
    found = False
    total = 0.0
    for org in snap.get("organizations", {}).values():
        recs = (org.get("metrics") or {}).get(df_name, [])
        if not isinstance(recs, list):
            continue
        for r in recs:
            if str(r.get("practitioner_id")) == str(pid):
                v = r.get(col)
                if v is None:
                    continue
                try:
                    total += float(v)
                    found = True
                except (TypeError, ValueError):
                    pass
                break  # one row per practitioner per df per org
    return total if found else None


def _snap_metric_mean(snap: dict[str, Any], pid: int,
                       df_name: str, col: str) -> float | None:
    """Average across orgs instead of sum (used for rates)."""
    vals: list[float] = []
    for org in snap.get("organizations", {}).values():
        recs = (org.get("metrics") or {}).get(df_name, [])
        if not isinstance(recs, list):
            continue
        for r in recs:
            if str(r.get("practitioner_id")) == str(pid):
                v = r.get(col)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
                break
    return sum(vals) / len(vals) if vals else None


# -------------------------------------------------------------------
# Admin-tool monthly aggregates (keyed by practitioner NAME, not id)
# -------------------------------------------------------------------
def _admin_punct_pct_by_month(name: str) -> dict[str, float]:
    """Punctuality within 10 min %, aggregated per calendar month.
    Weekly rows in data/punctuality/*.csv → month → %."""
    df = load_punctuality()
    if df.empty or "practitioner" not in df.columns:
        return {}
    sub = df[df["practitioner"].astype(str) == name].copy()
    if sub.empty:
        return {}
    for c in ("bucket_0_5", "bucket_6_10", "bucket_11_14", "bucket_15_plus"):
        sub[c] = pd.to_numeric(sub.get(c), errors="coerce").fillna(0)
    sub["within_10"] = sub["bucket_0_5"] + sub["bucket_6_10"]
    sub["total"] = sub["within_10"] + sub["bucket_11_14"] + sub["bucket_15_plus"]
    sub["ym"] = pd.to_datetime(sub["week_starting"], errors="coerce").dt.strftime("%Y-%m")
    sub = sub.dropna(subset=["ym"])
    grp = sub.groupby("ym").agg(within=("within_10", "sum"), total=("total", "sum"))
    grp = grp[grp["total"] > 0]
    return {ym: round(100.0 * r["within"] / r["total"], 1) for ym, r in grp.iterrows()}


def _admin_nps_by_month(name: str) -> dict[str, float]:
    """NPS score per month for this practitioner. Admin tool stores it with
    week_starting = first-of-month, so one row per month per practitioner."""
    df = load_nps()
    if df.empty or "practitioner" not in df.columns:
        return {}
    sub = df[df["practitioner"].astype(str) == name].copy()
    if sub.empty:
        return {}
    sub["ym"] = pd.to_datetime(sub["week_starting"], errors="coerce").dt.strftime("%Y-%m")
    sub = sub.dropna(subset=["ym"])
    sub["nps"] = pd.to_numeric(sub.get("nps"), errors="coerce")
    sub = sub.dropna(subset=["nps"])
    # Last entry per month wins.
    sub = sub.sort_values("week_starting").groupby("ym", as_index=False).last()
    return {r.ym: float(r.nps) for r in sub.itertuples()}


def _admin_audit_pct_by_month(name: str) -> dict[str, float]:
    """Audit % per month: sum(score_y) ÷ sum(score_denom) across all patient
    rows for this practitioner in that month."""
    df = kpi_sync.load_admin_audits()
    if df.empty or "practitioner" not in df.columns:
        return {}
    sub = df[df["practitioner"].astype(str) == name].copy()
    if sub.empty:
        return {}
    sub["score_y"] = pd.to_numeric(sub.get("score_y"), errors="coerce").fillna(0)
    sub["score_denom"] = pd.to_numeric(sub.get("score_denom"), errors="coerce").fillna(0)
    sub["ym"] = sub["month"].astype(str)
    grp = sub.groupby("ym").agg(y=("score_y", "sum"), denom=("score_denom", "sum"))
    grp = grp[grp["denom"] > 0]
    return {ym: round(100.0 * r["y"] / r["denom"], 1) for ym, r in grp.iterrows()}


# -------------------------------------------------------------------
# Tab UI
# -------------------------------------------------------------------
def trends_tab(filters: dict) -> None:
    st.subheader("📈 Practitioner trends — last 6 months")
    st.caption(
        "Line charts of each KPI across the last six completed calendar months. "
        "Clinic-derived metrics come from monthly snapshots; punctuality / NPS / "
        "audit come from the admin tool's shared CSVs."
    )

    practs = filters.get("practitioners")
    if practs is None or practs.empty:
        st.warning("No practitioners loaded from Cliniko.")
        return

    labels = sorted(practs["label"].tolist())
    label_to_id = dict(zip(practs["label"], practs["id"]))
    chosen = st.selectbox("Practitioner", labels)
    pid = label_to_id[chosen]
    name = chosen  # the "label" is usually the practitioner's display name

    months = _last_n_months(6)
    month_labels = [_month_label(y, m) for y, m in months]

    # Load each month's snapshot once.
    snaps: dict[str, dict | None] = {}
    missing: list[tuple[int, int]] = []
    for (y, m) in months:
        s = load_snapshot(y, m)
        snaps[_month_label(y, m)] = s
        if s is None:
            missing.append((y, m))

    if missing:
        st.warning(
            "Missing snapshots for: "
            + ", ".join(_month_label(y, m) for y, m in missing)
            + ". Those months will show as gaps on the charts. "
            "Click the button below to backfill — this hits the Cliniko API "
            "once per missing month, so it may take a minute."
        )
        if st.button("📸 Snapshot missing months now", key="trends_backfill"):
            with st.spinner("Fetching from Cliniko…"):
                for (y, m) in missing:
                    try:
                        create_snapshot(y, m)
                        st.toast(f"Snapshotted {y}-{m:02d}", icon="📸")
                    except Exception as e:
                        st.error(f"{y}-{m:02d}: {e}")
            st.rerun()

    # Build a monthly series per KPI.
    def _snap_series(df_name: str, col: str, reducer: str = "sum") -> pd.Series:
        vals: list[float | None] = []
        for lbl in month_labels:
            s = snaps[lbl]
            if s is None:
                vals.append(None)
                continue
            if reducer == "mean":
                vals.append(_snap_metric_mean(s, pid, df_name, col))
            else:
                vals.append(_snap_metric_sum(s, pid, df_name, col))
        return pd.Series(vals, index=month_labels, dtype="float64")

    # Admin-tool monthly maps keyed by ym; align to months list.
    punct_pct = _admin_punct_pct_by_month(name)
    nps_vals = _admin_nps_by_month(name)
    audit_pct = _admin_audit_pct_by_month(name)

    def _admin_series(d: dict[str, float]) -> pd.Series:
        return pd.Series([d.get(lbl) for lbl in month_labels],
                          index=month_labels, dtype="float64")

    # KPI definitions: (chart title, Series, help text).
    charts: list[tuple[str, pd.Series, str]] = [
        (
            "Service hours per month",
            _snap_series("service_hours", "service_hours"),
            "Billable face-to-face hours delivered by this practitioner.",
        ),
        (
            "Consults delivered per month",
            _snap_series("consults", "consults_delivered"),
            "Count of consults completed.",
        ),
        (
            "Days worked per month",
            _snap_series("service_hours", "days_worked"),
            "Clinical days worked (≥6 service hrs = 1, 1–5.9 = ½).",
        ),
        (
            "Utilisation %",
            _snap_series("utilisation", "utilisation", reducer="mean"),
            "Service hours ÷ clinically-available hours.",
        ),
        (
            "PVA (patient visit average)",
            _snap_series("pva", "pva", reducer="mean"),
            "Consults ÷ new patients. Higher = better retention.",
        ),
        (
            "Cx + DNA combined rate %",
            _snap_series("cx_dna", "cx_dna_combined_rate", reducer="mean"),
            "Cancellations + no-shows as % of booked appts. Lower is better.",
        ),
        (
            "Notes completion %",
            _snap_series("notes", "notes_completion", reducer="mean"),
            "Share of consults with a finalised treatment note. Target 100%.",
        ),
        (
            "Punctuality (within 10 min) %",
            _admin_series(punct_pct),
            "From admin-tool tally sheets. 0–5 min + 6–10 min buckets ÷ total.",
        ),
        (
            "NPS",
            _admin_series(nps_vals),
            "Monthly NPS score entered on the admin tool.",
        ),
        (
            "Audit score %",
            _admin_series(audit_pct),
            "Monthly audit score (Y ÷ (Y + N) across patients, N/A excluded).",
        ),
    ]

    # Render — 2 columns of charts.
    for row_start in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, (title, series, help_text) in zip(cols, charts[row_start:row_start + 2]):
            with col:
                st.markdown(f"**{title}**")
                st.caption(help_text)
                nonnull = series.dropna()
                if nonnull.empty:
                    st.info("No data for this metric in the last 6 months.")
                else:
                    # Direction indicator — compare first vs last populated point.
                    first, last = nonnull.iloc[0], nonnull.iloc[-1]
                    if last > first * 1.05:
                        arrow = "📈 trending up"
                    elif last < first * 0.95:
                        arrow = "📉 trending down"
                    else:
                        arrow = "➡️ flat"
                    st.caption(
                        f"{arrow} · first {first:.1f} → latest {last:.1f}"
                    )
                    st.line_chart(series, use_container_width=True)

    with st.expander("Raw monthly values"):
        tbl = pd.DataFrame({title: s for title, s, _ in charts})
        tbl.index.name = "month"
        st.dataframe(tbl, use_container_width=True)
