"""Admin-tool audit scores → per-practitioner DataFrame for scoring.

Reads the admin-tool-synced ``audits.csv`` (patient-level rows) and rolls up
to per-practitioner aggregate scores in the shape ``merge_per_practitioner``
already expects:

    practitioner_id, audit_pct, audit_epc_pct, audit_private_pct, patients_audited

Name-matching maps admin "practitioner" → Cliniko ``practitioner_id`` via the
practitioners reference DataFrame.

v27.6.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from dashboard import kpi_sync


def _range_bounds(dr) -> tuple[date | None, date | None]:
    try:
        s = dr.start_date  # type: ignore[attr-defined]
    except Exception:
        s = dr.start.date() if hasattr(dr, "start") else None
    try:
        e = dr.end_date_inclusive  # type: ignore[attr-defined]
    except Exception:
        e = dr.end.date() if hasattr(dr, "end") else None
    return s, e


def admin_audit_dataframe(dr, practitioners_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame keyed on practitioner_id with audit aggregates for
    the given date range. Empty frame if nothing to show."""
    cols = ["practitioner_id", "audit_pct", "audit_epc_pct", "audit_private_pct",
            "patients_audited"]
    df = kpi_sync.load_admin_audits()
    if df.empty or "practitioner" not in df.columns:
        return pd.DataFrame(columns=cols)

    # Filter to the dashboard's selected date range by month.
    df = df.copy()
    df["m"] = pd.to_datetime(df["month"].astype(str) + "-01",
                              errors="coerce").dt.date
    df = df.dropna(subset=["m"])
    start, end = _range_bounds(dr)
    if start is not None and end is not None:
        # Include any audit month that overlaps the selected range.
        df = df[(df["m"] >= date(start.year, start.month, 1))
                & (df["m"] <= date(end.year, end.month, 1))]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["score_y"] = pd.to_numeric(df.get("score_y"), errors="coerce").fillna(0)
    df["score_denom"] = pd.to_numeric(df.get("score_denom"), errors="coerce").fillna(0)

    # Overall per-practitioner
    overall = (df.groupby("practitioner", as_index=False)
                 .agg(y=("score_y", "sum"),
                      denom=("score_denom", "sum"),
                      patients_audited=("patient_name", "nunique")))
    overall["audit_pct"] = overall.apply(
        lambda r: r["y"] / r["denom"] if r["denom"] else 0.0, axis=1,
    )

    # Per-funding split
    def _subscore(funding: str) -> pd.DataFrame:
        sub = df[df["funding"].astype(str).str.lower() == funding.lower()]
        if sub.empty:
            return pd.DataFrame(columns=["practitioner", "pct"])
        g = (sub.groupby("practitioner", as_index=False)
                .agg(y=("score_y", "sum"), denom=("score_denom", "sum")))
        g["pct"] = g.apply(
            lambda r: r["y"] / r["denom"] if r["denom"] else 0.0, axis=1,
        )
        return g[["practitioner", "pct"]]

    epc = _subscore("EPC").rename(columns={"pct": "audit_epc_pct"})
    priv = _subscore("Private").rename(columns={"pct": "audit_private_pct"})
    merged = (overall
              .merge(epc, on="practitioner", how="left")
              .merge(priv, on="practitioner", how="left"))
    merged["audit_epc_pct"] = merged["audit_epc_pct"].fillna(0.0)
    merged["audit_private_pct"] = merged["audit_private_pct"].fillna(0.0)

    # Map name → Cliniko practitioner_id using the dashboard's fuzzy
    # resolver — handles first-name-only entries (e.g. admin tool has
    # "Hamzeh" / "Jas" / "Isaac" while Cliniko labels are full names).
    if practitioners_df is None or practitioners_df.empty \
            or "label" not in practitioners_df.columns:
        return pd.DataFrame(columns=cols)
    from dashboard.manual import build_practitioner_name_index, resolve_practitioner_id
    idx = build_practitioner_name_index(practitioners_df, None)
    merged["practitioner_id"] = merged["practitioner"].apply(
        lambda n: resolve_practitioner_id(n, idx))
    merged = merged.dropna(subset=["practitioner_id"])
    if merged.empty:
        return pd.DataFrame(columns=cols)
    merged["practitioner_id"] = merged["practitioner_id"].astype(str)
    return merged[cols].reset_index(drop=True)
