"""Admin-tool-driven utilisation.

Replaces the Cliniko-roster denominator with the admin team's manually-entered
``availability.csv`` hours. Cliniko rosters at some clinics are overblown or
inconsistently labelled, which drags utilisation down artificially. The admin
team already types each practitioner's actual clinically-available hours per
day, so this gives a cleaner ground truth:

    utilisation = service_hours (Cliniko) ÷ availability_hours (admin tool)

For per-clinic rollups the same ratio is grouped by business, matching the
clinic name on the Cliniko business label (e.g. "Enhance Physio Albury").

Called from ``overview_tab`` after the live/snapshot metrics are loaded and
exclusions have been applied, before scoring.

v27.5.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from dashboard import kpi_sync


# -------------------------------------------------------------------
# Date range helpers
# -------------------------------------------------------------------
def _range_bounds(dr) -> tuple[date, date]:
    """Pull start/end dates from a dashboard DateRange object."""
    try:
        s = dr.start_date  # type: ignore[attr-defined]
    except Exception:
        s = dr.start.date() if hasattr(dr, "start") else None
    try:
        e = dr.end_date_inclusive  # type: ignore[attr-defined]
    except Exception:
        e = dr.end.date() if hasattr(dr, "end") else None
    return s, e


# -------------------------------------------------------------------
# Admin availability → per-practitioner / per-clinic hours in range
# -------------------------------------------------------------------
def admin_availability_hours(dr) -> pd.DataFrame:
    """Return a DataFrame of ``[practitioner, clinic, total_hours]`` for the
    date range. Empty if the admin tool hasn't synced any availability.

    Status rows (off/leave/sick/ph/na) contribute 0 hours — the admin tool
    already encodes this (``hours`` is set to the clinical hours only).
    """
    df = kpi_sync.load_admin_availability()
    if df.empty:
        return pd.DataFrame(columns=["practitioner", "clinic", "total_hours"])
    df = df.copy()
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0)
    df["d"] = pd.to_datetime(df.get("date"), errors="coerce").dt.date
    df = df.dropna(subset=["d"])

    start, end = _range_bounds(dr)
    if start is not None and end is not None:
        df = df[(df["d"] >= start) & (df["d"] <= end)]
    if df.empty:
        return pd.DataFrame(columns=["practitioner", "clinic", "total_hours"])

    grp = (df.groupby(["practitioner", "clinic"], as_index=False)
             .agg(total_hours=("hours", "sum")))
    return grp[grp["total_hours"] > 0].reset_index(drop=True)


# -------------------------------------------------------------------
# Clinic name → business_id resolution via the Cliniko businesses frame.
# -------------------------------------------------------------------
def _business_ids_for_clinic_name(businesses_df: pd.DataFrame, clinic: str) -> set[str]:
    if businesses_df is None or businesses_df.empty:
        return set()
    needle = clinic.strip().lower()
    out: set[str] = set()
    for _, row in businesses_df.iterrows():
        label = str(row.get("label") or row.get("business_name") or "").lower()
        if needle in label:
            bid = row.get("id")
            if bid is not None:
                out.add(str(bid))
    return out


# -------------------------------------------------------------------
# Service hours per clinic (from the appointments table on MetricResult)
# -------------------------------------------------------------------
def _service_hours_by_business(result) -> dict[str, float]:
    """Return {business_id (str): delivered hours} from the appointments
    DataFrame on a MetricResult. Delivered = not cancelled, not DNA."""
    appts = getattr(result, "appointments", None)
    if appts is None or appts.empty:
        return {}
    df = appts.copy()
    # Normalise business_id to string for safe matching.
    if "business_id" not in df.columns:
        return {}
    df["business_id"] = df["business_id"].astype(str)
    # Drop cancelled + DNA.
    if "cancelled_at" in df.columns:
        df = df[df["cancelled_at"].isna() | (df["cancelled_at"] == "")]
    if "did_not_arrive" in df.columns:
        df = df[~df["did_not_arrive"].fillna(False).astype(bool)]
    if df.empty or "starts_at" not in df.columns or "ends_at" not in df.columns:
        return {}
    df["starts_at"] = pd.to_datetime(df["starts_at"], utc=True, errors="coerce")
    df["ends_at"] = pd.to_datetime(df["ends_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["starts_at", "ends_at"])
    df["duration_h"] = (df["ends_at"] - df["starts_at"]).dt.total_seconds() / 3600.0
    out = df.groupby("business_id")["duration_h"].sum().round(2).to_dict()
    return {str(k): float(v) for k, v in out.items()}


# -------------------------------------------------------------------
# Main entry — overlay admin availability onto MetricResult
# -------------------------------------------------------------------
def apply_admin_utilisation(result, dr, practitioners_df: pd.DataFrame,
                              businesses_df: pd.DataFrame):
    """Return a NEW MetricResult with ``utilisation`` and
    ``utilisation_by_clinic`` recomputed from the admin tool's
    ``availability.csv`` hours. If no admin availability is present for
    the date range the result is returned unchanged (so the dashboard
    falls back to Cliniko-roster-based numbers)."""
    if result is None:
        return result
    hours_df = admin_availability_hours(dr)
    if hours_df.empty:
        return result

    from dashboard.metrics import MetricResult

    # ---- Per-practitioner ----
    # service_hours DataFrame already keys by practitioner_id.
    # Map admin practitioner names → Cliniko ids via practitioners_df.label.
    label_to_id = {}
    if practitioners_df is not None and not practitioners_df.empty \
            and "label" in practitioners_df.columns and "id" in practitioners_df.columns:
        label_to_id = {str(l).strip().lower(): i
                       for l, i in zip(practitioners_df["label"], practitioners_df["id"])}

    sh = getattr(result, "service_hours", pd.DataFrame())
    if sh is None or sh.empty or "service_hours" not in sh.columns:
        per_pract_service = {}
    else:
        per_pract_service = {str(pid): float(hrs) for pid, hrs
                             in zip(sh["practitioner_id"], sh["service_hours"].fillna(0))}

    # Group admin hours by practitioner (sum across clinics for the period).
    adm_by_pract = (hours_df.groupby("practitioner", as_index=False)
                             .agg(total_hours=("total_hours", "sum")))
    rows = []
    for _, r in adm_by_pract.iterrows():
        name = str(r["practitioner"]).strip()
        pid = label_to_id.get(name.lower())
        if pid is None:
            continue  # practitioner isn't in Cliniko — ignore
        denom = float(r["total_hours"])
        if denom <= 0:
            continue
        numer = per_pract_service.get(str(pid), 0.0)
        util = min(1.0, numer / denom) if denom > 0 else None
        rows.append({"practitioner_id": pid, "utilisation": util})
    new_util = pd.DataFrame(rows, columns=["practitioner_id", "utilisation"])

    # ---- Per-clinic ----
    sh_by_biz = _service_hours_by_business(result)   # {business_id: delivered_h}
    adm_by_clinic = (hours_df.groupby("clinic", as_index=False)
                               .agg(total_hours=("total_hours", "sum")))
    clinic_rows = []
    for _, r in adm_by_clinic.iterrows():
        clinic = str(r["clinic"]).strip()
        denom = float(r["total_hours"])
        if denom <= 0:
            continue
        biz_ids = _business_ids_for_clinic_name(businesses_df, clinic)
        if not biz_ids:
            continue
        numer = sum(sh_by_biz.get(bid, 0.0) for bid in biz_ids)
        util = min(1.0, numer / denom) if denom > 0 else None
        # One row per matching business so the existing per-clinic chart
        # (keyed on business_id) picks it up unchanged.
        for bid in biz_ids:
            clinic_rows.append({"business_id": bid, "utilisation": util})
    new_util_by_clinic = pd.DataFrame(clinic_rows, columns=["business_id", "utilisation"])

    # ---- Rebuild MetricResult with overrides ----
    fields = {}
    for name in result.__dataclass_fields__:
        if name == "utilisation":
            fields[name] = new_util if not new_util.empty else getattr(result, name)
        elif name == "utilisation_by_clinic":
            fields[name] = new_util_by_clinic if not new_util_by_clinic.empty else getattr(result, name)
        else:
            fields[name] = getattr(result, name)
    return MetricResult(**fields)
