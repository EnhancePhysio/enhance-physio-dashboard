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
    """Return a DataFrame of ``[practitioner, clinic, total_hours, total_days]``
    for the date range. Empty if the admin tool hasn't synced any availability.

    Status rows (off/leave/sick/ph/na) contribute 0 hours — the admin tool
    already encodes this (``hours`` is set to the clinical hours only).

    ``total_days`` is the sum of the ``day_worked`` fractions the admin tool
    emits per entry (≥6 hrs = 1, 1-5.9 hrs = 0.5, else 0). Used downstream to
    override the Cliniko-based ``days_worked`` which is often inflated.
    """
    df = kpi_sync.load_admin_availability()
    if df.empty:
        return pd.DataFrame(columns=["practitioner", "clinic", "total_hours", "total_days"])
    df = df.copy()
    df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0)
    df["day_worked"] = pd.to_numeric(df.get("day_worked"), errors="coerce").fillna(0)
    df["d"] = pd.to_datetime(df.get("date"), errors="coerce").dt.date
    df = df.dropna(subset=["d"])

    start, end = _range_bounds(dr)
    if start is not None and end is not None:
        df = df[(df["d"] >= start) & (df["d"] <= end)]
    if df.empty:
        return pd.DataFrame(columns=["practitioner", "clinic", "total_hours", "total_days"])

    grp = (df.groupby(["practitioner", "clinic"], as_index=False)
             .agg(total_hours=("hours", "sum"),
                  total_days=("day_worked", "sum")))
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
    # Map admin practitioner names → Cliniko ids via the fuzzy resolver so
    # first-name-only entries and nicknames match the same way the audits
    # and punctuality merges do.
    from dashboard.manual import build_practitioner_name_index, resolve_practitioner_id
    name_idx = build_practitioner_name_index(practitioners_df, None) \
        if practitioners_df is not None and not practitioners_df.empty else {}

    def _resolve(name: str) -> str | None:
        if not name_idx:
            return None
        return resolve_practitioner_id(str(name), name_idx)

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
        pid = _resolve(name)
        if pid is None:
            continue  # practitioner isn't in Cliniko — ignore
        denom = float(r["total_hours"])
        if denom <= 0:
            continue
        numer = per_pract_service.get(str(pid), 0.0)
        # Guard against override producing 0 when service hours haven't come
        # through yet. Skip the override in that case so the Cliniko-based
        # value remains visible instead of a confusing 0%.
        if numer <= 0:
            continue
        util = min(1.0, numer / denom)
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
        # Guard: skip override if we can't match service hours (keeps the
        # Cliniko-roster value visible instead of a confusing 0%).
        if numer <= 0:
            continue
        util = min(1.0, numer / denom)
        # One row per matching business so the existing per-clinic chart
        # (keyed on business_id) picks it up unchanged.
        for bid in biz_ids:
            clinic_rows.append({"business_id": bid, "utilisation": util})
    new_util_by_clinic = pd.DataFrame(clinic_rows, columns=["business_id", "utilisation"])

    # ---- Override days_worked + avg_hours_per_day from admin data ----
    # Cliniko-based days_worked derives from white_space_hours / 6.6 which
    # over-counts for part-timers (admin/non-clinical blocks still add to
    # white space). Admin tool stores a direct day_worked fraction per day
    # (≥6 hrs = 1, 1-5.9 = ½, 0 = 0); the sum across the period is the
    # ground truth. Rewriting here keeps service_hours (numerator) from
    # Cliniko unchanged.
    sh_df = getattr(result, "service_hours", None)
    new_sh_df = None
    if sh_df is not None and not sh_df.empty and name_idx:
        days_by_pract = (hours_df.groupby("practitioner", as_index=False)
                                    .agg(total_days=("total_days", "sum")))
        day_overrides: dict[str, float] = {}
        for _, r in days_by_pract.iterrows():
            name = str(r["practitioner"]).strip()
            pid = _resolve(name)
            if pid is None:
                continue
            days = float(r["total_days"])
            if days > 0:
                day_overrides[str(pid)] = days
        if day_overrides:
            new_sh_df = sh_df.copy()
            new_sh_df["practitioner_id"] = new_sh_df["practitioner_id"].astype(str)
            # Apply override row-by-row so practitioners without admin data
            # keep Cliniko-based days.
            def _apply_days(row):
                pid = str(row["practitioner_id"])
                if pid in day_overrides:
                    row["days_worked"] = day_overrides[pid]
                    sh_hours = float(row.get("service_hours") or 0)
                    row["avg_hours_per_day"] = (
                        sh_hours / day_overrides[pid] if day_overrides[pid] > 0 else 0
                    )
                return row
            new_sh_df = new_sh_df.apply(_apply_days, axis=1)

    # ---- Rebuild MetricResult with merge-over-base semantics ----
    # Only override practitioners/clinics that have admin availability data.
    # Everyone else keeps the Cliniko-roster-based number so they don't
    # suddenly read as 0.
    def _merge_override(base_df: pd.DataFrame | None, new_df: pd.DataFrame,
                         key: str) -> pd.DataFrame:
        if base_df is None or base_df.empty:
            return new_df
        if new_df.empty:
            return base_df
        base = base_df.copy()
        base[key] = base[key].astype(str)
        new_df = new_df.copy()
        new_df[key] = new_df[key].astype(str)
        overridden_keys = set(new_df[key].tolist())
        keep = base[~base[key].isin(overridden_keys)]
        return pd.concat([keep, new_df], ignore_index=True)

    fields = {}
    for name in result.__dataclass_fields__:
        if name == "utilisation":
            fields[name] = _merge_override(getattr(result, name), new_util,
                                             "practitioner_id")
        elif name == "utilisation_by_clinic":
            fields[name] = _merge_override(getattr(result, name),
                                             new_util_by_clinic, "business_id")
        elif name == "service_hours" and new_sh_df is not None:
            fields[name] = new_sh_df
        else:
            fields[name] = getattr(result, name)
    return MetricResult(**fields)
