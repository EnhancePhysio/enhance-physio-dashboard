"""Metric calculators. Each function returns a pandas DataFrame keyed by practitioner_id."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from .cliniko import ClinikoClient, starts_at_range_params
from .config import load_settings
from .date_ranges import DateRange
from .reference_data import extract_linked_id


# -------------------------------------------------------------------
# Fetchers — all return raw DataFrames scoped to the date range
# -------------------------------------------------------------------
def _iter_appointments(client: ClinikoClient, dr: DateRange,
                       business_ids: list[int] | None = None,
                       practitioner_ids: list[int] | None = None) -> Iterable[dict[str, Any]]:
    params = starts_at_range_params(dr.start_iso_utc, dr.end_iso_utc)
    # Cliniko supports q[]=business_id:= and q[]=practitioner_id:= — apply narrowest first.
    q = list(params["q[]"])
    if business_ids and len(business_ids) == 1:
        q.append(f"business_id:={business_ids[0]}")
    if practitioner_ids and len(practitioner_ids) == 1:
        q.append(f"practitioner_id:={practitioner_ids[0]}")
    params = {"q[]": q}
    yield from client.paginate("individual_appointments", params=params)


def fetch_appointments(client: ClinikoClient, dr: DateRange,
                       business_ids: list[int] | None = None,
                       practitioner_ids: list[int] | None = None) -> pd.DataFrame:
    rows = []
    for a in _iter_appointments(client, dr, business_ids, practitioner_ids):
        rows.append({
            "id": a.get("id"),
            "patient_id": extract_linked_id(a.get("patient"), "self")
                           or extract_linked_id(a.get("links"), "patient"),
            "practitioner_id": extract_linked_id(a.get("practitioner"), "self")
                               or extract_linked_id(a.get("links"), "practitioner"),
            "business_id": extract_linked_id(a.get("business"), "self")
                            or extract_linked_id(a.get("links"), "business"),
            "appointment_type_id": extract_linked_id(a.get("appointment_type"), "self")
                                    or extract_linked_id(a.get("links"), "appointment_type"),
            "starts_at": a.get("starts_at"),
            "ends_at": a.get("ends_at"),
            "cancelled_at": a.get("cancelled_at"),
            "did_not_arrive": a.get("did_not_arrive"),
            "cancellation_reason": a.get("cancellation_reason"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["starts_at"] = pd.to_datetime(df["starts_at"], utc=True, errors="coerce")
    df["ends_at"] = pd.to_datetime(df["ends_at"], utc=True, errors="coerce")
    df["cancelled_at"] = pd.to_datetime(df["cancelled_at"], utc=True, errors="coerce")
    # Post-filter for multi-select (server only supports single equality cleanly)
    if business_ids:
        df = df[df["business_id"].isin(business_ids)]
    if practitioner_ids:
        df = df[df["practitioner_id"].isin(practitioner_ids)]
    return df


def _is_delivered(df: pd.DataFrame) -> pd.Series:
    cancelled = df["cancelled_at"].notna()
    dna = df["did_not_arrive"].fillna(False).astype(bool)
    return ~(cancelled | dna)


# -------------------------------------------------------------------
# 4.1 New Patients
# -------------------------------------------------------------------
def new_patients(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange) -> pd.DataFrame:
    """Count distinct patients whose first attended appointment falls in the range.

    To be strictly accurate we'd need every patient's complete appointment history.
    For prototype scope we approximate via the in-range delivered appointments and
    cross-check each candidate by fetching that patient's earliest appointment.
    That's expensive (one call per candidate) so we cache per practitioner.
    """
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "new_patients"])

    delivered = appts[_is_delivered(appts)].copy()
    if delivered.empty:
        return pd.DataFrame(columns=["practitioner_id", "new_patients"])

    # Earliest in-range delivered appointment per (practitioner, patient)
    delivered = delivered.sort_values("starts_at")
    first_in_range = delivered.groupby(["practitioner_id", "patient_id"]).first().reset_index()

    # For each candidate patient, verify no prior delivered appointment exists
    # (this catches returning patients whose record pre-dates the range)
    new_rows = []
    for _, row in first_in_range.iterrows():
        earlier = list(client.paginate(
            "individual_appointments",
            params={
                "q[]": [
                    f"patient_id:={row['patient_id']}",
                    f"starts_at:<{row['starts_at'].strftime('%Y-%m-%dT%H:%M:%SZ')}",
                ],
                "per_page": 1,
            },
        ))
        is_new = not any(
            (e.get("cancelled_at") is None and not e.get("did_not_arrive", False))
            for e in earlier
        )
        if is_new:
            new_rows.append(row["practitioner_id"])

    if not new_rows:
        return pd.DataFrame(columns=["practitioner_id", "new_patients"])
    s = pd.Series(new_rows).value_counts()
    return s.rename_axis("practitioner_id").reset_index(name="new_patients")


# -------------------------------------------------------------------
# 4.2 Total Consults Delivered
# -------------------------------------------------------------------
def total_consults(appts: pd.DataFrame) -> pd.DataFrame:
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "consults_delivered"])
    delivered = appts[_is_delivered(appts)]
    g = delivered.groupby("practitioner_id").size().rename("consults_delivered").reset_index()
    return g


# -------------------------------------------------------------------
# 4.3 Service Hours (delivered appointment time + avg per day worked)
# -------------------------------------------------------------------
def service_hours(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange,
                  practitioner_ids: list[int] | None = None) -> pd.DataFrame:
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "service_hours",
                                     "days_worked", "avg_hours_per_day"])

    delivered = appts[_is_delivered(appts)].copy()
    delivered["duration_h"] = (delivered["ends_at"] - delivered["starts_at"]).dt.total_seconds() / 3600.0
    by_prac = delivered.groupby("practitioner_id").agg(
        service_hours=("duration_h", "sum"),
    ).reset_index()

    # Days worked — days with any availability OR any delivered appt
    days_map: dict[int, set] = {}
    # Cheap proxy: days with delivered appts
    tmp = delivered.dropna(subset=["starts_at"]).copy()
    tmp["d"] = tmp["starts_at"].dt.tz_convert("UTC").dt.date
    for pid, grp in tmp.groupby("practitioner_id"):
        days_map.setdefault(pid, set()).update(grp["d"].unique())

    # Also count availability_blocks as 'worked' if available
    try:
        avail = list(client.paginate(
            "availability_blocks",
            params={"q[]": [
                f"starts_at:>={dr.start_iso_utc}",
                f"starts_at:<{dr.end_iso_utc}",
            ]},
        ))
        for b in avail:
            pid = (extract_linked_id(b.get("practitioner"), "self")
                   or extract_linked_id(b.get("links"), "practitioner"))
            if pid is None:
                continue
            try:
                d = datetime.fromisoformat(b["starts_at"].replace("Z", "+00:00")).date()
                days_map.setdefault(pid, set()).add(d)
            except (KeyError, ValueError):
                pass
    except Exception:
        # availability_blocks endpoint is optional; skip silently if unavailable
        pass

    by_prac["days_worked"] = by_prac["practitioner_id"].map(
        lambda pid: max(len(days_map.get(pid, set())), 1)
    )
    by_prac["avg_hours_per_day"] = by_prac["service_hours"] / by_prac["days_worked"]
    return by_prac


# -------------------------------------------------------------------
# 4.4 PVA
# -------------------------------------------------------------------
def pva(appts: pd.DataFrame) -> pd.DataFrame:
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "pva"])
    delivered = appts[_is_delivered(appts)]
    agg = delivered.groupby("practitioner_id").agg(
        consults=("id", "count"),
        patients=("patient_id", "nunique"),
    ).reset_index()
    agg["pva"] = (agg["consults"] / agg["patients"]).where(agg["patients"] > 0, 0)
    return agg[["practitioner_id", "pva"]]


# -------------------------------------------------------------------
# 4.5 PPVA — Private + EPC together
# -------------------------------------------------------------------
def _match_any(name: str, patterns: list[str]) -> bool:
    if not name:
        return False
    return any(re.search(p, name) for p in patterns)


def ppva(appts: pd.DataFrame, appt_types: pd.DataFrame) -> pd.DataFrame:
    if appts.empty or appt_types.empty:
        return pd.DataFrame(columns=["practitioner_id", "ppva"])
    settings = load_settings()["ppva"]
    include = settings["private_name_patterns"]
    exclude = settings["excluded_name_patterns"]

    types = appt_types.copy()
    types["is_private"] = types["name"].apply(
        lambda n: _match_any(n, include) and not _match_any(n, exclude)
    )
    types["is_initial"] = types["name"].str.contains("initial", case=False, na=False)
    private_ids = set(types.loc[types["is_private"], "id"])
    initial_private_ids = set(types.loc[types["is_private"] & types["is_initial"], "id"])

    delivered = appts[_is_delivered(appts)].copy()
    delivered["is_private"] = delivered["appointment_type_id"].isin(private_ids)
    delivered["is_initial_private"] = delivered["appointment_type_id"].isin(initial_private_ids)

    by = delivered.groupby("practitioner_id").agg(
        private_consults=("is_private", "sum"),
        initial_private=("is_initial_private", "sum"),
    ).reset_index()
    by["ppva"] = (by["private_consults"] / by["initial_private"]).where(by["initial_private"] > 0, 0)
    return by[["practitioner_id", "ppva"]]


# -------------------------------------------------------------------
# 4.6 Cx / DNA rates
# -------------------------------------------------------------------
def cx_dna_rates(appts: pd.DataFrame) -> pd.DataFrame:
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "scheduled", "cancelled",
                                     "dna", "cx_rate", "dna_rate", "cx_dna_combined_rate"])
    df = appts.copy()
    df["cancelled"] = df["cancelled_at"].notna()
    df["dna"] = df["did_not_arrive"].fillna(False).astype(bool)
    g = df.groupby("practitioner_id").agg(
        scheduled=("id", "count"),
        cancelled=("cancelled", "sum"),
        dna=("dna", "sum"),
    ).reset_index()
    g["cx_rate"] = g["cancelled"] / g["scheduled"].where(g["scheduled"] > 0, 1)
    g["dna_rate"] = g["dna"] / g["scheduled"].where(g["scheduled"] > 0, 1)
    g["cx_dna_combined_rate"] = (g["cancelled"] + g["dna"]) / g["scheduled"].where(g["scheduled"] > 0, 1)
    return g


# -------------------------------------------------------------------
# 4.7 Utilisation
# -------------------------------------------------------------------
def utilisation(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange) -> pd.DataFrame:
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "utilisation"])
    settings = load_settings()["utilisation"]
    qualify = [k.lower() for k in settings["qualifying_keywords"]]
    exclude = [k.lower() for k in settings["excluded_keywords"]]

    def _classify(label: str | None) -> str:
        text = (label or "").lower()
        if any(k in text for k in exclude):
            return "excluded"
        if any(k in text for k in qualify):
            return "qualifying"
        return "other"

    # Available minutes per practitioner (from availability_blocks)
    avail_minutes: dict[int, float] = {}
    try:
        for b in client.paginate("availability_blocks",
                                  params={"q[]": [f"starts_at:>={dr.start_iso_utc}",
                                                   f"starts_at:<{dr.end_iso_utc}"]}):
            pid = (extract_linked_id(b.get("practitioner"), "self")
                   or extract_linked_id(b.get("links"), "practitioner"))
            if pid is None:
                continue
            try:
                s = datetime.fromisoformat(b["starts_at"].replace("Z", "+00:00"))
                e = datetime.fromisoformat(b["ends_at"].replace("Z", "+00:00"))
                mins = (e - s).total_seconds() / 60.0
                avail_minutes[pid] = avail_minutes.get(pid, 0.0) + mins
            except (KeyError, ValueError):
                pass
    except Exception:
        pass

    # Qualifying unavailable blocks
    qualifying_minutes: dict[int, float] = {}
    try:
        for b in client.paginate("unavailable_blocks",
                                  params={"q[]": [f"starts_at:>={dr.start_iso_utc}",
                                                   f"starts_at:<{dr.end_iso_utc}"]}):
            pid = (extract_linked_id(b.get("practitioner"), "self")
                   or extract_linked_id(b.get("links"), "practitioner"))
            if pid is None:
                continue
            label = b.get("name") or b.get("notes") or ""
            cls = _classify(label)
            if cls != "qualifying":
                continue
            try:
                s = datetime.fromisoformat(b["starts_at"].replace("Z", "+00:00"))
                e = datetime.fromisoformat(b["ends_at"].replace("Z", "+00:00"))
                mins = (e - s).total_seconds() / 60.0
                qualifying_minutes[pid] = qualifying_minutes.get(pid, 0.0) + mins
            except (KeyError, ValueError):
                pass
    except Exception:
        pass

    # Delivered appointment minutes
    delivered = appts[_is_delivered(appts)].copy()
    delivered["duration_min"] = (delivered["ends_at"] - delivered["starts_at"]).dt.total_seconds() / 60.0
    appt_minutes = delivered.groupby("practitioner_id")["duration_min"].sum().to_dict()

    pids = set(appt_minutes) | set(qualifying_minutes) | set(avail_minutes)
    rows = []
    for pid in pids:
        numer = appt_minutes.get(pid, 0.0) + qualifying_minutes.get(pid, 0.0)
        denom = avail_minutes.get(pid, 0.0)
        if denom <= 0:
            util = 0.0
        else:
            util = min(numer / denom, 1.0)
        rows.append({"practitioner_id": pid, "utilisation": util})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# 4.10 Notes Completion — within 24h
# -------------------------------------------------------------------
def notes_completion(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange) -> pd.DataFrame:
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "notes_completion"])

    settings = load_settings()
    window_h = settings.get("notes_completion_window_hours", 24)

    delivered = appts[_is_delivered(appts)].copy()
    if delivered.empty:
        return pd.DataFrame(columns=["practitioner_id", "notes_completion"])

    # Fetch treatment notes for each patient in scope (bulk endpoint if supported)
    notes_by_appt: dict[int, list[dict]] = {}
    try:
        for n in client.paginate("treatment_notes",
                                  params={"q[]": [f"created_at:>={dr.start_iso_utc}",
                                                   f"created_at:<{dr.end_iso_utc}"]}):
            appt_id = (extract_linked_id(n.get("appointment"), "self")
                       or extract_linked_id(n.get("links"), "appointment"))
            if appt_id is None:
                continue
            notes_by_appt.setdefault(appt_id, []).append(n)
    except Exception:
        # Fallback — query per-patient only if needed
        for pid in delivered["patient_id"].dropna().unique():
            try:
                for n in client.paginate(f"patients/{int(pid)}/treatment_notes"):
                    appt_id = (extract_linked_id(n.get("appointment"), "self")
                               or extract_linked_id(n.get("links"), "appointment"))
                    if appt_id:
                        notes_by_appt.setdefault(appt_id, []).append(n)
            except Exception:
                continue

    def _pass(row) -> bool | None:
        notes = notes_by_appt.get(row["id"], [])
        if not notes:
            return False
        for n in notes:
            if n.get("draft", False):
                continue
            up = n.get("updated_at")
            if not up:
                continue
            try:
                up_dt = datetime.fromisoformat(up.replace("Z", "+00:00"))
            except ValueError:
                continue
            delta_h = (up_dt - row["starts_at"].to_pydatetime()).total_seconds() / 3600.0
            if 0 <= delta_h <= window_h:
                return True
        return False

    delivered["notes_pass"] = delivered.apply(_pass, axis=1)
    g = delivered.groupby("practitioner_id")["notes_pass"].agg(
        lambda s: float(s.sum()) / float(len(s)) if len(s) else 0.0
    ).rename("notes_completion").reset_index()
    return g


# -------------------------------------------------------------------
# Combiner
# -------------------------------------------------------------------
@dataclass
class MetricResult:
    appointments: pd.DataFrame
    new_patients: pd.DataFrame
    consults: pd.DataFrame
    service_hours: pd.DataFrame
    pva: pd.DataFrame
    ppva: pd.DataFrame
    cx_dna: pd.DataFrame
    utilisation: pd.DataFrame
    notes: pd.DataFrame


def compute_core_metrics(client: ClinikoClient, dr: DateRange, appt_types: pd.DataFrame,
                          business_ids: list[int] | None = None,
                          practitioner_ids: list[int] | None = None,
                          include_new_patients_check: bool = False) -> MetricResult:
    appts = fetch_appointments(client, dr, business_ids, practitioner_ids)
    np_df = (new_patients(client, appts, dr)
             if include_new_patients_check
             else pd.DataFrame(columns=["practitioner_id", "new_patients"]))
    return MetricResult(
        appointments=appts,
        new_patients=np_df,
        consults=total_consults(appts),
        service_hours=service_hours(client, appts, dr, practitioner_ids),
        pva=pva(appts),
        ppva=ppva(appts, appt_types),
        cx_dna=cx_dna_rates(appts),
        utilisation=utilisation(client, appts, dr),
        notes=notes_completion(client, appts, dr),
    )


def merge_per_practitioner(result: MetricResult,
                           practitioners: pd.DataFrame,
                           manual_nps: pd.DataFrame | None = None,
                           manual_punct: pd.DataFrame | None = None,
                           audit: pd.DataFrame | None = None) -> pd.DataFrame:
    """Produce a wide DataFrame — one row per practitioner, every raw metric in columns."""
    base = practitioners[["id", "label"]].rename(columns={"id": "practitioner_id"})
    for df, cols in [
        (result.new_patients, ["practitioner_id", "new_patients"]),
        (result.consults, ["practitioner_id", "consults_delivered"]),
        (result.service_hours, ["practitioner_id", "service_hours", "days_worked", "avg_hours_per_day"]),
        (result.pva, ["practitioner_id", "pva"]),
        (result.ppva, ["practitioner_id", "ppva"]),
        (result.cx_dna, ["practitioner_id", "cx_rate", "dna_rate", "cx_dna_combined_rate"]),
        (result.utilisation, ["practitioner_id", "utilisation"]),
        (result.notes, ["practitioner_id", "notes_completion"]),
    ]:
        if df is not None and not df.empty:
            base = base.merge(df[cols], on="practitioner_id", how="left")
    if manual_nps is not None and not manual_nps.empty:
        base = base.merge(manual_nps, on="practitioner_id", how="left")
    if manual_punct is not None and not manual_punct.empty:
        base = base.merge(manual_punct, on="practitioner_id", how="left")
    if audit is not None and not audit.empty:
        base = base.merge(audit, on="practitioner_id", how="left")
    # Fill missing numeric columns with 0 so downstream scoring doesn't blow up
    numeric_defaults = {
        "new_patients": 0, "consults_delivered": 0,
        "service_hours": 0.0, "days_worked": 0, "avg_hours_per_day": 0.0,
        "pva": 0.0, "ppva": 0.0, "cx_rate": 0.0, "dna_rate": 0.0,
        "cx_dna_combined_rate": 0.0, "utilisation": 0.0,
        "notes_completion": 0.0, "nps": 0.0,
        "punctuality_within_15": 0.0, "audit_pct": 0.0,
    }
    for col, default in numeric_defaults.items():
        if col in base.columns:
            base[col] = base[col].fillna(default)
    return base
