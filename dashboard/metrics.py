"""Metric calculators. Each function returns a pandas DataFrame keyed by practitioner_id."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import pandas as pd
import pytz

from dashboard.cliniko import ClinikoClient, starts_at_range_params
from dashboard.config import load_settings, timezone_name
from dashboard.date_ranges import DateRange
from dashboard.reference_data import extract_linked_id


# -------------------------------------------------------------------
# Fetchers — all return raw DataFrames scoped to the date range
# -------------------------------------------------------------------
def _iter_appointments(client: ClinikoClient, dr: DateRange,
                       business_ids: list[int] | None = None,
                       practitioner_ids: list[int] | None = None,
                       extra_q: list[str] | None = None) -> Iterable[dict[str, Any]]:
    params = starts_at_range_params(dr.start_iso_utc, dr.end_iso_utc)
    # Cliniko supports q[]=business_id:= and q[]=practitioner_id:= — apply narrowest first.
    q = list(params["q[]"])
    if business_ids and len(business_ids) == 1:
        q.append(f"business_id:={business_ids[0]}")
    if practitioner_ids and len(practitioner_ids) == 1:
        q.append(f"practitioner_id:={practitioner_ids[0]}")
    if extra_q:
        q.extend(extra_q)
    params = {"q[]": q}
    yield from client.paginate("individual_appointments", params=params)


def _appt_row(a: dict[str, Any]) -> dict[str, Any]:
    aid = a.get("id")
    return {
        # Keep ID as string — appointment IDs are referenced by treatment_notes
        # via a URL link that extract_linked_id parses as str, so both sides of
        # the join must match as strings.
        "id": None if aid is None else str(aid),
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
        # Appointment's own updated_at — used as a fallback signal for
        # "when was the treatment note finalised" when the /treatment_notes
        # endpoint returns nothing (Cliniko flips treatment_note_status as
        # part of the appt record update, so updated_at moves when the
        # note transitions 20→90).
        "appt_updated_at": a.get("updated_at"),
        "cancelled_at": a.get("cancelled_at"),
        "archived_at": a.get("archived_at"),
        "did_not_arrive": a.get("did_not_arrive"),
        "cancellation_reason": a.get("cancellation_reason"),
        # Cliniko's own treatment-note status on the appointment record.
        # Values observed: 10 = N/A, 20 = pending, 30 = draft, 40 = overdue,
        # 90 = finalised / complete. We use this directly rather than
        # fetching /treatment_notes per appt — much faster and more reliable.
        "treatment_note_status": a.get("treatment_note_status"),
        "patient_arrived": a.get("patient_arrived"),
    }


def fetch_appointments(client: ClinikoClient, dr: DateRange,
                       business_ids: list[int] | None = None,
                       practitioner_ids: list[int] | None = None) -> pd.DataFrame:
    """Fetch in-range individual_appointments, including cancelled, excluding archived.

    Two-pass fetch:
      1. Default query — active, non-archived, non-cancelled.
      2. Explicit cancelled pass — Cliniko's default filter excludes records
         with cancelled_at set, so we force inclusion via a range filter.
    De-dupe by id. Archived records are excluded (per Matt: archived = NA,
    not counted as delivered/cancelled/DNA).
    """
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add(a: dict[str, Any]) -> None:
        row = _appt_row(a)
        rid = row["id"]
        if rid is None or rid in seen_ids:
            return
        seen_ids.add(rid)
        rows.append(row)

    # Pass 1 — default (active appointments)
    for a in _iter_appointments(client, dr, business_ids, practitioner_ids):
        _add(a)

    # Pass 2 — force cancelled appointments to surface. Cliniko excludes rows
    # with cancelled_at set from the default query; a range filter on
    # cancelled_at flips that to "cancelled_at is set and >= epoch", which
    # returns every cancellation in the starts_at window.
    try:
        for a in _iter_appointments(
            client, dr, business_ids, practitioner_ids,
            extra_q=["cancelled_at:>=1970-01-01T00:00:00Z"],
        ):
            _add(a)
    except Exception:
        pass

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["starts_at"] = pd.to_datetime(df["starts_at"], utc=True, errors="coerce")
    df["ends_at"] = pd.to_datetime(df["ends_at"], utc=True, errors="coerce")
    df["cancelled_at"] = pd.to_datetime(df["cancelled_at"], utc=True, errors="coerce")
    df["archived_at"] = pd.to_datetime(df["archived_at"], utc=True, errors="coerce")
    if "appt_updated_at" in df.columns:
        df["appt_updated_at"] = pd.to_datetime(
            df["appt_updated_at"], utc=True, errors="coerce"
        )
    # Archived = NA. Drop entirely so they don't count anywhere.
    df = df[df["archived_at"].isna()].copy()
    # Post-filter for multi-select (server only supports single equality cleanly)
    if business_ids:
        df = df[df["business_id"].isin([str(b) for b in business_ids])]
    if practitioner_ids:
        df = df[df["practitioner_id"].isin([str(p) for p in practitioner_ids])]
    return df


def _is_delivered(df: pd.DataFrame) -> pd.Series:
    """A delivered consult = not cancelled AND patient arrived.

    `archived_at` alone does NOT disqualify — plenty of clinics archive
    completed appointments for tidiness. Only cancellation or DNA removes
    an appointment from the 'delivered' set.
    """
    cancelled = df["cancelled_at"].notna()
    dna = df["did_not_arrive"].fillna(False).astype(bool)
    return ~(cancelled | dna)


# -------------------------------------------------------------------
# 4.1 New Patients
# -------------------------------------------------------------------
def new_patients(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange,
                  appt_types: pd.DataFrame | None = None) -> pd.DataFrame:
    """Count distinct NEW CLIENTS seen by each practitioner in the range.

    Two signals decide whether a patient is "new" in the range:
      1. They attended a delivered appointment whose type matches the
         Initial naming pattern (captures Physio, Massage, etc. Initials).
      2. Their Cliniko `created_at` falls inside the range — Cliniko creates
         the patient record at their first-ever booking, so created_at in
         range ≈ first-time patient of the practice. This is the signal for
         DVA / NDIS / TAC / NSW WorkCover patients, which don't have
         dedicated "Initial" appointment types.

    If either signal fires, the patient's delivered appts in the range get
    counted per-practitioner (distinct patient IDs). Same patient seen by
    two practitioners counts once for each — matches Cliniko's own
    Practitioner Performance report attribution.
    """
    if appts.empty or appt_types is None or appt_types.empty:
        return pd.DataFrame(columns=["practitioner_id", "new_patients"])

    # -- Signal 1: Initial-type appts --
    initial_pattern = re.compile(
        r"\b(initial|new\s*patient|new\s*client|assessment|first\s*visit)\b",
        re.IGNORECASE,
    )
    initial_type_ids = {
        str(row["id"]) for _, row in appt_types.iterrows()
        if row.get("name") and initial_pattern.search(str(row["name"]))
    }

    # -- Signal 2: patients created inside the range --
    # OPT-IN because it paginates /patients with updated_at:>=start, which
    # on busy clinics returns thousands of records and 20-50 API calls.
    # Behind settings.new_patients.scan_patients so Matt can toggle it
    # when he wants DVA/NDIS/TAC/WC first-timers included (they don't have
    # dedicated "Initial" appointment types). Default is OFF — fast loads,
    # Initial-type signal only. Page cap prevents any chance of an infinite
    # pagination loop dragging the dashboard down.
    created_in_range: set[str] = set()
    settings = load_settings()
    np_cfg = settings.get("new_patients", {}) if isinstance(settings, dict) else {}
    scan_patients = bool(np_cfg.get("scan_patients", False))
    max_pages = int(np_cfg.get("scan_patients_max_pages", 20))
    if scan_patients:
        try:
            start_utc_dt = datetime.fromisoformat(dr.start_iso_utc.replace("Z", "+00:00"))
            end_utc_dt = datetime.fromisoformat(dr.end_iso_utc.replace("Z", "+00:00"))
            pages_seen = 0
            records_seen = 0
            records_per_page = 100  # matches cliniko.page_size
            for p in client.paginate(
                "patients",
                params={"q[]": [f"updated_at:>={dr.start_iso_utc}"]},
            ):
                records_seen += 1
                if records_seen // records_per_page > pages_seen:
                    pages_seen = records_seen // records_per_page
                    if pages_seen >= max_pages:
                        break  # safety cap — don't hang the dashboard
                pid = p.get("id")
                c = p.get("created_at")
                if not pid or not c:
                    continue
                try:
                    c_dt = datetime.fromisoformat(c.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if start_utc_dt <= c_dt < end_utc_dt:
                    created_in_range.add(str(pid))
        except Exception:
            # If Cliniko throws, we still fall back to the Initial-type signal.
            pass

    if not initial_type_ids and not created_in_range:
        return pd.DataFrame(columns=["practitioner_id", "new_patients"])

    delivered = appts[_is_delivered(appts)]
    if delivered.empty:
        return pd.DataFrame(columns=["practitioner_id", "new_patients"])

    is_initial_type = (
        delivered["appointment_type_id"].astype(str).isin(initial_type_ids)
        if initial_type_ids else pd.Series(False, index=delivered.index)
    )
    is_date_new = (
        delivered["patient_id"].astype(str).isin(created_in_range)
        if created_in_range else pd.Series(False, index=delivered.index)
    )
    new_rows = delivered[is_initial_type | is_date_new]
    if new_rows.empty:
        return pd.DataFrame(columns=["practitioner_id", "new_patients"])

    g = (
        new_rows.groupby("practitioner_id")["patient_id"]
        .nunique()
        .rename("new_patients")
        .reset_index()
    )
    g["practitioner_id"] = g["practitioner_id"].astype(str)
    return g


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
    """Service hours + days worked.

    `days_worked` counts unique LOCAL-timezone dates on which the practitioner
    delivered at least one appointment. We dropped the availability_blocks
    heuristic — it was over-counting because Cliniko returns roster-pattern
    blocks that include days the practitioner never actually consulted
    (which produced values like 26 days in a 30-day window).
    """
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "service_hours",
                                     "days_worked", "avg_hours_per_day"])

    tz = pytz.timezone(timezone_name())
    delivered = appts[_is_delivered(appts)].copy()
    delivered["duration_h"] = (delivered["ends_at"] - delivered["starts_at"]).dt.total_seconds() / 3600.0
    by_prac = delivered.groupby("practitioner_id").agg(
        service_hours=("duration_h", "sum"),
    ).reset_index()

    # Days worked — unique LOCAL dates with at least one delivered appt
    days_map: dict[str, set] = {}
    tmp = delivered.dropna(subset=["starts_at"]).copy()
    # Convert from UTC to the clinic timezone so an 8am Sydney appt counts as
    # that calendar day, not the previous UTC date.
    tmp["d"] = tmp["starts_at"].dt.tz_convert(tz).dt.date
    for pid, grp in tmp.groupby("practitioner_id"):
        days_map.setdefault(pid, set()).update(grp["d"].unique())

    by_prac["days_worked"] = by_prac["practitioner_id"].map(
        lambda pid: max(len(days_map.get(pid, set())), 1)
    )
    by_prac["avg_hours_per_day"] = by_prac["service_hours"] / by_prac["days_worked"]
    return by_prac


# -------------------------------------------------------------------
# 4.4 PVA
# -------------------------------------------------------------------
def pva(appts: pd.DataFrame, new_patients_df: pd.DataFrame) -> pd.DataFrame:
    """PVA = attended appointments ÷ new clients seen (Cliniko's definition).

    Matches the Cliniko Practitioner Performance report exactly.
    Verified against real data:
        Alesha: 545 attended / 58 new = 9.4  ✓
        Ben:    112 / 35                = 3.2 ✓
        Emma:   568 / 46                = 12.3 ✓
    """
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "pva"])
    delivered = appts[_is_delivered(appts)]
    consults = (
        delivered.groupby("practitioner_id").size().rename("consults").reset_index()
    )
    if new_patients_df is None or new_patients_df.empty:
        consults["new_patients"] = 0
    else:
        consults = consults.merge(new_patients_df, on="practitioner_id", how="left")
        consults["new_patients"] = consults["new_patients"].fillna(0)
    consults["pva"] = consults.apply(
        lambda r: (r["consults"] / r["new_patients"]) if r["new_patients"] > 0 else 0.0,
        axis=1,
    )
    return consults[["practitioner_id", "pva"]]


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
    """Utilisation = (delivered + qualifying admin) ÷ white-block minutes.

    "White-block minutes" = the practitioner's actual working window per day
    on the Cliniko calendar, i.e. the span from their first to last
    appointment MINUS the grey (unavailable) blocks sitting between those
    appointments for lunch / admin / "do not book".

    The `availability_blocks` endpoint is too sparse to be useful (47 blocks
    across 16 practitioners over 30 days = 30 min/day max denominator), so
    we infer the working window from the actual calendar. Qualifying
    unavailable_blocks (billable report, case conference, mentoring, etc.)
    add back to the numerator so that productive admin time isn't counted
    against utilisation — matches Matt's scoring model.

    Returns NaN (not 0) when a practitioner has no appointments in the
    range, so the rubric can skip them rather than dragging them to the
    bottom of the rank.
    """
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "utilisation"])
    settings = load_settings()["utilisation"]
    qualify = [k.lower() for k in settings["qualifying_keywords"]]
    exclude = [k.lower() for k in settings["excluded_keywords"]]

    def _classify(label: str | None) -> str:
        text = (label or "").lower()
        if any(k in text for k in qualify):
            return "qualifying"
        if any(k in text for k in exclude):
            return "non_qualifying"
        # Anything else (lunch, break, generic hold, etc.) is non-qualifying
        # — i.e. it's part of the working window but NOT counted as work
        # output. Subtract it from the denominator.
        return "non_qualifying"

    tz = pytz.timezone(timezone_name())

    # Pull ALL unavailable_blocks in the range. We'll split them per-day
    # and classify each one as qualifying or non-qualifying.
    unavail_blocks: list[dict[str, Any]] = []
    try:
        for b in client.paginate(
            "unavailable_blocks",
            params={"q[]": [
                f"starts_at:>={dr.start_iso_utc}",
                f"starts_at:<{dr.end_iso_utc}",
            ]},
        ):
            pid = (extract_linked_id(b.get("practitioner"), "self")
                   or extract_linked_id(b.get("links"), "practitioner"))
            if pid is None:
                continue
            try:
                s = datetime.fromisoformat(b["starts_at"].replace("Z", "+00:00"))
                e = datetime.fromisoformat(b["ends_at"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            label = b.get("name") or b.get("notes") or ""
            unavail_blocks.append({
                "practitioner_id": str(pid),
                "start": s,
                "end": e,
                "kind": _classify(label),
                "label": label,
            })
    except Exception:
        pass

    # Per-practitioner per-LOCAL-date:
    #   window   = [first_appt_start .. last_appt_end]
    #   denom    = window_minutes - non_qualifying_unavail_minutes_in_window
    #   numer    = delivered_minutes + qualifying_unavail_minutes_in_window
    delivered = appts[_is_delivered(appts)].copy()
    if delivered.empty:
        return pd.DataFrame(columns=["practitioner_id", "utilisation"])
    delivered = delivered.dropna(subset=["starts_at", "ends_at"]).copy()
    delivered["local_date"] = delivered["starts_at"].dt.tz_convert(tz).dt.date

    numer_mins: dict[str, float] = {}
    denom_mins: dict[str, float] = {}

    for (pid, day), grp in delivered.groupby(["practitioner_id", "local_date"]):
        pid = str(pid)
        day_start = grp["starts_at"].min().to_pydatetime()
        day_end = grp["ends_at"].max().to_pydatetime()
        window_min = (day_end - day_start).total_seconds() / 60.0
        if window_min <= 0:
            continue

        # Delivered-appt minutes for this practitioner-day (numerator)
        delivered_min = (
            (grp["ends_at"] - grp["starts_at"]).dt.total_seconds().sum() / 60.0
        )

        # Unavailable blocks overlapping this practitioner-day window
        qual_overlap_min = 0.0
        nonqual_overlap_min = 0.0
        for b in unavail_blocks:
            if b["practitioner_id"] != pid:
                continue
            overlap_start = max(b["start"], day_start)
            overlap_end = min(b["end"], day_end)
            if overlap_end <= overlap_start:
                continue
            mins = (overlap_end - overlap_start).total_seconds() / 60.0
            if b["kind"] == "qualifying":
                qual_overlap_min += mins
            else:
                nonqual_overlap_min += mins

        denom = window_min - nonqual_overlap_min
        numer = delivered_min + qual_overlap_min
        if denom <= 0:
            continue
        numer_mins[pid] = numer_mins.get(pid, 0.0) + numer
        denom_mins[pid] = denom_mins.get(pid, 0.0) + denom

    rows = []
    for pid, denom in denom_mins.items():
        numer = numer_mins.get(pid, 0.0)
        util = min(numer / denom, 1.0) if denom > 0 else None
        rows.append({"practitioner_id": pid, "utilisation": util})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# 4.10 Notes Completion — FINALISED WITHIN 24H (medicolegal requirement)
# -------------------------------------------------------------------
def notes_completion(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange) -> pd.DataFrame:
    """Medicolegal notes compliance = finalised within 24h ÷ notes expected.

    Implementation is deliberately fast — no per-appointment or per-patient
    network calls. Every signal we need is already on the appointment record
    Cliniko returned in the primary fetch.

    Denominator: delivered appts where `treatment_note_status != 10`
        10 = N/A (no note expected)
        20 = pending  ·  30 = draft  ·  40 = overdue  ·  90 = finalised

    Numerator: appts where `treatment_note_status == 90` AND the appt's own
    `updated_at` is within 24h (± small buffer) of its `starts_at`.
    Cliniko bumps appt.updated_at whenever the status flips to 90, so this
    is a close proxy for "when was the note finalised". Not 100% perfect —
    if someone edits the appt later for an unrelated reason, updated_at
    moves past the true finalisation time — but it avoids the
    tens-of-thousands of API calls a note-level check would need against a
    month of data. We can add note-payload verification later behind a
    "deep audit" button if Matt wants per-response precision.
    """
    if appts.empty:
        return pd.DataFrame(columns=["practitioner_id", "notes_completion"])

    settings = load_settings()
    window_h = float(settings.get("notes_completion_window_hours", 24))

    delivered = appts[_is_delivered(appts)].copy()
    if delivered.empty:
        return pd.DataFrame(columns=["practitioner_id", "notes_completion"])

    status = pd.to_numeric(delivered["treatment_note_status"], errors="coerce")
    # Treat missing/NaN status as "expected" — Cliniko populates it for
    # every real consult, so a missing value usually means a partial API
    # response; conservative here is to still require a note.
    delivered["_expected"] = status.fillna(20).astype(int) != 10
    delivered["_status_is_final"] = status.fillna(0).astype(int) == 90

    # Hours between appt start and last update to the appt record. Requires
    # appt_updated_at to be populated (added in _appt_row). If missing for
    # some reason, the row fails the 24h check — conservative.
    if "appt_updated_at" in delivered.columns:
        delta_h = (
            (delivered["appt_updated_at"] - delivered["starts_at"]).dt.total_seconds()
            / 3600.0
        )
    else:
        delta_h = pd.Series([float("nan")] * len(delivered), index=delivered.index)

    within_window = delta_h.between(-1.0, window_h, inclusive="both").fillna(False)
    delivered["_passed_24h"] = (
        delivered["_expected"] & delivered["_status_is_final"] & within_window
    )

    g = delivered.groupby("practitioner_id").agg(
        expected=("_expected", "sum"),
        passed_24h=("_passed_24h", "sum"),
    ).reset_index()

    g["notes_completion"] = g.apply(
        lambda r: (float(r["passed_24h"]) / float(r["expected"]))
                  if r["expected"] > 0 else float("nan"),
        axis=1,
    )
    return g[["practitioner_id", "notes_completion"]]


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
                          practitioner_ids: list[int] | None = None) -> MetricResult:
    """Pull every metric for the given range. New-patients is cheap now
    (one extra call to /patients with a created_at filter), so it's always
    computed — PVA depends on it."""
    appts = fetch_appointments(client, dr, business_ids, practitioner_ids)
    np_df = new_patients(client, appts, dr, appt_types)
    return MetricResult(
        appointments=appts,
        new_patients=np_df,
        consults=total_consults(appts),
        service_hours=service_hours(client, appts, dr, practitioner_ids),
        pva=pva(appts, np_df),
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
    base = practitioners[["id", "label"]].rename(columns={"id": "practitioner_id"}).copy()
    # Belt-and-braces: coerce the merge key on BOTH sides to strings.
    # pandas dtype inference has bitten us before — if one side ends up as
    # int64 and the other as object-of-str, the merge silently drops rows
    # (NaN → fillna(0) → metric reads 0). Forcing str on every df before
    # merging eliminates that whole class of bug.
    base["practitioner_id"] = base["practitioner_id"].astype(str)
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
            right = df[cols].copy()
            right["practitioner_id"] = right["practitioner_id"].astype(str)
            base = base.merge(right, on="practitioner_id", how="left")
        else:
            # Source DataFrame is empty (no data in this date range, or API
            # returned nothing). Still create the expected columns so
            # downstream code can reference them safely.
            for c in cols:
                if c != "practitioner_id" and c not in base.columns:
                    base[c] = pd.NA
    # Same dtype-coercion hygiene for the supplementary frames. Without this,
    # audit_df's practitioner_id (produced via groupby on PatientAudit
    # objects) can land as object-of-int-like-strings and silently fail
    # to join against base's explicit str column — the exact failure mode
    # that made "audits are done but don't show in raw data scores".
    if manual_nps is not None and not manual_nps.empty:
        m = manual_nps.copy()
        m["practitioner_id"] = m["practitioner_id"].astype(str)
        base = base.merge(m, on="practitioner_id", how="left")
    if manual_punct is not None and not manual_punct.empty:
        m = manual_punct.copy()
        m["practitioner_id"] = m["practitioner_id"].astype(str)
        base = base.merge(m, on="practitioner_id", how="left")
    if audit is not None and not audit.empty:
        a = audit.copy()
        a["practitioner_id"] = a["practitioner_id"].astype(str)
        base = base.merge(a, on="practitioner_id", how="left")
    # Fill missing numeric columns with 0 so downstream scoring doesn't blow up
    numeric_defaults = {
        "new_patients": 0, "consults_delivered": 0,
        "service_hours": 0.0, "days_worked": 0, "avg_hours_per_day": 0.0,
        "pva": 0.0, "ppva": 0.0, "cx_rate": 0.0, "dna_rate": 0.0,
        "cx_dna_combined_rate": 0.0, "utilisation": 0.0,
        "notes_completion": 0.0, "nps": 0.0,
        "punctuality_within_15": 0.0, "audit_pct": 0.0,
        "audit_epc_pct": 0.0, "audit_private_pct": 0.0, "patients_audited": 0,
    }
    for col, default in numeric_defaults.items():
        if col in base.columns:
            base[col] = base[col].fillna(default)
    return base
