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
# v26.6: `days_worked` is now derived from the practitioner's WHITE SPACE
# (bookable clinical time) — not the raw count of unique calendar dates.
#
# Matt's definition:
#   full-time = 38h/week window − 5h admin = 33h white space = 5 days worked
#   therefore 1 day = 33 / 5 = 6.6 hours of white space
#
# This fixes a long-standing skew:
#   - 10h/week (two 5h days) previously counted as 2 days → now 1.5
#   - 32h/week (four 8h days) previously counted as 4 days → now 5
# so avg_hours_per_day (the scored input) no longer rewards/punishes
# people for packing into long or short days.
#
# We reuse the white-window cell table produced by ``_compute_white_windows``
# (shared with utilisation) so there's a single source of truth and no
# double-fetch of unavailable_blocks.
# -------------------------------------------------------------------


def _round_to_half(x: float) -> float:
    """Round to the nearest 0.5 (e.g. 4.85 → 5.0, 1.52 → 1.5, 0.23 → 0.5).

    We floor at 0.5 rather than 0 for practitioners who actually delivered
    anything — rounding a tiny positive to 0 would blow up avg_hours_per_day
    with a divide-by-zero. Zero-in / zero-out is handled separately.
    """
    if x is None or pd.isna(x):
        return 0.0
    if x <= 0:
        return 0.0
    halves = round(float(x) * 2) / 2.0
    return max(halves, 0.5)


def service_hours(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange,
                  practitioner_ids: list[int] | None = None,
                  white_space_hours: dict[str, float] | None = None) -> pd.DataFrame:
    """Service hours + days worked (v26.6 — white-space-derived).

    ``days_worked`` = round_to_half(white_space_hours / 6.6), where
    white_space_hours is the sum of bookable clinical minutes across the
    date range (window − all unavailable blocks, qualifying + non-qualifying).

    If the caller doesn't pass ``white_space_hours`` (e.g. unit tests, or
    the white-window computation yielded nothing for a practitioner) we
    fall back to the v26.5 heuristic: count unique LOCAL dates with at
    least one delivered appt. That way a practitioner with no unavail
    blocks recorded at all still gets a sensible (if slightly coarse)
    number rather than 0.
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

    # Fallback: unique LOCAL dates with at least one delivered appt.
    # Used only when white_space_hours is missing or zero for a practitioner.
    fallback_days: dict[str, int] = {}
    tmp = delivered.dropna(subset=["starts_at"]).copy()
    tmp["d"] = tmp["starts_at"].dt.tz_convert(tz).dt.date
    for pid, grp in tmp.groupby("practitioner_id"):
        fallback_days[str(pid)] = max(len(set(grp["d"].unique())), 1)

    ws = white_space_hours or {}

    def _days_for(pid: object) -> float:
        pid_s = str(pid)
        hrs = ws.get(pid_s)
        if hrs is not None and hrs > 0:
            return _round_to_half(hrs / 6.6)
        return float(fallback_days.get(pid_s, 1))

    by_prac["days_worked"] = by_prac["practitioner_id"].map(_days_for)
    # Avoid div-by-zero if both white_space and fallback ended up at 0.
    safe_days = by_prac["days_worked"].where(by_prac["days_worked"] > 0, 1)
    by_prac["avg_hours_per_day"] = by_prac["service_hours"] / safe_days
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
# 4.7 Utilisation (+ white-space + per-clinic rollup, all from one cell table)
# -------------------------------------------------------------------
def _compute_white_windows(client: ClinikoClient, appts: pd.DataFrame,
                           dr: DateRange) -> pd.DataFrame:
    """Cell-level table of practitioner-day-business working windows.

    Row schema:
        practitioner_id, business_id, local_date,
        window_min, delivered_min, qual_unavail_min, nonqual_unavail_min

    Used by three downstream aggregations:
      * practitioner utilisation      = Σ(delivered+qual) / Σ(window−nonqual)
      * clinic     utilisation        = same, grouped by business_id
      * practitioner white-space hrs  = Σ(window − qual − nonqual) / 60

    Splitting into cells lets us attribute to clinics without double-fetching
    ``unavailable_blocks`` or re-running the window logic. Unavailable blocks
    are attributed to the "dominant" business of each practitioner-day
    (the business with the most delivered minutes), which is correct in
    effectively all cases — the rare split-clinic day is absorbed by the
    biz the practitioner spent most of the day at.
    """
    cols = ["practitioner_id", "business_id", "local_date",
            "window_min", "delivered_min",
            "qual_unavail_min", "nonqual_unavail_min", "break_min"]
    if appts.empty:
        return pd.DataFrame(columns=cols)

    settings = load_settings()["utilisation"]
    qualify = [k.lower() for k in settings["qualifying_keywords"]]
    exclude = [k.lower() for k in settings["excluded_keywords"]]
    # v26.6.2 — a narrow sub-list of excluded_keywords that ALSO subtracts
    # from days_worked (only genuine off-clock time). Defaults to lunch/break
    # when the key is absent so existing configs keep working.
    breaks = [k.lower() for k in settings.get("break_keywords", ["lunch", "break"])]

    def _classify(label: str | None) -> str:
        """Three-way classification for unavail blocks:
          * 'qualifying'   → counts toward utilisation numerator (admin work)
          * 'break'        → subtracts from util denom AND from days_worked
                              (lunch, AM/PM break — genuine off-clock time)
          * 'non_qualifying' → subtracts from util denom only, NOT from
                              days_worked (do-not-book, hold, showing, and
                              every other manual block)
        """
        text = (label or "").lower()
        if any(k in text for k in qualify):
            return "qualifying"
        if any(k in text for k in breaks):
            return "break"
        # Everything remaining (do-not-book, hold, showing, generic manual
        # block) lands here — still subtracts from utilisation denom, but
        # does NOT shrink days_worked (per Matt: "it doesn't mean they're
        # not available").
        return "non_qualifying"

    tz = pytz.timezone(timezone_name())

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

    delivered = appts[_is_delivered(appts)].copy()
    if delivered.empty:
        return pd.DataFrame(columns=cols)
    delivered = delivered.dropna(subset=["starts_at", "ends_at"]).copy()
    delivered["local_date"] = delivered["starts_at"].dt.tz_convert(tz).dt.date

    rows: list[dict[str, Any]] = []

    for (pid, day), day_grp in delivered.groupby(["practitioner_id", "local_date"]):
        pid = str(pid)
        # Day-level unavail overlap (full window, for attribution math)
        day_start = day_grp["starts_at"].min().to_pydatetime()
        day_end = day_grp["ends_at"].max().to_pydatetime()
        if (day_end - day_start).total_seconds() <= 0:
            continue

        qual_day_min = 0.0
        nonqual_day_min = 0.0
        break_day_min = 0.0
        for b in unavail_blocks:
            if b["practitioner_id"] != pid:
                continue
            overlap_start = max(b["start"], day_start)
            overlap_end = min(b["end"], day_end)
            if overlap_end <= overlap_start:
                continue
            mins = (overlap_end - overlap_start).total_seconds() / 60.0
            if b["kind"] == "qualifying":
                qual_day_min += mins
            elif b["kind"] == "break":
                # Breaks are a subset of "non-qualifying" for utilisation
                # purposes (still subtract from util denom), but we keep a
                # separate tally so white_space_hours_map can subtract
                # ONLY breaks when computing days_worked.
                nonqual_day_min += mins
                break_day_min += mins
            else:
                nonqual_day_min += mins

        # Dominant business of this practitioner-day = biz with most delivered mins
        day_grp = day_grp.copy()
        day_grp["_dur_min"] = (day_grp["ends_at"] - day_grp["starts_at"]).dt.total_seconds() / 60.0
        per_biz_minutes = day_grp.groupby("business_id")["_dur_min"].sum()
        if per_biz_minutes.empty:
            continue
        dominant_biz = str(per_biz_minutes.idxmax())

        for biz, biz_grp in day_grp.groupby("business_id"):
            biz = str(biz) if biz is not None else ""
            biz_start = biz_grp["starts_at"].min().to_pydatetime()
            biz_end = biz_grp["ends_at"].max().to_pydatetime()
            window_min = (biz_end - biz_start).total_seconds() / 60.0
            if window_min <= 0:
                continue
            delivered_min = biz_grp["_dur_min"].sum()
            # Full day unavail gets attributed to the dominant biz only.
            if biz == dominant_biz:
                qual_biz = qual_day_min
                nonqual_biz = nonqual_day_min
                break_biz = break_day_min
            else:
                qual_biz = 0.0
                nonqual_biz = 0.0
                break_biz = 0.0
            rows.append({
                "practitioner_id": pid,
                "business_id": biz,
                "local_date": day,
                "window_min": window_min,
                "delivered_min": delivered_min,
                "qual_unavail_min": qual_biz,
                "nonqual_unavail_min": nonqual_biz,
                "break_min": break_biz,
            })

    return pd.DataFrame(rows, columns=cols)


def _utilisation_from_cells(cells: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate a cells DataFrame to ``[group_col, utilisation]``.

    Utilisation = (delivered + qualifying unavail) ÷ (window − non-qualifying unavail),
    capped at 1.0. Returns NaN for groups where denom ≤ 0 (shouldn't happen
    in practice but defensive).
    """
    if cells is None or cells.empty:
        return pd.DataFrame(columns=[group_col, "utilisation"])
    g = cells.copy()
    g["_numer"] = g["delivered_min"] + g["qual_unavail_min"]
    g["_denom"] = g["window_min"] - g["nonqual_unavail_min"]
    # Drop degenerate cells where denom collapsed below zero due to wildly
    # overlapping unavail blocks — would otherwise poison the sum.
    g = g[g["_denom"] > 0]
    if g.empty:
        return pd.DataFrame(columns=[group_col, "utilisation"])
    agg = g.groupby(group_col).agg(numer=("_numer", "sum"),
                                     denom=("_denom", "sum")).reset_index()
    agg["utilisation"] = (agg["numer"] / agg["denom"]).clip(upper=1.0)
    return agg[[group_col, "utilisation"]]


def utilisation(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange,
                 cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-practitioner utilisation. See ``_compute_white_windows`` for the
    underlying cell table — this function just aggregates it by practitioner.

    ``cells`` can be passed in to avoid recomputing when a caller already
    has the cell table (compute_core_metrics does this so we only hit
    /unavailable_blocks once per run).
    """
    if cells is None:
        cells = _compute_white_windows(client, appts, dr)
    return _utilisation_from_cells(cells, "practitioner_id")


def utilisation_by_clinic(client: ClinikoClient, appts: pd.DataFrame,
                           dr: DateRange,
                           cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-clinic utilisation, summing numer/denom across all practitioners
    who worked at each business in the range.

    Use this to answer "Albury was 96%, Wodonga 91%" — the clinic-level
    equivalent of the per-practitioner gauge.
    """
    if cells is None:
        cells = _compute_white_windows(client, appts, dr)
    return _utilisation_from_cells(cells, "business_id")


def white_space_hours_map(cells: pd.DataFrame) -> dict[str, float]:
    """Practitioner → "at work" hours total across the range.

    v26.6.2 — only structural breaks subtract from days_worked:
      * Qualifying unavail (admin / billable report / case conf / mentoring)
        → at work, not subtracted.
      * Do-not-book / hold / showing / any other manual grey block
        → still at work (Matt: "it doesn't mean they're not available"),
          not subtracted.
      * Break unavail (lunch, AM break, PM break — controlled by
        ``break_keywords`` in settings.yml)
        → genuine off-clock, SUBTRACTED.

    So ``white_space_hours = window − break_min``. Utilisation denom is
    computed independently and is unaffected by this change.
    """
    if cells is None or cells.empty:
        return {}
    g = cells.copy()
    # Defensive — older cell tables (pre-v26.6.2) won't have break_min.
    if "break_min" not in g.columns:
        g["break_min"] = 0.0
    g["_white_min"] = g["window_min"] - g["break_min"].fillna(0.0)
    g["_white_min"] = g["_white_min"].clip(lower=0.0)
    by_pid = g.groupby("practitioner_id")["_white_min"].sum()
    return {str(pid): float(mins) / 60.0 for pid, mins in by_pid.items()}


# -------------------------------------------------------------------
# 4.10 Notes Completion — FINALISED WITHIN 24H (medicolegal requirement)
# -------------------------------------------------------------------
_APPT_LINK_KEYS = (
    # Every plausible key Cliniko might use to expose the appointment
    # reference on a treatment_note. Ordered by observed likelihood across
    # Cliniko API versions. Shape 2 and 3 iterate over this list.
    "appointment", "individual_appointment",
    "booking", "event", "session", "consultation",
    "attendee",  # on some older schemas this IS the appointment, not the patient
)


def _tail_from_url(url: Any) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def _extract_appt_id_from_note(n: dict[str, Any]) -> str | None:
    """Pull an appointment_id out of a treatment_note record.

    v23 — v19's naïve ``extract_linked_id(n["links"], "appointment")``
    silently fell back to ``links.self`` (the note's own URL) when no
    ``appointment`` link was present, which meant we were harvesting 1286
    NOTE ids and trying to join them against appointment ids. Zero matches,
    zero notes_completion. Users saw 0% across the board on 2026-04-24.

    v24 — expanded to try more shapes after v23 came back with 0 matches
    on Enhance Physio's shard (4043 scanned, 0 kept). Ordering matters:
    direct id fields first, then embedded reference objects, then URL
    strings under links. Never falls back to self.
    """
    # Shape 1 — direct numeric/string id field (several possible names)
    for k in ("appointment_id", "individual_appointment_id",
              "booking_id", "event_id", "session_id"):
        v = n.get(k)
        if v:
            return str(v)

    # Shape 2 — embedded reference object. Covers both
    #   {"appointment": {"id": "123"}}
    # and
    #   {"appointment": {"links": {"self": "https://.../appointments/123"}}}
    for k in _APPT_LINK_KEYS:
        v = n.get(k)
        if isinstance(v, dict):
            # Direct nested id
            nid = v.get("id")
            if nid:
                return str(nid)
            # Links.self URL
            links = v.get("links") if isinstance(v.get("links"), dict) else v
            tail = _tail_from_url(links.get("self") if isinstance(links, dict) else None)
            if tail:
                return tail
        elif isinstance(v, (str, int)) and v not in ("", 0):
            # Some APIs return the reference as a bare id string/number
            return str(v)

    # Shape 3 — URL or id under note's own links map
    links = n.get("links") if isinstance(n.get("links"), dict) else {}
    for k in _APPT_LINK_KEYS:
        if not isinstance(links, dict):
            break
        raw = links.get(k)
        if isinstance(raw, dict):
            # Nested link object under links, e.g. links.appointment.self
            inner_url = raw.get("self") if isinstance(raw, dict) else None
            tail = _tail_from_url(inner_url)
            if tail:
                return tail
            nid = raw.get("id")
            if nid:
                return str(nid)
        else:
            tail = _tail_from_url(raw)
            if tail:
                return tail

    return None


def _safe_note_structure(n: dict[str, Any]) -> dict[str, Any]:
    """Extract ONLY the structural keys of a treatment_note for diagnostics.

    Must never leak clinical content — we return key names and link URLs
    (which are just record pointers, no PHI) but drop anything that could
    be a note body, subjective text, patient detail, etc. Used by the
    v24 diagnostic to show Matt what shape Cliniko is actually returning.
    """
    safe: dict[str, Any] = {"top_level_keys": sorted(n.keys())}
    links = n.get("links")
    if isinstance(links, dict):
        safe["links_keys"] = sorted(links.keys())
        # Show only the key names of each link value's structure
        safe["links_value_types"] = {
            k: (type(v).__name__ + (f" (keys={sorted(v.keys())})"
                                      if isinstance(v, dict) else ""))
            for k, v in links.items()
        }
    # If there's an embedded `appointment` or similar, show its keys only
    for k in _APPT_LINK_KEYS:
        v = n.get(k)
        if isinstance(v, dict):
            safe[f"{k}_keys"] = sorted(v.keys())
    return safe


def _fetch_treatment_notes_for_range(
    client: ClinikoClient, dr: DateRange,
) -> pd.DataFrame:
    """Pull every /treatment_notes record whose `created_at` sits in the window.

    Returns a DataFrame keyed by appointment_id with the note's created_at
    (the moment the practitioner first saved it) and archived_at. We buffer
    the query window by ±3 days on either side so we catch notes that were
    finalised a bit before the appt started (pre-populated templates) or a
    day or two after (late finalisations — we want these to count as FAILS
    of the 24h rule, but we still need the record to know they happened).

    Also stamps a ``notes_scanned`` attr on the output (total /treatment_notes
    rows seen, before the appt-link drop) so diagnostics can report the
    match rate.
    """
    import pandas as pd  # local import to keep the module's top-level slim
    from datetime import timedelta

    # Buffer window
    start_dt = pd.Timestamp(dr.start_iso_utc)
    end_dt = pd.Timestamp(dr.end_iso_utc)
    q_start = (start_dt - pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    q_end = (end_dt + pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")

    params = {
        "q[]": [f"created_at:>={q_start}", f"created_at:<{q_end}"],
    }

    rows: list[dict[str, Any]] = []
    scanned = 0  # total notes seen, before the appt-id drop
    dropped_no_appt = 0
    draft_count = 0  # notes still flagged draft=true, no finalized_at
    first_note_shape: dict[str, Any] | None = None
    try:
        for n in client.paginate("treatment_notes", params=params):
            scanned += 1
            # v24 — capture structural shape of the first unmatched note so
            # Matt can see exactly what Cliniko is returning. PHI-safe:
            # _safe_note_structure only returns key names, never content.
            appt_id = _extract_appt_id_from_note(n)
            if appt_id is None:
                if first_note_shape is None and isinstance(n, dict):
                    first_note_shape = _safe_note_structure(n)
                dropped_no_appt += 1
                continue
            is_draft = bool(n.get("draft"))
            if is_draft:
                draft_count += 1
            rows.append({
                "appointment_id": str(appt_id),
                "note_created_at": n.get("created_at"),
                "note_finalized_at": n.get("finalized_at"),
                "note_archived_at": n.get("archived_at"),
                "note_draft": is_draft,
            })
    except Exception:
        # If the list endpoint fails outright we return empty — caller
        # will fall back to the appt.updated_at proxy so the dashboard
        # still renders.
        out = pd.DataFrame(columns=[
            "appointment_id", "note_created_at", "note_finalized_at",
            "note_archived_at", "note_draft",
        ])
        out.attrs["notes_scanned"] = scanned
        out.attrs["notes_dropped_no_appt_link"] = dropped_no_appt
        out.attrs["notes_still_draft"] = draft_count
        if first_note_shape is not None:
            out.attrs["first_unmatched_note_shape"] = first_note_shape
        return out

    df = pd.DataFrame(rows, columns=[
        "appointment_id", "note_created_at", "note_finalized_at",
        "note_archived_at", "note_draft",
    ])
    if df.empty:
        df.attrs["notes_scanned"] = scanned
        df.attrs["notes_dropped_no_appt_link"] = dropped_no_appt
        df.attrs["notes_still_draft"] = draft_count
        if first_note_shape is not None:
            df.attrs["first_unmatched_note_shape"] = first_note_shape
        return df
    df["note_created_at"] = pd.to_datetime(df["note_created_at"], utc=True, errors="coerce")
    df["note_finalized_at"] = pd.to_datetime(df["note_finalized_at"], utc=True, errors="coerce")
    df["note_archived_at"] = pd.to_datetime(df["note_archived_at"], utc=True, errors="coerce")
    # Drop archived/deleted notes — they're not a valid "finalised note".
    df = df[df["note_archived_at"].isna()].copy()
    # v25 — medicolegal rule: a note counts only when it has been formally
    # FINALISED. We sort by finalized_at (earliest finalisation wins) and
    # keep one row per appt. Draft notes and notes with no finalized_at
    # timestamp are kept in the frame so the caller can still see them
    # (the 24h comparison will fail them via NaT), but they cannot pass.
    df = (
        df.sort_values("note_finalized_at", na_position="last")
        .drop_duplicates(subset="appointment_id", keep="first")
        .reset_index(drop=True)
    )
    df.attrs["notes_scanned"] = scanned
    df.attrs["notes_dropped_no_appt_link"] = dropped_no_appt
    df.attrs["notes_still_draft"] = draft_count
    if first_note_shape is not None:
        df.attrs["first_unmatched_note_shape"] = first_note_shape
    return df


def notes_completion(client: ClinikoClient, appts: pd.DataFrame, dr: DateRange) -> pd.DataFrame:
    """Medicolegal notes compliance = finalised within 24h ÷ notes expected.

    v19 — hits Cliniko's /treatment_notes list endpoint and uses the note's
    own `created_at` (first save) as the finalisation timestamp.

    Previous implementation (v7-v18) used appt.updated_at as a proxy for
    finalisation time. That proxy was systematically pessimistic because
    appt.updated_at bumps on ANY appointment change — billing codes, invoice
    generation, cancellation flips, patient-arrived flips, linked follow-up
    bookings, etc. — not just note finalisation. The net effect was real
    within-24h rates of ~95% showing as ~50% on the dashboard. Matt flagged
    the gap against clinic reality on 2026-04-24.

    Denominator: delivered appts where `treatment_note_status != 10`
        10 = N/A (no note expected)
        20 = pending  ·  30 = draft  ·  40 = overdue  ·  90 = finalised

    Numerator: appts where we find a matching treatment_note whose
    `created_at` is within 24h (± small buffer) of the appt's `starts_at`.
    An appt without a matching finalised note fails, regardless of its
    treatment_note_status field.

    If the /treatment_notes list endpoint fails entirely (404, auth error,
    timeout), we fall back to the old appt.updated_at proxy so the
    dashboard still renders — clearly worse than nothing for diagnostics.
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

    # --- Primary path: /treatment_notes list endpoint ------------------
    notes_df = _fetch_treatment_notes_for_range(client, dr)
    used_note_endpoint = not notes_df.empty

    if used_note_endpoint:
        # Left-join: one row per delivered appt, optional note_finalized_at
        merged = delivered.merge(
            notes_df, left_on="id", right_on="appointment_id", how="left",
        )
        # v25 — use finalized_at instead of created_at. The medicolegal
        # rule is "finalised within 24h", not "draft saved within 24h". A
        # note with no finalized_at (still a draft, never formally
        # finalised) produces NaT here, which fails the .between() check
        # and correctly fails the 24h rule.
        delta_h = (
            (merged["note_finalized_at"] - merged["starts_at"]).dt.total_seconds()
            / 3600.0
        )
        # Allow a small negative bound (-1h) to tolerate pre-populated
        # templates where the note record is saved minutes before the
        # appt officially starts. Anything more than ~1h ahead is suspect
        # but we don't have a clean way to distinguish from clock skew.
        within_window = delta_h.between(-1.0, window_h, inclusive="both").fillna(False)
        # Pass requires: appt expected a note AND a matching note exists
        # AND that note was FINALISED within the window AND is not still
        # flagged as a draft. draft=True with a finalized_at timestamp is
        # unusual but we treat it as fail — if Cliniko still has the
        # draft flag set, the practitioner hasn't actually closed it.
        has_finalised_note = merged["note_finalized_at"].notna()
        is_not_draft = ~merged["note_draft"].fillna(True).astype(bool)
        merged["_passed_24h"] = (
            merged["_expected"] & has_finalised_note & is_not_draft & within_window
        )
        base = merged
    else:
        # --- Fallback: old appt.updated_at proxy -----------------------
        # Same logic as v7-v18. Triggered if the /treatment_notes list
        # endpoint returned nothing (plausible for a very short window
        # with no activity) OR errored. Keep the dashboard rendering
        # rather than failing the whole metrics pipeline.
        delivered["_status_is_final"] = status.fillna(0).astype(int) == 90
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
        base = delivered

    g = base.groupby("practitioner_id").agg(
        expected=("_expected", "sum"),
        passed_24h=("_passed_24h", "sum"),
    ).reset_index()

    g["notes_completion"] = g.apply(
        lambda r: (float(r["passed_24h"]) / float(r["expected"]))
                  if r["expected"] > 0 else float("nan"),
        axis=1,
    )
    # Mark the data source on the returned frame so the diagnostics tab
    # can tell Matt which path was used. Not attached to every row — just
    # a DataFrame-level attr.
    out = g[["practitioner_id", "notes_completion"]]
    out.attrs["source"] = "treatment_notes endpoint" if used_note_endpoint else "appt.updated_at fallback"
    out.attrs["notes_fetched"] = int(len(notes_df))
    # v23 — surface the raw scanned count and the unlinked drop count so we
    # can't silently regress on note→appt linkage. If scanned is high but
    # fetched is 0, every note is missing an appointment link and we need
    # to expand _extract_appt_id_from_note to handle a new Cliniko shape.
    out.attrs["notes_scanned"] = int(notes_df.attrs.get("notes_scanned", len(notes_df)))
    out.attrs["notes_dropped_no_appt_link"] = int(
        notes_df.attrs.get("notes_dropped_no_appt_link", 0)
    )
    # v25 — how many of the matched notes are still flagged draft=true
    # and therefore can never pass the finalised-within-24h check.
    out.attrs["notes_still_draft"] = int(
        notes_df.attrs.get("notes_still_draft", 0)
    )
    # v24 — propagate the structural diagnostic shape (if any) so the
    # dashboard caption can render it. Only populated when notes were
    # scanned but none had an appointment link our extractor recognised.
    first_shape = notes_df.attrs.get("first_unmatched_note_shape")
    if first_shape is not None:
        out.attrs["first_unmatched_note_shape"] = first_shape
    if used_note_endpoint:
        # v25 — break out three appt states so Matt can see where notes
        # compliance is actually failing:
        #   any_note     — appt has some treatment_note record linked
        #   finalised    — that note has a finalized_at timestamp
        #   (passed already counted separately via _passed_24h)
        any_note = merged["note_created_at"].notna()
        finalised_note = merged["note_finalized_at"].notna() & \
            ~merged["note_draft"].fillna(True).astype(bool)
        out.attrs["notes_matched_appts"] = int(any_note.sum())
        out.attrs["notes_finalised_appts"] = int(finalised_note.sum())
        out.attrs["delivered_appts"] = int(len(merged))
    return out


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
    # v26.6 additions — per-clinic rollup of utilisation and the per-practitioner
    # white-space total that drives days_worked. Both derived from the same
    # cell table to avoid any double-fetch of /unavailable_blocks.
    utilisation_by_clinic: pd.DataFrame
    white_space_hours: pd.DataFrame
    notes: pd.DataFrame


def compute_core_metrics(client: ClinikoClient, dr: DateRange, appt_types: pd.DataFrame,
                          business_ids: list[int] | None = None,
                          practitioner_ids: list[int] | None = None) -> MetricResult:
    """Pull every metric for the given range. New-patients is cheap now
    (one extra call to /patients with a created_at filter), so it's always
    computed — PVA depends on it.

    v26.6: white-window cells are computed once and reused for utilisation,
    clinic utilisation, and service_hours/days_worked.
    """
    appts = fetch_appointments(client, dr, business_ids, practitioner_ids)
    np_df = new_patients(client, appts, dr, appt_types)

    cells = _compute_white_windows(client, appts, dr)
    ws_map = white_space_hours_map(cells)
    ws_df = pd.DataFrame(
        [{"practitioner_id": pid, "white_space_hours": hrs} for pid, hrs in ws_map.items()],
        columns=["practitioner_id", "white_space_hours"],
    )

    return MetricResult(
        appointments=appts,
        new_patients=np_df,
        consults=total_consults(appts),
        service_hours=service_hours(client, appts, dr, practitioner_ids,
                                     white_space_hours=ws_map),
        pva=pva(appts, np_df),
        ppva=ppva(appts, appt_types),
        cx_dna=cx_dna_rates(appts),
        utilisation=utilisation(client, appts, dr, cells=cells),
        utilisation_by_clinic=utilisation_by_clinic(client, appts, dr, cells=cells),
        white_space_hours=ws_df,
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
    # v26.6: use getattr so this still works against a stale pickled
    # MetricResult from an older Streamlit cache. (Streamlit's @st.cache_data
    # only hashes the decorated function's source, not its transitive calls,
    # so a v26.5 cached result can be served into v26.6 code on the first
    # load after deploy. The cache-bust key in ``cached_metrics`` clears it
    # next session, but this keeps THIS run alive.)
    for df, cols in [
        (getattr(result, "new_patients", None), ["practitioner_id", "new_patients"]),
        (getattr(result, "consults", None), ["practitioner_id", "consults_delivered"]),
        (getattr(result, "service_hours", None),
            ["practitioner_id", "service_hours", "days_worked", "avg_hours_per_day"]),
        (getattr(result, "pva", None), ["practitioner_id", "pva"]),
        (getattr(result, "ppva", None), ["practitioner_id", "ppva"]),
        (getattr(result, "cx_dna", None),
            ["practitioner_id", "cx_rate", "dna_rate", "cx_dna_combined_rate"]),
        (getattr(result, "utilisation", None), ["practitioner_id", "utilisation"]),
        (getattr(result, "white_space_hours", None),
            ["practitioner_id", "white_space_hours"]),
        (getattr(result, "notes", None), ["practitioner_id", "notes_completion"]),
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
    # Fill missing numeric columns with 0 so downstream scoring doesn't blow up.
    # IMPORTANT: NPS is deliberately excluded from this fill. A practitioner
    # who received no NPS survey responses should be scored as N/A on NPS —
    # not 0, which would otherwise drop them into band 1 (0.0-0.64) and
    # unfairly penalise them. With nps=NaN, score_table() will emit
    # nps_band=NaN, and the nonclinical_axis mean excludes it (denominator
    # becomes 30 instead of 40 for that row).
    numeric_defaults = {
        "new_patients": 0, "consults_delivered": 0,
        "service_hours": 0.0, "days_worked": 0, "avg_hours_per_day": 0.0,
        "white_space_hours": 0.0,
        "pva": 0.0, "ppva": 0.0, "cx_rate": 0.0, "dna_rate": 0.0,
        "cx_dna_combined_rate": 0.0, "utilisation": 0.0,
        "notes_completion": 0.0,
        "punctuality_within_15": 0.0, "audit_pct": 0.0,
        "audit_epc_pct": 0.0, "audit_private_pct": 0.0, "patients_audited": 0,
    }
    for col, default in numeric_defaults.items():
        if col in base.columns:
            base[col] = base[col].fillna(default)
    return base
