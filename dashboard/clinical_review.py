"""Clinical Review — long-duration / under-servicing patient flagging.

v26.9 — surfaces patients who need a clinical review:

  Over-servicing (still being seen but past funding-bucket threshold):
    * Private/EPC:        >20 appts OR >90 days since initial
    * VIC WorkCover:      >36 appts OR >252 days since initial (3× PMP @ 12wk)
    * NSW WorkCover:      >40 appts (5× AHTR @ 8 appts)
    * TAC / DVA / NDIS:   >30 appts each

  Under-servicing (initial only, no follow-up):
    * Patient's only delivered appt is an initial-style appt
    * No future booked appt
    * >14 days since the initial

"Active" is the gate for over-servicing — we only flag patients who are
still in care (delivered appt in last 14 days OR future booking). A
patient who got 25 appts six months ago and isn't coming back doesn't
need clinical review; we'd just be making noise.

The whole tab pulls 12 months of delivered appts in one paginated
fetch (~5–10k records, ~30–60s first time, cached 30 min) plus a
forward-looking fetch for future bookings. Patient names are fetched
per-flagged-patient as a final small batch to avoid pulling /patients
for the whole clinic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable

import pandas as pd
import pytz

from dashboard.cliniko import ClinikoClient
from dashboard.config import load_settings, timezone_name
from dashboard.date_ranges import DateRange


# -------------------------------------------------------------------
# Bucket classification — appt_type_name → funding bucket name
# -------------------------------------------------------------------
# Order matters: more specific patterns first (e.g. "vic workcover"
# before "private", because some clinics name a type "Private VIC WC"
# and we want WorkCover to win).
_BUCKET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("VIC WorkCover", re.compile(r"\bvic(?:torian)?\s*work\s*cover\b", re.IGNORECASE)),
    ("NSW WorkCover", re.compile(r"\bnsw\s*work\s*cover\b", re.IGNORECASE)),
    ("NDIS",          re.compile(r"\bNDIS\b", re.IGNORECASE)),
    ("TAC",           re.compile(r"\bTAC\b", re.IGNORECASE)),
    ("DVA",           re.compile(r"\bDVA\b", re.IGNORECASE)),
    # EPC and Private collapse into one bucket per Matt — same threshold.
    ("Private/EPC",   re.compile(r"\b(?:EPC|private)\b", re.IGNORECASE)),
]


# Default thresholds — overridable via settings.yml `clinical_review.buckets`.
# `max_appts` and `max_days` are EITHER triggers — flag if either exceeded.
# None = ignore that axis (e.g. NSW WorkCover is appt-count-only).
_DEFAULT_THRESHOLDS: dict[str, dict[str, int | None]] = {
    "Private/EPC":   {"max_appts": 20, "max_days": 90},
    "VIC WorkCover": {"max_appts": 36, "max_days": 252},
    "NSW WorkCover": {"max_appts": 40, "max_days": None},
    "TAC":           {"max_appts": 30, "max_days": None},
    "DVA":           {"max_appts": 30, "max_days": None},
    "NDIS":          {"max_appts": 30, "max_days": None},
}


_INITIAL_TYPE_PATTERN = re.compile(
    r"\b(initial|new\s*patient|new\s*client|assessment|first\s*visit)\b",
    re.IGNORECASE,
)


def classify_bucket(appt_type_name: str | None) -> str | None:
    """Return funding bucket for an appointment-type name, or None if it
    doesn't match a tracked funding category (private therapy outside
    EPC/insurance, group classes, hydrotherapy, etc — we don't flag
    those for over-servicing review)."""
    if not appt_type_name:
        return None
    for bucket, pattern in _BUCKET_PATTERNS:
        if pattern.search(appt_type_name):
            return bucket
    return None


def _is_initial_type(appt_type_name: str | None) -> bool:
    return bool(appt_type_name and _INITIAL_TYPE_PATTERN.search(appt_type_name))


def _settings_thresholds() -> dict[str, dict[str, int | None]]:
    """Read thresholds from settings.yml, falling back to defaults
    per-bucket so a partial config doesn't drop buckets entirely."""
    cr = load_settings().get("clinical_review") or {}
    out: dict[str, dict[str, int | None]] = {}
    user_buckets = cr.get("buckets") or {}
    for bucket, defaults in _DEFAULT_THRESHOLDS.items():
        user = user_buckets.get(bucket) or {}
        out[bucket] = {
            "max_appts": user.get("max_appts", defaults["max_appts"]),
            "max_days":  user.get("max_days",  defaults["max_days"]),
        }
    return out


def _settings_active_window_days() -> int:
    cr = load_settings().get("clinical_review") or {}
    return int(cr.get("active_window_days", 14))


def _settings_under_servicing_min_days() -> int:
    cr = load_settings().get("clinical_review") or {}
    return int(cr.get("under_servicing_min_days", 14))


# -------------------------------------------------------------------
# Cliniko fetches
# -------------------------------------------------------------------
def _fetch_appts_in_window(client: ClinikoClient, dr: DateRange,
                            ) -> pd.DataFrame:
    """Wrapper around metrics.fetch_appointments — kept here so this
    module's API surface is self-describing. Imports inline to avoid a
    circular dependency with metrics.py."""
    from dashboard.metrics import fetch_appointments
    return fetch_appointments(client, dr)


def _fetch_patient_names(client: ClinikoClient,
                          patient_ids: Iterable[str]) -> dict[str, str]:
    """Best-effort patient-name lookup. One Cliniko call per patient,
    so callers should pass only the flagged subset (typically <100)."""
    names: dict[str, str] = {}
    for pid in patient_ids:
        if not pid:
            continue
        try:
            p = client.get(f"patients/{pid}")
        except Exception:
            continue
        if not isinstance(p, dict):
            continue
        first = (p.get("first_name") or "").strip()
        last = (p.get("last_name") or "").strip()
        full = " ".join(s for s in (first, last) if s)
        names[str(pid)] = full or f"Patient {pid}"
    return names


def _build_patient_link(patient_id: str, shard: str | None = None) -> str:
    """Construct a Cliniko deep-link for the patient's profile.

    The clinic's shard is in the base URL we already use for API calls;
    we surface a clickable link in the table so reviewers can jump
    straight to the chart. Shard defaults to "au1" if unknown.
    """
    shard = shard or "au1"
    return f"https://enhance-physiotherapy-albury-wodonga.{shard}.cliniko.com/#/patients/{patient_id}"


# -------------------------------------------------------------------
# Core computation
# -------------------------------------------------------------------
@dataclass
class _Episode:
    patient_id: str
    bucket: str
    initial_date: date
    last_appt_date: date
    appts_count: int
    has_future_appt: bool
    practitioner_id: str
    initial_appt_type: str
    last_appt_type: str


def _is_delivered(row: dict[str, Any] | pd.Series) -> bool:
    """A delivered appt = not cancelled, not archived, patient arrived
    (or arrival flag is None — older Cliniko records omit it)."""
    if row.get("archived_at"):
        return False
    if row.get("cancelled_at"):
        return False
    if row.get("did_not_arrive"):
        return False
    return True


def compute_clinical_review(client: ClinikoClient,
                              today: date | None = None,
                              lookback_months: int = 12,
                              future_months: int = 6,
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (over_servicing_df, under_servicing_df).

    Pulls 12 months back of delivered appts plus 6 months forward of
    booked appts. Groups by patient, classifies the most recent funded
    appt's bucket, applies the bucket's thresholds, and flags actives
    that exceed them.
    """
    today = today or date.today()

    # --- Pull data --------------------------------------------------
    # DateRange wants timezone-aware datetimes (the API helpers convert
    # them to UTC ISO). Anchor the windows at start-of-day local time.
    tz = pytz.timezone(timezone_name())
    today_start_local = tz.localize(datetime(today.year, today.month, today.day))
    tomorrow_start_local = today_start_local + timedelta(days=1)

    look_start = today_start_local - timedelta(days=lookback_months * 31)
    look_end = tomorrow_start_local  # exclusive upper bound = end of today
    fwd_start = tomorrow_start_local
    fwd_end = tomorrow_start_local + timedelta(days=future_months * 31)

    look_dr = DateRange(look_start, look_end)
    fwd_dr = DateRange(fwd_start, fwd_end)

    delivered = _fetch_appts_in_window(client, look_dr)
    future = _fetch_appts_in_window(client, fwd_dr)

    # Empty? Bail out with empty frames; UI will show a friendly message.
    if delivered.empty:
        empty = pd.DataFrame()
        return empty, empty

    # Drop non-delivered (cancelled/DNA/archived).
    delivered = delivered[delivered.apply(_is_delivered, axis=1)].copy()
    if delivered.empty:
        empty = pd.DataFrame()
        return empty, empty
    delivered["starts_at"] = pd.to_datetime(delivered["starts_at"], utc=True,
                                              errors="coerce")
    delivered = delivered.dropna(subset=["starts_at", "patient_id"])

    # Active future bookings = patient ids with any appt > today, not cancelled.
    if not future.empty:
        future["starts_at"] = pd.to_datetime(future["starts_at"], utc=True,
                                                errors="coerce")
        future = future.dropna(subset=["starts_at", "patient_id"])
        future_active = future[~future.get("cancelled_at").fillna("").astype(bool)
                                & ~future.get("archived_at").fillna("").astype(bool)]
        future_patient_ids: set[str] = set(
            future_active["patient_id"].astype(str).unique()
        )
    else:
        future_patient_ids = set()

    # Map appt_type_id → name (for bucket classification) using existing helper
    from dashboard.reference_data import load_appointment_types
    appt_types = load_appointment_types(client)
    type_to_name: dict[str, str] = {}
    if not appt_types.empty:
        for _, r in appt_types.iterrows():
            tid = str(r.get("id") or "")
            name = r.get("name") or ""
            if tid:
                type_to_name[tid] = str(name)

    delivered["appt_type_name"] = (delivered["appointment_type_id"]
                                     .astype(str).map(type_to_name).fillna(""))
    delivered["bucket"] = delivered["appt_type_name"].apply(classify_bucket)

    thresholds = _settings_thresholds()
    active_window = _settings_active_window_days()
    under_min_days = _settings_under_servicing_min_days()
    # appt starts_at are tz-aware UTC after the to_datetime call above,
    # so today_ts must be tz-aware UTC too for safe arithmetic.
    today_ts = pd.Timestamp(today).tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59)

    over_rows: list[dict[str, Any]] = []
    under_rows: list[dict[str, Any]] = []

    for pid, group in delivered.groupby("patient_id"):
        pid_str = str(pid)
        group = group.sort_values("starts_at")

        # ---- Under-servicing pass first (cheap) ----
        if len(group) == 1:
            only = group.iloc[0]
            if pid_str not in future_patient_ids:
                days_since = (today_ts - only["starts_at"]).days
                if days_since >= under_min_days:
                    under_rows.append({
                        "patient_id": pid_str,
                        "initial_date": only["starts_at"].date(),
                        "days_since_initial": int(days_since),
                        "initial_appt_type": only["appt_type_name"] or "—",
                        "practitioner_id": str(only.get("practitioner_id") or ""),
                        "is_initial_type": _is_initial_type(only["appt_type_name"]),
                    })
            # Single-appt patient — they can't be over-servicing too,
            # so move on.
            continue

        # ---- Over-servicing pass ----
        funded = group[group["bucket"].notna()]
        if funded.empty:
            continue  # not in any tracked funding category

        latest = funded.iloc[-1]
        latest_bucket = str(latest["bucket"])
        bucket_appts = funded[funded["bucket"] == latest_bucket].sort_values(
            "starts_at"
        )

        last_appt_ts = bucket_appts.iloc[-1]["starts_at"]
        is_recently_active = (today_ts - last_appt_ts).days <= active_window
        has_future = pid_str in future_patient_ids
        if not (is_recently_active or has_future):
            continue  # finished / churned, no review needed

        initial_ts = bucket_appts.iloc[0]["starts_at"]
        initial_date = initial_ts.date()
        last_appt_date = last_appt_ts.date()
        days_since_initial = (today_ts - initial_ts).days
        appts_count = len(bucket_appts)

        rules = thresholds.get(latest_bucket, {})
        max_appts = rules.get("max_appts")
        max_days = rules.get("max_days")

        flag_reasons: list[str] = []
        if max_appts is not None and appts_count > max_appts:
            flag_reasons.append(f">{max_appts} appts ({appts_count})")
        if max_days is not None and days_since_initial > max_days:
            flag_reasons.append(f">{max_days} days since initial ({days_since_initial})")

        if not flag_reasons:
            continue

        over_rows.append({
            "patient_id": pid_str,
            "bucket": latest_bucket,
            "initial_date": initial_date,
            "last_appt_date": last_appt_date,
            "days_since_initial": int(days_since_initial),
            "appts_count": int(appts_count),
            "has_future_appt": has_future,
            "practitioner_id": str(bucket_appts.iloc[-1].get("practitioner_id") or ""),
            "flag_reason": "; ".join(flag_reasons),
        })

    over_df = pd.DataFrame(over_rows)
    under_df = pd.DataFrame(under_rows)

    # Resolve patient names for the flagged subset (one API hit each).
    flagged_pids = (set(over_df["patient_id"]) if not over_df.empty else set()) | \
                   (set(under_df["patient_id"]) if not under_df.empty else set())
    name_map = _fetch_patient_names(client, flagged_pids)

    if not over_df.empty:
        over_df["patient"] = over_df["patient_id"].map(name_map).fillna(
            over_df["patient_id"].apply(lambda p: f"Patient {p}")
        )
        over_df["cliniko_url"] = over_df["patient_id"].apply(_build_patient_link)
    if not under_df.empty:
        under_df["patient"] = under_df["patient_id"].map(name_map).fillna(
            under_df["patient_id"].apply(lambda p: f"Patient {p}")
        )
        under_df["cliniko_url"] = under_df["patient_id"].apply(_build_patient_link)

    return over_df, under_df


# -------------------------------------------------------------------
# Practitioner-name resolution helper (shared with the UI)
# -------------------------------------------------------------------
def attach_practitioner_names(df: pd.DataFrame,
                                practitioners: pd.DataFrame | None) -> pd.DataFrame:
    """Add a 'practitioner' column to df by joining on practitioner_id.
    Mutates a copy; returns the new frame. Safe on empty input."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if practitioners is None or practitioners.empty:
        out["practitioner"] = out.get("practitioner_id", "")
        return out
    name_map: dict[str, str] = {}
    for _, r in practitioners.iterrows():
        pid = str(r.get("id"))
        label = r.get("label") or r.get("display_name") or r.get("name") or pid
        name_map[pid] = str(label)
    out["practitioner"] = out["practitioner_id"].astype(str).map(name_map).fillna(
        out["practitioner_id"].astype(str)
    )
    return out
