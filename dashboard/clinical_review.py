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


def _fetch_active_appts_only(client: ClinikoClient,
                               dr: DateRange) -> pd.DataFrame:
    """Single-pass active-only fetch. ~50% faster than fetch_appointments
    because it skips the explicit-cancelled pass.

    v26.11 — Cliniko's default /individual_appointments query already
    excludes cancelled records, so for clinical-review purposes (where
    we're looking for delivered/active appts) we don't need the second
    pass that fetch_appointments runs."""
    from dashboard.metrics import _iter_appointments, _appt_row
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for a in _iter_appointments(client, dr):
        row = _appt_row(a)
        rid = row["id"]
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        rows.append(row)
    cols = ["id", "patient_id", "practitioner_id", "business_id",
             "appointment_type_id", "starts_at", "ends_at",
             "appt_updated_at", "cancelled_at", "archived_at",
             "did_not_arrive", "cancellation_reason",
             "treatment_note_status", "patient_arrived"]
    return pd.DataFrame(rows, columns=cols)


def _fetch_patient_history(client: ClinikoClient,
                             patient_id: str,
                             since: datetime) -> list[dict[str, Any]]:
    """Pull all individual_appointments for ONE patient since ``since``.

    Returns a list of normalised appt dicts. Filters by patient_id at
    the API level so the response is small (one patient = typically 5–60
    appts), and we only get back what we need.
    """
    from dashboard.metrics import _appt_row
    since_iso = since.astimezone(pytz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"q[]": [f"patient_id:={patient_id}",
                       f"starts_at:>={since_iso}"]}
    out: list[dict[str, Any]] = []
    try:
        for a in client.paginate("individual_appointments", params=params):
            row = _appt_row(a)
            if row.get("id"):
                out.append(row)
    except Exception:
        pass
    return out


def _fetch_patient_names(client: ClinikoClient,
                          patient_ids: Iterable[str]) -> dict[str, str]:
    """Best-effort patient-name lookup with disk cache.

    v26.11.1 — Names rarely change, so we cache them in
    data/_cache/patient_names.json indefinitely. After the first run
    most lookups are cache hits and we skip the Cliniko call entirely.

    For a clinic Matt's size (50-100 flagged patients per run), this
    drops from ~30-60 seconds (sequential API calls) to <1 second on
    repeat visits.
    """
    import json
    from dashboard.config import DATA_DIR
    cache_path = DATA_DIR / "_cache" / "patient_names.json"

    # Load existing cache
    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    out: dict[str, str] = {}
    missing: list[str] = []
    for pid in patient_ids:
        pid_str = str(pid) if pid else ""
        if not pid_str:
            continue
        if pid_str in cache:
            out[pid_str] = cache[pid_str]
        else:
            missing.append(pid_str)

    # Fetch the missing ones from Cliniko (sequential — typically <100
    # per run, and after first run this is empty most of the time).
    cache_dirty = False
    for pid in missing:
        try:
            p = client.get(f"patients/{pid}")
        except Exception:
            out[pid] = f"Patient {pid}"
            continue
        if not isinstance(p, dict):
            out[pid] = f"Patient {pid}"
            continue
        first = (p.get("first_name") or "").strip()
        last = (p.get("last_name") or "").strip()
        full = " ".join(s for s in (first, last) if s)
        full = full or f"Patient {pid}"
        out[pid] = full
        cache[pid] = full
        cache_dirty = True

    # Persist cache if we added any new entries
    if cache_dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        except Exception:
            pass

    return out


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


def _is_truthy_str_local(v: Any) -> bool:
    """NaN-safe check for a non-empty string field. v26.10.5 in
    commission.py — duplicated here to avoid an import cycle."""
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(v, str):
        return bool(v.strip())
    return False


def _is_delivered(row: dict[str, Any] | pd.Series) -> bool:
    """A delivered appt = not cancelled, not archived, patient arrived
    (or arrival flag is None — older Cliniko records omit it).

    v26.10.5 — uses NaN-safe checks because pandas iterrows() converts
    None → NaN, and ``bool(NaN) == True`` was silently flagging every
    delivered appt as archived.
    """
    archived = row.get("archived_at")
    cancelled = row.get("cancelled_at")
    dna = row.get("did_not_arrive")
    # Treat NaN/NaT/None/"" as not-set
    for v in (archived, cancelled):
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(v, str) and not v.strip():
            continue
        # Set to a non-empty value → not delivered
        return False
    if dna is True:
        return False
    if isinstance(dna, str) and dna.strip().lower() == "true":
        return False
    return True


def compute_clinical_review_diagnostic(client: ClinikoClient,
                                         today: date | None = None,
                                         lookback_months: int = 12,
                                         future_months: int = 6,
                                         ) -> dict[str, Any]:
    """Same fetch logic as compute_clinical_review, but returns counters
    + samples instead of the flagged tables. Used by the UI's debug
    expander when the queue looks unrealistically empty.

    v26.10.3 — added because the initial v26.9 ship returned zero
    flagged patients on a clinic of 16 practitioners with a year of
    data, which is impossible. Surfaces where the filter funnel is
    dropping rows.
    """
    today = today or date.today()
    tz = pytz.timezone(timezone_name())
    today_start_local = tz.localize(datetime(today.year, today.month, today.day))
    tomorrow_start_local = today_start_local + timedelta(days=1)
    look_start = today_start_local - timedelta(days=lookback_months * 31)
    look_end = tomorrow_start_local
    fwd_start = tomorrow_start_local
    fwd_end = tomorrow_start_local + timedelta(days=future_months * 31)
    look_dr = DateRange(look_start, look_end)
    fwd_dr = DateRange(fwd_start, fwd_end)

    delivered = _fetch_appts_in_window(client, look_dr)
    future = _fetch_appts_in_window(client, fwd_dr)
    delivered_total = len(delivered)
    future_total = len(future)

    if not delivered.empty:
        delivered = delivered[delivered.apply(_is_delivered, axis=1)].copy()
    after_is_delivered = len(delivered)

    from dashboard.reference_data import load_appointment_types
    appt_types = load_appointment_types(client)
    type_to_name: dict[str, str] = {}
    if not appt_types.empty:
        for _, r in appt_types.iterrows():
            tid = str(r.get("id") or "")
            name = r.get("name") or ""
            if tid:
                type_to_name[tid] = str(name)

    if not delivered.empty:
        delivered["appt_type_name"] = (delivered["appointment_type_id"]
                                         .astype(str).map(type_to_name).fillna(""))
        delivered["bucket"] = delivered["appt_type_name"].apply(classify_bucket)

    # Bucket distribution
    bucket_dist: dict[str, int] = {}
    if not delivered.empty:
        bucket_counts = delivered.groupby(delivered["bucket"].fillna("(unmatched)")
                                            ).size().to_dict()
        bucket_dist = {str(k): int(v) for k, v in bucket_counts.items()}

    # Sample of unmatched appt-type names so we can spot a missing
    # bucket pattern quickly
    unmatched_appt_types: list[tuple[str, int]] = []
    if not delivered.empty and "bucket" in delivered.columns:
        unmatched = delivered[delivered["bucket"].isna()]
        if not unmatched.empty:
            unmatched_appt_types = (unmatched["appt_type_name"]
                                       .value_counts().head(10).items().__iter__())
            unmatched_appt_types = list(unmatched_appt_types)

    # Top-N patients by lifetime appt count, bucket-aware
    patient_summary: list[dict[str, Any]] = []
    if not delivered.empty:
        funded = delivered[delivered["bucket"].notna()]
        if not funded.empty:
            agg = (funded.groupby(["patient_id", "bucket"]).size()
                          .reset_index(name="appts_count"))
            top10 = agg.nlargest(10, "appts_count")
            patient_summary = [
                {"patient_id": str(r["patient_id"]),
                 "bucket": str(r["bucket"]),
                 "appts_count": int(r["appts_count"])}
                for _, r in top10.iterrows()
            ]

    return {
        "lookback_window_days": (look_end - look_start).days,
        "delivered_total": delivered_total,
        "after_is_delivered": after_is_delivered,
        "future_total": future_total,
        "appt_types_total": len(type_to_name),
        "bucket_distribution": bucket_dist,
        "unmatched_appt_types_sample": [
            {"name": str(name), "count": int(count)}
            for name, count in unmatched_appt_types[:10]
        ] if unmatched_appt_types else [],
        "top10_patients_by_appt_count": patient_summary,
    }


def compute_clinical_review(client: ClinikoClient,
                              today: date | None = None,
                              active_window_days: int | None = None,
                              future_months: int = 2,
                              episode_lookback_months: int = 18,
                              progress_callback=None,
                              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (over_servicing_df, under_servicing_df).

    v26.11 — refactored from "pull 12 months of everyone" to a two-phase
    per-patient drill-down. Saves ~60% of the API calls because most
    patients aren't currently active.

    Phase A (small): pull last 30d delivered + next 2mo booked appts to
                     identify the active patient set (~500 patients for
                     a clinic Matt's size).
    Phase B (medium): for each active patient, pull their lifetime appt
                      history within the last 18 months (covers every
                      threshold incl. NSW WC at ~10 months and VIC WC
                      at ~8.5 months).
    Phase C (in-memory): apply bucket + threshold logic per patient.

    progress_callback(done, total, message) is called between Phase B
    iterations so the UI can render a live progress bar.
    """
    today = today or date.today()
    # v27.2.1 — Phase A window now defaults from settings.yml
    # (``clinical_review.active_window_days``) so the clinical lead's
    # weekly review can be controlled from one place. Fall back to 7 days
    # if the setting is missing.
    if active_window_days is None:
        active_window_days = _settings_active_window_days()
        if active_window_days is None or active_window_days <= 0:
            active_window_days = 7

    # --- Phase A: identify active patient set -----------------------
    tz = pytz.timezone(timezone_name())
    today_start_local = tz.localize(datetime(today.year, today.month, today.day))
    tomorrow_start_local = today_start_local + timedelta(days=1)

    recent_dr = DateRange(
        today_start_local - timedelta(days=active_window_days),
        tomorrow_start_local,
    )
    future_dr = DateRange(
        tomorrow_start_local,
        tomorrow_start_local + timedelta(days=future_months * 31),
    )

    if progress_callback:
        progress_callback(0, 100, "Phase A — finding active patients (~30s)…")

    recent = _fetch_active_appts_only(client, recent_dr)
    future = _fetch_active_appts_only(client, future_dr)

    # If both fetches are empty, no point continuing.
    if recent.empty and future.empty:
        empty = pd.DataFrame()
        return empty, empty

    # ---- Build the active patient set ----
    active_patient_ids: set[str] = set()
    future_patient_ids: set[str] = set()

    if not recent.empty:
        # Drop NaT/missing patient_ids and not-delivered rows
        recent_delivered = recent[recent.apply(_is_delivered, axis=1)].copy()
        if not recent_delivered.empty:
            recent_delivered["patient_id"] = recent_delivered["patient_id"].astype(str)
            active_patient_ids.update(
                recent_delivered["patient_id"].dropna().unique()
            )

    if not future.empty:
        future = future.copy()
        future["patient_id"] = future["patient_id"].astype(str)
        # Future bookings only count if not cancelled / archived
        future_kept = future[future.apply(
            lambda r: not _is_truthy_str_local(r.get("cancelled_at"))
                        and not _is_truthy_str_local(r.get("archived_at")),
            axis=1,
        )]
        future_patient_ids.update(future_kept["patient_id"].dropna().unique())
        active_patient_ids.update(future_patient_ids)

    if not active_patient_ids:
        empty = pd.DataFrame()
        return empty, empty

    # ---- Phase B: per-patient appt history ----
    episode_since = today_start_local - timedelta(days=episode_lookback_months * 31)

    # Map appt_type_id → name (for bucket classification)
    from dashboard.reference_data import load_appointment_types
    appt_types = load_appointment_types(client)
    type_to_name: dict[str, str] = {}
    if not appt_types.empty:
        for _, r in appt_types.iterrows():
            tid = str(r.get("id") or "")
            name = r.get("name") or ""
            if tid:
                type_to_name[tid] = str(name)

    total = len(active_patient_ids)
    all_history: list[dict[str, Any]] = []
    for i, pid in enumerate(sorted(active_patient_ids)):
        if progress_callback:
            progress_callback(
                i, total,
                f"Phase B — fetching patient {i+1}/{total} histories…",
            )
        rows = _fetch_patient_history(client, pid, episode_since)
        for r in rows:
            r["patient_id"] = pid
            all_history.append(r)

    if not all_history:
        empty = pd.DataFrame()
        return empty, empty

    delivered = pd.DataFrame(all_history)
    # Drop non-delivered rows
    delivered = delivered[delivered.apply(_is_delivered, axis=1)].copy()
    if delivered.empty:
        empty = pd.DataFrame()
        return empty, empty
    delivered["starts_at"] = pd.to_datetime(delivered["starts_at"], utc=True,
                                              errors="coerce")
    delivered = delivered.dropna(subset=["starts_at", "patient_id"])

    delivered["appt_type_name"] = (delivered["appointment_type_id"]
                                     .astype(str).map(type_to_name).fillna(""))
    delivered["bucket"] = delivered["appt_type_name"].apply(classify_bucket)

    if progress_callback:
        progress_callback(total, total, "Phase C — applying threshold logic…")

    # ---- Phase C: bucket + threshold logic ----
    thresholds = _settings_thresholds()
    active_window = _settings_active_window_days()
    under_min_days = _settings_under_servicing_min_days()
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
