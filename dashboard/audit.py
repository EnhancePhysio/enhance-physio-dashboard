"""Audit engine — 5 consistent checks per new EPC / Private patient.

Checks:
  1. RAP attachment
  2. Wibbi exercises attachment
  3. Correspondence to referrer (N/A for social/self-referral)
  4. Upcoming appointment OR recall
  5. Notes match appointments attended
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from dashboard.cliniko import ClinikoClient
from dashboard.config import load_settings, load_rap_exempt
from dashboard.date_ranges import DateRange
from dashboard.reference_data import extract_linked_id


@dataclass
class CheckResult:
    name: str
    passed: bool | None   # True = pass, False = fail, None = N/A
    reason: str = ""


@dataclass
class PatientAudit:
    patient_id: str
    patient_name: str
    practitioner_id: str
    business_id: str | None
    cohort: str   # "EPC" or "Private"
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passes(self) -> int:
        return sum(1 for c in self.checks if c.passed is True)

    @property
    def applicable(self) -> int:
        return sum(1 for c in self.checks if c.passed is not None)

    @property
    def score(self) -> float:
        return self.passes / self.applicable if self.applicable else 0.0

    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if c.passed is False]


# -------------------------------------------------------------------
# Pool selection
# -------------------------------------------------------------------
def _classify_appointment_type(name: str) -> tuple[bool, bool]:
    """Return (is_initial_epc, is_initial_private)."""
    if not name:
        return (False, False)
    n = name.lower()
    if "initial" not in n:
        return (False, False)
    if "epc" in n:
        return (True, False)
    if "private" in n:
        return (False, True)
    return (False, False)


def select_audit_pool(appointments: pd.DataFrame,
                      appt_types: pd.DataFrame) -> pd.DataFrame:
    """Pick every patient whose first-ever appointment with a practitioner
    (inside the date range) was an Initial EPC or Initial Private type.

    Returns a DataFrame: patient_id, practitioner_id, cohort (EPC|Private).
    """
    if appointments.empty or appt_types.empty:
        return pd.DataFrame(columns=["patient_id", "practitioner_id", "business_id", "cohort"])
    type_map = dict(zip(appt_types["id"], appt_types["name"].fillna("")))
    df = appointments.copy()
    df = df[df["cancelled_at"].isna() & ~df["did_not_arrive"].fillna(False).astype(bool)]
    df["type_name"] = df["appointment_type_id"].map(type_map).fillna("")
    df[["is_epc_init", "is_priv_init"]] = df["type_name"].apply(
        lambda n: pd.Series(_classify_appointment_type(n))
    )
    df = df[df["is_epc_init"] | df["is_priv_init"]]
    # First appointment per (practitioner, patient) in the pool
    df = df.sort_values("starts_at").drop_duplicates(subset=["practitioner_id", "patient_id"])
    df["cohort"] = df.apply(lambda r: "EPC" if r["is_epc_init"] else "Private", axis=1)
    return df[["patient_id", "practitioner_id", "business_id", "cohort"]].reset_index(drop=True)


# -------------------------------------------------------------------
# Helpers for fetching per-patient data
# -------------------------------------------------------------------
# Module-level diagnostic counters. Reset per run by run_audit().
# Structure: {"attachments": {"ok": 12, "empty": 3, "error": 1, "last_err": "..."}, ...}
FETCH_STATS: dict[str, dict] = {}

# v21 — endpoint-unavailable short-circuits. When an endpoint 404s on the
# first patient (confirmed against Matt's Cliniko shard 2026-04-24 for
# `letters` and `patient_recalls`), there's no point hammering it 371 more
# times. We flip these flags on first 404 and treat the feature as
# "unavailable on this Cliniko account" for the rest of the run, which
# surfaces in checks 3/4 as N/A instead of False.
# v26.4 — added `communications` (now the primary source for Check 3 since
# Cliniko support confirmed /letters doesn't exist on Enhance Physio's plan
# but /communications does, with q[]=patient_id:= filter whitelisted).
# Reset per run by run_audit().
ENDPOINT_UNAVAILABLE: dict[str, bool] = {
    "letters": False, "recalls": False, "communications": False,
}


def _reset_fetch_stats() -> None:
    FETCH_STATS.clear()
    for key in ("patient", "attachments", "letters", "communications",
                "notes", "appointments", "recalls"):
        FETCH_STATS[key] = {"ok": 0, "empty": 0, "error": 0, "last_err": ""}
    for key in ENDPOINT_UNAVAILABLE:
        ENDPOINT_UNAVAILABLE[key] = False


def _is_404(err: Exception) -> bool:
    """True iff the ClinikoError message indicates a 404 response."""
    msg = str(err)
    return "404" in msg or "Not Found" in msg


def _record(key: str, result: Any, err: Exception | None = None) -> None:
    s = FETCH_STATS.setdefault(key, {"ok": 0, "empty": 0, "error": 0, "last_err": ""})
    if err is not None:
        s["error"] += 1
        s["last_err"] = f"{type(err).__name__}: {err}"[:200]
    elif result is None:
        s["error"] += 1
    elif isinstance(result, list) and len(result) == 0:
        s["empty"] += 1
    else:
        s["ok"] += 1


def _get_patient(client: ClinikoClient, patient_id: str) -> dict[str, Any] | None:
    """Fetch a single patient. Returns None (not raise) for archived/deleted
    patients so run_audit can record a clear reason instead of a generic
    'fetch error'."""
    try:
        p = client.get(f"patients/{patient_id}")
        _record("patient", p)
        return p
    except Exception as e:
        _record("patient", None, e)
        return None


def _get_attachments(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    try:
        out = list(client.paginate(f"patients/{patient_id}/patient_attachments"))
        _record("attachments", out)
        return out
    except Exception as e:
        _record("attachments", None, e)
        return []


def _get_communications(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    """Fetch every communication logged for a patient.

    v26.4 — Replaces _get_letters. Cliniko support confirmed that /letters
    doesn't exist on Enhance Physio's plan but /communications does, and
    `patient_id` is one of the officially-whitelisted q[] filter fields
    per the OpenAPI spec (alongside archived_at, category_code, created_at,
    id, letter_id, type_code, updated_at).
    """
    if ENDPOINT_UNAVAILABLE["communications"]:
        return []
    try:
        out = list(client.paginate(
            "communications",
            params={"q[]": [f"patient_id:={patient_id}"]},
        ))
        _record("communications", out)
        return out
    except Exception as e:
        _record("communications", None, e)
        if _is_404(e):
            ENDPOINT_UNAVAILABLE["communications"] = True
        return []


def _get_patient_treatment_notes(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    # NOTE: treatment_notes DOES support the nested /patients/{id}/treatment_notes
    # path (confirmed 278/278 ok against au1). Don't "fix" this to match the
    # letters pattern — it would actually break.
    try:
        out = list(client.paginate(f"patients/{patient_id}/treatment_notes"))
        _record("notes", out)
        return out
    except Exception as e:
        _record("notes", None, e)
        return []


def _get_patient_appointments(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    # v20 — /patients/{id}/individual_appointments 404s. Use top-level
    # collection with a patient_id filter.
    try:
        out = list(client.paginate(
            "individual_appointments",
            params={"q[]": [f"patient_id:={patient_id}"]},
        ))
        _record("appointments", out)
        return out
    except Exception as e:
        _record("appointments", None, e)
        return []


def _get_recalls(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    # v20 — /patients/{id}/patient_recalls 404s; use top-level with filter.
    # v21 — on some accounts the top-level /patient_recalls also 404s
    # (Enhance Physio 2026-04-24). Short-circuit on first 404 and let the
    # check 4 logic treat "no future appt + endpoint unavailable" as N/A
    # rather than an unverified Fail.
    if ENDPOINT_UNAVAILABLE["recalls"]:
        return []
    try:
        out = list(client.paginate(
            "patient_recalls",
            params={"q[]": [f"patient_id:={patient_id}"]},
        ))
        _record("recalls", out)
        return out
    except Exception as e:
        _record("recalls", None, e)
        if _is_404(e):
            ENDPOINT_UNAVAILABLE["recalls"] = True
        return []


def _coerce_ref_field(v: Any) -> str:
    """Normalise a patient referrer-ish field into a lowercase-able string.

    Cliniko returns these fields inconsistently — sometimes a plain string
    ("Dr Jenkins"), sometimes a reference object
    (``{"id": "123", "type": "Contact", "links": {...}}``), sometimes None.
    We extract a name-like payload from objects and coerce everything else
    to str so downstream lower()/in-checks don't blow up.
    """
    if v is None:
        return ""
    if isinstance(v, dict):
        # Prefer human-readable name fields; fall back to id/type so we at
        # least have *something* for referrer-matching to key on.
        for k in ("name", "full_name", "display_name", "description"):
            nv = v.get(k)
            if nv:
                return str(nv)
        # Nothing name-like; swallow the object silently rather than
        # leaking {"id":"...","type":"Contact"} into the blob.
        return ""
    return str(v)


def _has_clinical_fallback(notes: list[dict[str, Any]], keywords: list[str]) -> tuple[bool, str]:
    kws = [k.lower() for k in keywords]
    for n in notes:
        blob = " ".join(str(n.get(f, "")) for f in ("content", "subjective", "objective",
                                                     "assessment", "plan", "notes",
                                                     "body", "text"))
        blob = blob.lower()
        for kw in kws:
            if kw in blob:
                return True, kw
    return False, ""


def _practitioner_name_from_id(practitioners: pd.DataFrame, pid: int) -> str:
    row = practitioners[practitioners["id"] == pid]
    if row.empty:
        return ""
    return f"{row.iloc[0].get('first_name', '')} {row.iloc[0].get('last_name', '')}".strip()


def _is_rap_exempt(practitioners: pd.DataFrame, pid: int) -> bool:
    exempt = load_rap_exempt()
    if not exempt:
        return False
    name = _practitioner_name_from_id(practitioners, pid)
    name_lower = name.lower()
    for entry in exempt:
        cid = entry.get("cliniko_id")
        if cid and str(cid) == str(pid):
            return True
        ename = (entry.get("name") or "").lower()
        if ename and ename in name_lower:
            return True
    return False


def _first_word_lower(s: str | None) -> str:
    if not s:
        return ""
    return s.strip().split()[0].lower() if s.strip() else ""


def _is_contact_doctor_referral(patient: dict[str, Any]) -> bool:
    """True iff the patient's referral source is a Cliniko Contact with a name.

    Cliniko represents Contact-type referrals as a reference object:
        {"id": "123", "type": "Contact", "links": {...}}
    That's the "referred by a specific doctor" case Matt cares about. We
    also treat a non-empty `referring_doctor` string as a doctor referral
    (older patient records predate the structured Contact link).
    """
    for field_name in ("referral_source", "referrer"):
        v = patient.get(field_name)
        if isinstance(v, dict) and str(v.get("type", "")).lower() == "contact":
            return True
    rd = patient.get("referring_doctor")
    if isinstance(rd, str) and rd.strip():
        return True
    return False


def _letter_required(patient: dict[str, Any], cohort: str,
                      na_ref_values: list[str]) -> tuple[bool, str]:
    """Matt's v26.4 rule for Check 3:

    * EPC cohort → letter always required (no generic-referrer exemption).
    * Private cohort → letter required IFF the referral source is a
      Cliniko Contact with a doctor name attached.
    * Private cohort with a generic/social referrer (Google, Facebook,
      sporting clubs, self, etc.) → N/A, no letter expected.
    * Private cohort with a blank / unrecognised referrer → N/A
      (conservative — don't flag a clinician for a data-entry hole).

    Returns (required, reason).
    """
    if cohort == "EPC":
        return True, "EPC cohort — letter required regardless of referrer"

    # Private cohort from here on.
    if _is_contact_doctor_referral(patient):
        return True, "Private patient referred by a Contact (doctor) — letter required"

    referrer_blob = " ".join(
        _coerce_ref_field(patient.get(f))
        for f in ("referral_source", "referral_source_other",
                  "referring_doctor", "referrer")
    ).lower().strip()
    if not referrer_blob:
        return False, "Private patient with no referrer on record — N/A"
    ref_first = _first_word_lower(referrer_blob)
    is_generic = any(
        (ref_first == v) or (v and v in referrer_blob)
        for v in na_ref_values
    )
    if is_generic:
        return False, f"Private patient, generic referral ({referrer_blob[:40]}) — N/A"
    return False, f"Private patient, non-Contact referral ({referrer_blob[:40]}) — N/A"


def _refresh_check_4(audit: "PatientAudit",
                      recall_lookup: dict[str, Any] | None) -> "PatientAudit":
    """Update a cached ``Upcoming / recall`` check against the fresh
    recall lookup, leaving other checks untouched.

    Called on every cache-hit in run_audit. Without this, a patient who
    was cached under v26.4 (when the recalls endpoint 404'd) keeps their
    "unverified N/A" forever, even after Matt uploads the recalls list.

    We don't have the patient's mobile/name in cache (only the audit
    result), so the best we can do for a cached FAIL without an API
    re-fetch is: if the cache was a definite "not on list" fail and the
    patient_name substring matches any recalled_name in the CSV, flip to
    a provisional pass. Otherwise leave the cached state.

    The safe path is: v26.4 cached audits will all say "No future appt;
    recall status unverified" (their N/A reason). We convert those into
    a firm fail (we now have an authoritative list). Matt can hit
    'Force refresh all' once to rebuild from scratch and get name/mobile
    comparisons; from then on cache entries are already v26.5-consistent.

    Scope is deliberately narrow: we only REWRITE cached fails or N/As,
    never cached passes. An upcoming-appt pass can't retroactively become
    invalid from cache alone.
    """
    if recall_lookup is None or not recall_lookup.get("row_count"):
        return audit
    # Name-based retry from cache — patient_name is stored on the audit
    names = recall_lookup.get("names") or set()
    pname = (audit.patient_name or "").strip().lower()
    parts = pname.split()
    name_hit = False
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        if (first, last) in names:
            name_hit = True
    new_checks = []
    changed = False
    for c in audit.checks:
        if c.name != "Upcoming / recall" or c.passed is True:
            new_checks.append(c)
            continue
        if name_hit:
            new_checks.append(CheckResult(
                "Upcoming / recall", True,
                "No future appointment, but patient is on the "
                "uploaded Cliniko recalls list (matched by name)",
            ))
        else:
            new_checks.append(CheckResult(
                "Upcoming / recall", False,
                "No future appointment and patient not on uploaded "
                "Cliniko recalls list (cached audit — force-refresh for "
                "mobile-based match)",
            ))
        changed = True
    if not changed:
        return audit
    from dataclasses import replace
    return replace(audit, checks=new_checks)


# -------------------------------------------------------------------
# Per-patient auditor
# -------------------------------------------------------------------
def audit_patient(client: ClinikoClient,
                   patient_id: str,
                   practitioner_id: str,
                   business_id: str | None,
                   cohort: str,
                   practitioners: pd.DataFrame,
                   recall_lookup: dict[str, Any] | None = None) -> PatientAudit:
    """Run all 5 checks on a single patient.

    Parameters
    ----------
    recall_lookup : dict | None
        Pre-built set-lookup from dashboard.recalls.build_recall_lookup —
        contains ``mobiles`` and ``names`` sets derived from the uploaded
        Cliniko Recalls CSV (v26.5+). Check 4 passes if the patient's
        mobile or name is in those sets, even when there's no upcoming
        appointment. If None, Check 4 degrades to N/A ("please upload
        the recalls CSV") rather than failing — we don't penalise Matt
        for not having uploaded yet.
    """
    settings = load_settings()
    audit_cfg = settings["audit"]
    rap_pattern = re.compile(audit_cfg["rap_attachment_pattern"])
    wibbi_name_pattern = re.compile(audit_cfg["wibbi_name_pattern"])
    wibbi_uploader = (audit_cfg.get("wibbi_uploader_name") or "").lower()
    na_ref_values = [v.lower() for v in audit_cfg.get("correspondence_na_referrer_values", [])]
    clin_fallback_keywords = audit_cfg.get("clinical_fallback_keywords", [])

    patient = _get_patient(client, patient_id)
    if patient is None:
        # Patient archived/deleted/inaccessible — bail out with a clear reason
        # so the diagnostic view can show exactly what happened, rather than
        # a generic "fetch error" that hides the root cause.
        return PatientAudit(
            patient_id=patient_id,
            patient_name="(patient not accessible)",
            practitioner_id=practitioner_id,
            business_id=business_id,
            cohort=cohort,
            checks=[CheckResult(
                "Error", False,
                f"Patient lookup failed — /patients/{patient_id} returned "
                f"no record (likely archived/deleted). "
                f"Last error: {FETCH_STATS.get('patient', {}).get('last_err', 'unknown')}"
            )],
        )
    name = f"{patient.get('first_name','')} {patient.get('last_name','')}".strip()

    attachments = _get_attachments(client, patient_id)
    communications = _get_communications(client, patient_id)
    notes = _get_patient_treatment_notes(client, patient_id)
    appointments = _get_patient_appointments(client, patient_id)
    # v26.5 — stop hitting /patient_recalls; it 404s on this Cliniko plan.
    # Instead, check the patient (already fetched) against the uploaded
    # recalls CSV's mobile/name sets.
    from dashboard.recalls import patient_is_in_recalls
    in_recall_list = patient_is_in_recalls(patient, recall_lookup)

    # --- Check 1: RAP ---
    rap_hit = any(rap_pattern.search(str(a.get("name") or "")
                                     + " " + str(a.get("description") or ""))
                  for a in attachments)
    if rap_hit:
        c1 = CheckResult("RAP", True)
    else:
        # Exempt practitioners on bulk-bill EPC patients → N/A
        if cohort == "EPC" and _is_rap_exempt(practitioners, practitioner_id):
            c1 = CheckResult("RAP", None, "Practitioner is RAP-exempt (new grad) on bulk-bill EPC")
        else:
            # Clinical-reason fallback in notes
            found, kw = _has_clinical_fallback(notes, clin_fallback_keywords)
            if found:
                c1 = CheckResult("RAP", None, f"N/A via clinical-reason fallback: '{kw}'")
            else:
                c1 = CheckResult("RAP", False, "No RAP attachment and no clinical-reason fallback")

    # --- Check 2: Wibbi exercises ---
    wibbi_hit = False
    for a in attachments:
        text = (str(a.get("name") or "") + " " + str(a.get("description") or "")).strip()
        uploader = str(a.get("uploaded_by") or a.get("created_by") or "").lower()
        if wibbi_uploader and wibbi_uploader in uploader:
            wibbi_hit = True
            break
        if wibbi_name_pattern.search(text):
            wibbi_hit = True
            break
    if wibbi_hit:
        c2 = CheckResult("Wibbi exercises", True)
    else:
        found, kw = _has_clinical_fallback(notes, clin_fallback_keywords)
        if found:
            c2 = CheckResult("Wibbi exercises", None, f"N/A via clinical-reason fallback: '{kw}'")
        else:
            c2 = CheckResult("Wibbi exercises", False, "No Wibbi attachment and no clinical-reason fallback")

    # --- Check 3: Correspondence to referrer (v26.4) ---
    # Rule per Matt: letter required for ALL EPC patients, plus Private
    # patients whose referral source is a Cliniko Contact (doctor with
    # name attached). Generic/social referrers (Google, social media,
    # sporting clubs, self, etc.) are N/A. Data source is now /communications
    # instead of /letters — the latter doesn't exist on Enhance Physio's
    # Cliniko plan, and Cliniko support directed us to /communications.
    letter_needed, letter_reason = _letter_required(patient, cohort, na_ref_values)
    if not letter_needed:
        c3 = CheckResult("Correspondence to referrer", None, letter_reason)
    elif ENDPOINT_UNAVAILABLE["communications"]:
        c3 = CheckResult("Correspondence to referrer", None,
                         "Communications endpoint unavailable on this Cliniko "
                         "account — check API key scope / permissions")
    else:
        has_communication = len(communications) > 0
        c3 = CheckResult(
            "Correspondence to referrer", has_communication,
            letter_reason if has_communication
            else f"{letter_reason}, but no communication logged for this patient",
        )

    # --- Check 4: Upcoming appointment OR recall (v26.5) ---
    # Matt's rule: pass if there's a future appointment OR the patient is
    # on the uploaded Cliniko Recalls CSV (bookmarklet-exported list).
    # Fails otherwise. The v21 ENDPOINT_UNAVAILABLE["recalls"] N/A branch is
    # gone — now that we ingest recalls via CSV, "no upcoming + not in list"
    # is a real fail, not an unverified maybe.
    now = datetime.utcnow()
    has_future = False
    for a in appointments:
        try:
            s = a.get("starts_at")
            if not s:
                continue
            dt = datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
            if dt > now and a.get("cancelled_at") is None:
                has_future = True
                break
        except ValueError:
            continue
    if has_future:
        c4 = CheckResult("Upcoming / recall", True)
    elif in_recall_list:
        c4 = CheckResult("Upcoming / recall", True,
                         "No future appointment, but patient is on the "
                         "uploaded Cliniko recalls list")
    elif recall_lookup is None or not recall_lookup.get("row_count"):
        # No recalls CSV has been uploaded this session — degrade to N/A
        # rather than fail, so we don't punish Matt for a missing upload.
        c4 = CheckResult("Upcoming / recall", None,
                         "No future appointment; recall list not uploaded "
                         "this session — upload the Cliniko Recalls CSV on "
                         "the Manual tab to verify this check.")
    else:
        c4 = CheckResult("Upcoming / recall", False,
                         "No future appointment and patient not on uploaded "
                         "Cliniko recalls list")

    # --- Check 5: Notes match appointments attended ---
    attended = [a for a in appointments
                if a.get("cancelled_at") is None
                and not a.get("did_not_arrive", False)]
    past_attended = []
    for a in attended:
        try:
            dt = datetime.fromisoformat(a["starts_at"].replace("Z", "+00:00")).replace(tzinfo=None)
            if dt <= now:
                past_attended.append(a)
        except (KeyError, ValueError):
            continue
    non_draft_notes = [n for n in notes if not n.get("draft", False)]
    match = len(non_draft_notes) >= len(past_attended) and len(past_attended) > 0
    if len(past_attended) == 0:
        c5 = CheckResult("Notes match appts", None, "Patient has no attended appointments yet — N/A")
    else:
        c5 = CheckResult("Notes match appts", match,
                         "" if match else
                         f"{len(non_draft_notes)} non-draft notes vs {len(past_attended)} attended appts")

    return PatientAudit(
        patient_id=patient_id,
        patient_name=name,
        practitioner_id=practitioner_id,
        business_id=business_id,
        cohort=cohort,
        checks=[c1, c2, c3, c4, c5],
    )


# -------------------------------------------------------------------
# Run audit across a pool
# -------------------------------------------------------------------
def run_audit(client: ClinikoClient,
              pool: pd.DataFrame,
              practitioners: pd.DataFrame,
              progress_cb=None,
              use_cache: bool = True,
              force_refresh: bool = False,
              cache_ttl_days: int | None = None,
              on_result=None,
              recall_lookup: dict[str, Any] | None = None) -> list[PatientAudit]:
    """Audit every patient in ``pool`` (possibly cached).

    Parameters
    ----------
    use_cache : bool
        If True, consult the persistent cache (data/audit_cache/audits.jsonl)
        before hitting Cliniko. Results freshly generated during this run
        are also written back to the cache for future reuse.
    force_refresh : bool
        If True, ignore the cache on read but still write fresh results
        back. Useful when Matt wants to re-check a date range after
        clinicians have had time to upload missing RAPs / correspondence.
    cache_ttl_days : int | None
        Freshness window. Entries older than this are treated as absent.
        Defaults to ``settings.audit.cache_ttl_days`` (30).
    on_result : callable(audit) | None
        Invoked after each patient completes, with the PatientAudit. Used
        by the Streamlit UI to checkpoint ``st.session_state`` so a
        mid-run dropout doesn't lose completed work.
    progress_cb : callable(i, total, patient_name, cohort) | None
        Progress ping; ``patient_name`` + ``cohort`` are optional trailing
        args so older 2-arg callbacks still work.
    """
    # Lazy import to avoid a circular dep at module-import time
    from dashboard import audit_cache

    settings = load_settings()
    if cache_ttl_days is None:
        cache_ttl_days = int(settings.get("audit", {}).get("cache_ttl_days", 30))

    # Reset per-endpoint diagnostic counters so the UI can show a clean
    # picture of THIS run (cached patients aren't counted, which is correct
    # — they didn't hit the API).
    _reset_fetch_stats()

    cached = audit_cache.load_all() if use_cache and not force_refresh else {}

    results: list[PatientAudit] = []
    total = len(pool)
    for i, row in enumerate(pool.itertuples(index=False), start=1):
        pid = str(row.patient_id)
        hit = audit_cache.get_fresh(cached, pid, cache_ttl_days) if not force_refresh else None
        if hit is not None:
            # v26.5 — cached Check 4 is derived from the OLD recall state.
            # Re-evaluate it against the fresh recall_lookup without
            # re-running the expensive checks 1/2/3/5.
            r = _refresh_check_4(hit, recall_lookup)
        else:
            try:
                r = audit_patient(
                    client,
                    pid,
                    str(row.practitioner_id),
                    str(row.business_id) if pd.notna(row.business_id) else None,
                    row.cohort,
                    practitioners,
                    recall_lookup=recall_lookup,
                )
            except Exception as e:
                # Log and carry on — one patient shouldn't sink the whole run
                r = PatientAudit(
                    patient_id=pid,
                    patient_name="(fetch error)",
                    practitioner_id=str(row.practitioner_id),
                    business_id=str(row.business_id) if pd.notna(row.business_id) else None,
                    cohort=row.cohort,
                    checks=[CheckResult("Error", False, str(e))],
                )
            # Only cache successful runs (not fetch errors)
            if use_cache and r.checks and r.checks[0].name != "Error":
                try:
                    audit_cache.save_audit(r)
                except Exception:  # disk full, permission, whatever — don't fail the run
                    pass

        results.append(r)
        if on_result is not None:
            try:
                on_result(r)
            except Exception:
                pass
        if progress_cb:
            try:
                progress_cb(i, total, r.patient_name, r.cohort)
            except TypeError:
                # Backward-compat: legacy 2-arg callback
                progress_cb(i, total)
    return results


# -------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------
def aggregate_audit(audits: list[PatientAudit]) -> pd.DataFrame:
    """Per-practitioner EPC %, Private %, Overall %."""
    if not audits:
        return pd.DataFrame(columns=["practitioner_id", "audit_epc_pct",
                                     "audit_private_pct", "audit_pct", "patients_audited"])
    rows = [{
        "practitioner_id": a.practitioner_id,
        "cohort": a.cohort,
        "passes": a.passes,
        "applicable": a.applicable,
        "patient_id": a.patient_id,
    } for a in audits]
    df = pd.DataFrame(rows)

    def _pct(frame: pd.DataFrame, cohort: str) -> pd.Series:
        sub = frame[frame["cohort"] == cohort]
        g = sub.groupby("practitioner_id").agg(passes=("passes", "sum"),
                                                applicable=("applicable", "sum"))
        g["pct"] = g.apply(lambda r: r["passes"] / r["applicable"] if r["applicable"] else 0.0, axis=1)
        return g["pct"].rename(f"audit_{cohort.lower()}_pct")

    epc = _pct(df, "EPC")
    priv = _pct(df, "Private")
    combined = pd.concat([epc, priv], axis=1).fillna(0.0)
    combined["audit_pct"] = combined[["audit_epc_pct", "audit_private_pct"]].mean(axis=1)
    counts = df.groupby("practitioner_id")["patient_id"].nunique().rename("patients_audited")
    combined = combined.join(counts, how="outer").fillna(0)
    return combined.reset_index()


def aggregate_by_check(audits: list[PatientAudit]) -> pd.DataFrame:
    """% failed per check type, per practitioner — for the training-pattern view."""
    rows = []
    for a in audits:
        for c in a.checks:
            rows.append({
                "practitioner_id": a.practitioner_id,
                "check": c.name,
                "status": "pass" if c.passed is True
                          else "fail" if c.passed is False
                          else "na",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pivot = df.pivot_table(
        index=["practitioner_id", "check"],
        columns="status",
        values="check",
        aggfunc="count",
        fill_value=0,
    ).reset_index()
    for col in ("pass", "fail", "na"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["applicable"] = pivot["pass"] + pivot["fail"]
    pivot["fail_rate"] = pivot.apply(
        lambda r: r["fail"] / r["applicable"] if r["applicable"] else 0.0, axis=1
    )
    return pivot


# -------------------------------------------------------------------
# Diagnostics — what the per-endpoint fetch results actually looked like
# -------------------------------------------------------------------
def fetch_stats_snapshot() -> dict[str, dict]:
    """Return a deep-enough copy of FETCH_STATS for UI rendering.

    Each key (``patient``, ``attachments``, ``letters``, ``notes``,
    ``appointments``, ``recalls``) maps to a dict with keys
    ``ok`` / ``empty`` / ``error`` / ``last_err``. Useful for telling the
    difference between "endpoint returns nothing for this patient" (empty,
    which is a normal audit fail) and "endpoint is broken" (error, which
    points at API / pattern / permissions problems).
    """
    return {k: dict(v) for k, v in FETCH_STATS.items()}


def fetch_stats_summary(stats: dict[str, dict] | None = None) -> pd.DataFrame:
    """Flatten FETCH_STATS into a display-ready DataFrame."""
    data = stats if stats is not None else FETCH_STATS
    rows = []
    for endpoint, s in data.items():
        rows.append({
            "endpoint": endpoint,
            "ok": s.get("ok", 0),
            "empty": s.get("empty", 0),
            "error": s.get("error", 0),
            "last_error": s.get("last_err", "") or "—",
        })
    return pd.DataFrame(rows)


def error_reasons_summary(audits: list[PatientAudit], limit: int = 10) -> pd.DataFrame:
    """Count distinct error reasons across a batch of audits.

    When audit_patient can't even fetch the patient (archived, deleted,
    403, rate limit...), the PatientAudit has a single check named
    "Error" with the exception message. Grouping those gives a quick
    picture of the failure modes actually affecting the run.
    """
    rows = []
    for a in audits:
        for c in a.checks:
            if c.name == "Error":
                rows.append({"reason": c.reason or "(no detail)"})
    if not rows:
        return pd.DataFrame(columns=["reason", "count"])
    df = pd.DataFrame(rows)
    g = df.groupby("reason").size().reset_index(name="count").sort_values("count", ascending=False)
    return g.head(limit).reset_index(drop=True)
