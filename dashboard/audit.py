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
def _get_patient(client: ClinikoClient, patient_id: str) -> dict[str, Any]:
    return client.get(f"patients/{patient_id}")


def _get_attachments(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    try:
        return list(client.paginate(f"patients/{patient_id}/patient_attachments"))
    except Exception:
        return []


def _get_letters(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    try:
        return list(client.paginate(f"patients/{patient_id}/letters"))
    except Exception:
        return []


def _get_patient_treatment_notes(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    try:
        return list(client.paginate(f"patients/{patient_id}/treatment_notes"))
    except Exception:
        return []


def _get_patient_appointments(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    try:
        return list(client.paginate(f"patients/{patient_id}/individual_appointments"))
    except Exception:
        return []


def _get_recalls(client: ClinikoClient, patient_id: str) -> list[dict[str, Any]]:
    try:
        return list(client.paginate(f"patients/{patient_id}/patient_recalls"))
    except Exception:
        return []


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


# -------------------------------------------------------------------
# Per-patient auditor
# -------------------------------------------------------------------
def audit_patient(client: ClinikoClient,
                   patient_id: str,
                   practitioner_id: str,
                   business_id: str | None,
                   cohort: str,
                   practitioners: pd.DataFrame) -> PatientAudit:
    settings = load_settings()
    audit_cfg = settings["audit"]
    rap_pattern = re.compile(audit_cfg["rap_attachment_pattern"])
    wibbi_name_pattern = re.compile(audit_cfg["wibbi_name_pattern"])
    wibbi_uploader = (audit_cfg.get("wibbi_uploader_name") or "").lower()
    na_ref_values = [v.lower() for v in audit_cfg.get("correspondence_na_referrer_values", [])]
    clin_fallback_keywords = audit_cfg.get("clinical_fallback_keywords", [])

    patient = _get_patient(client, patient_id)
    name = f"{patient.get('first_name','')} {patient.get('last_name','')}".strip()

    attachments = _get_attachments(client, patient_id)
    letters = _get_letters(client, patient_id)
    notes = _get_patient_treatment_notes(client, patient_id)
    appointments = _get_patient_appointments(client, patient_id)
    recalls = _get_recalls(client, patient_id)

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

    # --- Check 3: Correspondence to referrer ---
    referrer_blob = (
        (patient.get("referral_source") or "") + " " +
        (patient.get("referral_source_other") or "") + " " +
        (patient.get("referring_doctor") or "") + " " +
        (patient.get("referrer") or "")
    ).lower().strip()
    ref_first = _first_word_lower(referrer_blob)
    na_referrer = any(
        (ref_first == v) or (v and v in referrer_blob)
        for v in na_ref_values
    )
    if na_referrer or not referrer_blob:
        c3 = CheckResult("Correspondence to referrer", None, "Referrer is generic/social/self — N/A")
    else:
        has_letter = len(letters) > 0
        c3 = CheckResult("Correspondence to referrer", has_letter,
                         "" if has_letter else "No letter found on patient record")

    # --- Check 4: Upcoming appointment OR recall ---
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
    has_recall = len(recalls) > 0
    if has_future or has_recall:
        c4 = CheckResult("Upcoming / recall", True)
    else:
        c4 = CheckResult("Upcoming / recall", False, "No future appointment and no recall set")

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
              progress_cb=None) -> list[PatientAudit]:
    results: list[PatientAudit] = []
    total = len(pool)
    for i, row in enumerate(pool.itertuples(index=False), start=1):
        try:
            r = audit_patient(
                client,
                str(row.patient_id),
                str(row.practitioner_id),
                str(row.business_id) if pd.notna(row.business_id) else None,
                row.cohort,
                practitioners,
            )
            results.append(r)
        except Exception as e:
            # Log and carry on — one patient shouldn't sink the whole run
            results.append(PatientAudit(
                patient_id=str(row.patient_id),
                patient_name="(fetch error)",
                practitioner_id=str(row.practitioner_id),
                business_id=str(row.business_id) if pd.notna(row.business_id) else None,
                cohort=row.cohort,
                checks=[CheckResult("Error", False, str(e))],
            ))
        if progress_cb:
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
