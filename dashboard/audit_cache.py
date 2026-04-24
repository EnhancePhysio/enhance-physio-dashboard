"""Persistent on-disk cache of per-patient audit results.

Storage format: JSON Lines at ``data/audit_cache/audits.jsonl``. One line per
audit run, newest-wins on duplicate ``patient_id`` (enforced at read time, and
compacted to a single line per patient by ``compact()``).

Why JSONL: append-only writes are atomic on POSIX, safe to interrupt
mid-write, easy to inspect with ``cat`` / ``jq``, and tiny even at tens of
thousands of audits.

Design rationale — what belongs in the cache:
* Audit results are *per-patient* snapshots, independent of the date range
  Matt chose when he ran the audit. So once a patient is audited, their
  result can be reused for any subsequent run that includes them in the pool
  (until the TTL expires).
* A 30-day TTL (configurable in settings.yml via ``audit.cache_ttl_days``)
  balances speed against freshness — a patient who was "failing RAP" 31
  days ago should be re-checked in case the clinician has since uploaded it.
* Matt can "force refresh" a run from the UI to bypass the cache entirely.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from dashboard.audit import CheckResult, PatientAudit
from dashboard.config import DATA_DIR


CACHE_DIR = DATA_DIR / "audit_cache"
CACHE_FILE = CACHE_DIR / "audits.jsonl"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------
# (De)serialisation
# ---------------------------------------------------------------
def _serialise(audit: PatientAudit, scored_at: datetime) -> dict:
    return {
        "patient_id": str(audit.patient_id),
        "patient_name": audit.patient_name,
        "practitioner_id": str(audit.practitioner_id),
        "business_id": str(audit.business_id) if audit.business_id else None,
        "cohort": audit.cohort,
        "scored_at": scored_at.isoformat(),
        "checks": [
            {"name": c.name, "passed": c.passed, "reason": c.reason}
            for c in audit.checks
        ],
    }


def _deserialise(d: dict) -> tuple[PatientAudit, datetime]:
    audit = PatientAudit(
        patient_id=str(d["patient_id"]),
        patient_name=d.get("patient_name", ""),
        practitioner_id=str(d["practitioner_id"]),
        business_id=d.get("business_id"),
        cohort=d.get("cohort", ""),
        checks=[
            CheckResult(
                name=c["name"],
                passed=c.get("passed"),
                reason=c.get("reason") or c.get("detail") or "",
            )
            for c in d.get("checks", [])
        ],
    )
    try:
        ts = datetime.fromisoformat(d.get("scored_at", ""))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        ts = datetime.min.replace(tzinfo=timezone.utc)
    return audit, ts


# ---------------------------------------------------------------
# Read / write
# ---------------------------------------------------------------
def save_audit(audit: PatientAudit, scored_at: datetime | None = None) -> None:
    """Append one audit to the cache file. Dedup happens on read + compact()."""
    ts = scored_at or datetime.now(timezone.utc)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_serialise(audit, ts), ensure_ascii=False, default=str)
    with CACHE_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_all() -> dict[str, tuple[PatientAudit, datetime]]:
    """Load every cached audit. Returns {patient_id: (audit, scored_at)}.

    If a patient has multiple entries (from weekly re-audits), the newest
    wins. Corrupt lines are silently skipped so a single bad write can't
    brick the cache.
    """
    out: dict[str, tuple[PatientAudit, datetime]] = {}
    if not CACHE_FILE.exists():
        return out
    with CACHE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            audit, ts = _deserialise(d)
            pid = audit.patient_id
            existing = out.get(pid)
            if existing is None or ts > existing[1]:
                out[pid] = (audit, ts)
    return out


def _is_poisoned(audit: PatientAudit) -> bool:
    """Is this a cached audit from a broken-endpoint run that we should
    silently re-audit instead of serving?

    v20 — the audit pool ran against three sub-resource endpoints that
    hard-404 on Matt's Cliniko shard (letters, individual_appointments,
    patient_recalls). Patients that went through the broken pipeline
    landed in the cache either as:
      1. Single 'Error' check PatientAudit — explicit bail-out, OR
      2. A 5-check audit where checks 3, 4, 5 are all forced-fail because
         the three fetchers returned [] (404 -> try/except -> []).

    Both end up wrong. We can't distinguish case 2 cleanly from real
    failures without re-running, so we only auto-evict case 1 here (any
    single-'Error'-check cached audit) and rely on Matt clicking the new
    'Wipe audit cache' button (added to the UI in v20) to force a clean
    re-run if he wants to discard the 5-check garbage too.
    """
    checks = audit.checks or []
    if len(checks) != 1:
        return False
    c = checks[0]
    return c.name == "Error"


def get_fresh(cache: dict[str, tuple[PatientAudit, datetime]],
               patient_id: str,
               ttl_days: int) -> PatientAudit | None:
    """Return the cached audit iff it exists and is within TTL."""
    entry = cache.get(str(patient_id))
    if entry is None:
        return None
    audit, ts = entry
    # v20 self-heal: ignore cached audits that are clearly the result of
    # the v19-era endpoint bugs so v20's corrected fetchers can re-audit.
    if _is_poisoned(audit):
        return None
    if ttl_days <= 0:
        return audit  # 0 or negative disables expiry
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    if ts >= cutoff:
        return audit
    return None


def compact() -> int:
    """Rewrite the cache file keeping only the newest entry per patient.

    Returns the number of entries retained. Safe to run any time — worst
    case the file gets re-linearised with the same content.
    """
    cache = load_all()
    if not cache:
        return 0
    tmp = CACHE_FILE.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for pid, (audit, ts) in cache.items():
            f.write(json.dumps(_serialise(audit, ts), ensure_ascii=False, default=str) + "\n")
    tmp.replace(CACHE_FILE)
    return len(cache)


def clear() -> int:
    """Delete the cache file. Returns the number of entries that were in it."""
    n = 0
    if CACHE_FILE.exists():
        n = sum(1 for _ in CACHE_FILE.open("r", encoding="utf-8") if _.strip())
        CACHE_FILE.unlink()
    return n


def stats() -> dict[str, int | str]:
    """Quick summary for the UI diagnostics panel."""
    cache = load_all()
    if not cache:
        return {"entries": 0, "oldest": "—", "newest": "—"}
    timestamps = [ts for (_, ts) in cache.values()]
    return {
        "entries": len(cache),
        "oldest": min(timestamps).isoformat(timespec="minutes"),
        "newest": max(timestamps).isoformat(timespec="minutes"),
    }
