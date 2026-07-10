"""Persistent "reviewed" flags for the Clinical Review queue.

v27.2 — Once a clinical lead ticks a patient as reviewed, they stay
hidden from the weekly queue for ``reviewed_auto_unhide_days`` (default
90 days). After that they'll reappear if they still meet the
over/under-servicing criteria — that guarantees long-duration cases
get periodic re-review rather than being permanently forgotten.

Storage: single JSON at ``data/clinical_review_reviewed.json``. Pushed
to GitHub for permanence + hydrated on cold starts (same pattern as
punctuality/NPS/recalls).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dashboard.config import DATA_DIR, load_settings


_REVIEWED_FILE = DATA_DIR / "clinical_review_reviewed.json"


def _auto_unhide_days() -> int:
    cr = load_settings().get("clinical_review") or {}
    return int(cr.get("reviewed_auto_unhide_days", 90))


def _load_all() -> dict[str, Any]:
    if not _REVIEWED_FILE.exists():
        return {}
    try:
        with open(_REVIEWED_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_all(data: dict[str, Any]) -> None:
    _REVIEWED_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_REVIEWED_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        return
    # Push to GitHub so flags survive Streamlit Cloud redeploys
    try:
        from dashboard.github_persistence import save_file_to_github
        save_file_to_github(
            _REVIEWED_FILE,
            commit_msg="[clinical review] reviewed flags update",
        )
    except Exception:
        pass


def mark_reviewed(patient_id: str, note: str = "",
                    reviewer: str = "") -> None:
    """Mark a patient as reviewed *now*. Overwrites any prior flag."""
    data = _load_all()
    data[str(patient_id)] = {
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "note": note or "",
        "reviewer": reviewer or "",
    }
    _save_all(data)


def unmark_reviewed(patient_id: str) -> None:
    """Remove a reviewed flag (undo)."""
    data = _load_all()
    if str(patient_id) in data:
        data.pop(str(patient_id))
        _save_all(data)


def hidden_patient_ids(as_of: datetime | None = None) -> set[str]:
    """Patient IDs currently hidden from the queue (reviewed within
    the auto-unhide window). After the window, they're not returned
    here, so they reappear in the queue if they still meet criteria.
    """
    as_of = as_of or datetime.now(timezone.utc)
    cutoff_days = _auto_unhide_days()
    hidden: set[str] = set()
    for pid, meta in _load_all().items():
        raw = meta.get("reviewed_at") if isinstance(meta, dict) else None
        if not raw:
            continue
        try:
            reviewed_at = datetime.fromisoformat(raw)
        except (TypeError, ValueError):
            continue
        if reviewed_at.tzinfo is None:
            reviewed_at = reviewed_at.replace(tzinfo=timezone.utc)
        if (as_of - reviewed_at).days < cutoff_days:
            hidden.add(str(pid))
    return hidden


def get_flag(patient_id: str) -> dict[str, Any] | None:
    """Return the reviewed metadata for a patient, or None."""
    data = _load_all()
    return data.get(str(patient_id))


def hydrate_from_github() -> None:
    """Pull the reviewed-flags file from GitHub if the local copy is
    missing (called on app startup)."""
    if _REVIEWED_FILE.exists():
        return
    try:
        from dashboard.github_persistence import hydrate_directory_from_github
        hydrate_directory_from_github(_REVIEWED_FILE.parent)
    except Exception:
        pass
