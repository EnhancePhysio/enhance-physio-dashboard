"""Monthly metric snapshots for long-term storage.

v27.0.1 — Phase 2 of the multi-org rollout. Each snapshot captures
every practitioner's metrics for one calendar month across every
configured Cliniko org, saved as a single JSON file. Snapshots live
in ``data/snapshots/`` and are pushed to GitHub for permanence.

Two access modes:
  * ``create_snapshot(year, month)`` — synchronous, computes fresh
    from Cliniko. Used by the manual "Snapshot now" button and by the
    auto-trigger.
  * ``load_snapshot(year, month)`` — reads a saved snapshot. Returns
    None if we haven't captured that month yet.

Auto-trigger (``maybe_auto_snapshot()``) runs on app startup. If we're
in a new month and the previous month hasn't been snapshotted, it
does so in the background. Because Streamlit Cloud sleeps apps between
visits, "background" really means "first visitor of the 1st carries
the cost" — which is acceptable given managers reload Monday morning
anyway.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytz

from dashboard.config import DATA_DIR, load_settings
from dashboard.cliniko import (
    ClinikoError, get_configured_orgs, get_client_for_org,
)


SNAPSHOT_DIR = DATA_DIR / "snapshots"
_LAST_META_FILE = SNAPSHOT_DIR / "_last_snapshot.json"


def snapshot_path(year: int, month: int) -> Path:
    return SNAPSHOT_DIR / f"{year}_{month:02d}.json"


def _month_range(year: int, month: int):
    """Local-tz month range compatible with DateRange."""
    from dashboard.date_ranges import DateRange
    tz = pytz.timezone(load_settings().get("timezone", "Australia/Sydney"))
    start = tz.localize(datetime(year, month, 1))
    if month == 12:
        end = tz.localize(datetime(year + 1, 1, 1))
    else:
        end = tz.localize(datetime(year, month + 1, 1))
    return DateRange(start, end)


def _df_to_records(obj: Any) -> Any:
    """Convert any pandas structure to a JSON-serialisable form."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _df_to_records(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_df_to_records(x) for x in obj]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    return obj


def create_snapshot(year: int, month: int,
                     push_to_github: bool = True) -> dict[str, Any]:
    """Compute every configured org's metrics for the given month and
    save to snapshot_path(year, month). Returns the snapshot dict.

    On success, updates ``_LAST_META_FILE`` so ``maybe_auto_snapshot``
    knows the job's done.

    If ``push_to_github`` is True (default), also commits the snapshot
    to the connected GitHub repo — that's the "stored forever" bit.
    """
    dr = _month_range(year, month)
    all_org_data: dict[str, Any] = {}

    for org in get_configured_orgs():
        org_data: dict[str, Any] = {
            "org_name": org.display_name,
            "org_key": org.key,
        }
        try:
            client = get_client_for_org(org)
            from dashboard.metrics import compute_core_metrics
            from dashboard.reference_data import (
                load_appointment_types, load_practitioners, load_businesses,
            )
            appt_types = load_appointment_types(client)
            practitioners = load_practitioners(client)
            businesses = load_businesses(client)
            result, _ = compute_core_metrics(
                client, dr, appt_types,
                business_ids=[],
                practitioner_ids=[],
            )
            # Persist practitioners + businesses reference tables so
            # downstream renders can label rows without needing another
            # Cliniko fetch.
            org_data["reference"] = {
                "practitioners": _df_to_records(practitioners),
                "businesses":    _df_to_records(businesses),
            }
            org_data["metrics"] = {}
            # Snapshot every DataFrame attribute of the result object.
            # This future-proofs against new metrics being added — as
            # long as they're pandas DataFrames on MetricResult, they
            # get captured automatically.
            for attr_name in dir(result):
                if attr_name.startswith("_"):
                    continue
                try:
                    v = getattr(result, attr_name)
                except Exception:
                    continue
                if isinstance(v, pd.DataFrame):
                    org_data["metrics"][attr_name] = _df_to_records(v)
                elif isinstance(v, dict) and v and all(
                    isinstance(x, (int, float, str, bool, type(None)))
                    for x in v.values()
                ):
                    org_data["metrics"][attr_name] = v
        except ClinikoError as e:
            org_data["error"] = f"Cliniko error: {e}"
        except Exception as e:
            org_data["error"] = f"{type(e).__name__}: {e}"

        all_org_data[org.key] = org_data

    snapshot = {
        "year": year,
        "month": month,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "organizations": all_org_data,
        "schema_version": 1,
    }

    # v27.0.1 — Only write the snapshot + marker if at least one org
    # actually produced data. This prevents a zero-org sandbox / broken
    # keys situation from silently "completing" the auto-trigger.
    orgs_with_data = [k for k, v in all_org_data.items()
                        if isinstance(v, dict) and "metrics" in v]
    if not orgs_with_data:
        raise RuntimeError(
            "No configured orgs returned data — snapshot NOT saved. "
            "Check that at least one CLINIKO_API_KEY* secret is set."
        )

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path(year, month), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)

    # Update last-snapshot marker
    try:
        with open(_LAST_META_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "year": year, "month": month,
                "at": datetime.utcnow().isoformat() + "Z",
                "orgs_captured": orgs_with_data,
            }, f)
    except Exception:
        pass

    # Push to GitHub so the snapshot survives Streamlit Cloud redeploys.
    # Uses the same GitHub persistence layer as recalls / NPS / punctuality.
    if push_to_github:
        try:
            from dashboard.github_persistence import save_file_to_github
            save_file_to_github(
                snapshot_path(year, month),
                commit_msg=f"[snapshot] {year}-{month:02d} auto-monthly",
            )
        except Exception:
            pass  # never let GitHub failure crash the auto-trigger

    return snapshot


def load_snapshot(year: int, month: int) -> dict[str, Any] | None:
    """Return the saved snapshot for (year, month), or None if we
    haven't captured that month yet.

    Also tries to pull from GitHub if the local file's missing but the
    repo has one (e.g. fresh Streamlit Cloud deploy where the disk
    was wiped).
    """
    p = snapshot_path(year, month)
    if not p.exists():
        # Try hydrating from GitHub
        try:
            from dashboard.github_persistence import hydrate_directory_from_github
            hydrate_directory_from_github(SNAPSHOT_DIR)
        except Exception:
            pass
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _prev_month(today: date) -> tuple[int, int]:
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


def maybe_auto_snapshot(today: date | None = None) -> tuple[bool, str]:
    """Called on app startup. If today is in a new month and last
    month hasn't been snapshotted yet, do so now.

    Returns ``(did_snapshot, message)`` for logging. Never raises.
    """
    today = today or date.today()
    target_year, target_month = _prev_month(today)

    # Have we already snapshotted this month?
    if _LAST_META_FILE.exists():
        try:
            with open(_LAST_META_FILE, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (int(meta.get("year", 0)) == target_year
                    and int(meta.get("month", 0)) == target_month):
                return False, f"already snapshotted {target_year}-{target_month:02d}"
        except Exception:
            pass

    # File-level fallback — if the local marker's missing but the
    # snapshot file exists (e.g. hydrated from GitHub), skip.
    if snapshot_path(target_year, target_month).exists():
        return False, f"snapshot file exists for {target_year}-{target_month:02d}"

    try:
        create_snapshot(target_year, target_month)
        return True, f"created snapshot for {target_year}-{target_month:02d}"
    except Exception as e:
        return False, f"snapshot failed: {type(e).__name__}: {e}"


def list_snapshots() -> list[dict[str, Any]]:
    """Return a summary of every snapshot on disk, newest first."""
    if not SNAPSHOT_DIR.exists():
        return []
    rows: list[dict[str, Any]] = []
    for f in sorted(SNAPSHOT_DIR.glob("*.json"), reverse=True):
        if f.name.startswith("_"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            rows.append({
                "year": int(data.get("year", 0)),
                "month": int(data.get("month", 0)),
                "created_at": data.get("created_at"),
                "orgs": list(data.get("organizations", {}).keys()),
                "path": str(f),
            })
        except Exception:
            continue
    return rows
