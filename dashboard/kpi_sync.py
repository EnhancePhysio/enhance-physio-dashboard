"""Pull KPI inputs from the admin tool's GitHub data repo.

The admin tool at https://enhancekpitracker.netlify.app writes four CSVs to a
private GitHub repo (default: ``EnhancePhysio/enhance-kpi-data``) under
``data/``: ``punctuality.csv``, ``availability.csv``, ``audits.csv``, ``nps.csv``.

Each time the dashboard boots, this module fetches the latest versions of those
CSVs, converts them into the schema the existing dashboard loaders already
expect (``data/punctuality/``, ``data/nps/``) and drops the raw audit +
availability files into ``data/kpi_sync/`` for ad-hoc UI consumption.

Config via env / Streamlit secrets:
    KPI_DATA_TOKEN   — fine-grained PAT with Contents: Read on the data repo
    KPI_DATA_REPO    — owner/repo (default: EnhancePhysio/enhance-kpi-data)
    KPI_DATA_BRANCH  — default: main
    KPI_DATA_DIR     — subdir inside the repo (default: data)

If KPI_DATA_TOKEN is missing we skip the sync silently so the dashboard still
works with any pre-existing manual-upload data on disk.
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.config import DATA_DIR, _ensure_env


_API_BASE = "https://api.github.com"
_USER_AGENT = "EnhancePhysioDashboard-KPISync/1.0"

KPI_SYNC_DIR = DATA_DIR / "kpi_sync"
PUNCT_DIR = DATA_DIR / "punctuality"
NPS_DIR = DATA_DIR / "nps"
KPI_SYNC_DIR.mkdir(parents=True, exist_ok=True)
PUNCT_DIR.mkdir(parents=True, exist_ok=True)
NPS_DIR.mkdir(parents=True, exist_ok=True)

_STATUS_FILE = KPI_SYNC_DIR / "_sync_status.json"
_PUNCT_SYNC_FILENAME = "SYNC_admin_tool_punctuality.csv"
_NPS_SYNC_FILENAME = "SYNC_admin_tool_nps.csv"


@dataclass(frozen=True)
class _Config:
    token: str
    repo: str
    branch: str
    data_dir: str


def _config() -> _Config | None:
    _ensure_env()
    token = os.environ.get("KPI_DATA_TOKEN", "").strip()
    repo = os.environ.get("KPI_DATA_REPO", "").strip() or "EnhancePhysio/enhance-kpi-data"
    branch = os.environ.get("KPI_DATA_BRANCH", "").strip() or "main"
    data_dir = os.environ.get("KPI_DATA_DIR", "").strip() or "data"
    if not token:
        return None
    return _Config(token=token, repo=repo, branch=branch, data_dir=data_dir.strip("/"))


def _github_get(cfg: _Config, path: str) -> str | None:
    """Fetch raw file text from the data repo. Returns None if 404."""
    url = f"{_API_BASE}/repos/{cfg.repo}/contents/{path}?ref={cfg.branch}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {cfg.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": _USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    if payload.get("encoding") != "base64":
        return None
    return base64.b64decode(payload["content"]).decode("utf-8")


# -------------------------------------------------------------------
# CSV adapters: convert admin-tool schema → dashboard schema
# -------------------------------------------------------------------
_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _week_start(d: date) -> date:
    """Monday of the ISO week containing d."""
    return d - timedelta(days=d.weekday())


def adapt_punctuality(text: str) -> pd.DataFrame:
    """Admin tool: date, clinic, practitioner, b05, b610, b1114, b15, status
       Dashboard:  week_starting, clinic, practitioner, day,
                   bucket_0_5, bucket_6_10, bucket_11_14, bucket_15_plus, status
    """
    if not text or not text.strip():
        return pd.DataFrame(columns=[
            "week_starting", "clinic", "practitioner", "day",
            "bucket_0_5", "bucket_6_10", "bucket_11_14", "bucket_15_plus", "status",
        ])
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df["week_starting"] = df["date"].map(_week_start).map(lambda d: d.isoformat())
    df["day"] = df["date"].map(lambda d: _DAY_NAMES[d.weekday()])
    out = df.rename(columns={
        "b05": "bucket_0_5",
        "b610": "bucket_6_10",
        "b1114": "bucket_11_14",
        "b15": "bucket_15_plus",
    })
    cols = ["week_starting", "clinic", "practitioner", "day",
            "bucket_0_5", "bucket_6_10", "bucket_11_14", "bucket_15_plus", "status"]
    for c in cols:
        if c not in out.columns:
            out[c] = "" if c in ("status",) else 0
    return out[cols]


def adapt_nps(text: str) -> pd.DataFrame:
    """Admin tool: month, scope, clinic, name, nps, responses
       Dashboard:  week_starting, clinic, practitioner, responses,
                   promoters, passives, detractors, nps

    The admin tool doesn't collect promoter/passive/detractor breakdowns, so we
    leave those blank — the dashboard already treats them as optional.
    'scope=clinic' rows get practitioner="<clinic> (whole clinic)" so they're
    distinguishable from per-practitioner rows.
    """
    if not text or not text.strip():
        return pd.DataFrame(columns=[
            "week_starting", "clinic", "practitioner", "responses",
            "promoters", "passives", "detractors", "nps",
        ])
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return df
    # Convert YYYY-MM → first-of-month date string
    df["week_starting"] = pd.to_datetime(df["month"] + "-01", errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=["week_starting"])

    def _prac_label(row):
        if (row.get("scope") or "").lower() == "clinic":
            return f"{row.get('clinic', '')} (whole clinic)"
        return row.get("name", "")

    df["practitioner"] = df.apply(_prac_label, axis=1)
    out = pd.DataFrame({
        "week_starting": df["week_starting"],
        "clinic": df["clinic"].fillna(""),
        "practitioner": df["practitioner"].fillna(""),
        "responses": pd.to_numeric(df.get("responses"), errors="coerce"),
        "promoters": pd.NA,
        "passives": pd.NA,
        "detractors": pd.NA,
        "nps": pd.to_numeric(df.get("nps"), errors="coerce"),
    })
    return out


# -------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------
def sync_now() -> dict[str, Any]:
    """Fetch latest CSVs from the data repo, adapt + write to the dashboard's
    expected locations. Returns a status dict (also persisted to disk for the
    sidebar indicator)."""
    cfg = _config()
    status: dict[str, Any] = {
        "enabled": bool(cfg),
        "stamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "ok": False,
        "detail": {},
    }
    if not cfg:
        status["detail"]["reason"] = "KPI_DATA_TOKEN not set — skipping sync."
        _write_status(status)
        return status

    try:
        raw_punct = _github_get(cfg, f"{cfg.data_dir}/punctuality.csv")
        raw_avail = _github_get(cfg, f"{cfg.data_dir}/availability.csv")
        raw_audit = _github_get(cfg, f"{cfg.data_dir}/audits.csv")
        raw_nps   = _github_get(cfg, f"{cfg.data_dir}/nps.csv")
    except Exception as e:
        status["detail"]["error"] = f"{type(e).__name__}: {e}"
        _write_status(status)
        return status

    counts: dict[str, int] = {}

    # Punctuality — write as a single synced file that the existing loader
    # already picks up (loader globs *.csv in PUNCT_DIR).
    if raw_punct:
        df = adapt_punctuality(raw_punct)
        (PUNCT_DIR / _PUNCT_SYNC_FILENAME).write_text(df.to_csv(index=False))
        counts["punctuality_rows"] = len(df)
    # Clear stale sync file if the source has disappeared
    elif (PUNCT_DIR / _PUNCT_SYNC_FILENAME).exists():
        (PUNCT_DIR / _PUNCT_SYNC_FILENAME).unlink()

    if raw_nps:
        df = adapt_nps(raw_nps)
        (NPS_DIR / _NPS_SYNC_FILENAME).write_text(df.to_csv(index=False))
        counts["nps_rows"] = len(df)
    elif (NPS_DIR / _NPS_SYNC_FILENAME).exists():
        (NPS_DIR / _NPS_SYNC_FILENAME).unlink()

    # Availability + audits don't have an existing loader; keep raw copies in
    # kpi_sync/ for later use.
    if raw_avail:
        (KPI_SYNC_DIR / "availability.csv").write_text(raw_avail)
        counts["availability_bytes"] = len(raw_avail)
    if raw_audit:
        (KPI_SYNC_DIR / "audits.csv").write_text(raw_audit)
        counts["audits_bytes"] = len(raw_audit)

    status["ok"] = True
    status["detail"] = {"repo": cfg.repo, "branch": cfg.branch, "counts": counts}
    _write_status(status)
    return status


def _write_status(status: dict[str, Any]) -> None:
    try:
        _STATUS_FILE.write_text(json.dumps(status, indent=2))
    except Exception:
        pass


def last_status() -> dict[str, Any]:
    try:
        return json.loads(_STATUS_FILE.read_text())
    except Exception:
        return {"enabled": False, "ok": False, "stamp": Null, "detail": {}}


# -------------------------------------------------------------------
# Audit convenience readers (raw admin-tool audits.csv)
# -------------------------------------------------------------------
def load_admin_audits() -> pd.DataFrame:
    """Return the raw audits CSV as a DataFrame, or empty."""
    p = KPI_SYNC_DIR / "audits.csv"
    if not p.exists():
        return pd.DataFrame(columns=[
            "month", "clinic", "practitioner", "funding", "patient_name",
            "notes", "upcoming", "rap", "exercises", "correspondence",
            "score_y", "score_denom",
        ])
    df = pd.read_csv(p)
    for col in ("score_y", "score_denom"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def load_admin_availability() -> pd.DataFrame:
    p = KPI_SYNC_DIR / "availability.csv"
    if not p.exists():
        return pd.DataFrame(columns=["date", "clinic", "practitioner", "hours", "status", "day_worked"])
    df = pd.read_csv(p)
    if "hours" in df.columns:
        df["hours"] = pd.to_numeric(df["hours"], errors="coerce").fillna(0)
    if "day_worked" in df.columns:
        df["day_worked"] = pd.to_numeric(df["day_worked"], errors="coerce").fillna(0)
    return df
