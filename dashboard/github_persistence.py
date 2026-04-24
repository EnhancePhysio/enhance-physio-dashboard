"""GitHub-backed persistence for uploaded data.

Streamlit Cloud has an ephemeral filesystem — every app reboot wipes anything
we wrote to ``data/punctuality/``, ``data/nps/``, etc. That's fatal for a
dashboard where Matt uploads weekly data that needs to accumulate over months.

This module solves that by treating the app's own GitHub repo as the source
of truth for uploaded data:

* **On save** — we write the file to local disk (so it's immediately usable
  in the current session) AND push it to the repo via the GitHub REST API.
  The commit is tagged ``[data] <filename>`` so history is human-readable.
* **On startup** — if the local directory is empty (post-reboot), we pull
  every file from the repo back into the ephemeral filesystem. After that
  ``load_punctuality()`` / ``load_nps()`` work unchanged.

The only thing the user needs to configure is:

1. ``GITHUB_TOKEN``   — a fine-grained PAT with **Contents: Read & Write**
                        scope on this one repo.
2. ``GITHUB_REPO``    — ``owner/repo`` slug, e.g. ``mattx/physio-dashboard``.
3. ``GITHUB_BRANCH``  — usually ``main`` (default if unset).

If any of those are missing, we degrade to local-only behaviour (same as v25
and earlier) and surface a banner in the UI so the user knows syncing is
disabled. Nothing crashes if GitHub is misconfigured or unreachable.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dashboard.config import DATA_DIR, _ensure_env


_API_BASE = "https://api.github.com"
_USER_AGENT = "EnhancePhysioDashboard-Persistence/1.0"


@dataclass(frozen=True)
class _GitHubConfig:
    token: str
    repo: str     # "owner/repo"
    branch: str   # "main"


def _config() -> _GitHubConfig | None:
    """Read GitHub settings from env/Streamlit secrets. Returns None when
    any required field is missing — callers should treat that as "sync
    disabled" rather than an error."""
    _ensure_env()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPO", "").strip()
    branch = os.environ.get("GITHUB_BRANCH", "").strip() or "main"
    if not token or not repo or "/" not in repo:
        return None
    return _GitHubConfig(token=token, repo=repo, branch=branch)


def is_configured() -> bool:
    return _config() is not None


def status_description() -> str:
    """Human-readable one-liner for the sidebar badge."""
    cfg = _config()
    if cfg is None:
        return "⚠️ GitHub sync disabled (no GITHUB_TOKEN/GITHUB_REPO set)"
    return f"✅ GitHub sync on: `{cfg.repo}@{cfg.branch}`"


def _request(method: str, path: str, body: dict[str, Any] | None = None,
              token: str | None = None) -> tuple[int, dict[str, Any] | None]:
    """Tiny wrapper around urllib for the few REST calls we need.

    Returns (status_code, parsed_json_body_or_None). Never raises on HTTP
    errors — returns the status so the caller can decide what to do
    (e.g. 404 = file doesn't exist yet, treat as "create").
    """
    url = f"{_API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", _USER_AGENT)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(raw) if raw else None
            except json.JSONDecodeError:
                return resp.status, None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else ""
        try:
            return e.code, json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return e.code, None
    except urllib.error.URLError:
        return 0, None  # network failure


def _repo_path_for(local_path: Path) -> str:
    """Convert an absolute local path like /.../data/punctuality/x.csv to
    the repo-relative equivalent: ``data/punctuality/x.csv``."""
    try:
        rel = local_path.relative_to(DATA_DIR.parent)
    except ValueError:
        # Fallback — at least keep the last two segments
        rel = Path(*local_path.parts[-3:])
    return str(rel).replace("\\", "/")


def save_file_to_github(local_path: Path, commit_msg: str | None = None) -> tuple[bool, str]:
    """Push the contents of ``local_path`` to the configured GitHub repo.

    Returns (ok, message). If GitHub sync isn't configured, returns
    ``(False, "sync disabled")``. Non-destructive — if the file already
    exists at the same path the call is an update; otherwise a create.
    """
    cfg = _config()
    if cfg is None:
        return False, "sync disabled"
    if not local_path.exists():
        return False, f"local file missing: {local_path}"

    repo_path = _repo_path_for(local_path)
    content_b64 = base64.b64encode(local_path.read_bytes()).decode()

    # Check if the file exists at this path on the target branch — need
    # the `sha` for an update, omitted for a create.
    status, body = _request(
        "GET",
        f"/repos/{cfg.repo}/contents/{repo_path}?ref={cfg.branch}",
        token=cfg.token,
    )
    existing_sha: str | None = None
    if status == 200 and isinstance(body, dict):
        existing_sha = body.get("sha")
    elif status == 404:
        existing_sha = None
    elif status == 0:
        return False, "network error contacting GitHub"
    elif status not in (200, 404):
        return False, f"GitHub GET failed: HTTP {status}"

    put_body: dict[str, Any] = {
        "message": commit_msg or f"[data] {repo_path}",
        "content": content_b64,
        "branch": cfg.branch,
    }
    if existing_sha:
        put_body["sha"] = existing_sha

    status, resp = _request(
        "PUT",
        f"/repos/{cfg.repo}/contents/{repo_path}",
        body=put_body,
        token=cfg.token,
    )
    if status in (200, 201):
        action = "updated" if existing_sha else "created"
        return True, f"GitHub: {action} {repo_path}"
    if status == 0:
        return False, "network error contacting GitHub"
    return False, f"GitHub PUT failed: HTTP {status}"


def hydrate_directory_from_github(local_dir: Path) -> tuple[int, str]:
    """Pull every file under ``local_dir`` (relative to the repo root) into
    the local filesystem. Used on app startup after a Streamlit reboot has
    wiped the ephemeral disk.

    Returns (files_pulled, message). No-op if:
      - sync is not configured
      - ``local_dir`` already has files (assume disk survived)
      - the path doesn't exist in the repo
    """
    cfg = _config()
    if cfg is None:
        return 0, "sync disabled"
    local_dir.mkdir(parents=True, exist_ok=True)
    # Skip hydration if the local dir already has CSVs — avoids hammering
    # the API on every rerun once the session is warm.
    existing = list(local_dir.glob("*.csv"))
    if existing:
        return 0, f"local already has {len(existing)} file(s); skipping hydration"

    repo_path = _repo_path_for(local_dir)
    status, body = _request(
        "GET",
        f"/repos/{cfg.repo}/contents/{repo_path}?ref={cfg.branch}",
        token=cfg.token,
    )
    if status == 404:
        return 0, f"repo path '{repo_path}' doesn't exist yet"
    if status == 0:
        return 0, "network error contacting GitHub"
    if status != 200 or not isinstance(body, list):
        return 0, f"GitHub GET failed: HTTP {status}"

    pulled = 0
    for entry in body:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "file":
            continue
        name = entry.get("name", "")
        if not name.endswith(".csv"):
            continue
        # Download the raw content
        download_url = entry.get("download_url")
        if not download_url:
            # Fall back to the contents API for the individual file
            f_status, f_body = _request(
                "GET",
                f"/repos/{cfg.repo}/contents/{repo_path}/{name}?ref={cfg.branch}",
                token=cfg.token,
            )
            if f_status == 200 and isinstance(f_body, dict):
                encoded = f_body.get("content", "")
                try:
                    raw_bytes = base64.b64decode(encoded)
                except Exception:
                    continue
                (local_dir / name).write_bytes(raw_bytes)
                pulled += 1
            continue
        # Use urllib directly for the raw download (no auth needed for
        # public repos but we include the token so private repos work
        # equally).
        req = urllib.request.Request(download_url)
        req.add_header("User-Agent", _USER_AGENT)
        req.add_header("Authorization", f"Bearer {cfg.token}")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                (local_dir / name).write_bytes(resp.read())
                pulled += 1
        except urllib.error.URLError:
            continue  # skip this file, keep trying the rest
    return pulled, f"pulled {pulled} file(s) from {repo_path}"


def hydrate_all() -> dict[str, str]:
    """Hydrate every tracked data directory. Intended to be called once
    per session startup (wrapped in @st.cache_data for rate-limit safety).
    Returns a dict of dir_name -> status message."""
    results: dict[str, str] = {}
    for name in ("punctuality", "nps", "audit_cache"):
        path = DATA_DIR / name
        _, msg = hydrate_directory_from_github(path)
        results[name] = msg
    return results
