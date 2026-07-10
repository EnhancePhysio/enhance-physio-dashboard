"""Cliniko REST API client — auth, rate limiting, pagination.

v27.0 — Multi-org support. See ``load_organizations()`` and
``get_client_for_org()`` for the new APIs. Existing single-org callers
(``ClinikoClient()`` / ``get_client()``) still work — they now resolve
to the org marked ``default: true`` in settings.yml.
"""
from __future__ import annotations

import base64
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx  # noqa: F401  — runtime import inside __init__

from dashboard.config import (
    _ensure_env,
    cliniko_api_key,
    cliniko_shard,
    cliniko_user_agent,
    load_settings,
)


class ClinikoError(RuntimeError):
    """Raised for non-retryable Cliniko API errors."""


# -------------------------------------------------------------------
# v27.0 — Multi-org configuration
# -------------------------------------------------------------------
@dataclass(frozen=True)
class ClinikoOrg:
    """One Cliniko organisation (business tenant)."""
    key: str                    # short slug used as dict key / cache key
    name: str                   # short display label ("Albury/Wodonga")
    display_name: str           # full clinic name for KPI titles etc
    api_key_env: str            # env var / Streamlit secret name
    shard: str                  # e.g. "au1"
    is_default: bool = False

    @property
    def api_key(self) -> str | None:
        """Read the API key at call-time so secret rotations take effect
        without a restart."""
        _ensure_env()
        return os.environ.get(self.api_key_env) or None


def load_organizations() -> list[ClinikoOrg]:
    """Return the list of configured Cliniko orgs from settings.yml.

    Falls back to a single "default" org built from the legacy
    ``CLINIKO_API_KEY`` env var if the ``cliniko_organizations`` block
    is missing — this keeps v26.x setups working while v27.x is being
    rolled out.
    """
    settings = load_settings()
    raw = settings.get("cliniko_organizations") or []
    if not raw:
        # Legacy single-org fallback
        return [ClinikoOrg(
            key="default",
            name="Default",
            display_name="Default",
            api_key_env="CLINIKO_API_KEY",
            shard=cliniko_shard(),
            is_default=True,
        )]
    orgs: list[ClinikoOrg] = []
    for r in raw:
        orgs.append(ClinikoOrg(
            key=str(r.get("key") or "").strip(),
            name=str(r.get("name") or r.get("key") or ""),
            display_name=str(r.get("display_name") or r.get("name") or ""),
            api_key_env=str(r.get("api_key_env") or "CLINIKO_API_KEY"),
            shard=str(r.get("shard") or cliniko_shard()),
            is_default=bool(r.get("default", False)),
        ))
    if not any(o.is_default for o in orgs) and orgs:
        # If nobody's marked default, promote the first one
        orgs[0] = ClinikoOrg(**{**orgs[0].__dict__, "is_default": True})
    return orgs


def get_default_org() -> ClinikoOrg:
    """The org used by legacy single-org call sites (whichever is
    marked ``default: true`` in settings.yml)."""
    for o in load_organizations():
        if o.is_default:
            return o
    # Should never happen — load_organizations always ensures one default
    return load_organizations()[0]


def get_client_for_org(org: ClinikoOrg | str) -> "ClinikoClient":
    """Build a client bound to a specific org's key + shard.

    Accepts either a ClinikoOrg or a string org key. Raises
    ClinikoError if the org's API key isn't set in the environment.
    """
    if isinstance(org, str):
        matches = [o for o in load_organizations() if o.key == org]
        if not matches:
            raise ClinikoError(f"Unknown Cliniko org key: {org!r}")
        org = matches[0]
    if not org.api_key:
        raise ClinikoError(
            f"Cliniko API key for org '{org.key}' isn't set. "
            f"Add {org.api_key_env} to Streamlit Cloud → Secrets."
        )
    return ClinikoClient(api_key=org.api_key, shard=org.shard)


def get_configured_orgs() -> list[ClinikoOrg]:
    """Return only orgs whose API key IS set in the environment.

    Useful for the sidebar filter — we don't want to offer a clinic
    the app can't actually fetch from. Silently drops orgs with
    missing secrets so the sidebar doesn't error out mid-render.
    """
    return [o for o in load_organizations() if o.api_key]


class _RateLimiter:
    """Simple rolling-window limiter: N requests per 60 seconds."""

    def __init__(self, max_per_minute: int) -> None:
        self.max = max_per_minute
        self.window = 60.0
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        cutoff = now - self.window
        while self._calls and self._calls[0] < cutoff:
            self._calls.popleft()
        if len(self._calls) >= self.max:
            sleep_for = self.window - (now - self._calls[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
            return self.acquire()
        self._calls.append(now)


class ClinikoClient:
    """Thin Cliniko API v1 client.

    Usage:
        client = ClinikoClient()
        for appt in client.paginate("individual_appointments",
                                    params={"q[]": "starts_at:>=2026-01-01T00:00:00Z"}):
            ...
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 shard: Optional[str] = None,
                 user_agent: Optional[str] = None,
                 timeout: float = 30.0) -> None:
        import httpx  # deferred so smoke tests can import this module without httpx
        settings = load_settings()
        self.api_key = api_key or cliniko_api_key()
        self.shard = shard or cliniko_shard()
        self.user_agent = user_agent or cliniko_user_agent()
        self.base_url = settings["cliniko"]["base_url_template"].format(shard=self.shard)
        self.page_size = settings["cliniko"]["page_size"]
        self._rate = _RateLimiter(settings["cliniko"]["requests_per_minute"])

        token = base64.b64encode(f"{self.api_key}:".encode()).decode()
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Basic {token}",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ClinikoClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---------------------------------------------------------------
    # Low-level
    # ---------------------------------------------------------------
    def _request(self, method: str, path: str,
                 params: Optional[dict[str, Any]] = None,
                 max_retries: int = 5) -> dict[str, Any]:
        import httpx  # local to keep module import cheap
        self._rate.acquire()
        url = path if path.startswith("http") else path
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._client.request(method, url, params=params)
            except httpx.HTTPError as e:
                if attempt >= max_retries:
                    raise ClinikoError(f"Network error after {attempt} tries: {e}") from e
                time.sleep(min(2 ** attempt, 30))
                continue

            if resp.status_code == 429:
                # Respect Retry-After if present, else back off
                retry_after = float(resp.headers.get("Retry-After", "5"))
                time.sleep(retry_after)
                if attempt >= max_retries:
                    raise ClinikoError("Rate-limited repeatedly; giving up")
                continue
            if 500 <= resp.status_code < 600:
                if attempt >= max_retries:
                    raise ClinikoError(
                        f"Server error {resp.status_code} on {method} {url}: {resp.text[:200]}"
                    )
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code >= 400:
                raise ClinikoError(
                    f"Cliniko {resp.status_code} on {method} {url}: {resp.text[:400]}"
                )
            return resp.json()

    # ---------------------------------------------------------------
    # Pagination
    # ---------------------------------------------------------------
    def paginate(self, resource: str,
                 params: Optional[dict[str, Any]] = None,
                 max_pages: int = 500) -> Iterator[dict[str, Any]]:
        """Yield every record across all pages for the given resource.

        resource is either a path like 'individual_appointments' or a full URL
        (used internally for `next.href` continuation).

        `max_pages` is a hard safety cap — no Cliniko query should ever
        return more than ~50k records in a single call, and if one tries to
        (because of a server bug or a pathological filter), we want the
        dashboard to surface an error rather than hang indefinitely.
        """
        p = dict(params or {})
        p.setdefault("per_page", self.page_size)

        path = f"/{resource}" if not resource.startswith("http") and not resource.startswith("/") else resource

        pages = 0
        while True:
            pages += 1
            if pages > max_pages:
                # Don't raise — just stop. Raising would kill the whole
                # dashboard; stopping silently means metrics are computed
                # against a truncated dataset, which is visible in the
                # diagnostics tab.
                return
            data = self._request("GET", path, params=p)
            # Cliniko envelope: {"<resource>":[...], "links":{"self":"...", "next":"..."}, "total_entries":N}
            key = self._resource_key(data, resource)
            if key is None:
                return
            for item in data.get(key, []):
                yield item
            next_link = (data.get("links") or {}).get("next")
            if not next_link:
                return
            # Use next link as absolute URL; drop query params we originally set
            path = next_link
            p = None

    @staticmethod
    def _resource_key(data: dict[str, Any], resource: str) -> Optional[str]:
        """Find the envelope key holding the list."""
        # Envelope key normally matches the final path segment
        candidate = resource.rstrip("/").rsplit("/", 1)[-1]
        if candidate in data:
            return candidate
        # Otherwise find the first list-valued key (ignoring 'links')
        for k, v in data.items():
            if isinstance(v, list):
                return k
        return None

    # ---------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------
    def get(self, resource: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        path = f"/{resource}" if not resource.startswith("/") else resource
        return self._request("GET", path, params=params)

    def list_all(self, resource: str,
                 params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        return list(self.paginate(resource, params))

    # ---------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------
    def ping(self) -> dict[str, Any]:
        """Call /account — cheap way to verify key + shard + network."""
        return self.get("account")


# -------------------------------------------------------------------
# Query helpers
# -------------------------------------------------------------------
def starts_at_range_params(start_iso: str, end_iso: str) -> dict[str, list[str]]:
    """Cliniko uses repeated q[] params. httpx serialises list values as repeats."""
    return {
        "q[]": [f"starts_at:>={start_iso}", f"starts_at:<{end_iso}"],
    }


def updated_at_range_params(start_iso: str, end_iso: str) -> dict[str, list[str]]:
    return {
        "q[]": [f"updated_at:>={start_iso}", f"updated_at:<{end_iso}"],
    }
