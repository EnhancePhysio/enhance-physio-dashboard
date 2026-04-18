"""Cliniko REST API client — auth, rate limiting, pagination."""
from __future__ import annotations

import base64
import time
from collections import deque
from typing import Any, Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx  # noqa: F401  — runtime import inside __init__

from .config import (
    cliniko_api_key,
    cliniko_shard,
    cliniko_user_agent,
    load_settings,
)


class ClinikoError(RuntimeError):
    """Raised for non-retryable Cliniko API errors."""


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
                 params: Optional[dict[str, Any]] = None) -> Iterator[dict[str, Any]]:
        """Yield every record across all pages for the given resource.

        resource is either a path like 'individual_appointments' or a full URL
        (used internally for `next.href` continuation).
        """
        p = dict(params or {})
        p.setdefault("per_page", self.page_size)

        path = f"/{resource}" if not resource.startswith("http") and not resource.startswith("/") else resource

        while True:
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
