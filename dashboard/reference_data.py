"""Load long-lived reference data (practitioners, businesses, appointment types)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from dashboard.cliniko import ClinikoClient


def _str_id(value: Any) -> str | None:
    """Coerce a Cliniko id to a string, preserving None.

    Cliniko sometimes returns ids as numbers and sometimes as strings (newer
    snowflake-style ids overflow int64). `extract_linked_id` always returns
    strings, so every reference table must store ids as strings too —
    otherwise downstream merges on `practitioner_id` / `business_id` silently
    produce NaN and every metric collapses to zero.
    """
    if value is None:
        return None
    return str(value)


def load_practitioners(client: ClinikoClient) -> pd.DataFrame:
    rows = list(client.paginate("practitioners"))
    if not rows:
        return pd.DataFrame(columns=["id", "label", "first_name", "last_name", "active"])
    df = pd.DataFrame([{
        "id": _str_id(r.get("id")),
        "first_name": r.get("first_name", ""),
        "last_name": r.get("last_name", ""),
        "label": r.get("label") or f"{r.get('first_name','')} {r.get('last_name','')}".strip(),
        "active": r.get("active", True),
        "designation": r.get("designation", ""),
    } for r in rows])
    df["id"] = df["id"].astype(str)
    return df


def load_businesses(client: ClinikoClient) -> pd.DataFrame:
    rows = list(client.paginate("businesses"))
    if not rows:
        return pd.DataFrame(columns=["id", "label", "business_name"])
    df = pd.DataFrame([{
        "id": _str_id(r.get("id")),
        "label": r.get("label") or r.get("business_name") or str(r.get("id")),
        "business_name": r.get("business_name"),
    } for r in rows])
    df["id"] = df["id"].astype(str)
    return df


def load_appointment_types(client: ClinikoClient) -> pd.DataFrame:
    rows = list(client.paginate("appointment_types"))
    if not rows:
        return pd.DataFrame(columns=["id", "name", "duration_in_minutes"])
    df = pd.DataFrame([{
        "id": _str_id(r.get("id")),
        "name": r.get("name", ""),
        "duration_in_minutes": r.get("duration_in_minutes"),
        "online_bookings_enabled": r.get("online_bookings_enabled"),
    } for r in rows])
    df["id"] = df["id"].astype(str)
    return df


def extract_linked_id(obj: dict[str, Any] | None, key: str = "self") -> str | None:
    """Extract trailing ID from a Cliniko link structure.

    Cliniko IDs are strings (newer snowflake-style IDs can exceed int64),
    so we keep them as strings throughout.

    Handles both shapes:
        {"links": {"self": "https://.../practitioners/123"}}   # nested (most common)
        {"self": "https://.../practitioners/123"}              # flat
    """
    if not isinstance(obj, dict):
        return None
    inner = obj
    if "links" in obj and isinstance(obj["links"], dict):
        inner = obj["links"]
    url = inner.get(key)
    if not url and key != "self":
        url = inner.get("self")
    if not isinstance(url, str) or not url:
        return None
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or None
