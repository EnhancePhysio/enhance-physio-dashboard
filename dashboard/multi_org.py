"""Multi-org helpers for tabs that aggregate data across all Cliniko
orgs (Albury-Wodonga + Mudgeeraba + Mulgrave).

v27.0 — Phase 1 of the multi-org rollout. Provides:
  * ``for_all_orgs(fetch_fn)`` — run a fetch across every configured
    org and combine results (either concat DataFrames or merge dicts).
  * ``for_selected_org(org_key, fetch_fn)`` — same but for one org only.
  * Cache keys are per-org so a "Mudgeeraba only" filter doesn't
    invalidate the Albury-Wodonga cache.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import pandas as pd

from dashboard.cliniko import (
    ClinikoClient, ClinikoError, ClinikoOrg,
    get_client_for_org, get_configured_orgs, load_organizations,
)


def for_all_orgs(
    fetch_fn: Callable[[ClinikoClient, ClinikoOrg], Any],
    orgs: Iterable[ClinikoOrg] | None = None,
    tag_column: str = "organization_key",
) -> pd.DataFrame:
    """Call ``fetch_fn(client, org)`` for every configured org and
    concatenate the results into a single DataFrame.

    Each row is tagged with the org's ``key`` in the ``tag_column``
    (default: ``organization_key``) so downstream code can group /
    filter by clinic.

    If ``fetch_fn`` returns None or an empty DataFrame for an org,
    that org contributes nothing (rather than crashing the aggregate).
    """
    orgs = list(orgs) if orgs is not None else get_configured_orgs()
    if not orgs:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for org in orgs:
        try:
            client = get_client_for_org(org)
            df = fetch_fn(client, org)
        except ClinikoError:
            # Missing key or bad org — skip silently. UI shows a
            # warning banner via `get_configured_orgs()` filtering.
            continue
        except Exception:
            # A single-org failure shouldn't break the whole aggregate
            continue
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
        if isinstance(df, pd.DataFrame):
            df = df.copy()
            df[tag_column] = org.key
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def for_selected_orgs(
    org_keys: Iterable[str],
    fetch_fn: Callable[[ClinikoClient, ClinikoOrg], Any],
    tag_column: str = "organization_key",
) -> pd.DataFrame:
    """Same as ``for_all_orgs`` but filtered to a list of org keys.

    Empty ``org_keys`` iterable means "no orgs selected" (returns
    empty DataFrame — not a bug, tab UI should handle this state).
    """
    selected = [o for o in load_organizations() if o.key in set(org_keys)]
    return for_all_orgs(fetch_fn, orgs=selected, tag_column=tag_column)


def resolve_clinic_filter(clinic_selection: str | None) -> list[str]:
    """Convert a sidebar clinic-filter value into a list of org keys.

    Convention:
      * ``None`` / ``""`` / ``"All"`` → every configured org
      * a valid org key                → just that one
      * anything else                  → empty list (nothing matches)

    Returned list is safe to pass to ``for_selected_orgs``.
    """
    orgs = get_configured_orgs()
    if not clinic_selection or clinic_selection.lower() in ("all", "all clinics"):
        return [o.key for o in orgs]
    for o in orgs:
        if clinic_selection == o.key or clinic_selection == o.name or clinic_selection == o.display_name:
            return [o.key]
    return []
