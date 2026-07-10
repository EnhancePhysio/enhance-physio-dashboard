"""Commission Calculator (v26.10).

Monthly bonus calculation for clinicians. Formula (per practitioner per month):

    base_hours    = Σ (DOW occurrences × hours[DOW]) for the calendar month
    paid_hours    = base_hours − recurring_deductions − manual_adjustment
    base_pay      = paid_hours × hourly_rate           (pre-super, ex GST)
    base_super    = base_pay × super_rate              (12% → 12.5% Jul 2026)
    base_cost     = base_pay + base_super              (total clinic cost)

    revenue       = invoices raised in the month, EXCLUDING:
                      • product line items (no appointment link)
                      • DNA / cancellation fees (appt.did_not_arrive)
                      • room hire fees (appt type matches exclusion list)

    target_total  = revenue × commission_pct

    if target_total > base_cost:
        bonus_total_cost = target_total − base_cost
        bonus_pre_super  = bonus_total_cost / (1 + super_rate)
        ↑ this is what gets entered in Xero; Xero adds super on top.
    else:
        no bonus this month.

Only Cliniko data needed — no Xero integration in v1. Hourly rates
and commission percentages live in config/practitioner_pay.yml.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pytz
import yaml

from dashboard.cliniko import ClinikoClient
from dashboard.config import CONFIG_DIR, timezone_name
from dashboard.date_ranges import DateRange


PAY_CONFIG_PATH = CONFIG_DIR / "practitioner_pay.yml"

DOWS = ["Monday", "Tuesday", "Wednesday", "Thursday",
         "Friday", "Saturday", "Sunday"]


# Default room-hire / DNA / fee patterns to exclude from revenue.
# Override via settings.yml `commission.excluded_appointment_type_patterns`
# if Matt names them differently. Case-insensitive substring match.
_DEFAULT_EXCLUDED_APPT_TYPE_PATTERNS = [
    "room hire",
    "cancellation fee",
    "no show",
    "did not arrive",
    "dna fee",
]

# v26.11.2 — Cliniko often bills DNA/room-hire/products as line items on
# a regular appointment (so filtering by appt type misses them). These
# patterns match the invoice item's `name` field, case-insensitive.
_DEFAULT_EXCLUDED_ITEM_NAME_PATTERNS = [
    r"did not arrive",
    r"\bDNA\b",
    r"room hire",
    r"MLCOA",             # MLCOA Room Hire, MLCOA Video Assessment (both are non-service)
]

# Match against the invoice item's `code` field (Cliniko billing code).
_DEFAULT_EXCLUDED_ITEM_CODE_PATTERNS = [
    r"^DNA$",
    r"^MLCOARH$",
    r"^MLCOAVA$",
]


# -------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------
@dataclass
class PractitionerPay:
    name: str                                          # canonical name
    hourly_rate: float
    commission_pct: float
    hours_per_day: dict[str, float]                    # DOW → hours
    recurring_deductions: dict[str, float] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    region: str = "albury_wodonga"                     # v27.0 default

    @property
    def total_recurring_deduction_hours(self) -> float:
        return float(sum(self.recurring_deductions.values()))

    @property
    def all_names(self) -> list[str]:
        return [self.name] + list(self.aliases)


@dataclass
class PayConfig:
    super_rate_default: float
    super_rate_changes: list[dict[str, Any]]           # [{effective_from, rate}]
    default_recurring_deductions: dict[str, float]
    practitioners: list[PractitionerPay]

    def super_rate_for(self, target_date: date) -> float:
        """Return the SG rate applicable on ``target_date`` (e.g. for the
        first day of the calendar month being calculated)."""
        applicable = self.super_rate_default
        # Apply changes in chronological order — last applicable wins
        for change in sorted(self.super_rate_changes,
                              key=lambda c: c["effective_from"]):
            eff = change["effective_from"]
            if isinstance(eff, str):
                eff = date.fromisoformat(eff)
            if target_date >= eff:
                applicable = float(change["rate"])
        return applicable


def load_pay_config(path: Path | None = None) -> PayConfig:
    """Read config/practitioner_pay.yml into a structured PayConfig."""
    p = Path(path) if path else PAY_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    super_default = float(raw.get("super_rate_default", 0.12))
    super_changes = list(raw.get("super_rate_changes") or [])
    defaults = dict(raw.get("default_recurring_deductions") or {})
    # v26.11.4 — additional deductions for practitioners flagged as manager
    manager_extras = dict(raw.get("manager_extra_recurring_deductions") or {})
    practitioners: list[PractitionerPay] = []
    for name, body in (raw.get("practitioners") or {}).items():
        body = body or {}
        # If the practitioner block has its own recurring_deductions key,
        # use it verbatim (even if empty {} — that means "no deductions").
        # Otherwise inherit defaults.
        if "recurring_deductions" in body:
            ded = dict(body["recurring_deductions"] or {})
        else:
            ded = dict(defaults)
        # v26.11.4 — merge in manager extras when is_manager: true.
        # These add on top of whatever recurring_deductions were set,
        # so a manager with a custom deduction block still gets the
        # quarterly Matt catch-up.
        if body.get("is_manager"):
            for k, v in manager_extras.items():
                ded[k] = ded.get(k, 0.0) + float(v)
        practitioners.append(PractitionerPay(
            name=name,
            hourly_rate=float(body.get("hourly_rate", 0.0)),
            commission_pct=float(body.get("commission_pct", 0.0)),
            hours_per_day={dow: float(body.get("hours_per_day", {}).get(dow, 0) or 0)
                            for dow in DOWS},
            recurring_deductions=ded,
            aliases=list(body.get("aliases") or []),
            region=str(body.get("region") or "albury_wodonga"),
        ))
    return PayConfig(
        super_rate_default=super_default,
        super_rate_changes=super_changes,
        default_recurring_deductions=defaults,
        practitioners=practitioners,
    )


# -------------------------------------------------------------------
# Day-of-week month math
# -------------------------------------------------------------------
def dow_occurrences(year: int, month: int) -> dict[str, int]:
    """Return {DOW: count} for the calendar month.

    e.g. April 2026 → {Mon:4, Tue:4, Wed:5, Thu:4, Fri:4, Sat:4, Sun:4}
    """
    _, last_day = calendar.monthrange(year, month)
    counts = {dow: 0 for dow in DOWS}
    for d in range(1, last_day + 1):
        dow_idx = date(year, month, d).weekday()  # Mon=0 .. Sun=6
        counts[DOWS[dow_idx]] += 1
    return counts


def hours_for_month(year: int, month: int,
                     hours_per_day: dict[str, float]) -> float:
    """Multiply DOW counts by per-DOW hours and sum.

    Handles missing keys gracefully (treat as 0).
    """
    counts = dow_occurrences(year, month)
    return float(sum(counts[dow] * float(hours_per_day.get(dow, 0) or 0)
                      for dow in DOWS))


# -------------------------------------------------------------------
# Cliniko revenue fetch
# -------------------------------------------------------------------
def _fetch_active_appts_only(client: ClinikoClient,
                               dr: DateRange) -> pd.DataFrame:
    """Fetch /individual_appointments for the date range — single pass,
    no cancelled-pass round-trip. Cuts wall-clock time ~50% vs
    metrics.fetch_appointments since we filter cancelled out anyway.
    """
    from dashboard.metrics import _iter_appointments, _appt_row
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for a in _iter_appointments(client, dr):
        row = _appt_row(a)
        rid = row["id"]
        if rid is None or rid in seen:
            continue
        seen.add(rid)
        rows.append(row)
    cols = ["id", "patient_id", "practitioner_id", "business_id",
             "appointment_type_id", "starts_at", "ends_at",
             "appt_updated_at", "cancelled_at", "archived_at",
             "did_not_arrive", "cancellation_reason",
             "treatment_note_status", "patient_arrived"]
    return pd.DataFrame(rows, columns=cols)


def _month_range(year: int, month: int) -> DateRange:
    """DateRange for [first second of month, first second of next month)
    in Australia/Sydney local time, formatted to UTC ISO."""
    tz = pytz.timezone(timezone_name())
    start_local = tz.localize(datetime(year, month, 1))
    if month == 12:
        next_local = tz.localize(datetime(year + 1, 1, 1))
    else:
        next_local = tz.localize(datetime(year, month + 1, 1))
    return DateRange(start_local, next_local)


def _link_tail(url: Any) -> str | None:
    """Extract the trailing id from a Cliniko self-link URL."""
    if not isinstance(url, str) or not url:
        return None
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or None


def _is_truthy_str(v: Any) -> bool:
    """v26.10.5 — strict truthy-string check that handles pandas NaN/NaT
    correctly. The naive ``bool(value)`` flagged delivered appts as DNA
    because pandas missing-value sentinels are truthy in some contexts.

    Returns True only if v is a non-empty string. Everything else
    (None, NaN, NaT, "", False, 0) returns False.
    """
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        # Not pandas-compatible — fall through to string check
        pass
    if isinstance(v, str):
        return bool(v.strip())
    return False


def _is_truthy_bool(v: Any) -> bool:
    """v26.10.5 — strict bool check. True only if v is True (or "true").
    Handles None/NaN/NaT/""/False all → False."""
    if v is None or v is False:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return False


def _appt_id_from_invoice_item(item: dict[str, Any]) -> str | None:
    """[DEPRECATED in v26.10.4 — left for diagnostic compatibility]
    An invoice_item linked to an appointment can expose it as either
    a nested object {"id": ...} or as a self-URL under links.appointment.
    Try both before giving up."""
    for k in ("appointment", "individual_appointment"):
        v = item.get(k)
        if isinstance(v, dict):
            aid = v.get("id")
            if aid:
                return str(aid)
            url = (v.get("links") or {}).get("self") if isinstance(v.get("links"), dict) else None
            if isinstance(url, str):
                t = _link_tail(url)
                if t:
                    return t
    links = item.get("links")
    if isinstance(links, dict):
        url = links.get("appointment")
        if isinstance(url, str):
            t = _link_tail(url)
            if t:
                return t
    return None


def _invoice_id_from_invoice_item(item: dict[str, Any]) -> str | None:
    """Extract invoice_id from an invoice_item via its links.invoice URL."""
    inv = item.get("invoice")
    if isinstance(inv, dict):
        iid = inv.get("id")
        if iid:
            return str(iid)
        links = inv.get("links")
        if isinstance(links, dict):
            url = links.get("self")
            if isinstance(url, str):
                t = _link_tail(url)
                if t:
                    return t
    return None


def _appt_id_from_invoice(inv: dict[str, Any]) -> str | None:
    """Extract appointment_id from an invoice — Cliniko links it via
    `appointment` field (which may be a nested object or a links.self
    URL)."""
    for k in ("appointment", "individual_appointment"):
        v = inv.get(k)
        if isinstance(v, dict):
            aid = v.get("id")
            if aid:
                return str(aid)
            links = v.get("links")
            if isinstance(links, dict):
                url = links.get("self")
                if isinstance(url, str):
                    t = _link_tail(url)
                    if t:
                        return t
    links = inv.get("links")
    if isinstance(links, dict):
        for k in ("appointment", "individual_appointment"):
            url = links.get(k)
            if isinstance(url, str):
                t = _link_tail(url)
                if t:
                    return t
    return None


def _practitioner_id_from_invoice(inv: dict[str, Any]) -> str | None:
    """Cliniko exposes practitioner directly on the invoice via
    practitioner.links.self URL (per diag in v26.10.4)."""
    v = inv.get("practitioner")
    if isinstance(v, dict):
        pid = v.get("id")
        if pid:
            return str(pid)
        links = v.get("links")
        if isinstance(links, dict):
            url = links.get("self")
            if isinstance(url, str):
                t = _link_tail(url)
                if t:
                    return t
    links = inv.get("links")
    if isinstance(links, dict):
        url = links.get("practitioner")
        if isinstance(url, str):
            t = _link_tail(url)
            if t:
                return t
    return None


def _practitioner_id_from_appointment(appt: dict[str, Any]) -> str | None:
    for k in ("practitioner",):
        v = appt.get(k)
        if isinstance(v, dict):
            pid = v.get("id")
            if pid:
                return str(pid)
            url = (v.get("links") or {}).get("self") if isinstance(v.get("links"), dict) else None
            if isinstance(url, str):
                t = _link_tail(url)
                if t:
                    return t
    links = appt.get("links")
    if isinstance(links, dict):
        url = links.get("practitioner")
        if isinstance(url, str):
            t = _link_tail(url)
            if t:
                return t
    return None


def fetch_invoice_items_diagnostic(client: ClinikoClient,
                                     year: int, month: int,
                                     excluded_patterns: list[str] | None = None,
                                     ) -> dict[str, Any]:
    """Same fetch as the main function, but returns a dict of diagnostic
    counters and a sample raw shape so the UI can show exactly where
    items are being dropped. Used by the Commission tab's debug expander.
    """
    excluded_patterns = excluded_patterns or _DEFAULT_EXCLUDED_APPT_TYPE_PATTERNS
    excluded_res = [re.compile(p, re.IGNORECASE) for p in excluded_patterns]
    dr = _month_range(year, month)
    params = {"q[]": [f"created_at:>={dr.start_iso_utc}",
                       f"created_at:<{dr.end_iso_utc}"]}
    # v26.10.6 — only the active pass (we filter cancelled out anyway,
    # so skipping the cancelled pass cuts wall-clock ~50%).
    appts = _fetch_active_appts_only(client, dr)
    appt_map: dict[str, dict[str, Any]] = {}
    type_name_map: dict[str, str] = {}
    for _, row in appts.iterrows():
        aid_raw = row.get("id")
        aid = str(aid_raw) if not pd.isna(aid_raw) else ""
        if not aid:
            continue
        prac_raw = row.get("practitioner_id")
        type_raw = row.get("appointment_type_id")
        appt_map[aid] = {
            "practitioner_id": (str(prac_raw)
                                  if prac_raw is not None and not pd.isna(prac_raw)
                                  else ""),
            "did_not_arrive": _is_truthy_bool(row.get("did_not_arrive")),
            "archived":       _is_truthy_str(row.get("archived_at")),
            "cancelled":      _is_truthy_str(row.get("cancelled_at")),
            "appointment_type_id": (str(type_raw)
                                      if type_raw is not None and not pd.isna(type_raw)
                                      else ""),
        }
    from dashboard.reference_data import load_appointment_types
    types = load_appointment_types(client)
    if not types.empty:
        for _, r in types.iterrows():
            tid = str(r.get("id") or "")
            if tid:
                type_name_map[tid] = str(r.get("name") or "")

    # v26.10.5 — pull invoices, capture both appt and practitioner.
    invoices = fetch_invoices_for_month(client, year, month)
    invoice_meta: dict[str, dict[str, str]] = {}
    sample_invoice: dict[str, Any] | None = None
    for _, r in invoices.iterrows():
        iid = str(r.get("invoice_id") or "")
        if not iid:
            continue
        invoice_meta[iid] = {
            "appointment_id": str(r.get("appointment_id") or ""),
            "practitioner_id": str(r.get("practitioner_id_direct") or ""),
        }

    # Also grab a sample invoice raw shape for the UI
    try:
        for inv in client.paginate("invoices", params=params):
            if isinstance(inv, dict):
                sample_invoice = _safe_shape(inv)
                break
    except Exception:
        pass

    # v26.11.2 — item-name/code exclusion patterns (same as production)
    from dashboard.config import load_settings
    cfg2 = load_settings().get("commission", {}) or {}
    item_name_patterns = cfg2.get("excluded_item_name_patterns",
                                     _DEFAULT_EXCLUDED_ITEM_NAME_PATTERNS)
    item_code_patterns = cfg2.get("excluded_item_code_patterns",
                                     _DEFAULT_EXCLUDED_ITEM_CODE_PATTERNS)
    item_name_res_diag = [re.compile(p, re.IGNORECASE) for p in item_name_patterns]
    item_code_res_diag = [re.compile(p) for p in item_code_patterns]

    counters = {
        "scanned": 0,
        "no_invoice_link": 0,
        "invoice_not_in_window": 0,
        "invoice_no_appt": 0,
        "appt_outside_range": 0,
        "dna": 0,
        "archived": 0,
        "cancelled": 0,
        "excluded_appt_type": 0,
        "excluded_item_name": 0,
        "excluded_item_code": 0,
        "kept": 0,
    }
    first_raw: dict[str, Any] | None = None
    sample_kept: list[dict[str, Any]] = []
    err: str | None = None
    try:
        for it in client.paginate("invoice_items", params=params):
            counters["scanned"] += 1
            if not isinstance(it, dict):
                continue
            if first_raw is None:
                first_raw = _safe_shape(it)
            inv_id = _invoice_id_from_invoice_item(it)
            if not inv_id:
                counters["no_invoice_link"] += 1
                continue
            inv_info = invoice_meta.get(inv_id)
            if inv_info is None:
                counters["invoice_not_in_window"] += 1
                continue
            aid = inv_info.get("appointment_id") or ""
            prac_id = inv_info.get("practitioner_id") or ""
            if not aid:
                counters["invoice_no_appt"] += 1
                continue
            meta = appt_map.get(aid)
            if meta is None:
                counters["appt_outside_range"] += 1
                # but if invoice has practitioner directly, we'd still
                # keep it in production code — diagnostic skips for
                # clarity
                continue
            # Granular drop reasons
            if meta["did_not_arrive"]:
                counters["dna"] += 1
                continue
            if meta["archived"]:
                counters["archived"] += 1
                continue
            if meta["cancelled"]:
                counters["cancelled"] += 1
                continue
            type_name = type_name_map.get(meta["appointment_type_id"], "")
            if any(r.search(type_name) for r in excluded_res):
                counters["excluded_appt_type"] += 1
                continue
            # v26.11.2 — item-name / item-code exclusion
            item_name_val = str(it.get("name") or "")
            item_code_val = str(it.get("code") or "")
            if any(r.search(item_name_val) for r in item_name_res_diag):
                counters["excluded_item_name"] += 1
                continue
            if item_code_val and any(r.search(item_code_val) for r in item_code_res_diag):
                counters["excluded_item_code"] += 1
                continue
            # Ex-tax amount (matches Cliniko's "Amount ex. tax" report column)
            price = it.get("net_price")
            qty = it.get("quantity", 1)
            if price is None:
                price = it.get("unit_price") or 0
            try:
                amount = float(price) * float(qty or 1)
            except (TypeError, ValueError):
                continue
            counters["kept"] += 1
            if len(sample_kept) < 5:
                sample_kept.append({
                    "invoice_id": inv_id,
                    "appointment_id": aid,
                    "practitioner_id": meta["practitioner_id"],
                    "appt_type_name": type_name,
                    "item_name": item_name_val[:60],
                    "item_code": item_code_val,
                    "amount_ex_tax": round(amount, 2),
                })
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    return {
        "counters": counters,
        "first_raw_shape": first_raw,
        "first_invoice_shape": sample_invoice,
        "sample_kept_items": sample_kept,
        "invoices_fetched": len(invoice_to_appt),
        "appts_fetched": len(appt_map),
        "appt_types_fetched": len(type_name_map),
        "error": err,
    }


def _safe_shape(obj: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Return a structural-only preview of a Cliniko payload — keys and
    types, not values — so we can debug shape without leaking PHI."""
    if depth >= max_depth:
        return f"<{type(obj).__name__}>"
    if isinstance(obj, dict):
        return {k: _safe_shape(v, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        if not obj:
            return []
        return [_safe_shape(obj[0], depth + 1, max_depth), f"… +{len(obj)-1} more"]
    if isinstance(obj, str):
        # Show short strings (likely IDs / status codes), redact long text.
        return obj if len(obj) < 40 else f"<str len={len(obj)}>"
    return type(obj).__name__


def fetch_invoices_for_month(client: ClinikoClient,
                               year: int, month: int) -> pd.DataFrame:
    """Pull /invoices created in the month with their appointment +
    practitioner links intact. Excludes archived/deleted invoices.

    v26.10.5 — also captures practitioner_id directly from the invoice
    (Cliniko exposes it as a top-level link), so we don't have to detour
    through the appointment for attribution. Faster + works even when
    the appt fetch missed the appointment.

    Returned DataFrame columns:
      invoice_id, appointment_id, practitioner_id_direct, status, total
    """
    dr = _month_range(year, month)
    params = {"q[]": [f"created_at:>={dr.start_iso_utc}",
                       f"created_at:<{dr.end_iso_utc}"]}
    rows: list[dict[str, Any]] = []
    try:
        for inv in client.paginate("invoices", params=params):
            if not isinstance(inv, dict):
                continue
            if inv.get("archived_at") or inv.get("deleted_at"):
                continue
            # Status can be int or string in Cliniko payloads
            status_raw = inv.get("status_description") or inv.get("status") or ""
            status = str(status_raw).lower()
            if status in ("cancelled", "void"):
                continue
            appt_id = _appt_id_from_invoice(inv)
            prac_id = _practitioner_id_from_invoice(inv)
            total = inv.get("total_amount")
            if total is None:
                total = inv.get("total_including_tax")
            if total is None:
                total = inv.get("total") or 0
            try:
                total = float(total)
            except (TypeError, ValueError):
                continue
            rows.append({
                "invoice_id": str(inv.get("id") or ""),
                "appointment_id": appt_id or "",
                "practitioner_id_direct": prac_id or "",
                "status": status,
                "total": total,
            })
    except Exception:
        pass
    return pd.DataFrame(rows, columns=[
        "invoice_id", "appointment_id", "practitioner_id_direct",
        "status", "total",
    ])


def fetch_invoice_items_for_month(client: ClinikoClient,
                                    year: int, month: int,
                                    excluded_patterns: list[str] | None = None,
                                    ) -> pd.DataFrame:
    """v26.10.4 — joins /invoice_items → /invoices → /appointments to
    attribute revenue to a practitioner.

    Cliniko's invoice_item doesn't expose an appointment link directly;
    the path goes invoice_item.invoice → invoice.appointment. So we
    pull /invoices once for the month (small set), build an
    invoice_id → appointment_id map, then stream /invoice_items and
    look up each item's invoice → appt → practitioner.

    Excludes (per Matt's spec):
      * items with no linked appointment   → products / desk fees
      * items where appt.did_not_arrive    → DNA / cancellation fees
      * items where appt is cancelled or archived
      * items where appt's appointment_type name matches an excluded pattern
        (room hire, DNA fee, no-show, etc — configurable)

    The remaining items are the appointment-based revenue we credit to
    the practitioner who delivered the appointment.
    """
    excluded_patterns = excluded_patterns or _DEFAULT_EXCLUDED_APPT_TYPE_PATTERNS
    excluded_res = [re.compile(p, re.IGNORECASE) for p in excluded_patterns]

    dr = _month_range(year, month)
    params = {"q[]": [f"created_at:>={dr.start_iso_utc}",
                       f"created_at:<{dr.end_iso_utc}"]}

    # ---- Fetch only the ACTIVE pass of appointments ---------------
    # v26.10.6 — fetch_appointments does two passes (active + cancelled).
    # For commission we filter out cancelled anyway, so we skip pass 2
    # entirely. Cuts wall-clock time roughly in half.
    appts = _fetch_active_appts_only(client, dr)

    # v26.11.2 — pre-compile item-name / item-code exclusion regexes
    # (Cliniko bills DNA/room-hire/products as line items on regular
    # appts, so appointment-type filtering alone misses them).
    from dashboard.config import load_settings
    cfg = load_settings().get("commission", {}) or {}
    item_name_patterns = cfg.get("excluded_item_name_patterns",
                                    _DEFAULT_EXCLUDED_ITEM_NAME_PATTERNS)
    item_code_patterns = cfg.get("excluded_item_code_patterns",
                                    _DEFAULT_EXCLUDED_ITEM_CODE_PATTERNS)
    item_name_res = [re.compile(p, re.IGNORECASE) for p in item_name_patterns]
    item_code_res = [re.compile(p) for p in item_code_patterns]
    use_ex_tax = bool(cfg.get("use_ex_tax", True))

    # Map appt_id → (practitioner_id, did_not_arrive, archived, cancelled,
    # appointment_type_id) for fast lookup.
    # v26.10.5 — use NaN-safe truthiness helpers because pandas iterrows()
    # converts None → NaN, and bool(NaN) == True silently flagged every
    # delivered appt as archived/cancelled. The new helpers correctly
    # treat NaN/NaT/None/"" as "not set".
    appt_map: dict[str, dict[str, Any]] = {}
    for _, row in appts.iterrows():
        aid_raw = row.get("id")
        aid = str(aid_raw) if not pd.isna(aid_raw) else ""
        if not aid:
            continue
        prac_raw = row.get("practitioner_id")
        type_raw = row.get("appointment_type_id")
        appt_map[aid] = {
            "practitioner_id": (str(prac_raw)
                                  if prac_raw is not None and not pd.isna(prac_raw)
                                  else ""),
            "did_not_arrive": _is_truthy_bool(row.get("did_not_arrive")),
            "archived":       _is_truthy_str(row.get("archived_at")),
            "cancelled":      _is_truthy_str(row.get("cancelled_at")),
            "appointment_type_id": (str(type_raw)
                                      if type_raw is not None and not pd.isna(type_raw)
                                      else ""),
        }

    # Map appointment_type_id → name (so we can match exclusion patterns)
    from dashboard.reference_data import load_appointment_types
    types = load_appointment_types(client)
    type_name_map: dict[str, str] = {}
    if not types.empty:
        for _, r in types.iterrows():
            tid = str(r.get("id") or "")
            if tid:
                type_name_map[tid] = str(r.get("name") or "")

    # v26.10.5 — pull /invoices for the month, build invoice_id → {appt_id,
    # practitioner_id_direct}. Practitioner comes from the invoice itself
    # (faster + more reliable than going through appointments).
    invoices = fetch_invoices_for_month(client, year, month)
    invoice_meta: dict[str, dict[str, str]] = {}
    for _, r in invoices.iterrows():
        iid = str(r.get("invoice_id") or "")
        if not iid:
            continue
        invoice_meta[iid] = {
            "appointment_id": str(r.get("appointment_id") or ""),
            "practitioner_id": str(r.get("practitioner_id_direct") or ""),
        }

    rows: list[dict[str, Any]] = []
    try:
        for it in client.paginate("invoice_items", params=params):
            if not isinstance(it, dict):
                continue
            inv_id = _invoice_id_from_invoice_item(it)
            if not inv_id:
                continue
            inv_info = invoice_meta.get(inv_id)
            if inv_info is None:
                continue  # invoice not in our window, skip
            aid = inv_info.get("appointment_id") or ""
            prac_id = inv_info.get("practitioner_id") or ""
            if not aid:
                continue  # standalone product / desk-fee invoice
            meta = appt_map.get(aid)
            if meta is not None:
                # Use appt for DNA / type filtering. Trust appt's
                # practitioner_id only if the direct one was empty.
                if meta["did_not_arrive"] or meta["archived"] or meta["cancelled"]:
                    continue
                if not prac_id:
                    prac_id = meta["practitioner_id"]
                type_name = type_name_map.get(meta["appointment_type_id"], "")
                if any(r.search(type_name) for r in excluded_res):
                    continue
            else:
                # Appointment not in our fetched window — happens for
                # late-billed appts (delivered last month, invoiced this
                # month). We can still attribute via the invoice's direct
                # practitioner link, but skip the type-name filter.
                if not prac_id:
                    continue
                type_name = ""

            # v26.11.2 — item-level exclusion (DNA fee, MLCOA room hire,
            # products billed on regular appts). Filter by name/code
            # before summing.
            item_name = str(it.get("name") or "")
            item_code = str(it.get("code") or "")
            if any(r.search(item_name) for r in item_name_res):
                continue
            if item_code and any(r.search(item_code) for r in item_code_res):
                continue

            # v26.11.2 — use net_price (ex GST) not total_including_tax.
            # Matches the "Amount (ex. tax)" column in Cliniko's
            # Practitioner-Revenue-by-raised-invoices report. Toggleable
            # via commission.use_ex_tax in settings.yml.
            if use_ex_tax:
                # net_price = per-unit ex-tax; multiply by quantity
                price = it.get("net_price")
                qty = it.get("quantity", 1)
                if price is None:
                    price = it.get("unit_price") or 0
                try:
                    amount = float(price) * float(qty or 1)
                except (TypeError, ValueError):
                    amount = None
                if amount is None:
                    # Fall back to inc-tax minus tax_amount if net_price is missing
                    inc = it.get("total_including_tax") or it.get("total") or 0
                    tax = it.get("tax_amount") or 0
                    try:
                        amount = float(inc) - float(tax)
                    except (TypeError, ValueError):
                        continue
            else:
                amount = it.get("total_including_tax")
                if amount is None:
                    amount = it.get("total")
                if amount is None:
                    price = it.get("net_price") or it.get("unit_price") or 0
                    qty = it.get("quantity") or 1
                    try:
                        amount = float(price) * float(qty)
                    except (TypeError, ValueError):
                        amount = 0
                try:
                    amount = float(amount)
                except (TypeError, ValueError):
                    continue
            rows.append({
                "invoice_item_id": str(it.get("id") or ""),
                "invoice_id": inv_id,
                "appointment_id": aid,
                "practitioner_id": prac_id,
                "appointment_type_name": type_name,
                "item_name": str(it.get("name") or ""),
                "amount": amount,
            })
    except Exception:
        # If invoice_items endpoint fails, return what we've collected so
        # the UI can still render a partial table with a banner.
        pass

    return pd.DataFrame(rows, columns=[
        "invoice_item_id", "invoice_id", "appointment_id", "practitioner_id",
        "appointment_type_name", "item_name", "amount",
    ])


def revenue_per_practitioner(client: ClinikoClient,
                               year: int, month: int,
                               excluded_patterns: list[str] | None = None,
                               ) -> dict[str, float]:
    """Sum filtered invoice-item amounts by practitioner_id."""
    items = fetch_invoice_items_for_month(client, year, month,
                                            excluded_patterns)
    if items.empty:
        return {}
    by_prac = items.groupby("practitioner_id")["amount"].sum()
    return {str(pid): float(amt) for pid, amt in by_prac.items()}


# -------------------------------------------------------------------
# Commission calculation
# -------------------------------------------------------------------
@dataclass
class CommissionResult:
    name: str
    cliniko_practitioner_id: str | None
    base_hours: float
    deduction_hours: float
    paid_hours: float
    hourly_rate: float
    base_pay_pre_super: float
    base_super: float
    base_cost: float
    revenue: float
    commission_pct: float
    target_total: float
    bonus_total_cost: float
    bonus_pre_super: float
    total_clinic_cost: float
    super_rate: float

    def as_row(self) -> dict[str, Any]:
        """Flat dict for DataFrame rendering."""
        return {
            "Practitioner": self.name,
            "Base hours": round(self.base_hours, 2),
            "Deductions (hrs)": round(self.deduction_hours, 2),
            "Paid hours": round(self.paid_hours, 2),
            "Rate ($/h)": round(self.hourly_rate, 2),
            "Base pay (pre-super)": round(self.base_pay_pre_super, 2),
            "Super on base": round(self.base_super, 2),
            "Base cost (incl super)": round(self.base_cost, 2),
            "Revenue invoiced": round(self.revenue, 2),
            "Commission %": f"{self.commission_pct * 100:.0f}%",
            "Target total cost": round(self.target_total, 2),
            "Bonus (pre-super, enter in Xero)": round(self.bonus_pre_super, 2),
            "Total clinic cost": round(self.total_clinic_cost, 2),
        }


def compute_commission_for_practitioner(
    prac: PractitionerPay,
    year: int, month: int,
    revenue: float,
    super_rate: float,
    manual_adjustment_hours: float = 0.0,
    paid_hours_override: float | None = None,
    cliniko_practitioner_id: str | None = None,
) -> CommissionResult:
    """Apply the formula end-to-end for one practitioner.

    v26.11.3 — added ``paid_hours_override``. If provided (non-None,
    positive), it REPLACES the computed paid_hours entirely, bypassing
    both recurring deductions and manual_adjustment_hours. Intended for
    pasting actual Xero timesheet hours which capture real-world sick
    days / short weeks that the theoretical schedule can't predict.
    """
    base_hours = hours_for_month(year, month, prac.hours_per_day)
    deduction_hours = prac.total_recurring_deduction_hours + float(manual_adjustment_hours or 0)
    if paid_hours_override is not None and paid_hours_override > 0:
        paid_hours = float(paid_hours_override)
    else:
        paid_hours = max(0.0, base_hours - deduction_hours)

    base_pay = paid_hours * prac.hourly_rate
    base_super = base_pay * super_rate
    base_cost = base_pay + base_super

    target_total = revenue * prac.commission_pct
    if target_total > base_cost:
        bonus_total_cost = target_total - base_cost
        bonus_pre_super = bonus_total_cost / (1 + super_rate)
    else:
        bonus_total_cost = 0.0
        bonus_pre_super = 0.0

    total_clinic_cost = base_cost + bonus_total_cost
    return CommissionResult(
        name=prac.name,
        cliniko_practitioner_id=cliniko_practitioner_id,
        base_hours=base_hours,
        deduction_hours=deduction_hours,
        paid_hours=paid_hours,
        hourly_rate=prac.hourly_rate,
        base_pay_pre_super=base_pay,
        base_super=base_super,
        base_cost=base_cost,
        revenue=revenue,
        commission_pct=prac.commission_pct,
        target_total=target_total,
        bonus_total_cost=bonus_total_cost,
        bonus_pre_super=bonus_pre_super,
        total_clinic_cost=total_clinic_cost,
        super_rate=super_rate,
    )


# -------------------------------------------------------------------
# Practitioner-name → Cliniko-id resolver
# -------------------------------------------------------------------
def _normalise(name: str) -> str:
    """Lowercase + collapse whitespace + drop punctuation, for fuzzy match."""
    return re.sub(r"[^a-z0-9 ]+", " ",
                   re.sub(r"\s+", " ", name.lower())).strip()


def resolve_cliniko_ids(pay_config: PayConfig,
                          cliniko_practitioners: pd.DataFrame
                          ) -> dict[str, str | None]:
    """Map each PayConfig practitioner name → Cliniko practitioner_id.

    Tries the canonical name first, then each alias. Falls back to a
    first/last-name token match. Returns {pay_config_name: cliniko_id_or_None}.

    v26.10.2 — added first-name-only fallback (so "Patrick Stow" matches
    Cliniko's "Patrick" even if the last-name display is missing) and
    also matches on any token overlap of length ≥ 2.
    """
    out: dict[str, str | None] = {}
    if cliniko_practitioners is None or cliniko_practitioners.empty:
        return {p.name: None for p in pay_config.practitioners}

    # Build a normalised lookup over Cliniko practitioners
    cliniko_by_norm: dict[str, str] = {}
    cliniko_first_name_to_ids: dict[str, list[str]] = {}
    cliniko_last_name_to_ids: dict[str, list[str]] = {}
    for _, row in cliniko_practitioners.iterrows():
        cid = str(row.get("id") or "")
        if not cid:
            continue
        for col in ("label", "display_name", "first_name", "name"):
            v = row.get(col)
            if isinstance(v, str) and v.strip():
                cliniko_by_norm[_normalise(v)] = cid
        first = (row.get("first_name") or "").strip()
        last = (row.get("last_name") or "").strip()
        if first or last:
            cliniko_by_norm[_normalise(f"{first} {last}")] = cid
            cliniko_by_norm[_normalise(f"{last} {first}")] = cid
        if first:
            cliniko_first_name_to_ids.setdefault(_normalise(first), []).append(cid)
        if last:
            cliniko_last_name_to_ids.setdefault(_normalise(last), []).append(cid)

    for prac in pay_config.practitioners:
        cid: str | None = None
        # 1) exact (normalised) match against canonical or alias
        for n in prac.all_names:
            norm = _normalise(n)
            if norm in cliniko_by_norm:
                cid = cliniko_by_norm[norm]
                break

        # 2) token-set fallback (handles "Last, First" formats etc)
        if cid is None:
            tokens = set(_normalise(prac.name).split())
            best_match_score = 0
            for known_norm, known_cid in cliniko_by_norm.items():
                known_tokens = set(known_norm.split())
                overlap = len(tokens & known_tokens)
                if overlap >= 2 and overlap > best_match_score:
                    cid = known_cid
                    best_match_score = overlap

        # 3) last-name fallback — unique last names only
        if cid is None:
            tokens = _normalise(prac.name).split()
            if len(tokens) >= 2:
                last_tok = tokens[-1]
                ids = cliniko_last_name_to_ids.get(last_tok, [])
                if len(ids) == 1:
                    cid = ids[0]

        # 4) first-name fallback — unique first names only (incl aliases)
        if cid is None:
            for n in prac.all_names:
                tokens = _normalise(n).split()
                if not tokens:
                    continue
                first_tok = tokens[0]
                ids = cliniko_first_name_to_ids.get(first_tok, [])
                if len(ids) == 1:
                    cid = ids[0]
                    break

        out[prac.name] = cid
    return out


# -------------------------------------------------------------------
# Top-level entry-point used by the UI
# -------------------------------------------------------------------
def compute_commission_table(
    client: ClinikoClient,
    year: int, month: int,
    pay_config: PayConfig | None = None,
    cliniko_practitioners: pd.DataFrame | None = None,
    manual_adjustments: dict[str, float] | None = None,
    paid_hours_overrides: dict[str, float] | None = None,
) -> tuple[list[CommissionResult], dict[str, str | None]]:
    """End-to-end: returns (results, name_to_cliniko_id_map).

    ``manual_adjustments`` is a {practitioner_name: extra_deduction_hours}
    dict so the UI can let Matt override per-month.
    """
    pay_config = pay_config or load_pay_config()
    manual_adjustments = manual_adjustments or {}

    # Apply super rate that's effective at the START of the calculation month
    super_rate = pay_config.super_rate_for(date(year, month, 1))

    revenue_map = revenue_per_practitioner(client, year, month)

    # Map pay-config name → Cliniko id (for the revenue join)
    if cliniko_practitioners is None:
        from dashboard.reference_data import load_practitioners
        cliniko_practitioners = load_practitioners(client)
    id_map = resolve_cliniko_ids(pay_config, cliniko_practitioners)

    paid_hours_overrides = paid_hours_overrides or {}
    results: list[CommissionResult] = []
    for prac in pay_config.practitioners:
        cid = id_map.get(prac.name)
        revenue = float(revenue_map.get(cid, 0.0)) if cid else 0.0
        manual_h = float(manual_adjustments.get(prac.name, 0.0))
        override = paid_hours_overrides.get(prac.name)
        override_val = float(override) if override else None
        results.append(compute_commission_for_practitioner(
            prac, year, month,
            revenue=revenue,
            super_rate=super_rate,
            paid_hours_override=override_val,
            manual_adjustment_hours=manual_h,
            cliniko_practitioner_id=cid,
        ))
    return results, id_map
