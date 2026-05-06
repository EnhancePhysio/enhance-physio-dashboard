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
        practitioners.append(PractitionerPay(
            name=name,
            hourly_rate=float(body.get("hourly_rate", 0.0)),
            commission_pct=float(body.get("commission_pct", 0.0)),
            hours_per_day={dow: float(body.get("hours_per_day", {}).get(dow, 0) or 0)
                            for dow in DOWS},
            recurring_deductions=ded,
            aliases=list(body.get("aliases") or []),
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


def _appt_id_from_invoice_item(item: dict[str, Any]) -> str | None:
    """An invoice_item linked to an appointment can expose it as either
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


def fetch_invoice_items_for_month(client: ClinikoClient,
                                    year: int, month: int,
                                    excluded_patterns: list[str] | None = None,
                                    ) -> pd.DataFrame:
    """Pull every /invoice_items in the month, drop excluded, return a
    DataFrame keyed by item_id with practitioner_id + amount.

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

    # ---- Fetch every appointment id we'll need, in two passes ----
    # We need delivered/non-cancelled/non-DNA appts in the same month
    # window. Reuse the existing fetch_appointments helper which does
    # the cancellation pass too.
    from dashboard.metrics import fetch_appointments
    appts = fetch_appointments(client, dr)

    # Map appt_id → (practitioner_id, did_not_arrive, archived, cancelled,
    # appointment_type_id) for fast lookup.
    appt_map: dict[str, dict[str, Any]] = {}
    for _, row in appts.iterrows():
        aid = str(row.get("id") or "")
        if not aid:
            continue
        appt_map[aid] = {
            "practitioner_id": str(row.get("practitioner_id") or ""),
            "did_not_arrive": bool(row.get("did_not_arrive") or False),
            "archived": bool(row.get("archived_at") or False),
            "cancelled": bool(row.get("cancelled_at") or False),
            "appointment_type_id": str(row.get("appointment_type_id") or ""),
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

    rows: list[dict[str, Any]] = []
    try:
        for it in client.paginate("invoice_items", params=params):
            if not isinstance(it, dict):
                continue
            aid = _appt_id_from_invoice_item(it)
            if not aid:
                continue  # product / desk fee — no appt link
            meta = appt_map.get(aid)
            if meta is None:
                continue  # appt outside our fetched range, skip
            if meta["did_not_arrive"] or meta["archived"] or meta["cancelled"]:
                continue
            type_name = type_name_map.get(meta["appointment_type_id"], "")
            if any(r.search(type_name) for r in excluded_res):
                continue  # excluded appointment type (room hire etc)
            # Collect the amount. Cliniko exposes a few fields; prefer
            # `total` (post-discount, post-tax). Fall back to price×qty.
            amount = it.get("total")
            if amount is None:
                price = it.get("price") or 0
                qty = it.get("quantity") or 1
                amount = float(price) * float(qty)
            try:
                amount = float(amount)
            except (TypeError, ValueError):
                continue
            rows.append({
                "invoice_item_id": str(it.get("id") or ""),
                "appointment_id": aid,
                "practitioner_id": meta["practitioner_id"],
                "appointment_type_name": type_name,
                "amount": amount,
            })
    except Exception:
        # If invoice_items endpoint fails, return what we've collected so
        # the UI can still render a partial table with a banner.
        pass

    return pd.DataFrame(rows, columns=[
        "invoice_item_id", "appointment_id", "practitioner_id",
        "appointment_type_name", "amount",
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
    cliniko_practitioner_id: str | None = None,
) -> CommissionResult:
    """Apply the formula end-to-end for one practitioner."""
    base_hours = hours_for_month(year, month, prac.hours_per_day)
    deduction_hours = prac.total_recurring_deduction_hours + float(manual_adjustment_hours or 0)
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
    """
    out: dict[str, str | None] = {}
    if cliniko_practitioners is None or cliniko_practitioners.empty:
        return {p.name: None for p in pay_config.practitioners}

    # Build a normalised lookup over Cliniko practitioners
    cliniko_by_norm: dict[str, str] = {}
    for _, row in cliniko_practitioners.iterrows():
        cid = str(row.get("id") or "")
        if not cid:
            continue
        for col in ("label", "display_name", "first_name", "name"):
            v = row.get(col)
            if isinstance(v, str) and v.strip():
                cliniko_by_norm[_normalise(v)] = cid
        # Combined first + last
        first = row.get("first_name") or ""
        last = row.get("last_name") or ""
        if first or last:
            cliniko_by_norm[_normalise(f"{first} {last}")] = cid
            cliniko_by_norm[_normalise(f"{last} {first}")] = cid

    for prac in pay_config.practitioners:
        cid: str | None = None
        for n in prac.all_names:
            norm = _normalise(n)
            if norm in cliniko_by_norm:
                cid = cliniko_by_norm[norm]
                break
        if cid is None:
            # Token fallback — match on any first-name token
            tokens = _normalise(prac.name).split()
            for known_norm, known_cid in cliniko_by_norm.items():
                known_tokens = set(known_norm.split())
                if known_tokens.issuperset(tokens) or set(tokens).issubset(known_tokens):
                    cid = known_cid
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

    results: list[CommissionResult] = []
    for prac in pay_config.practitioners:
        cid = id_map.get(prac.name)
        revenue = float(revenue_map.get(cid, 0.0)) if cid else 0.0
        manual_h = float(manual_adjustments.get(prac.name, 0.0))
        results.append(compute_commission_for_practitioner(
            prac, year, month,
            revenue=revenue,
            super_rate=super_rate,
            manual_adjustment_hours=manual_h,
            cliniko_practitioner_id=cid,
        ))
    return results, id_map
