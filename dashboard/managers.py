"""Manager KPIs tab.

Each clinic has a manager who's tracked against ten quarterly KPIs. Some
pull automatically from Cliniko / the admin tool, others are self-entered
in the UI and persisted to GitHub so the state is shared across visitors.

Manager → clinic assignments shift on 2027-01-01 (Patrick & Alesha swap,
Thurgoona opens with Mitch), so a date-aware helper resolves the current
mapping.

v27.4.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from dashboard import kpi_sync
from dashboard.config import DATA_DIR
from dashboard.manual import load_nps, load_punctuality


# -------------------------------------------------------------------
# Manager → clinic mapping (date-aware)
# -------------------------------------------------------------------
# Each entry: (effective_from, clinic, manager_name).
# The newest entry whose effective_from <= today wins for each clinic.
MANAGER_HISTORY: list[tuple[date, str, str]] = [
    # Pre-2027 roster
    (date(2024, 1, 1), "Albury",     "Alesha Harmer"),
    (date(2024, 1, 1), "Lavington",  "Patrick Stow"),
    (date(2024, 1, 1), "Wodonga",    "Maya Davis"),
    (date(2024, 1, 1), "Mulgrave",   "James Serong"),
    (date(2024, 1, 1), "Mudgeeraba", "Karen Main"),
    # 2027 shuffle — Patrick ↔ Alesha swap, Thurgoona opens with Mitch.
    (date(2027, 1, 1), "Albury",     "Patrick Stow"),
    (date(2027, 1, 1), "Lavington",  "Alesha Harmer"),
    (date(2027, 1, 1), "Thurgoona",  "Mitch Wadley"),
]


def current_assignments(today: date | None = None) -> dict[str, str]:
    """{clinic: manager_name} as of ``today`` (defaults to today())."""
    today = today or date.today()
    out: dict[str, str] = {}
    for eff, clinic, manager in sorted(MANAGER_HISTORY, key=lambda t: t[0]):
        if eff <= today:
            out[clinic] = manager
    return out


def manager_for_clinic(clinic: str, today: date | None = None) -> str | None:
    return current_assignments(today).get(clinic)


# -------------------------------------------------------------------
# Quarter helpers
# -------------------------------------------------------------------
def quarter_of(d: date) -> int:
    return (d.month - 1) // 3 + 1


def quarter_bounds(year: int, q: int) -> tuple[date, date]:
    """Return (first_day, last_day) of calendar quarter q in year."""
    start_month = (q - 1) * 3 + 1
    end_month = start_month + 2
    start = date(year, start_month, 1)
    if end_month == 12:
        end = date(year, 12, 31)
    else:
        import calendar as _cal
        end = date(year, end_month, _cal.monthrange(year, end_month)[1])
    return start, end


def quarter_label(year: int, q: int) -> str:
    return f"{year} Q{q}"


def recent_quarters(n: int = 8, today: date | None = None) -> list[tuple[int, int]]:
    """Return the last n quarters, oldest-first, including the current."""
    today = today or date.today()
    y, q = today.year, quarter_of(today)
    out = [(y, q)]
    for _ in range(n - 1):
        q -= 1
        if q == 0:
            q = 4; y -= 1
        out.append((y, q))
    out.reverse()
    return out


# -------------------------------------------------------------------
# KPI storage — CSV in data/, pushed to GitHub via github_persistence.
# -------------------------------------------------------------------
MANAGER_KPI_DIR = DATA_DIR / "manager_kpis"
MANAGER_KPI_DIR.mkdir(parents=True, exist_ok=True)
MANAGER_KPI_FILE = MANAGER_KPI_DIR / "entries.csv"


def _load_entries() -> pd.DataFrame:
    if not MANAGER_KPI_FILE.exists():
        return pd.DataFrame(columns=["manager", "clinic", "quarter",
                                       "kpi_key", "subkey", "value", "updated_at"])
    try:
        return pd.read_csv(MANAGER_KPI_FILE, dtype=str)
    except Exception:
        return pd.DataFrame(columns=["manager", "clinic", "quarter",
                                       "kpi_key", "subkey", "value", "updated_at"])


def _save_entries(df: pd.DataFrame) -> None:
    MANAGER_KPI_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANAGER_KPI_FILE, index=False)
    # Push to GitHub so other managers see each other's entries.
    try:
        from dashboard.github_persistence import save_file_to_github
        save_file_to_github(MANAGER_KPI_FILE,
                             commit_msg="[manager_kpis] update entries")
    except Exception:
        pass  # local-only fallback is fine


def _get(df: pd.DataFrame, manager: str, quarter: str, kpi: str,
          subkey: str = "", default: Any = None) -> Any:
    """Pull one stored value, or ``default``."""
    if df.empty:
        return default
    mask = (
        (df["manager"] == manager)
        & (df["quarter"] == quarter)
        & (df["kpi_key"] == kpi)
        & (df["subkey"].fillna("") == subkey)
    )
    hits = df[mask]
    if hits.empty:
        return default
    return hits.iloc[-1]["value"]


def _set(df: pd.DataFrame, manager: str, clinic: str, quarter: str,
          kpi: str, subkey: str, value: Any) -> pd.DataFrame:
    """Upsert one value, keyed on (manager, quarter, kpi, subkey)."""
    key_mask = (
        (df["manager"] == manager)
        & (df["quarter"] == quarter)
        & (df["kpi_key"] == kpi)
        & (df["subkey"].fillna("") == subkey)
    )
    df = df[~key_mask].copy()
    df = pd.concat([
        df,
        pd.DataFrame([{
            "manager": manager, "clinic": clinic, "quarter": quarter,
            "kpi_key": kpi, "subkey": subkey, "value": str(value),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }]),
    ], ignore_index=True)
    return df


# -------------------------------------------------------------------
# Auto-pull helpers
# -------------------------------------------------------------------
def _clinic_nps_for_quarter(clinic: str, year: int, q: int) -> float | None:
    """Latest NPS score for the whole clinic within the quarter."""
    df = load_nps()
    if df.empty or "practitioner" not in df.columns:
        return None
    start, end = quarter_bounds(year, q)
    whole = df[df["practitioner"].astype(str) == f"{clinic} (whole clinic)"].copy()
    if whole.empty:
        return None
    whole["ws"] = pd.to_datetime(whole["week_starting"], errors="coerce").dt.date
    whole = whole.dropna(subset=["ws"])
    whole = whole[(whole["ws"] >= start) & (whole["ws"] <= end)]
    whole["nps"] = pd.to_numeric(whole["nps"], errors="coerce")
    whole = whole.dropna(subset=["nps"]).sort_values("ws")
    if whole.empty:
        return None
    return float(whole.iloc[-1]["nps"])


def _clinic_punctuality_for_quarter(clinic: str, year: int, q: int) -> float | None:
    """% of appointments within 10 min for the clinic across the quarter."""
    df = load_punctuality()
    if df.empty or "clinic" not in df.columns:
        return None
    start, end = quarter_bounds(year, q)
    sub = df[df["clinic"].astype(str).str.lower() == clinic.lower()].copy()
    if sub.empty:
        return None
    for c in ("bucket_0_5", "bucket_6_10", "bucket_11_14", "bucket_15_plus"):
        sub[c] = pd.to_numeric(sub.get(c), errors="coerce").fillna(0)
    sub["ws"] = pd.to_datetime(sub["week_starting"], errors="coerce").dt.date
    sub = sub.dropna(subset=["ws"])
    sub = sub[(sub["ws"] >= start) & (sub["ws"] <= end)]
    if sub.empty:
        return None
    within = (sub["bucket_0_5"] + sub["bucket_6_10"]).sum()
    total = within + (sub["bucket_11_14"] + sub["bucket_15_plus"]).sum()
    if total == 0:
        return None
    return round(100.0 * within / total, 1)


def _clinic_audit_pct_for_quarter(clinic: str, year: int, q: int) -> float | None:
    """Audit pass % for the clinic within the quarter."""
    df = kpi_sync.load_admin_audits()
    if df.empty or "clinic" not in df.columns:
        return None
    start, end = quarter_bounds(year, q)
    sub = df[df["clinic"].astype(str).str.lower() == clinic.lower()].copy()
    if sub.empty:
        return None
    sub["m"] = pd.to_datetime(sub["month"].astype(str) + "-01", errors="coerce").dt.date
    sub = sub.dropna(subset=["m"])
    sub = sub[(sub["m"] >= start) & (sub["m"] <= end)]
    sub["score_y"] = pd.to_numeric(sub.get("score_y"), errors="coerce").fillna(0)
    sub["score_denom"] = pd.to_numeric(sub.get("score_denom"), errors="coerce").fillna(0)
    denom = sub["score_denom"].sum()
    if denom == 0:
        return None
    return round(100.0 * sub["score_y"].sum() / denom, 1)


def _clinic_cxdna_for_quarter(clinic: str, year: int, q: int) -> float | None:
    """Cx + DNA combined % for the clinic across the quarter, averaged
    monthly from snapshots."""
    from dashboard.snapshots import load_snapshot
    start, end = quarter_bounds(year, q)
    months = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1; y += 1
    rates: list[float] = []
    for (yy, mm) in months:
        snap = load_snapshot(yy, mm)
        if not snap:
            continue
        for org_data in snap.get("organizations", {}).values():
            metrics = (org_data.get("metrics") or {})
            # cx_dna has practitioner_id + rate; utilisation_by_clinic has business_id.
            # Easiest: average the per-practitioner cx_dna_combined_rate for the clinic's
            # practitioners. We don't have clinic→practitioner mapping here cheaply, so
            # fall back to the clinic util table's cx_dna if present.
            for rec in metrics.get("cx_dna", []) or []:
                r = rec.get("cx_dna_combined_rate")
                if r is None: continue
                try: rates.append(float(r))
                except Exception: pass
    if not rates:
        return None
    return round(100.0 * sum(rates) / len(rates), 1)


def _clinic_consults_for_quarter(clinic: str, year: int, q: int) -> int | None:
    """Sum of individual consults delivered across all months in the quarter.
    Reads from monthly snapshots."""
    from dashboard.snapshots import load_snapshot
    start, end = quarter_bounds(year, q)
    months = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1; y += 1
    total = 0
    any_data = False
    for (yy, mm) in months:
        snap = load_snapshot(yy, mm)
        if not snap:
            continue
        for org_data in snap.get("organizations", {}).values():
            metrics = (org_data.get("metrics") or {})
            for rec in metrics.get("consults", []) or []:
                v = rec.get("consults_delivered")
                if v is None: continue
                try:
                    total += int(v); any_data = True
                except Exception: pass
    return total if any_data else None


def _consult_growth_yoy(clinic: str, year: int, q: int) -> float | None:
    """YoY % growth of individual consults delivered, this quarter vs the same quarter last year."""
    this = _clinic_consults_for_quarter(clinic, year, q)
    last = _clinic_consults_for_quarter(clinic, year - 1, q)
    if not this or not last:
        return None
    return round(100.0 * (this - last) / last, 1)


# -------------------------------------------------------------------
# KPI definitions
# -------------------------------------------------------------------
# Each KPI returns a dict with {status: "met"|"partial"|"missed"|"pending",
# detail: str, inputs: list of (subkey, label, type) for self-entry}.
# Clinic-specific targets live on the KPI functions themselves.

AW_CLINICS = {"Albury", "Wodonga", "Lavington", "Thurgoona"}
MELBOURNE_CLINICS = {"Mulgrave"}
QLD_CLINICS = {"Mudgeeraba"}


def _cxdna_target_for(clinic: str) -> float:
    """Per-document: ≤12% AW, ≤14% Melbourne. Mudgeeraba assumed 14% too."""
    if clinic in AW_CLINICS:
        return 12.0
    return 14.0


KPI_SPECS: list[dict[str, Any]] = [
    {
        "key": "referral_development",
        "title": "Referral Development",
        "target_text": "12 total contacts (6 new) · 2 new referrers · ≥75% medical/health",
        "mode": "self",
        "subfields": [
            ("total_contacts", "Total referrer contacts", "int"),
            ("new_contacts",   "New (first-time) contacts", "int"),
            ("new_referrers",  "New referrers who sent a patient", "int"),
            ("pct_medical",    "% from medical/health field", "pct"),
        ],
    },
    {
        "key": "clinic_reputation",
        "title": "Clinic Reputation",
        "target_text": "NPS ≥ 80% · 10+ new Google reviews",
        "mode": "mixed",
        "auto_subkeys": ["nps"],
        "subfields": [
            ("google_reviews", "New Google reviews this quarter", "int"),
        ],
    },
    {
        "key": "staff_satisfaction",
        "title": "Staff Satisfaction",
        "target_text": "Staff survey score ≥ 80%",
        "mode": "self",
        "subfields": [
            ("score", "Staff satisfaction score (%)", "pct"),
        ],
    },
    {
        "key": "consult_growth",
        "title": "Consult Growth",
        "target_text": "≥5% YoY growth (individual consults, excl groups)",
        "mode": "auto",
    },
    {
        "key": "cx_dna",
        "title": "Cancellation / DNA Rate",
        "target_text": "≤12% (AW) / ≤14% (QLD, VIC)",
        "mode": "auto",
    },
    {
        "key": "recall_effectiveness",
        "title": "Recall Effectiveness",
        "target_text": "Recalls >90% · Conversion >20%",
        "mode": "self",
        "subfields": [
            ("recalls_pct",    "Recalls completed (%)", "pct"),
            ("conversion_pct", "Recall → booked conversion (%)", "pct"),
        ],
    },
    {
        "key": "audits",
        "title": "Documentation / Audits",
        "target_text": "≥90% individual audit compliance",
        "mode": "auto",
    },
    {
        "key": "diary_management",
        "title": "Diary Management",
        "target_text": "Approved in advance · Initialled · Not end-of-day",
        "mode": "self",
        "subfields": [
            ("approved_advance", "Approved in advance (Y/N)", "bool"),
            ("initialled",       "Initialled (Y/N)", "bool"),
            ("not_eod",          "Not end-of-day (Y/N)", "bool"),
        ],
    },
    {
        "key": "punctuality",
        "title": "Punctuality",
        "target_text": "≥90% patient starts within 10 min",
        "mode": "auto",
    },
    {
        "key": "responsiveness",
        "title": "Responsiveness",
        "target_text": "Internal ≤2 hrs · External same day",
        "mode": "self",
        "subfields": [
            ("internal_2hr",  "Met internal 2 hr target (Y/N)", "bool"),
            ("external_same", "Met external same-day target (Y/N)", "bool"),
        ],
    },
]


# -------------------------------------------------------------------
# Per-KPI evaluation
# -------------------------------------------------------------------
def _evaluate_kpi(spec: dict, df: pd.DataFrame, manager: str, clinic: str,
                   quarter: str, year: int, q: int) -> dict:
    k = spec["key"]
    mode = spec["mode"]

    def _as_float(v):
        try: return float(v) if v not in (None, "", "nan") else None
        except Exception: return None

    def _as_int(v):
        try: return int(float(v)) if v not in (None, "", "nan") else None
        except Exception: return None

    if k == "referral_development":
        tot = _as_int(_get(df, manager, quarter, k, "total_contacts"))
        new = _as_int(_get(df, manager, quarter, k, "new_contacts"))
        nref = _as_int(_get(df, manager, quarter, k, "new_referrers"))
        pct = _as_float(_get(df, manager, quarter, k, "pct_medical"))
        parts = []
        all_ok = True
        for label, val, target in [
            (f"total {tot or 0}/12", tot, 12),
            (f"new {new or 0}/6",     new, 6),
            (f"new refs {nref or 0}/2", nref, 2),
            (f"%med {pct or 0:.0f}%/75%", pct, 75),
        ]:
            ok = val is not None and val >= target
            parts.append(("✅" if ok else "❌") + " " + label)
            if not ok: all_ok = False
        return {"status": "met" if all_ok else "missed",
                "detail": " · ".join(parts)}

    if k == "clinic_reputation":
        nps = _clinic_nps_for_quarter(clinic, year, q)
        reviews = _as_int(_get(df, manager, quarter, k, "google_reviews"))
        nps_ok = nps is not None and nps >= 80
        rev_ok = reviews is not None and reviews >= 10
        bits = [
            f"NPS: {nps:.0f}/80" if nps is not None else "NPS: —",
            f"Reviews: {reviews or 0}/10",
        ]
        bits = [("✅" if ok else "❌") + " " + b for ok, b in zip([nps_ok, rev_ok], bits)]
        return {"status": "met" if nps_ok and rev_ok else "missed",
                "detail": " · ".join(bits)}

    if k == "staff_satisfaction":
        score = _as_float(_get(df, manager, quarter, k, "score"))
        ok = score is not None and score >= 80
        return {"status": "met" if ok else ("missed" if score is not None else "pending"),
                "detail": f"Score: {score:.0f}%/80%" if score is not None else "Not entered yet"}

    if k == "consult_growth":
        growth = _consult_growth_yoy(clinic, year, q)
        if growth is None:
            return {"status": "pending",
                    "detail": "Need snapshots for this quarter AND the same quarter last year."}
        ok = growth >= 5.0
        return {"status": "met" if ok else "missed",
                "detail": f"YoY {growth:+.1f}% / target ≥5%"}

    if k == "cx_dna":
        rate = _clinic_cxdna_for_quarter(clinic, year, q)
        tgt = _cxdna_target_for(clinic)
        if rate is None:
            return {"status": "pending", "detail": "No snapshot data for this quarter yet."}
        ok = rate <= tgt
        return {"status": "met" if ok else "missed",
                "detail": f"Cx/DNA {rate:.1f}% / target ≤{tgt:.0f}%"}

    if k == "recall_effectiveness":
        r = _as_float(_get(df, manager, quarter, k, "recalls_pct"))
        c = _as_float(_get(df, manager, quarter, k, "conversion_pct"))
        r_ok = r is not None and r > 90
        c_ok = c is not None and c > 20
        bits = [
            ("✅" if r_ok else "❌") + f" Recalls {r:.0f}%/90%" if r is not None else "❌ Recalls: —",
            ("✅" if c_ok else "❌") + f" Convert {c:.0f}%/20%" if c is not None else "❌ Convert: —",
        ]
        return {"status": "met" if r_ok and c_ok else "missed",
                "detail": " · ".join(bits)}

    if k == "audits":
        pct = _clinic_audit_pct_for_quarter(clinic, year, q)
        if pct is None:
            return {"status": "pending", "detail": "No audit data entered for this quarter yet."}
        ok = pct >= 90
        return {"status": "met" if ok else "missed",
                "detail": f"Audit score {pct:.0f}%/90%"}

    if k == "diary_management":
        a = _get(df, manager, quarter, k, "approved_advance") == "True"
        b = _get(df, manager, quarter, k, "initialled") == "True"
        c = _get(df, manager, quarter, k, "not_eod") == "True"
        parts = [("✅ " if x else "❌ ") + lbl for x, lbl in
                 ((a, "approved"), (b, "initialled"), (c, "not EOD"))]
        return {"status": "met" if (a and b and c) else "missed",
                "detail": " · ".join(parts)}

    if k == "punctuality":
        pct = _clinic_punctuality_for_quarter(clinic, year, q)
        if pct is None:
            return {"status": "pending",
                    "detail": "No punctuality data for this clinic in this quarter."}
        ok = pct >= 90
        return {"status": "met" if ok else "missed",
                "detail": f"Within 10 min: {pct:.0f}%/90%"}

    if k == "responsiveness":
        a = _get(df, manager, quarter, k, "internal_2hr") == "True"
        b = _get(df, manager, quarter, k, "external_same") == "True"
        parts = [("✅ " if x else "❌ ") + lbl for x, lbl in
                 ((a, "internal 2hr"), (b, "external same day"))]
        return {"status": "met" if (a and b) else "missed",
                "detail": " · ".join(parts)}

    return {"status": "pending", "detail": ""}


# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------
def managers_tab(filters: dict) -> None:
    st.subheader("👥 Manager KPIs")
    st.caption(
        "Quarterly KPI tracker for each clinic manager. "
        "Metrics that have a data source (NPS, audits, punctuality, consults, Cx/DNA) "
        "auto-populate; the rest you fill in on this tab and the entries persist for everyone."
    )

    today = date.today()
    assignments = current_assignments(today)

    # Picker: clinic → surfaces the current manager; quarter picker; manager override.
    clinics = sorted(assignments.keys())
    if not clinics:
        st.warning("No manager assignments configured.")
        return

    col_a, col_b = st.columns([2, 1])
    with col_a:
        chosen_clinic = st.selectbox("Clinic", clinics,
                                       format_func=lambda c: f"{c} — {assignments.get(c, '(no manager)')}")
    with col_b:
        quarters = recent_quarters(8, today)
        q_default_idx = len(quarters) - 1  # current
        chosen_q = st.selectbox(
            "Quarter", quarters,
            index=q_default_idx,
            format_func=lambda yq: quarter_label(yq[0], yq[1]) + (" (current)" if yq == quarters[-1] else ""),
        )
    sel_year, sel_q = chosen_q
    manager_name = assignments[chosen_clinic]
    quarter_str = f"{sel_year}-Q{sel_q}"

    df = _load_entries()

    # Compute results per KPI
    results = []
    for spec in KPI_SPECS:
        res = _evaluate_kpi(spec, df, manager_name, chosen_clinic,
                              quarter_str, sel_year, sel_q)
        results.append((spec, res))

    met_count = sum(1 for _, r in results if r["status"] == "met")
    total_kpis = len(results)
    pending = sum(1 for _, r in results if r["status"] == "pending")

    # Header summary (no $ figures — Matt asked for them to be hidden here).
    c1, c2, c3 = st.columns(3)
    c1.metric("Manager", manager_name)
    c2.metric("Quarter", quarter_label(sel_year, sel_q))
    c3.metric("KPIs met", f"{met_count} / {total_kpis}",
              delta=f"{pending} pending" if pending else None, delta_color="off")

    st.markdown("---")

    # Per-KPI cards, two columns
    for i, (spec, res) in enumerate(results):
        if i % 2 == 0:
            cols = st.columns(2)
        with cols[i % 2]:
            status_icon = {"met": "✅", "missed": "❌", "pending": "⏳", "partial": "🟡"}.get(res["status"], "·")
            st.markdown(f"### {status_icon} {spec['title']}")
            st.caption(f"**Target:** {spec['target_text']}")
            st.write(res["detail"])

            # Self-entry form for the self/mixed KPIs.
            subfields = spec.get("subfields") or []
            if subfields:
                with st.form(f"form_{spec['key']}_{quarter_str}_{manager_name}",
                              clear_on_submit=False):
                    new_vals: dict[str, Any] = {}
                    for subkey, label, kind in subfields:
                        current = _get(df, manager_name, quarter_str, spec["key"], subkey, "")
                        if kind == "int":
                            default = int(float(current)) if current not in ("", None) else 0
                            new_vals[subkey] = st.number_input(label, min_value=0, step=1,
                                                                 value=default,
                                                                 key=f"inp_{spec['key']}_{subkey}_{quarter_str}")
                        elif kind == "pct":
                            default = float(current) if current not in ("", None) else 0.0
                            new_vals[subkey] = st.number_input(label, min_value=0.0, max_value=100.0,
                                                                 step=1.0, value=default,
                                                                 key=f"inp_{spec['key']}_{subkey}_{quarter_str}")
                        elif kind == "bool":
                            default = current == "True"
                            new_vals[subkey] = st.checkbox(label, value=default,
                                                             key=f"inp_{spec['key']}_{subkey}_{quarter_str}")
                        else:
                            new_vals[subkey] = st.text_input(label, value=str(current or ""),
                                                               key=f"inp_{spec['key']}_{subkey}_{quarter_str}")
                    if st.form_submit_button("Save"):
                        for sk, v in new_vals.items():
                            df = _set(df, manager_name, chosen_clinic, quarter_str,
                                        spec["key"], sk, v)
                        _save_entries(df)
                        st.success("Saved.")
                        st.rerun()

    # Admin view: show all managers' progress at a glance.
    st.markdown("---")
    with st.expander("All managers — this quarter at a glance", expanded=False):
        rows = []
        for clinic, mgr in assignments.items():
            met = 0
            for spec in KPI_SPECS:
                r = _evaluate_kpi(spec, df, mgr, clinic, quarter_str, sel_year, sel_q)
                if r["status"] == "met":
                    met += 1
            rows.append({"Clinic": clinic, "Manager": mgr,
                          "KPIs met": f"{met} / {total_kpis}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
