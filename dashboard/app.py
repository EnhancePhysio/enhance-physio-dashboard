"""Enhance Physio Reporting Dashboard — Streamlit entry point.

Run with:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `from dashboard.x` works when
# Streamlit runs this file directly (Streamlit Cloud executes
# `streamlit run dashboard/app.py`, which puts only `dashboard/` on the path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from dashboard.audit import aggregate_audit, aggregate_by_check, run_audit, select_audit_pool
from dashboard.cliniko import ClinikoClient, ClinikoError
from dashboard.config import dashboard_password, load_settings
from dashboard.date_ranges import PRESETS, resolve_preset
from dashboard.manual import (
    extract_punctuality_from_image, load_nps, load_punctuality,
    nps_per_practitioner, punctuality_per_practitioner,
    save_punctuality_csv, vision_response_to_dataframe,
    PUNCTUALITY_COLUMNS,
)
from dashboard.metrics import compute_core_metrics, merge_per_practitioner
from dashboard.reference_data import load_appointment_types, load_businesses, load_practitioners
from dashboard.scoring import score_table
from dashboard.ui import kpi_card, matrix_scatter, practitioner_detail, zone_summary_table


st.set_page_config(
    page_title="Enhance Physio — Performance Dashboard",
    page_icon="📊",
    layout="wide",
)


# -------------------------------------------------------------------
# Password gate (optional — enabled when DASHBOARD_PASSWORD is set)
# -------------------------------------------------------------------
def _require_password() -> None:
    pw = dashboard_password()
    if not pw:
        return  # gate disabled
    if st.session_state.get("_pw_ok"):
        return
    st.title("🔒 Enhance Physio Dashboard")
    st.caption("Enter the password to access the dashboard.")
    with st.form("pw", clear_on_submit=False):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if entered == pw:
            st.session_state["_pw_ok"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


# -------------------------------------------------------------------
# Cached data access
# -------------------------------------------------------------------
@st.cache_resource
def get_client() -> ClinikoClient:
    return ClinikoClient()


@st.cache_data(ttl=load_settings()["cliniko"]["cache_ttl_seconds"])
def cached_practitioners() -> pd.DataFrame:
    return load_practitioners(get_client())


@st.cache_data(ttl=load_settings()["cliniko"]["cache_ttl_seconds"])
def cached_businesses() -> pd.DataFrame:
    return load_businesses(get_client())


@st.cache_data(ttl=load_settings()["cliniko"]["cache_ttl_seconds"])
def cached_appointment_types() -> pd.DataFrame:
    return load_appointment_types(get_client())


@st.cache_data(ttl=load_settings()["cliniko"]["cache_ttl_seconds"])
def cached_metrics(preset: str, start: date | None, end: date | None,
                    business_ids: tuple[int, ...], practitioner_ids: tuple[int, ...]):
    client = get_client()
    dr = resolve_preset(preset, start, end)
    types = cached_appointment_types()
    result = compute_core_metrics(
        client, dr, types,
        business_ids=list(business_ids) if business_ids else None,
        practitioner_ids=list(practitioner_ids) if practitioner_ids else None,
    )
    return result, dr


# -------------------------------------------------------------------
# Sidebar — filters
# -------------------------------------------------------------------
def sidebar_filters() -> dict:
    st.sidebar.title("Filters")
    preset_keys = [k for k, _ in PRESETS]
    preset_labels = {k: label for k, label in PRESETS}
    preset = st.sidebar.selectbox(
        "Date range", preset_keys,
        format_func=lambda k: preset_labels[k],
        index=1,  # last 30 by default
    )
    custom_start = custom_end = None
    if preset == "custom":
        today = date.today()
        custom_start = st.sidebar.date_input("Start", today - timedelta(days=30))
        custom_end = st.sidebar.date_input("End", today)

    try:
        biz = cached_businesses()
        practs = cached_practitioners()
    except ClinikoError as e:
        st.sidebar.error(f"Cliniko error: {e}")
        biz, practs = pd.DataFrame(), pd.DataFrame()

    biz_ids: list[int] = []
    if not biz.empty:
        biz_labels = dict(zip(biz["id"], biz["label"]))
        chosen_biz = st.sidebar.multiselect(
            "Clinic", options=list(biz_labels.keys()),
            format_func=lambda i: biz_labels[i], default=[],
        )
        biz_ids = chosen_biz

    prac_ids: list[int] = []
    if not practs.empty:
        active = practs[practs["active"] != False]
        prac_labels = dict(zip(active["id"], active["label"]))
        chosen_prac = st.sidebar.multiselect(
            "Practitioner", options=list(prac_labels.keys()),
            format_func=lambda i: prac_labels[i], default=[],
        )
        prac_ids = chosen_prac

    st.sidebar.markdown("---")
    st.sidebar.caption("Cliniko shard: `{}`".format(get_client().shard))
    return {
        "preset": preset,
        "custom_start": custom_start,
        "custom_end": custom_end,
        "business_ids": biz_ids,
        "practitioner_ids": prac_ids,
        "businesses": biz,
        "practitioners": practs,
    }


# -------------------------------------------------------------------
# Tab: Overview
# -------------------------------------------------------------------
def overview_tab(filters: dict):
    try:
        result, dr = cached_metrics(
            filters["preset"], filters["custom_start"], filters["custom_end"],
            tuple(filters["business_ids"]), tuple(filters["practitioner_ids"]),
        )
    except ClinikoError as e:
        st.error(f"Cliniko error: {e}")
        return
    except RuntimeError as e:
        st.error(str(e))
        return

    st.caption(f"Range: **{dr.label()}** (tz: {load_settings()['timezone']})")

    # Manual data
    punct_df = load_punctuality()
    nps_df = load_nps()
    practs = filters["practitioners"]
    name_to_id = dict(zip(practs["label"], practs["id"])) if not practs.empty else {}
    punct_agg = punctuality_per_practitioner(punct_df, dr.start_date, dr.end_date_inclusive, name_to_id)
    nps_agg = nps_per_practitioner(nps_df, dr.start_date, dr.end_date_inclusive, name_to_id)

    # Audit (optional — only run on demand to save rate limits)
    audit_key = (filters["preset"], tuple(filters["business_ids"]),
                 tuple(filters["practitioner_ids"]))
    with st.expander("Audit (run on demand — calls Cliniko repeatedly)"):
        run_now = st.button("Run audit for this filter", key="run_audit")
        if run_now:
            client = get_client()
            types = cached_appointment_types()
            pool = select_audit_pool(result.appointments, types)
            st.write(f"Pool: **{len(pool)}** new patients")
            progress = st.progress(0)
            status = st.empty()

            def cb(i, total):
                progress.progress(i / total)
                status.caption(f"Auditing patient {i} of {total}...")

            results = run_audit(client, pool, practs, progress_cb=cb)
            audit_df = aggregate_audit(results)
            by_check = aggregate_by_check(results)
            st.session_state["audit_df"] = audit_df
            st.session_state["audit_by_check"] = by_check
            st.session_state["audit_key"] = audit_key
            st.session_state["audit_patients"] = results
            status.caption(f"Audit complete — {len(results)} patients scored.")
    audit_df = st.session_state.get("audit_df") if st.session_state.get("audit_key") == audit_key else None

    # Merge everything per-practitioner
    # Renames for punctuality + NPS
    manual_punct = (
        punct_agg.rename(columns={"punctuality_within_15": "punctuality_within_15"})
        [["practitioner_id", "punctuality_within_15"]]
        if "practitioner_id" in punct_agg.columns and not punct_agg["practitioner_id"].isna().all()
        else pd.DataFrame(columns=["practitioner_id", "punctuality_within_15"])
    )
    manual_nps = (
        nps_agg[["practitioner_id", "nps", "nps_raw"]]
        if "practitioner_id" in nps_agg.columns and not nps_agg.empty
        else pd.DataFrame(columns=["practitioner_id", "nps", "nps_raw"])
    )

    wide = merge_per_practitioner(
        result, practs, manual_nps=manual_nps, manual_punct=manual_punct, audit=audit_df,
    )

    # Apply practitioner filter to the visible set
    if filters["practitioner_ids"]:
        wide = wide[wide["practitioner_id"].isin(filters["practitioner_ids"])]

    scored = score_table(wide)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total consults",
                 f"{int(scored['consults_delivered'].fillna(0).sum()):,}")
    with c2:
        kpi_card("Service hours",
                 f"{scored['service_hours'].fillna(0).sum():.1f}")
    with c3:
        kpi_card("Avg Cx/DNA",
                 f"{scored['cx_dna_combined_rate'].fillna(0).mean() * 100:.1f}%")
    with c4:
        kpi_card("Practitioners in view",
                 str(len(scored)))

    st.markdown("---")
    st.subheader("Clinical × Non-clinical Matrix")
    st.plotly_chart(matrix_scatter(scored), use_container_width=True)

    st.subheader("Zone summary")
    st.dataframe(zone_summary_table(scored), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Practitioner breakdown")
    pick = st.selectbox("Practitioner",
                         options=[""] + scored["label"].tolist(),
                         format_func=lambda s: "(pick one)" if s == "" else s)
    if pick:
        practitioner_detail(scored, pick)

    st.markdown("---")
    with st.expander("Raw scored table"):
        st.dataframe(scored, use_container_width=True)

    if audit_df is not None and not audit_df.empty:
        with st.expander("Audit — training-pattern breakdown"):
            by_check = st.session_state.get("audit_by_check")
            if by_check is not None and not by_check.empty:
                labelled = by_check.merge(
                    practs[["id", "label"]].rename(columns={"id": "practitioner_id"}),
                    on="practitioner_id", how="left",
                )
                st.dataframe(labelled, use_container_width=True)
            st.markdown("**Failed patients (drill-down)**")
            patients = st.session_state.get("audit_patients", [])
            rows = []
            for pa in patients:
                if pa.failed_checks:
                    rows.append({
                        "Patient": pa.patient_name,
                        "Practitioner": practs.loc[practs["id"] == pa.practitioner_id, "label"].squeeze()
                                       if pa.practitioner_id in practs["id"].values else str(pa.practitioner_id),
                        "Cohort": pa.cohort,
                        "Score": f"{pa.passes}/{pa.applicable}",
                        "Failed": ", ".join(pa.failed_checks),
                        "Cliniko link": f"https://{get_client().shard}.cliniko.com/patients/{pa.patient_id}",
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No failed patients in this run.")


# -------------------------------------------------------------------
# Tab: Manual data
# -------------------------------------------------------------------
def manual_tab(filters: dict):
    st.subheader("Punctuality — upload paper sheet")
    st.caption("Drop a photo or PDF of the week's paper sheet. "
               "Claude Vision reads the circled totals; you confirm before save.")
    uploaded = st.file_uploader(
        "Punctuality photo (one clinic per file)",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=False,
        key="punct_upload",
    )
    if uploaded:
        st.image(uploaded.getvalue()) if uploaded.type.startswith("image/") else st.write(uploaded.name)
        if st.button("Extract with Claude Vision", key="vision_btn"):
            try:
                data = extract_punctuality_from_image(
                    uploaded.getvalue(),
                    media_type=uploaded.type or "image/jpeg",
                )
                df = vision_response_to_dataframe(data)
                st.session_state["punct_extracted"] = df
                st.success(f"Extracted {len(df)} rows. Review and edit, then Save below.")
            except Exception as e:
                st.error(f"Vision extraction failed: {e}")

    if "punct_extracted" in st.session_state:
        edited = st.data_editor(
            st.session_state["punct_extracted"],
            num_rows="dynamic",
            use_container_width=True,
            key="punct_editor",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save to data/punctuality/", key="punct_save"):
                week = edited["week_starting"].iloc[0] if not edited.empty else "unknown"
                clinic = edited["clinic"].iloc[0] if not edited.empty else "unknown"
                path = save_punctuality_csv(edited, str(week), str(clinic))
                st.success(f"Saved: {path.name}")
                del st.session_state["punct_extracted"]
        with col2:
            if st.button("Discard", key="punct_discard"):
                del st.session_state["punct_extracted"]

    st.markdown("---")
    st.subheader("Punctuality — manual CSV")
    st.caption("If you don't want to use vision extraction, fill this template and save "
               "to `data/punctuality/YYYY-MM-DD_<clinic>.csv`.")
    template = pd.DataFrame(columns=PUNCTUALITY_COLUMNS)
    st.dataframe(template, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("NPS — CSV upload (preferred)")
    st.caption("Export the NPS report from Cliniqapps and upload here, or save CSV directly to data/nps/.")
    nps_file = st.file_uploader("NPS CSV export",
                                  type=["csv"], key="nps_upload")
    if nps_file:
        try:
            new_nps = pd.read_csv(nps_file)
            st.dataframe(new_nps, use_container_width=True)
            name = st.text_input("Save as filename (YYYY-MM.csv)", value="nps.csv")
            if st.button("Save NPS CSV", key="nps_save"):
                from .manual import NPS_DIR
                out = NPS_DIR / name
                new_nps.to_csv(out, index=False)
                st.success(f"Saved: {out.name}")
        except Exception as e:
            st.error(f"NPS load failed: {e}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main() -> None:
    _require_password()
    st.title("Enhance Physio — Performance Dashboard")
    filters = sidebar_filters()

    tab_overview, tab_manual = st.tabs(["Overview", "Manual data"])
    with tab_overview:
        overview_tab(filters)
    with tab_manual:
        manual_tab(filters)


if __name__ == "__main__":
    main()
