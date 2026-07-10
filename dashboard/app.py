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

from dashboard.audit import (aggregate_audit, aggregate_by_check,
                              error_reasons_summary, fetch_stats_snapshot,
                              fetch_stats_summary, run_audit, select_audit_pool)
from dashboard.cliniko import ClinikoClient, ClinikoError
from dashboard.config import dashboard_password, load_settings
from dashboard.date_ranges import PRESETS, resolve_preset
from dashboard.manual import (
    aggregate_nps_from_individual_scores,
    extract_punctuality_from_image, load_nps, load_punctuality,
    nps_per_practitioner, punctuality_per_practitioner,
    save_nps_csv, save_punctuality_csv, vision_response_to_dataframe,
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
                    business_ids: tuple[int, ...], practitioner_ids: tuple[int, ...],
                    _schema_version: str = ""):
    """Cached wrapper around compute_core_metrics.

    ``_schema_version`` is a cache-bust key: Streamlit's @st.cache_data
    hashes THIS function's source, not the transitive call graph, so
    changes inside compute_core_metrics (e.g. adding fields to
    MetricResult) don't invalidate the cache on their own. Passing the
    dashboard's __version__ forces a fresh run every release, which
    prevents v26.5 pickled MetricResults from bleeding into v26.6 code
    and throwing AttributeError on the new fields.
    """
    del _schema_version  # only used as a cache-bust key
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

    # v27.0 — Multi-org clinic selector. Sits ABOVE the per-org business
    # multiselect (which is Cliniko's own concept of "business" within
    # one org). Chosen clinic filters ALL tabs.
    from dashboard.cliniko import (
        get_configured_orgs, load_organizations,
    )
    all_orgs = load_organizations()
    configured_orgs = get_configured_orgs()
    if len(all_orgs) > 1:
        # Show one option per org PLUS an "All clinics" aggregate.
        # Only include orgs whose API key is actually set.
        options = ["All clinics"] + [o.name for o in configured_orgs]
        missing = [o.name for o in all_orgs if not o.api_key]
        selected_clinic = st.sidebar.selectbox(
            "Clinic (multi-org)", options,
            index=0,
            help="Filter dashboard to one clinic, or aggregate all.",
        )
        if missing:
            st.sidebar.caption(
                f"⚠️ Missing API keys for: {', '.join(missing)}. "
                "Add secrets in Streamlit Cloud to enable those orgs."
            )
        # Resolve to org keys
        from dashboard.multi_org import resolve_clinic_filter
        selected_org_keys = resolve_clinic_filter(selected_clinic)
    else:
        selected_clinic = "All clinics"
        selected_org_keys = [o.key for o in configured_orgs]

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
    from dashboard import __version__ as _dashboard_version
    st.sidebar.caption(
        "Build: `v{}`  ·  Cliniko shard: `{}`".format(
            _dashboard_version, get_client().shard
        )
    )
    # v26 — surface GitHub persistence status in the sidebar so Matt knows
    # at a glance whether his weekly uploads are being persisted.
    try:
        from dashboard.github_persistence import status_description as _gh_status
        st.sidebar.caption(_gh_status())
    except Exception:
        pass

    return {
        "preset": preset,
        "custom_start": custom_start,
        "custom_end": custom_end,
        "business_ids": biz_ids,
        "practitioner_ids": prac_ids,
        "businesses": biz,
        "practitioners": practs,
        # v27.0 — multi-org filter
        "clinic_selection": selected_clinic,
        "selected_org_keys": selected_org_keys,
    }


@st.cache_data(ttl=300, show_spinner=False)
def _hydrate_data_from_github_once() -> dict[str, str]:
    """v26 — on startup, if Streamlit Cloud has wiped the ephemeral
    filesystem, pull back our persisted data directories from GitHub.

    Cached for 5 minutes so reruns don't hammer the API. Returns a dict
    of directory name -> status message, for optional UI display.
    """
    try:
        from dashboard.github_persistence import hydrate_all
        return hydrate_all()
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------------------
# Session-state NPS / punctuality helpers
# -------------------------------------------------------------------
# On Streamlit Cloud, `data/` is ephemeral — a redeploy (or even a
# long idle period) wipes any NPS/punct CSVs written via the "Save" buttons.
# v18 introduces a session-state bypass: Manual-tab uploads are mirrored into
# st.session_state so the Overview tab picks them up immediately, no Save
# button click required. Disk files still work unchanged — they just aren't
# the only path.
def _merged_nps_df() -> pd.DataFrame:
    # v26 — ensure the ephemeral filesystem is hydrated from GitHub before
    # we read from it. Cached so this is a cheap no-op after the first
    # rerun of the session.
    _hydrate_data_from_github_once()
    disk = load_nps()
    session = st.session_state.get("nps_session_df")
    if session is None or (isinstance(session, pd.DataFrame) and session.empty):
        return disk
    if disk.empty:
        return session.copy()
    # Concat, with session rows taking precedence for duplicate keys.
    # Key = (week_starting, clinic, practitioner) — same grain as NPS_COLUMNS.
    both = pd.concat([disk, session], ignore_index=True)
    key_cols = [c for c in ("week_starting", "clinic", "practitioner")
                if c in both.columns]
    if key_cols:
        both = both.drop_duplicates(subset=key_cols, keep="last")
    return both.reset_index(drop=True)


def _merged_punct_df() -> pd.DataFrame:
    _hydrate_data_from_github_once()
    disk = load_punctuality()
    session = st.session_state.get("punct_session_df")
    if session is None or (isinstance(session, pd.DataFrame) and session.empty):
        return disk
    if disk.empty:
        return session.copy()
    both = pd.concat([disk, session], ignore_index=True)
    key_cols = [c for c in ("week_starting", "clinic", "practitioner", "day")
                if c in both.columns]
    if key_cols:
        both = both.drop_duplicates(subset=key_cols, keep="last")
    return both.reset_index(drop=True)


# -------------------------------------------------------------------
# Tab: Overview
# -------------------------------------------------------------------
def overview_tab(filters: dict):
    try:
        from dashboard import __version__ as _v
        result, dr = cached_metrics(
            filters["preset"], filters["custom_start"], filters["custom_end"],
            tuple(filters["business_ids"]), tuple(filters["practitioner_ids"]),
            _schema_version=_v,
        )
    except ClinikoError as e:
        st.error(f"Cliniko error: {e}")
        return
    except RuntimeError as e:
        st.error(str(e))
        return

    st.caption(f"Range: **{dr.label()}** (tz: {load_settings()['timezone']})")

    # Manual data — combine disk-loaded files with in-session uploads.
    # On Streamlit Cloud the filesystem is ephemeral and wipes on redeploy,
    # so the Manual tab's uploads *also* get mirrored into session_state so
    # they flow straight through to scoring without requiring the "Save to
    # data/nps/" button (which stopped persisting when the app moved to
    # Streamlit Cloud).
    punct_df = _merged_punct_df()
    nps_df = _merged_nps_df()

    # Show a subtle banner if session-state uploads are active, so Matt
    # knows the numbers he's seeing include data that's only in memory.
    _session_nps = st.session_state.get("nps_session_df")
    _session_punct = st.session_state.get("punct_session_df")
    if (_session_nps is not None and not _session_nps.empty) or \
       (_session_punct is not None and not _session_punct.empty):
        bits = []
        if _session_nps is not None and not _session_nps.empty:
            bits.append(f"**{len(_session_nps)}** NPS row(s)")
        if _session_punct is not None and not _session_punct.empty:
            bits.append(f"**{len(_session_punct)}** punctuality row(s)")
        st.info(
            f"Using in-session uploads: {', '.join(bits)}. "
            "These are flowing through to scoring but are NOT saved to disk — "
            "use the Manual tab's Save buttons if you want them to persist "
            "across sessions."
        )
    practs = filters["practitioners"]
    name_to_id = dict(zip(practs["label"], practs["id"])) if not practs.empty else {}
    punct_agg = punctuality_per_practitioner(
        punct_df, dr.start_date, dr.end_date_inclusive, name_to_id, practitioners=practs,
    )
    nps_agg = nps_per_practitioner(
        nps_df, dr.start_date, dr.end_date_inclusive, name_to_id, practitioners=practs,
    )

    # Surface any CSV practitioner names we couldn't match to a Cliniko
    # practitioner — the old behaviour silently dropped these rows, which
    # made it look like NPS just wasn't working.
    _nps_unmatched = nps_agg.attrs.get("unmatched_names", []) if hasattr(nps_agg, "attrs") else []
    _punct_unmatched = punct_agg.attrs.get("unmatched_names", []) if hasattr(punct_agg, "attrs") else []
    if _nps_unmatched or _punct_unmatched:
        with st.expander("Manual-data name mismatches — expand to see", expanded=False):
            if _nps_unmatched:
                st.warning(
                    f"NPS CSV has **{len(_nps_unmatched)}** practitioner name(s) that "
                    f"didn't match any Cliniko practitioner:\n\n- " + "\n- ".join(sorted(_nps_unmatched))
                    + "\n\nFix the names in the CSV (or rename practitioners in Cliniko) and re-upload."
                )
            if _punct_unmatched:
                st.warning(
                    f"Punctuality CSV has **{len(_punct_unmatched)}** unmatched name(s):\n\n- "
                    + "\n- ".join(sorted(_punct_unmatched))
                )
            st.caption(
                "Cliniko practitioner labels (for reference): "
                + ", ".join(sorted(practs["label"].tolist())) if not practs.empty else ""
            )

    # Audit (optional — only run on demand to save rate limits)
    audit_key = (filters["preset"], tuple(filters["business_ids"]),
                 tuple(filters["practitioner_ids"]))
    with st.expander("Audit (run on demand — calls Cliniko repeatedly)"):
        # Show cache stats before running so Matt knows how much work is ahead.
        from dashboard import audit_cache
        cache_summary = audit_cache.stats()
        st.caption(
            f"Audit cache: **{cache_summary['entries']}** patients on disk "
            f"(oldest {cache_summary['oldest']}, newest {cache_summary['newest']}). "
            f"TTL: **{load_settings().get('audit', {}).get('cache_ttl_days', 30)}** days."
        )

        types = cached_appointment_types()
        pool_preview = select_audit_pool(result.appointments, types)
        ttl = int(load_settings().get("audit", {}).get("cache_ttl_days", 30))
        cache_all = audit_cache.load_all()
        cached_hits = 0
        for row in pool_preview.itertuples(index=False):
            if audit_cache.get_fresh(cache_all, str(row.patient_id), ttl) is not None:
                cached_hits += 1
        fresh_needed = len(pool_preview) - cached_hits
        st.write(
            f"Pool: **{len(pool_preview)}** new patients — "
            f"**{cached_hits}** already cached, **{fresh_needed}** to fetch fresh. "
            f"Estimated fresh-fetch time: **~{max(1, fresh_needed // 30)}–{max(1, fresh_needed // 20)} min** "
            f"(at 20-30 patients/min under the rate limit)."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            run_now = st.button("Run audit (use cache)", key="run_audit", type="primary")
        with col2:
            force_refresh = st.button("Force refresh all", key="run_audit_force",
                                       help="Ignore cache and re-audit every patient in the pool")
        with col3:
            if st.button("Wipe cache + re-audit (v20 fix)", key="run_audit_wipe",
                          help="One-click recovery after the v19 endpoint bugs — "
                               "clears the on-disk cache so every patient gets "
                               "freshly audited against the corrected endpoints."):
                from dashboard import audit_cache as _ac
                n = _ac.clear()
                st.success(
                    f"Cleared {n} cached entries. Click 'Force refresh all' or "
                    f"'Run audit (use cache)' to re-run from scratch."
                )

        if run_now or force_refresh:
            client = get_client()
            pool = pool_preview
            progress = st.progress(0)
            status = st.empty()

            partial_key = f"audit_partial_{hash(audit_key)}"
            st.session_state[partial_key] = []

            def cb(i, total, patient_name="", cohort=""):
                progress.progress(i / total)
                tag = f" — {cohort}: {patient_name}" if patient_name else ""
                status.caption(f"Auditing patient {i} of {total}{tag}...")

            def on_result(audit_obj):
                # Checkpoint to session state after every patient so a
                # mid-run crash / rerun / websocket drop doesn't wipe work.
                st.session_state[partial_key].append(audit_obj)

            # v26.5 — build the recall mobile+name lookup from whatever
            # CSV Matt has uploaded this session. audit_patient checks
            # its already-fetched patient object against these sets, so
            # no bulk patient fetch is required.
            recall_lookup = None
            try:
                from dashboard.recalls import build_recall_lookup
                recall_session_df = st.session_state.get("recalls_session_df")
                recall_lookup = build_recall_lookup(session_df=recall_session_df)
                if recall_lookup.get("row_count"):
                    st.caption(
                        f"Audit will use Cliniko Recalls CSV "
                        f"(**{recall_lookup['row_count']}** rows, "
                        f"source: {recall_lookup['source']}) for Check 4."
                    )
            except Exception as e:
                st.caption(f"(Recall-CSV lookup skipped: {type(e).__name__}: {e})")

            try:
                results = run_audit(
                    client, pool, practs,
                    progress_cb=cb,
                    on_result=on_result,
                    use_cache=True,
                    force_refresh=force_refresh,
                    recall_lookup=recall_lookup,
                )
            except Exception as e:
                # Preserve whatever completed before the crash
                results = list(st.session_state.get(partial_key, []))
                st.error(
                    f"Audit stopped early after {len(results)} of {len(pool)} "
                    f"patients: {type(e).__name__}: {e}. "
                    f"Results below are the partial run. Click 'Run audit' again "
                    f"to resume — cached patients won't be re-fetched."
                )

            if results:
                audit_df = aggregate_audit(results)
                by_check = aggregate_by_check(results)
                st.session_state["audit_df"] = audit_df
                st.session_state["audit_by_check"] = by_check
                st.session_state["audit_key"] = audit_key
                st.session_state["audit_patients"] = results
                # Snapshot per-endpoint fetch stats for this run so they
                # survive a rerun of the Streamlit page.
                st.session_state["audit_fetch_stats"] = fetch_stats_snapshot()
                status.caption(
                    f"Audit complete — **{len(results)}** patients scored "
                    f"({cached_hits if not force_refresh else 0} from cache, "
                    f"{len(results) - (cached_hits if not force_refresh else 0)} fresh)."
                )

        # --------------------------------------------------------
        # Endpoint-health diagnostic (v17)
        # --------------------------------------------------------
        # If we just ran an audit (or one's in session_state from a previous
        # click on this same filter), show per-endpoint ok/empty/error counts
        # plus a count of distinct error reasons across patients. This is the
        # view that tells Matt WHY patients are scoring 0 — e.g. the patient
        # endpoint 404s, or the letters endpoint is systematically erroring.
        # Only show stats for this filter. Switching filters shouldn't
        # display stale diagnostics from a previous audit of a different
        # date range / practitioner selection.
        if st.session_state.get("audit_key") == audit_key:
            stats_snapshot = st.session_state.get("audit_fetch_stats")
            cached_results = st.session_state.get("audit_patients")
        else:
            stats_snapshot = None
            cached_results = None
        if stats_snapshot or cached_results:
            st.markdown("**Endpoint health (this run)**")
            st.caption(
                "Only counts fresh API fetches — cached patients skip the API. "
                "`ok` = endpoint returned data. `empty` = endpoint returned an empty "
                "list (normal for patients with no letters/recalls). `error` = the "
                "endpoint raised — this is what breaks an audit."
            )
            if stats_snapshot:
                stats_df = fetch_stats_summary(stats_snapshot)
                if not stats_df.empty:
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
                else:
                    st.caption(
                        "No fresh API calls this run — every patient was served "
                        "from cache. If the scores still look wrong, click **Force "
                        "refresh all** to bypass the cache and re-collect stats."
                    )
            if cached_results:
                err_df = error_reasons_summary(cached_results, limit=15)
                if not err_df.empty:
                    n_errors = int(err_df["count"].sum())
                    n_total = len(cached_results)
                    st.markdown(
                        f"**Error reasons** — {n_errors}/{n_total} patients failed to audit"
                    )
                    st.dataframe(err_df, hide_index=True, use_container_width=True)
                else:
                    st.caption("No per-patient errors in this run's audits.")

        # Cache maintenance controls
        with st.popover("Cache maintenance"):
            st.caption(
                "The cache is a JSONL file at `data/audit_cache/audits.jsonl`. "
                "'Compact' dedupes patients who've been audited more than once "
                "(keeping only the newest). 'Clear' wipes everything."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Compact cache", key="compact_cache"):
                    kept = audit_cache.compact()
                    st.success(f"Cache compacted — {kept} unique patients.")
            with col_b:
                if st.button("Clear cache (danger)", key="clear_cache"):
                    n = audit_cache.clear()
                    st.warning(f"Cleared {n} entries.")
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

    # ------------------------------------------------------------------
    # v26.6 — Per-clinic utilisation rollup
    # Sums numerator/denominator across every practitioner who worked at
    # the business in the selected range. Filtered to the businesses in
    # the filter set (or all if none selected).
    # ------------------------------------------------------------------
    biz_util = getattr(result, "utilisation_by_clinic", None)
    if biz_util is not None and not biz_util.empty:
        biz_ref = filters.get("businesses")
        if biz_ref is not None and not biz_ref.empty:
            biz_util = biz_util.merge(
                biz_ref[["id", "label"]].rename(columns={"id": "business_id"}),
                on="business_id", how="left",
            )
        else:
            biz_util["label"] = biz_util["business_id"].astype(str)
        # Apply the sidebar business filter if one is active
        selected_biz_ids = filters.get("business_ids") or []
        if selected_biz_ids:
            selected_str = [str(b) for b in selected_biz_ids]
            biz_util = biz_util[biz_util["business_id"].isin(selected_str)]
        biz_util = biz_util.sort_values("utilisation", ascending=False).reset_index(drop=True)
        if not biz_util.empty:
            st.markdown("---")
            st.subheader("Clinic utilisation")
            cols = st.columns(min(len(biz_util), 4) or 1)
            for i, row in biz_util.iterrows():
                util_val = row.get("utilisation")
                label = row.get("label") or row.get("business_id")
                pct = f"{(util_val or 0) * 100:.0f}%" if pd.notna(util_val) else "—"
                with cols[i % len(cols)]:
                    kpi_card(str(label), pct)

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
        # v19 — surface which notes source drove notes_completion so Matt
        # can tell at a glance whether the new /treatment_notes path ran
        # or we fell back to the old appt.updated_at proxy.
        notes_attrs = getattr(result.notes, "attrs", {}) or {}
        notes_source = notes_attrs.get("source")
        notes_count = notes_attrs.get("notes_fetched")
        if notes_source:
            icon = "✅" if "treatment_notes" in notes_source else "⚠️"
            parts = [f"{icon} Notes source: **{notes_source}**"]
            # v23 — show scanned vs usable so we can spot the self-fallback
            # bug recurrence (scanned high but matched=0 means note→appt
            # linkage broke again).
            scanned = notes_attrs.get("notes_scanned")
            dropped = notes_attrs.get("notes_dropped_no_appt_link")
            matched = notes_attrs.get("notes_matched_appts")
            finalised = notes_attrs.get("notes_finalised_appts")
            still_draft = notes_attrs.get("notes_still_draft")
            delivered_n = notes_attrs.get("delivered_appts")
            if scanned is not None:
                parts.append(f"(scanned {scanned}, kept {notes_count}, "
                             f"dropped-no-appt-link {dropped})")
            # v25 — show finalised vs draft split. A note that has been
            # linked to an appt but is still in draft cannot pass the 24h
            # medicolegal check — this line tells you how many are in that
            # state, which is the actionable number for practitioner chasing.
            if matched is not None and delivered_n is not None:
                if finalised is not None:
                    parts.append(f"— matched {matched}/{delivered_n} "
                                 f"(finalised {finalised}, "
                                 f"still draft {still_draft or 0})")
                else:
                    parts.append(f"— matched {matched}/{delivered_n} delivered appts")
            if "fallback" in notes_source:
                parts.append(". Endpoint returned no data — numbers will be "
                             "pessimistic (same behaviour as v18 and earlier).")
            st.caption(" ".join(parts))
            # v24 — when the extractor couldn't find an appointment ref
            # on enough notes, dump the PHI-safe structural shape of the
            # first unmatched note so we can see where Cliniko is stashing
            # the appt link on this account. The shape contains only key
            # names and reference URLs — never clinical content.
            # v25.1 — only surface this when delivered-appt match rate is
            # below 95%. In normal operation some notes are standalone
            # patient records with no appt link and that's fine; the
            # diagnostic is noise. It resurfaces automatically if Cliniko
            # changes their schema and the match rate drops.
            first_shape = notes_attrs.get("first_unmatched_note_shape")
            _show_diag = False
            if first_shape:
                if matched is not None and delivered_n and delivered_n > 0:
                    _show_diag = (matched / delivered_n) < 0.95
                else:
                    # No match-rate context available — show it so we
                    # don't hide a genuine problem.
                    _show_diag = True
            if _show_diag:
                with st.expander(
                    "🔬 Diagnostic: unmatched treatment_note shape "
                    "(PHI-safe — keys only, no content)", expanded=False,
                ):
                    st.caption(
                        "Delivered-appt match rate is below 95% — Cliniko "
                        "may have changed their response shape. Copy the "
                        "JSON below and send it back so we can extend the "
                        "extractor."
                    )
                    st.json(first_shape)
        # Diagnostic: confirm the audit merge actually attached audit_pct
        # with real values (not all-zero). This is the check that would have
        # caught the dtype-mismatch bug that silently made audits invisible.
        audit_cols_present = [c for c in ["audit_pct", "audit_epc_pct",
                                            "audit_private_pct", "patients_audited"]
                              if c in scored.columns]
        if audit_df is None or audit_df.empty:
            st.caption("Audit not run yet for this filter — click 'Run audit' in the Audit expander above.")
        elif audit_cols_present:
            audited_rows = int((scored.get("patients_audited", pd.Series([0])) > 0).sum())
            mean_audit = float(scored.loc[scored.get("patients_audited", 0) > 0, "audit_pct"].mean() * 100) \
                if audited_rows else 0.0
            st.caption(
                f"Audit merged: **{audited_rows}/{len(scored)}** practitioners have audit data. "
                f"Mean audit % among audited practitioners: **{mean_audit:.1f}%**. "
                f"Audit columns in table: {audit_cols_present}."
            )
        else:
            st.warning(
                "Audit ran but audit columns didn't merge into the scored table. "
                "This usually means a practitioner_id dtype mismatch — please "
                "send me a screenshot and I'll dig in."
            )
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
    # Resolve the active date range so downstream NPS / punctuality upload
    # UIs can prefill week_starting defaults. If the preset fails to
    # resolve we keep going — the uploads still work, the default just
    # becomes blank.
    try:
        dr = resolve_preset(
            filters["preset"],
            filters.get("custom_start"),
            filters.get("custom_end"),
        )
    except Exception:
        dr = None

    st.subheader("Punctuality — upload paper sheet(s)")
    st.caption("Drop one or more photos/PDFs — ideally all three clinics "
               "(Albury, Wodonga, Lavington) at once each Friday. Claude "
               "Vision reads each sheet; you review and save per-clinic.")
    # v25.2 — accept multiple files so Matt can drop all three clinic
    # sheets at once each Friday instead of uploading one at a time.
    uploaded_list = st.file_uploader(
        "Punctuality photos (one clinic per file) — iPhone HEIC also accepted",
        type=["jpg", "jpeg", "png", "heic", "heif", "webp", "pdf"],
        accept_multiple_files=True,
        key="punct_upload",
    )
    if uploaded_list:
        st.caption(f"📁 {len(uploaded_list)} file(s) queued: "
                   + ", ".join(f.name for f in uploaded_list))
        if st.button(f"Extract all {len(uploaded_list)} with Claude Vision",
                      key="vision_btn"):
            extracted_by_file: dict[str, pd.DataFrame] = {}
            progress = st.progress(0.0, text="Starting vision extraction…")
            failures: list[tuple[str, str]] = []
            for i, f in enumerate(uploaded_list, start=1):
                progress.progress(
                    (i - 1) / len(uploaded_list),
                    text=f"Extracting {i}/{len(uploaded_list)}: {f.name}",
                )
                try:
                    data = extract_punctuality_from_image(
                        f.getvalue(),
                        media_type=f.type or "image/jpeg",
                        filename_hint=f.name,
                    )
                    df = vision_response_to_dataframe(data)
                    extracted_by_file[f.name] = df
                except Exception as e:
                    failures.append((f.name, str(e)))
            progress.progress(1.0, text=f"Extracted {len(extracted_by_file)}/{len(uploaded_list)}")
            if extracted_by_file:
                # Store all successfully-extracted frames keyed by filename so
                # each clinic's review block is independent.
                st.session_state["punct_extracted_multi"] = extracted_by_file
                st.success(
                    f"Extracted {sum(len(df) for df in extracted_by_file.values())} "
                    f"rows across {len(extracted_by_file)} sheet(s). "
                    "Review and save each below."
                )
            if failures:
                for name, err in failures:
                    st.error(f"❌ {name}: {err}")

    if "punct_extracted_multi" in st.session_state:
        # One review + save block per uploaded file — keeps clinics separate
        # so Matt can commit (or discard) each one independently.
        combined_session: list[pd.DataFrame] = []
        for fname, df in list(st.session_state["punct_extracted_multi"].items()):
            clinic_hint = df["clinic"].iloc[0] if not df.empty and "clinic" in df.columns else "?"
            week_hint = df["week_starting"].iloc[0] if not df.empty and "week_starting" in df.columns else "?"
            with st.expander(f"📋 {fname} — clinic: **{clinic_hint}** · week: **{week_hint}** · {len(df)} rows",
                              expanded=True):
                edited = st.data_editor(
                    df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"punct_editor_{fname}",
                )
                # Overwrite the per-file frame so edits stick across reruns.
                st.session_state["punct_extracted_multi"][fname] = edited
                if not edited.empty:
                    combined_session.append(edited)
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"💾 Save {fname} to data/punctuality/",
                                  key=f"punct_save_{fname}"):
                        week = edited["week_starting"].iloc[0] if not edited.empty else "unknown"
                        clinic = edited["clinic"].iloc[0] if not edited.empty else "unknown"
                        path = save_punctuality_csv(edited, str(week), str(clinic))
                        st.success(f"Saved: {path.name}")
                        # v26 — surface GitHub sync result alongside disk save
                        from dashboard.manual import last_github_sync as _lgs
                        sync = _lgs("punctuality")
                        if sync is not None:
                            ok, msg = sync
                            (st.success if ok else st.warning)(msg)
                        # Drop just this file's frame; leave the others for
                        # the user to handle independently.
                        st.session_state["punct_extracted_multi"].pop(fname, None)
                        st.rerun()
                with col2:
                    if st.button(f"🗑️ Discard {fname}", key=f"punct_discard_{fname}"):
                        st.session_state["punct_extracted_multi"].pop(fname, None)
                        st.rerun()
        # v18 — mirror the combined edits into session_state so the Overview
        # tab sees all three clinics immediately without requiring the Save
        # button (useful before Matt commits to disk).
        if combined_session:
            st.session_state["punct_session_df"] = pd.concat(
                combined_session, ignore_index=True,
            )
        # "Save all" and "Clear session" controls at the bottom
        st.markdown("&nbsp;")
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            if st.button("💾💾 Save ALL to data/punctuality/", key="punct_save_all"):
                saved = []
                for fname, df in list(st.session_state["punct_extracted_multi"].items()):
                    if df.empty:
                        continue
                    week = df["week_starting"].iloc[0]
                    clinic = df["clinic"].iloc[0]
                    path = save_punctuality_csv(df, str(week), str(clinic))
                    saved.append(path.name)
                if saved:
                    st.success(f"Saved {len(saved)}: {', '.join(saved)}")
                    from dashboard.manual import last_github_sync as _lgs
                    sync = _lgs("punctuality")
                    if sync is not None:
                        ok, msg = sync
                        (st.success if ok else st.warning)(
                            f"GitHub sync (last file): {msg}"
                        )
                    st.session_state["punct_extracted_multi"] = {}
                    st.rerun()
                else:
                    st.warning("Nothing to save — all frames were empty.")
        with bcol2:
            if st.button("Clear session punct", key="punct_clear_session",
                          help="Drop the in-memory punctuality upload"):
                st.session_state.pop("punct_session_df", None)
                st.info("Session punctuality cleared.")
        with bcol3:
            if st.button("🗑️ Discard all", key="punct_discard_all"):
                st.session_state["punct_extracted_multi"] = {}
                st.rerun()

    # v25.2 — panel showing what's currently persisted on disk, so Matt can
    # see his cumulative dataset (and notice if Streamlit Cloud has wiped it
    # on reboot).
    try:
        from dashboard.config import DATA_DIR as _DD
        punct_dir = _DD / "punctuality"
        on_disk = sorted(punct_dir.glob("*.csv")) if punct_dir.exists() else []
        if on_disk:
            by_week: dict[str, list[str]] = {}
            for p in on_disk:
                # filename pattern: YYYY-MM-DD_<clinic>.csv
                name = p.stem
                parts = name.split("_", 1)
                week = parts[0] if len(parts) == 2 else "?"
                clinic = parts[1] if len(parts) == 2 else name
                by_week.setdefault(week, []).append(clinic)
            summary_lines = [
                f"• **{w}** → {', '.join(sorted(by_week[w]))}"
                for w in sorted(by_week.keys(), reverse=True)[:8]
            ]
            st.caption(
                f"🗂️ **{len(on_disk)}** punctuality file(s) currently on disk "
                f"across **{len(by_week)}** week(s):\n\n" + "\n".join(summary_lines)
            )
        else:
            st.caption("🗂️ No punctuality files on disk yet for this session.")
    except Exception:
        pass

    st.markdown("---")
    st.subheader("Punctuality — manual CSV")
    st.caption("If you don't want to use vision extraction, fill this template and save "
               "to `data/punctuality/YYYY-MM-DD_<clinic>.csv`.")
    template = pd.DataFrame(columns=PUNCTUALITY_COLUMNS)
    st.dataframe(template, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("NPS — CSV upload")
    st.caption(
        "Two accepted shapes: **(1) individual scores** — one row per "
        "response with a practitioner name + a 0-10 score; the app buckets "
        "them into promoters (9-10), passives (7-8), detractors (0-6) and "
        "computes NPS = (P - D) ÷ total × 100. **(2) pre-aggregated** — "
        "already summarised per practitioner with columns "
        "responses/promoters/passives/detractors."
    )

    nps_mode = st.radio(
        "CSV format",
        options=["Individual scores (auto-calculate)", "Pre-aggregated"],
        index=0,
        key="nps_mode",
        horizontal=True,
    )
    nps_file = st.file_uploader(
        "NPS CSV export", type=["csv"], key="nps_upload"
    )
    if nps_file:
        try:
            new_nps_raw = pd.read_csv(nps_file)
            st.write(f"File loaded — **{len(new_nps_raw)}** rows, "
                     f"columns: {list(new_nps_raw.columns)}")
            st.dataframe(new_nps_raw.head(20), use_container_width=True)

            if nps_mode.startswith("Individual"):
                default_week_value = str(dr.start_date) if dr is not None else ""
                default_week = st.text_input(
                    "Default week_starting (YYYY-MM-DD) — used only if the "
                    "CSV has no date column",
                    value=default_week_value,
                    key="nps_default_week",
                )
                default_clinic_name = st.text_input(
                    "Default clinic name (used only if the CSV has no "
                    "clinic/location column)",
                    value="",
                    key="nps_default_clinic",
                )
                try:
                    aggregated = aggregate_nps_from_individual_scores(
                        new_nps_raw,
                        default_week_starting=default_week or None,
                        default_clinic=default_clinic_name,
                    )
                    # v18: mirror the aggregated frame into session_state so
                    # Overview tab picks it up on the next rerun without
                    # requiring the Save button. Streamlit Cloud's ephemeral
                    # disk means "Save to data/nps/" doesn't survive a
                    # redeploy, so session_state is now the primary path.
                    st.session_state["nps_session_df"] = aggregated.copy()
                    st.success(
                        f"Applied **{len(aggregated)}** aggregated rows to "
                        "this session — scores will show up in the Overview tab "
                        "immediately. Use the **Save** button below to also "
                        "persist to disk (optional; session survives until reload)."
                    )
                    st.markdown("**Aggregated per practitioner:**")
                    st.dataframe(aggregated, use_container_width=True)
                    if dr is not None:
                        default_name = f"nps_{dr.start_date}_{dr.end_date_inclusive}.csv"
                    else:
                        default_name = "nps.csv"
                    name = st.text_input(
                        "Save as filename", value=default_name, key="nps_save_name",
                    )
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        if st.button("Save aggregated NPS to data/nps/", key="nps_save_agg"):
                            out = save_nps_csv(aggregated, name)
                            st.success(f"Saved: {out.name} ({len(aggregated)} practitioner rows)")
                            from dashboard.manual import last_github_sync as _lgs
                            sync = _lgs("nps")
                            if sync is not None:
                                ok, msg = sync
                                (st.success if ok else st.warning)(msg)
                    with col_s2:
                        if st.button("Clear session upload", key="nps_clear_session",
                                      help="Remove the in-memory upload without touching disk files"):
                            st.session_state.pop("nps_session_df", None)
                            st.info("Session NPS cleared. Re-upload the CSV to re-apply.")
                except ValueError as e:
                    st.error(str(e))
            else:
                # v18: pre-aggregated uploads also auto-apply via session_state
                st.session_state["nps_session_df"] = new_nps_raw.copy()
                st.success(
                    f"Applied **{len(new_nps_raw)}** pre-aggregated rows to "
                    "this session — check the Overview tab."
                )
                name = st.text_input(
                    "Save as filename (YYYY-MM.csv)", value="nps.csv",
                    key="nps_save_name_preagg",
                )
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    if st.button("Save NPS CSV", key="nps_save_preagg"):
                        out = save_nps_csv(new_nps_raw, name)
                        st.success(f"Saved: {out.name}")
                        from dashboard.manual import last_github_sync as _lgs
                        sync = _lgs("nps")
                        if sync is not None:
                            ok, msg = sync
                            (st.success if ok else st.warning)(msg)
                with col_p2:
                    if st.button("Clear session upload", key="nps_clear_session_preagg"):
                        st.session_state.pop("nps_session_df", None)
                        st.info("Session NPS cleared.")
        except Exception as e:
            st.error(f"NPS load failed: {e}")

    # --------------------------------------------------------------
    # v26.5 — Cliniko Recalls CSV upload
    # --------------------------------------------------------------
    st.markdown("---")
    st.subheader("Cliniko Recalls — CSV upload (for audit Check 4)")
    st.caption(
        "Cliniko doesn't expose recalls via API, so we scrape the "
        "**Patient recalls** report page in your browser with a one-click "
        "bookmarklet. Check 4 then passes for any patient whose mobile or "
        "name appears in this CSV (in addition to those with upcoming "
        "appointments). Upload once a week; the file is auto-committed to "
        "GitHub so it survives Streamlit Cloud reboots."
    )

    with st.expander("How to get the CSV (first-time setup — 20 seconds)"):
        st.markdown(
            "**Option A — Console paste (no setup, quickest one-off):**\n\n"
            "1. Open Cliniko → Reports → **Patient recalls**. Widen the "
            "date range to today → +18-24 months. Leave *Hide recalled* "
            "unchecked.\n"
            "2. Press **F12** to open DevTools. Click the **Console** tab.\n"
            "3. Paste the script below, press Enter.\n"
            "4. Wait ~1-2 min while it auto-clicks Load More. A CSV will "
            "download to your Downloads folder.\n"
            "5. Drop it in the uploader below.\n\n"
            "**Option B — Bookmarklet (recommended for weekly use):**\n\n"
            "1. Show your bookmarks bar (**Ctrl/⌘+Shift+B**).\n"
            "2. Right-click the bookmarks bar → **Add page…** (or **Add "
            "bookmark**).\n"
            "3. Name it something like `Export Cliniko Recalls`. Paste the "
            "**Bookmarklet URL** (below) as the URL/address field.\n"
            "4. Save. From now on, open the Recalls report and click the "
            "bookmark — CSV downloads automatically."
        )
        st.code(_RECALL_BOOKMARKLET_CONSOLE, language="javascript")
        st.markdown("**Bookmarklet URL (one-line — paste into the bookmark's URL field):**")
        st.code(_RECALL_BOOKMARKLET_URL, language="text")

    recall_file = st.file_uploader(
        "Cliniko Recalls CSV (from the bookmarklet)",
        type=["csv"], key="recalls_upload",
    )
    if recall_file:
        try:
            from dashboard.recalls import (
                parse_recalls_csv, save_recalls_csv,
                last_github_sync as _lgs_recalls,
                build_recall_lookup,
            )
            new_df = parse_recalls_csv(recall_file)
            st.write(
                f"File loaded — **{len(new_df)}** recall rows. Preview:"
            )
            st.dataframe(new_df.head(15), use_container_width=True, hide_index=True)

            # Auto-apply to this session so the Overview tab's audit uses it
            st.session_state["recalls_session_df"] = new_df.copy()
            lookup = build_recall_lookup(session_df=new_df)
            st.success(
                f"Applied to this session — **{len(new_df)}** rows, "
                f"**{len(lookup['mobiles'])}** unique mobiles, "
                f"**{len(lookup['names'])}** unique (first, last) names. "
                f"Re-run the audit to see Check 4 scored against this list."
            )

            # Save + push to GitHub
            from datetime import date as _date
            default_label = f"recalls_{_date.today().isoformat()}"
            label = st.text_input(
                "Save as filename (without .csv)", value=default_label,
                key="recalls_save_label",
            )
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if st.button("Save recalls CSV", key="recalls_save"):
                    out = save_recalls_csv(new_df, label)
                    st.success(f"Saved: {out.name} ({len(new_df)} rows)")
                    sync = _lgs_recalls("recalls")
                    if sync is not None:
                        ok, msg = sync
                        (st.success if ok else st.warning)(msg)
            with col_r2:
                if st.button("Clear session recalls", key="recalls_clear_session"):
                    st.session_state.pop("recalls_session_df", None)
                    st.info("Session recalls cleared.")
        except Exception as e:
            st.error(f"Recalls CSV load failed: {e}")


# -------------------------------------------------------------------
# v26.5 — Cliniko Recalls bookmarklet
# -------------------------------------------------------------------
# Console form (readable, for the Option A paste-in-DevTools path).
_RECALL_BOOKMARKLET_CONSOLE = """(async () => {
  const all = (s) => Array.from(document.querySelectorAll(s));
  const findBtn = () => all('button, a').find(el =>
    /load\\s*more/i.test((el.textContent || '').trim()) && el.offsetParent !== null);
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  let last = 0, stable = 0;
  for (let i = 0; i < 300; i++) {
    const btn = findBtn();
    if (!btn) { console.log('No Load more button — list complete.'); break; }
    btn.scrollIntoView({block: 'center'}); btn.click();
    await sleep(700);
    const n = all('tr').length;
    console.log(`click ${i+1}: ${n} rows`);
    if (n === last) { if (++stable > 4) break; } else { stable = 0; last = n; }
  }

  const phoneRe = /0\\d{3}\\s\\d{3}\\s\\d{3}/;
  const dateRe  = /\\d{1,2}\\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+20\\d{2}/;
  const rows = all('tr').filter(tr => phoneRe.test(tr.textContent) && dateRe.test(tr.textContent));
  console.log(`Found ${rows.length} recall rows.`);

  const data = rows.map(tr => Array.from(tr.cells).map(c => c.textContent.trim().replace(/\\s+/g,' ')));
  const width = Math.max(...data.map(r => r.length));
  const csv = data.map(r => r.concat(Array(width - r.length).fill(''))
      .map(v => `"${v.replace(/"/g,'""')}"`).join(',')).join('\\n');
  const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8;'}));
  const a = document.createElement('a');
  a.href = url; a.download = 'cliniko_recalls.csv'; document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
  console.log('Saved cliniko_recalls.csv');
})();"""

# Bookmarklet form (single line, URL-safe, prefixed with `javascript:`).
_RECALL_BOOKMARKLET_URL = (
    "javascript:(async()=>{const a=s=>Array.from(document.querySelectorAll(s)),"
    "b=()=>a('button,a').find(e=>/load\\s*more/i.test((e.textContent||'').trim())&&e.offsetParent!==null),"
    "s=m=>new Promise(r=>setTimeout(r,m));let l=0,st=0;"
    "for(let i=0;i<300;i++){const k=b();if(!k)break;k.scrollIntoView({block:'center'});k.click();"
    "await s(700);const n=a('tr').length;if(n===l){if(++st>4)break}else{st=0;l=n}}"
    "const p=/0\\d{3}\\s\\d{3}\\s\\d{3}/,d=/\\d{1,2}\\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\\s+20\\d{2}/,"
    "r=a('tr').filter(t=>p.test(t.textContent)&&d.test(t.textContent)),"
    "dd=r.map(t=>Array.from(t.cells).map(c=>c.textContent.trim().replace(/\\s+/g,' '))),"
    "w=Math.max(...dd.map(x=>x.length)),"
    "c=dd.map(x=>x.concat(Array(w-x.length).fill('')).map(v=>`\"${v.replace(/\"/g,'\"\"')}\"`).join(',')).join('\\n'),"
    "u=URL.createObjectURL(new Blob([c],{type:'text/csv;charset=utf-8;'})),"
    "el=document.createElement('a');el.href=u;el.download='cliniko_recalls.csv';"
    "document.body.appendChild(el);el.click();document.body.removeChild(el);URL.revokeObjectURL(u);"
    "alert('Saved cliniko_recalls.csv ('+r.length+' rows)');})();"
)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main() -> None:
    _require_password()
    st.title("Enhance Physio — Performance Dashboard")
    filters = sidebar_filters()

    (tab_overview, tab_manual, tab_injuries, tab_review,
     tab_commission, tab_diag) = st.tabs(
        ["Overview", "Manual data", "🩻 Injuries",
         "🩺 Clinical Review", "💰 Commission", "🔧 Diagnostics"]
    )
    with tab_overview:
        overview_tab(filters)
    with tab_manual:
        manual_tab(filters)
    with tab_injuries:
        injuries_tab(filters)
    with tab_review:
        clinical_review_tab(filters)
    with tab_commission:
        commission_tab(filters)
    with tab_diag:
        diagnostics_tab(filters)


# -------------------------------------------------------------------
# Tab: Injuries (v26.8)
# -------------------------------------------------------------------
def _lazy_gate(tab_name: str, load_button_label: str,
                info_line: str) -> bool:
    """v26.11.5 — Lazy-load gate for heavy tabs.

    Returns True if the tab is authorised to run its heavy fetch.
    Returns False (after rendering a Load button) otherwise. Streamlit
    executes every tab function on every rerun; without a gate, the
    Commission tab has to wait for Overview + Clinical Review + Injuries
    to finish their fetches even when the user just wants Commission.

    Once loaded in a session, the tab stays loaded; the Force Refresh
    button (per-tab) can un-set it if needed.
    """
    key = f"loaded_{tab_name}"
    if st.session_state.get(key):
        return True
    st.info(info_line)
    if st.button(load_button_label, key=f"load_{tab_name}", type="primary"):
        st.session_state[key] = True
        st.rerun()
    return False


def injuries_tab(filters: dict) -> None:
    """Injury-area analytics from initial-consult treatment notes.

    Shows: bar chart of total counts per category, per-clinic
    breakdown table, monthly trend, optional per-practitioner drill.
    """
    st.subheader("🩻 Injuries seen — by area")
    if not _lazy_gate(
        "injuries",
        "Load Injuries data",
        "Click below to pull treatment notes for the selected range "
        "(~30–60s first time, then cached).",
    ):
        return
    from dashboard.injuries import (
        injuries_breakdown, total_by_category,
        by_category_and_clinic, by_category_monthly,
        by_category_and_practitioner,
    )
    from dashboard.metrics import fetch_appointments
    from dashboard.date_ranges import resolve_preset

    st.caption(
        "Pulled from the **Area of Injury** checkbox on initial-consult "
        "treatment notes. When checkboxes weren't ticked, falls back to "
        "keyword-scanning the **HoPC** paragraph. A patient with multiple "
        "areas counts in each — totals reflect 'things the clinic saw', "
        "not unique patients."
    )

    try:
        dr = resolve_preset(
            filters["preset"], filters["custom_start"], filters["custom_end"],
        )
    except Exception as e:
        st.error(f"Date range error: {e}")
        return

    st.caption(f"Range: **{dr.label()}**")

    with st.spinner("Pulling treatment notes from Cliniko…"):
        try:
            client = get_client()
            appt_types = cached_appointment_types()
            businesses = cached_businesses()
            practitioners = filters.get("practitioners")
            appts = fetch_appointments(
                client, dr,
                business_ids=list(filters.get("business_ids") or []) or None,
                practitioner_ids=list(filters.get("practitioner_ids") or []) or None,
            )
            breakdown = injuries_breakdown(
                client, appts, dr,
                appointment_types=appt_types,
                practitioners=practitioners,
            )
        except Exception as e:
            st.error(f"Couldn't load injury data: {e}")
            return

    if breakdown.empty:
        st.info(
            "No injury records found for this range. Either no initial "
            "consults occurred, or the notes don't have an Area of Injury "
            "checkbox / HoPC paragraph that the parser could read. Check "
            "the Diagnostics tab for the raw note shape."
        )
        return

    # 1. Totals bar chart
    totals = total_by_category(breakdown)
    st.markdown("### Total by category")
    n_notes = breakdown["appointment_id"].nunique()
    n_records = len(breakdown)
    st.caption(
        f"**{n_notes}** initial consults • **{n_records}** injury records "
        f"(some patients tick multiple areas)"
    )
    if not totals.empty:
        st.bar_chart(totals.set_index("category")["count"])
        st.dataframe(totals, hide_index=True, width="stretch")

    # 2. Per-clinic breakdown
    st.markdown("### By clinic")
    by_clinic = by_category_and_clinic(breakdown, businesses)
    if not by_clinic.empty:
        st.dataframe(by_clinic, width="stretch")

    # 3. Monthly trend
    st.markdown("### Monthly trend")
    monthly = by_category_monthly(breakdown)
    if not monthly.empty:
        # Pivot for chart: months on x-axis, one column per category
        pivot = monthly.pivot(index="month", columns="category",
                                values="count").fillna(0)
        st.line_chart(pivot)
    else:
        st.caption("Not enough data with valid dates for a monthly trend.")

    # 4. Per-practitioner drill-down (toggle)
    if st.toggle("Show per-practitioner breakdown", value=False,
                  key="injuries_per_prac"):
        st.markdown("### By practitioner")
        by_prac = by_category_and_practitioner(breakdown, practitioners)
        if not by_prac.empty:
            st.dataframe(by_prac, width="stretch")

    # 5. Diagnostic (v26.8.1) — surface the raw note JSON so the parser
    # can be tuned to whatever shape Cliniko actually returns. Useful
    # when totals look suspicious (e.g. every category equal).
    with st.expander("🔍 Diagnostic: raw note JSON shape", expanded=False):
        st.caption(
            "Shows the structured `content` of the first 2 initial-consult "
            "treatment notes. Use this to debug parser behaviour — paste "
            "back to Matt if the totals look wrong."
        )
        try:
            from dashboard.injuries import (
                _fetch_treatment_notes_with_content,
                _initial_appointment_ids, _appt_id_from_note,
                extract_injury_text,
            )
            initial_ids = _initial_appointment_ids(appts, appt_types)
            sample_notes = _fetch_treatment_notes_with_content(client, dr)
            shown = 0
            for note in sample_notes:
                aid = _appt_id_from_note(note)
                if aid is None or aid not in initial_ids:
                    continue
                cb_text, hopc_text = extract_injury_text(note)
                st.markdown(f"**Note (appointment id: `{aid}`)**")
                st.write(f"Parser found checkbox text: `{cb_text!r}`")
                st.write(f"Parser found HoPC text: `{hopc_text[:200]!r}` "
                          f"({len(hopc_text)} chars total)")
                st.json(note.get("content"), expanded=False)
                shown += 1
                if shown >= 2:
                    break
            if shown == 0:
                st.info("No initial-consult notes in the current range.")
        except Exception as e:
            st.error(f"Diagnostic failed: {e}")


# -------------------------------------------------------------------
# Tab: Clinical Review (v26.9)
# -------------------------------------------------------------------
def _cached_clinical_review(_schema_version: str = ""):
    """v26.11 — disk-only cache (reading a pickle is <100ms; no need
    for Streamlit's in-memory layer). The progress bar lives here
    because Streamlit's @st.cache_data doesn't play well with UI
    inside it.
    """
    from dashboard.clinical_review import compute_clinical_review
    from dashboard.config import DATA_DIR
    import pickle, time

    cache_dir = DATA_DIR / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "clinical_review.pkl"

    # Disk cache fast path — 12h TTL
    if cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        if age_seconds < 48 * 3600:
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

    # Cache miss — run with progress bar
    progress_bar = st.progress(0.0, text="Starting…")

    def cb(done: int, total: int, message: str):
        if total <= 0:
            progress_bar.progress(0.0, text=message)
        else:
            progress_bar.progress(min(1.0, done / max(total, 1)), text=message)

    result = compute_clinical_review(get_client(), progress_callback=cb)
    progress_bar.empty()

    # Persist for next cold start
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass
    return result


def _clear_clinical_review_cache() -> None:
    from dashboard.config import DATA_DIR
    cache_file = DATA_DIR / "_cache" / "clinical_review.pkl"
    try:
        cache_file.unlink(missing_ok=True)
    except Exception:
        pass
    _cached_clinical_review.clear()


def clinical_review_tab(filters: dict) -> None:
    """Surfaces patients flagged for clinical review:
      * Over-servicing — still-active patients past their bucket's
        appt-count or duration threshold.
      * Under-servicing — initial-only patients who never came back.

    Filterable by practitioner. The whole tab is independent of the
    sidebar date range — it always looks at the last 12 months back
    + 6 months forward (these are the windows we need to make the
    rules work). Practitioner filter still applies though, so a
    senior physio can review only their own caseload.
    """
    st.subheader("🩺 Clinical Review queue")
    if not _lazy_gate(
        "clinical_review",
        "Load Clinical Review",
        "Click below to pull 12 months of patient histories from Cliniko "
        "(~2–3 min first time, then cached for 48h).",
    ):
        return
    from dashboard.clinical_review import attach_practitioner_names
    from dashboard import __version__ as _v

    st.caption(
        "Patients still being seen but past their funding bucket's "
        "threshold (potential over-servicing — flag for clinical review, "
        "second opinion, or referral), plus initial-only patients who "
        "didn't return (potential under-servicing).  \n"
        "Active = delivered appt in last 14 days **OR** future booking. "
        "Thresholds editable in `settings.yml`."
    )

    practitioners = filters.get("practitioners")
    practitioner_filter = filters.get("practitioner_ids") or []

    # Force-refresh button up top
    if st.button("🔄 Force refresh", key="clinical_review_force_refresh",
                  help="Clear cache and re-fetch from Cliniko. "
                        "Cached data is up to 12h old."):
        _clear_clinical_review_cache()
        st.rerun()

    with st.spinner("Pulling 12 months of appointments from Cliniko (~30s first time)…"):
        try:
            over_df, under_df = _cached_clinical_review(_schema_version=_v)
        except Exception as e:
            st.error(f"Couldn't compute clinical review: {e}")
            return

    over_df = attach_practitioner_names(over_df, practitioners)
    under_df = attach_practitioner_names(under_df, practitioners)

    # Apply practitioner filter if set in the sidebar
    if practitioner_filter:
        pf = {str(p) for p in practitioner_filter}
        if not over_df.empty:
            over_df = over_df[over_df["practitioner_id"].astype(str).isin(pf)]
        if not under_df.empty:
            under_df = under_df[under_df["practitioner_id"].astype(str).isin(pf)]

    # ----- Over-servicing section -----
    st.markdown("### Over-servicing — past threshold")
    if over_df.empty:
        st.success("No patients flagged for over-servicing review. 🎉")
    else:
        n_total = len(over_df)
        # Quick KPI strip — counts per bucket
        bucket_counts = (over_df.groupby("bucket").size()
                                  .sort_values(ascending=False))
        cols = st.columns(len(bucket_counts) + 1)
        cols[0].metric("Total flagged", n_total)
        for i, (bucket, count) in enumerate(bucket_counts.items(), start=1):
            cols[i].metric(bucket, int(count))

        # Display table
        display_cols = ["patient", "bucket", "appts_count",
                          "days_since_initial", "initial_date",
                          "last_appt_date", "has_future_appt",
                          "practitioner", "flag_reason", "cliniko_url"]
        display = over_df[[c for c in display_cols if c in over_df.columns]].rename(
            columns={
                "patient": "Patient",
                "bucket": "Bucket",
                "appts_count": "Appts",
                "days_since_initial": "Days since initial",
                "initial_date": "Initial",
                "last_appt_date": "Last seen",
                "has_future_appt": "Future booked?",
                "practitioner": "Practitioner",
                "flag_reason": "Flag reason",
                "cliniko_url": "Cliniko",
            }
        )
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "Cliniko": st.column_config.LinkColumn(
                    "Cliniko", display_text="Open ↗"
                ),
            },
        )

    # ----- Under-servicing section -----
    st.markdown("### Under-servicing — initial only, no follow-up")
    if under_df.empty:
        st.success("No initial-only patients with missing follow-up. 🎉")
    else:
        st.metric("Patients with no follow-up after initial", len(under_df))
        display_cols_u = ["patient", "initial_appt_type", "initial_date",
                            "days_since_initial", "practitioner", "cliniko_url"]
        display_u = under_df[[c for c in display_cols_u if c in under_df.columns]].rename(
            columns={
                "patient": "Patient",
                "initial_appt_type": "Initial type",
                "initial_date": "Initial date",
                "days_since_initial": "Days since",
                "practitioner": "Practitioner",
                "cliniko_url": "Cliniko",
            }
        )
        st.dataframe(
            display_u,
            hide_index=True,
            width="stretch",
            column_config={
                "Cliniko": st.column_config.LinkColumn(
                    "Cliniko", display_text="Open ↗"
                ),
            },
        )

    st.caption(
        "Cache: 30 minutes. Open the **⋮ menu** above and pick **Clear cache** "
        "if you've just made schedule changes you want reflected immediately."
    )

    # --- Diagnostic expander (v26.10.3) -----------------------------
    with st.expander("🔍 Diagnostic: filter funnel + bucket coverage",
                       expanded=False):
        st.caption(
            "If the queue is unrealistically empty, click below to see "
            "where the filter chain is dropping patients."
        )
        if st.button("Run diagnostic (extra ~30s API call)",
                      key="clinical_review_diag_run"):
            from dashboard.clinical_review import compute_clinical_review_diagnostic
            with st.spinner("Pulling 12 months of appointments…"):
                try:
                    diag = compute_clinical_review_diagnostic(get_client())
                except Exception as e:
                    st.error(f"Diagnostic failed: {e}")
                    diag = None
            if diag:
                st.markdown("**Funnel counters**")
                st.json({
                    "Lookback window (days)":     diag["lookback_window_days"],
                    "Appts pulled (12 months)":   diag["delivered_total"],
                    "After cancellation/DNA filter": diag["after_is_delivered"],
                    "Future bookings pulled":     diag["future_total"],
                    "Appt types in clinic":       diag["appt_types_total"],
                })
                st.markdown("**Bucket distribution**")
                st.caption(
                    "How many delivered appts fell into each funding bucket. "
                    "If `(unmatched)` is huge, the bucket regex needs tweaking."
                )
                st.json(diag["bucket_distribution"])
                if diag["unmatched_appt_types_sample"]:
                    st.markdown("**Top unmatched appt-type names**")
                    st.caption(
                        "These would never trigger an over-servicing flag because "
                        "they're not classified into any bucket. If you see a "
                        "funded type here, tell me and I'll add the pattern."
                    )
                    st.dataframe(
                        pd.DataFrame(diag["unmatched_appt_types_sample"]),
                        hide_index=True, width="stretch",
                    )
                if diag["top10_patients_by_appt_count"]:
                    st.markdown("**Top 10 patients by lifetime appt count (any bucket)**")
                    st.caption(
                        "Anyone here exceeding their bucket's threshold should "
                        "appear in the over-servicing queue above. If they "
                        "don't, the active-window filter is probably the cause "
                        "(e.g. they finished treatment >14d ago with no future "
                        "booking)."
                    )
                    st.dataframe(
                        pd.DataFrame(diag["top10_patients_by_appt_count"]),
                        hide_index=True, width="stretch",
                    )
                else:
                    st.warning(
                        "No patients found with funded-bucket appts at all. "
                        "Either the appointments fetch returned nothing (check "
                        "Cliniko credentials) or every appt-type is unmatched "
                        "(check the bucket regex above)."
                    )


# -------------------------------------------------------------------
# Tab: Commission Calculator (v26.10)
# -------------------------------------------------------------------
# v26.10.6 — bumped from 30 min to 12 hours. Past-month invoice data is
# basically immutable (the month's already closed in payroll) so there's
# no value in re-fetching every 30 min. Force-refresh button below if
# you've just generated new invoices.
@st.cache_data(ttl=48 * 3600, show_spinner=False)
def _cached_commission(year: int, month: int, _schema_version: str = ""):
    """Cache around the heavy invoice fetch. Manual adjustments are NOT
    cached — they're applied after the cache layer.

    v26.10.6 — also persists a disk copy in DATA_DIR/_cache so the data
    survives Streamlit Cloud's app-sleep cycle. On a cold restart, we
    load from disk instead of re-fetching from Cliniko."""
    from dashboard.commission import (
        load_pay_config, revenue_per_practitioner, resolve_cliniko_ids,
    )
    from dashboard.reference_data import load_practitioners
    from dashboard.config import DATA_DIR
    import json, time
    cache_dir = DATA_DIR / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"commission_{year}_{month:02d}.json"

    pay_config = load_pay_config()
    client = get_client()

    # --- Disk-cache fast path: <12 hours old → load from disk ---
    if cache_file.exists():
        age_seconds = time.time() - cache_file.stat().st_mtime
        if age_seconds < 48 * 3600:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    blob = json.load(f)
                revenue_map = {str(k): float(v) for k, v in blob.get("revenue", {}).items()}
                id_map = {str(k): (str(v) if v else None)
                            for k, v in blob.get("id_map", {}).items()}
                return pay_config, revenue_map, id_map
            except Exception:
                pass  # corrupt cache file, fall through to refetch

    # --- Slow path: hit Cliniko ---
    revenue_map = revenue_per_practitioner(client, year, month)
    cliniko_pracs = load_practitioners(client)
    id_map = resolve_cliniko_ids(pay_config, cliniko_pracs)

    # Persist for next cold start
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({
                "fetched_at": time.time(),
                "revenue": {str(k): float(v) for k, v in revenue_map.items()},
                "id_map": {str(k): (str(v) if v else None)
                            for k, v in id_map.items()},
            }, f)
    except Exception:
        pass  # disk write failure shouldn't break the calculator

    return pay_config, revenue_map, id_map


def _clear_commission_cache(year: int, month: int) -> None:
    """Remove both the in-memory and on-disk cache for a specific month
    so the next render forces a fresh Cliniko fetch."""
    from dashboard.config import DATA_DIR
    cache_file = DATA_DIR / "_cache" / f"commission_{year}_{month:02d}.json"
    try:
        cache_file.unlink(missing_ok=True)
    except Exception:
        pass
    _cached_commission.clear()
    # v26.11.5 — keep the lazy-load flag set so we immediately re-fetch
    # after refresh instead of asking the user to click Load again.
    st.session_state["loaded_commission"] = True


def commission_tab(filters: dict) -> None:
    """Commission Calculator tab — monthly per-practitioner bonus math.

    v26.11.1 — gated behind a separate password (set via the
    COMMISSION_PASSWORD secret in Streamlit Cloud) so clinic managers
    with general dashboard access can't see pay data unless explicitly
    authorised.
    """
    import calendar as _cal
    from datetime import date as _date
    import os

    st.subheader("💰 Commission Calculator")
    # Lazy gate — no fetches until user clicks Load. Cache also survives
    # this gate (result is cached to disk), so subsequent months of the
    # same session are near-instant.
    if not _lazy_gate(
        "commission",
        "Load Commission",
        "Click below to pull invoices and appointments from Cliniko for "
        "the selected month (~30-60s first time, then cached for 48h).",
    ):
        return
    from dashboard.commission import (
        compute_commission_for_practitioner, hours_for_month, dow_occurrences,
    )
    from dashboard.config import _ensure_env
    from dashboard import __version__ as _v

    # ---- Password gate (v26.11.1) ----
    _ensure_env()
    expected_pw = (os.environ.get("COMMISSION_PASSWORD") or "").strip()

    if not expected_pw:
        st.error(
            "Commission tab is locked: the `COMMISSION_PASSWORD` secret "
            "isn't set. Add it via Streamlit Cloud's app settings → "
            "**Secrets**, then reboot."
        )
        return

    if not st.session_state.get("commission_authenticated"):
        st.warning(
            "🔒 This tab contains pay information and requires a separate "
            "password. Ask Matt for the commission password."
        )
        col_pw, col_btn = st.columns([3, 1])
        with col_pw:
            pw = st.text_input(
                "Commission password",
                type="password",
                key="commission_pw_input",
                label_visibility="collapsed",
                placeholder="Enter commission password",
            )
        with col_btn:
            unlock = st.button("Unlock", key="commission_unlock", type="primary")
        # Allow Enter-key submit by checking the value too
        if unlock or (pw and pw == expected_pw):
            if pw == expected_pw:
                st.session_state["commission_authenticated"] = True
                st.rerun()
            elif unlock:
                st.error("Incorrect password.")
        return

    # Lock button to surrender access in this session
    col_title_msg, col_lock = st.columns([5, 1])
    with col_lock:
        if st.button("🔒 Lock again", key="commission_lock_btn",
                      help="Re-locks the Commission tab for this session"):
            st.session_state["commission_authenticated"] = False
            st.rerun()

    st.caption(
        "Monthly per-practitioner commission. Hours from the schedule "
        "in `config/practitioner_pay.yml`, revenue from Cliniko invoices "
        "(appointment-linked items only — products/DNA/room-hire excluded), "
        "bonus = max(0, revenue × commission% − base cost) ÷ (1+super)."
    )

    # --- Month picker ---
    today = _date.today()
    # Default to the previous full calendar month
    if today.month == 1:
        default_year, default_month = today.year - 1, 12
    else:
        default_year, default_month = today.year, today.month - 1

    col_y, col_m, col_r = st.columns([1, 2, 1])
    with col_y:
        years = list(range(today.year - 2, today.year + 1))
        year = st.selectbox("Year", years, index=years.index(default_year))
    with col_m:
        months = [(i, _cal.month_name[i]) for i in range(1, 13)]
        month = st.selectbox(
            "Month",
            options=[m[0] for m in months],
            format_func=lambda m: _cal.month_name[m],
            index=default_month - 1,
        )
    with col_r:
        st.markdown("&nbsp;")  # vertical alignment
        if st.button("🔄 Force refresh", key="commission_force_refresh",
                      help="Clear cache and re-fetch from Cliniko. "
                            "Use this if you've just generated new invoices."):
            _clear_commission_cache(int(year), int(month))
            st.rerun()

    # --- Pull data ---
    with st.spinner("Pulling invoices from Cliniko (~30s first time)…"):
        try:
            pay_config, revenue_map, id_map = _cached_commission(
                int(year), int(month), _schema_version=_v,
            )
        except FileNotFoundError:
            st.error(
                "`config/practitioner_pay.yml` is missing from the repo. "
                "Upload it via GitHub before this tab can compute anything."
            )
            return
        except Exception as e:
            st.error(f"Couldn't compute commission: {e}")
            return

    super_rate = pay_config.super_rate_for(_date(year, month, 1))

    # --- Show the day-of-week breakdown so the user can sanity-check ---
    counts = dow_occurrences(year, month)
    st.caption(
        f"**{_cal.month_name[month]} {year}** weekdays: "
        + " · ".join(f"{c} {dow[:3]}" for dow, c in counts.items() if c)
        + f" · super rate **{super_rate * 100:.1f}%**"
    )

    # --- Manual adjustments (per-practitioner, per-month) ---
    # Stored in session_state keyed by (year, month, name) so editing
    # one practitioner doesn't blow away the others.
    adj_key_prefix = f"commission_manual_adj_{year}_{month}"

    # v26.11.3 — additional session-state key for paid-hours overrides
    override_key_prefix = f"commission_paid_hours_override_{year}_{month}"

    # Build initial table
    rows: list[dict] = []
    cliniko_id_warnings: list[str] = []
    for prac in pay_config.practitioners:
        cid = id_map.get(prac.name)
        if cid is None:
            cliniko_id_warnings.append(prac.name)
        revenue = float(revenue_map.get(cid or "", 0.0))
        manual_h = float(st.session_state.get(f"{adj_key_prefix}_{prac.name}", 0.0))
        override_raw = st.session_state.get(f"{override_key_prefix}_{prac.name}", 0.0)
        override_val = float(override_raw) if override_raw and float(override_raw) > 0 else None
        result = compute_commission_for_practitioner(
            prac, year, month,
            revenue=revenue,
            super_rate=super_rate,
            manual_adjustment_hours=manual_h,
            paid_hours_override=override_val,
            cliniko_practitioner_id=cid,
        )
        row = result.as_row()
        # Editable inputs come first
        row["Manual adj (hrs)"] = manual_h
        row["Paid hours (override)"] = float(override_raw or 0.0)
        rows.append(row)

    if cliniko_id_warnings:
        st.warning(
            "Couldn't match these pay-config names to a Cliniko practitioner — "
            "their revenue will show $0 until matched. Add an `aliases:` entry "
            "in `practitioner_pay.yml` or check the Cliniko practitioner "
            f"display name. Unmatched: {', '.join(cliniko_id_warnings)}"
        )

    df = pd.DataFrame(rows)
    # Reorder columns so the editable one sits next to deductions
    col_order = [
        "Practitioner", "Base hours", "Deductions (hrs)", "Manual adj (hrs)",
        "Paid hours (override)", "Paid hours",
        "Rate ($/h)", "Base pay (pre-super)", "Super on base",
        "Base cost (incl super)", "Revenue invoiced", "Commission %",
        "Target total cost", "Bonus (pre-super, enter in Xero)",
        "Total clinic cost",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    edited = st.data_editor(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Manual adj (hrs)": st.column_config.NumberColumn(
                "Manual adj (hrs)",
                help="Extra hours to deduct this month — GP meetings, "
                      "ad-hoc mentoring, sick leave (after-fact), etc.",
                step=0.5, min_value=0.0,
            ),
            "Paid hours (override)": st.column_config.NumberColumn(
                "Paid hours (override)",
                help="Enter ACTUAL paid hours from Xero timesheet. "
                      "Overrides the theoretical schedule + deductions. "
                      "Leave at 0 to use the computed value.",
                step=0.5, min_value=0.0,
            ),
            "Practitioner":             st.column_config.TextColumn(disabled=True),
            "Base hours":               st.column_config.NumberColumn(format="%.2f", disabled=True),
            "Deductions (hrs)":         st.column_config.NumberColumn(format="%.2f", disabled=True),
            "Paid hours":               st.column_config.NumberColumn(format="%.2f", disabled=True),
            "Rate ($/h)":               st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Base pay (pre-super)":     st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Super on base":            st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Base cost (incl super)":   st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Revenue invoiced":         st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Commission %":             st.column_config.TextColumn(disabled=True),
            "Target total cost":        st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Bonus (pre-super, enter in Xero)":
                                          st.column_config.NumberColumn(format="$%.2f", disabled=True),
            "Total clinic cost":        st.column_config.NumberColumn(format="$%.2f", disabled=True),
        },
        key=f"commission_editor_{year}_{month}",
    )

    # Persist edited Manual adj + Paid hours override back into
    # session_state and re-run if anything changed
    rerun = False
    for _, r in edited.iterrows():
        name = r["Practitioner"]
        new_adj = float(r.get("Manual adj (hrs)") or 0.0)
        ss_key = f"{adj_key_prefix}_{name}"
        if abs(new_adj - float(st.session_state.get(ss_key, 0.0))) > 1e-9:
            st.session_state[ss_key] = new_adj
            rerun = True
        new_override = float(r.get("Paid hours (override)") or 0.0)
        ov_key = f"{override_key_prefix}_{name}"
        if abs(new_override - float(st.session_state.get(ov_key, 0.0))) > 1e-9:
            st.session_state[ov_key] = new_override
            rerun = True
    if rerun:
        st.rerun()

    # --- Totals strip ---
    total_revenue = float(df["Revenue invoiced"].sum())
    total_base = float(df["Base cost (incl super)"].sum())
    total_bonus = float(df["Bonus (pre-super, enter in Xero)"].sum())
    total_cost = float(df["Total clinic cost"].sum())
    cols = st.columns(4)
    cols[0].metric("Total revenue", f"${total_revenue:,.0f}")
    cols[1].metric("Total base cost", f"${total_base:,.0f}")
    cols[2].metric("Total bonus (pre-super)", f"${total_bonus:,.0f}")
    cols[3].metric("Total clinic cost", f"${total_cost:,.0f}")

    st.caption(
        "Cache: 30 min on revenue. Manual-adj edits apply immediately. "
        "Use the **⋮ menu → Clear cache** if you've just generated new "
        "invoices in Cliniko and want them reflected straight away."
    )

    # --- Diagnostic expander (v26.10.2) -----------------------------
    # Shows where invoice items are being filtered out and how the
    # practitioner-name match resolved. Crucial for debugging $0
    # revenue across the board.
    with st.expander("🔍 Diagnostic: revenue fetch + name match", expanded=False):
        st.caption(
            "Use this when totals look wrong (e.g. $0 across the board). "
            "Counters show how invoice items survive each filter; the table "
            "below shows whether each pay-config practitioner matched a "
            "Cliniko id."
        )

        # 1) Practitioner ID match table
        st.markdown("**Practitioner-name → Cliniko ID match**")
        match_rows = []
        for prac in pay_config.practitioners:
            cid = id_map.get(prac.name)
            match_rows.append({
                "YAML name": prac.name,
                "Aliases": ", ".join(prac.aliases) if prac.aliases else "—",
                "Cliniko ID": cid or "—",
                "Matched?": "✅" if cid else "❌",
                "Revenue ($)": float(revenue_map.get(cid or "", 0.0)) if cid else 0.0,
            })
        st.dataframe(pd.DataFrame(match_rows), hide_index=True, width="stretch")

        # 2) Invoice items diagnostic
        if st.button("Run invoice-items diagnostic (extra ~30s API call)",
                      key="commission_diag_run"):
            from dashboard.commission import fetch_invoice_items_diagnostic
            with st.spinner("Pulling invoice items + appointments…"):
                diag = fetch_invoice_items_diagnostic(get_client(), int(year), int(month))
            st.markdown("**Filter funnel**")
            st.json(diag["counters"])
            st.caption(
                f"Invoices fetched in window: **{diag.get('invoices_fetched', 0)}** · "
                f"Appointments fetched: **{diag['appts_fetched']}** · "
                f"Appointment types: **{diag['appt_types_fetched']}**"
            )
            if diag.get("error"):
                st.error(f"Endpoint error: {diag['error']}")
            if diag.get("first_raw_shape"):
                st.markdown("**Raw shape of first invoice_item from Cliniko**")
                st.caption(
                    "Field structure only — values redacted to avoid leaking "
                    "patient data."
                )
                st.json(diag["first_raw_shape"])
            if diag.get("first_invoice_shape"):
                st.markdown("**Raw shape of first invoice from Cliniko**")
                st.caption(
                    "We use the invoice's `appointment` link to attribute "
                    "revenue to a practitioner."
                )
                st.json(diag["first_invoice_shape"])
            if diag.get("sample_kept_items"):
                st.markdown("**Sample kept items**")
                st.dataframe(pd.DataFrame(diag["sample_kept_items"]),
                              hide_index=True)
            else:
                st.warning(
                    "No invoice items survived the filters. Most likely "
                    "causes:\n"
                    "1. **Endpoint shape changed** — check the raw shape "
                    "above; if there's no `appointment` link, Cliniko's "
                    "field name differs and the parser needs to be updated.\n"
                    "2. **No appointment-linked invoices** — if you bill "
                    "outside Cliniko, this dashboard won't see revenue.\n"
                    "3. **All items match an excluded pattern** — try "
                    "removing terms from `_DEFAULT_EXCLUDED_APPT_TYPE_PATTERNS`."
                )


def diagnostics_tab(filters: dict) -> None:
    """Temporary debug tab: shows raw Cliniko responses so we can spot
    field-name mismatches. Safe to ignore once numbers look right."""
    from dashboard.cliniko import starts_at_range_params

    st.subheader("🔧 Cliniko API diagnostics")
    st.caption(
        "This tab shows raw samples from Cliniko so we can debug why the "
        "main dashboard might show zeros. Safe to ignore once numbers look right."
    )

    try:
        dr = resolve_preset(
            filters["preset"],
            filters.get("custom_start"),
            filters.get("custom_end"),
        )
    except Exception as e:
        st.error(f"Could not resolve date range: {e}")
        return
    st.write(f"**Date range:** {dr.start_iso_utc} → {dr.end_iso_utc}")
    st.write(f"**Practitioner filter:** "
             f"{filters['practitioner_ids'] or '(all)'}")
    st.write(f"**Clinic filter:** "
             f"{filters['business_ids'] or '(all)'}")

    try:
        client = ClinikoClient()
    except Exception as e:
        st.error(f"Could not build Cliniko client: {e}")
        return

    with st.expander("1. Auth check (practitioners?per_page=1)", expanded=True):
        try:
            data = client.get("practitioners", params={"per_page": 1})
            st.json(data)
        except Exception as e:
            st.error(f"Auth check failed: {e}")
            st.warning("Continuing with remaining sections anyway.")

    with st.expander("2. First practitioner — raw record"):
        try:
            prs = list(client.paginate("practitioners"))
            st.write(f"Total practitioners returned: **{len(prs)}**")
            if prs:
                st.json(prs[0])
        except Exception as e:
            st.error(f"Practitioners call failed: {e}")

    with st.expander("3. Businesses (clinics)"):
        try:
            bs = list(client.paginate("businesses"))
            st.write(f"Total businesses: **{len(bs)}**")
            if bs:
                st.json(bs[0])
        except Exception as e:
            st.error(f"Businesses call failed: {e}")

    with st.expander("4. First appointment type — raw record"):
        try:
            ats = list(client.paginate("appointment_types"))
            st.write(f"Total appointment types: **{len(ats)}**")
            if ats:
                st.json(ats[0])
        except Exception as e:
            st.error(f"Appointment types call failed: {e}")

    with st.expander("5. First 3 individual_appointments in range", expanded=True):
        params = starts_at_range_params(dr.start_iso_utc, dr.end_iso_utc)
        try:
            apps_iter = client.paginate("individual_appointments", params=params)
            sample = []
            for i, a in enumerate(apps_iter):
                sample.append(a)
                if i >= 2:
                    break
            st.write(f"Sample size: **{len(sample)}**")
            if not sample:
                st.warning(
                    "Cliniko returned ZERO appointments for this date range. "
                    "Try a bigger window (Last 90 days, or This FY)."
                )
            for a in sample:
                st.json(a)
        except Exception as e:
            st.error(f"Appointments call failed: {e}")

    with st.expander("6. First 2 patients created in range"):
        params = {"q[]": [
            f"created_at:>={dr.start_iso_utc}",
            f"created_at:<{dr.end_iso_utc}",
        ]}
        try:
            it = client.paginate("patients", params=params)
            sample = []
            for i, p in enumerate(it):
                sample.append(p)
                if i >= 1:
                    break
            st.write(f"Sample size: **{len(sample)}**")
            for p in sample:
                st.json(p)
        except Exception as e:
            st.error(f"Patients call failed: {e}")

    with st.expander("7. Cancelled appointments in range (2nd-pass fetch)"):
        cx_params = {"q[]": [
            f"starts_at:>={dr.start_iso_utc}",
            f"starts_at:<{dr.end_iso_utc}",
            "cancelled_at:>=1970-01-01T00:00:00Z",
        ]}
        try:
            it = client.paginate("individual_appointments", params=cx_params)
            total = 0
            sample = []
            for i, a in enumerate(it):
                total += 1
                if i < 2:
                    sample.append(a)
            st.write(f"Cancelled appointments found: **{total}**")
            if total == 0:
                st.warning(
                    "Zero cancelled appts returned. If Cliniko's own report "
                    "shows cancellations for this range, the q-filter syntax "
                    "may need adjustment."
                )
            for a in sample:
                st.json(a)
        except Exception as e:
            st.error(f"Cancelled-appointments call failed: {e}")

    with st.expander("8. availability_blocks in range (utilisation denominator)"):
        params = {"q[]": [
            f"starts_at:>={dr.start_iso_utc}",
            f"starts_at:<{dr.end_iso_utc}",
        ]}
        try:
            total = 0
            sample = []
            practitioners_seen: set = set()
            for i, b in enumerate(client.paginate("availability_blocks", params=params)):
                total += 1
                from dashboard.reference_data import extract_linked_id as _xid
                pid = _xid(b.get("practitioner"), "self") or _xid(b.get("links"), "practitioner")
                if pid:
                    practitioners_seen.add(pid)
                if i < 2:
                    sample.append(b)
            st.write(f"Total availability_blocks: **{total}** (across "
                     f"**{len(practitioners_seen)}** practitioners)")
            if total == 0:
                st.warning(
                    "Zero availability_blocks found. Utilisation will be NaN "
                    "for every practitioner. Either this endpoint isn't "
                    "populated in your Cliniko account, or the date range "
                    "is outside your roster data."
                )
            for b in sample:
                st.json(b)
        except Exception as e:
            st.error(f"availability_blocks call failed: {e}")

    with st.expander("9. unavailable_blocks in range (qualifying admin time)"):
        params = {"q[]": [
            f"starts_at:>={dr.start_iso_utc}",
            f"starts_at:<{dr.end_iso_utc}",
        ]}
        try:
            total = 0
            sample = []
            for i, b in enumerate(client.paginate("unavailable_blocks", params=params)):
                total += 1
                if i < 2:
                    sample.append(b)
            st.write(f"Total unavailable_blocks: **{total}**")
            for b in sample:
                st.json(b)
        except Exception as e:
            st.error(f"unavailable_blocks call failed: {e}")

    with st.expander("10. Metric pipeline audit (share with Matt if data looks off)",
                     expanded=True):
        try:
            from dashboard.metrics import fetch_appointments as _fetch
            from dashboard.metrics import _is_delivered as _isd
            types = cached_appointment_types()
            appts = _fetch(
                client, dr,
                business_ids=filters.get("business_ids") or None,
                practitioner_ids=filters.get("practitioner_ids") or None,
            )
            st.write(f"Total appts in df after 2-pass fetch + archive filter: "
                     f"**{len(appts)}**")
            if not appts.empty:
                cancelled_ct = int(appts["cancelled_at"].notna().sum())
                dna_ct = int(appts["did_not_arrive"].fillna(False).astype(bool).sum())
                delivered_ct = int(_isd(appts).sum())
                st.write(f"• Cancelled rows (cancelled_at set): **{cancelled_ct}**")
                st.write(f"• DNA rows (did_not_arrive=True): **{dna_ct}**")
                st.write(f"• Delivered (neither): **{delivered_ct}**")
                st.write(f"• Unique practitioner_ids in appts: "
                         f"**{appts['practitioner_id'].nunique()}**")
                st.write("Sample practitioner_ids from appts (shows dtype):")
                st.code(repr(appts["practitioner_id"].dropna().head(5).tolist()))

                # Treatment-note status distribution — feeds notes_completion.
                # 10 = N/A, 20 = pending, 30 = draft, 40 = overdue, 90 = finalised.
                if "treatment_note_status" in appts.columns:
                    delivered_only = appts[_isd(appts)].copy()
                    status_series = pd.to_numeric(
                        delivered_only["treatment_note_status"], errors="coerce"
                    )
                    status_labels = {
                        10: "10 — N/A (no note expected)",
                        20: "20 — pending",
                        30: "30 — draft",
                        40: "40 — overdue",
                        90: "90 — finalised",
                    }
                    counts = status_series.value_counts(dropna=False).sort_index()
                    st.write(
                        "**treatment_note_status distribution** "
                        "(delivered appts only — drives notes_completion):"
                    )
                    rows_ts = []
                    for k, v in counts.items():
                        label = (
                            status_labels.get(int(k), f"{int(k)} — unknown")
                            if pd.notna(k) else "missing / None"
                        )
                        rows_ts.append({"status": label, "count": int(v)})
                    st.dataframe(pd.DataFrame(rows_ts), hide_index=True)

                    expected = int((status_series.fillna(20).astype(int) != 10).sum())
                    finalised_ever = int((status_series.fillna(0).astype(int) == 90).sum())
                    st.write(
                        f"Of {len(delivered_only)} delivered appts: "
                        f"**{expected}** expect a note (status ≠ 10), "
                        f"**{finalised_ever}** ever finalised (status = 90)."
                    )

                    # Within-24h proxy check using appt.updated_at. Only an
                    # approximation — the real metric also looks at the note
                    # payload — but useful to eyeball.
                    if "appt_updated_at" in delivered_only.columns:
                        d24 = delivered_only.dropna(subset=["starts_at"]).copy()
                        d24["_h"] = (
                            (d24["appt_updated_at"] - d24["starts_at"]).dt.total_seconds()
                            / 3600.0
                        )
                        within_24 = int(
                            ((status_series.reindex(d24.index).fillna(0).astype(int) == 90)
                             & (d24["_h"] >= -1.0)
                             & (d24["_h"] <= 24.0)).sum()
                        )
                        pct24 = (within_24 / expected * 100.0) if expected else float("nan")
                        st.write(
                            f"Rough within-24h proxy (appt.updated_at vs starts_at): "
                            f"**{within_24}/{expected}** → **{pct24:.1f}%**. "
                            "The live metric uses treatment-note timestamps where "
                            "available (more accurate)."
                        )
                else:
                    st.warning(
                        "`treatment_note_status` column missing from appts df — "
                        "notes_completion will read as 0. Re-deploy latest code."
                    )

            import re as _re
            pat = _re.compile(
                r"\b(initial|new\s*patient|new\s*client|assessment|first\s*visit)\b",
                _re.IGNORECASE,
            )
            if not types.empty:
                initials = types[types["name"].fillna("").apply(
                    lambda n: bool(pat.search(n))
                )]
                st.write(f"Appointment types matched as 'Initial': **{len(initials)}**")
                if not initials.empty:
                    st.dataframe(initials[["id", "name"]], hide_index=True)

            # Created-in-range signal (catches DVA/NDIS/TAC/WC first-timers)
            # Only scan /patients when the user asks — it's the slow path.
            settings_now = load_settings()
            np_cfg = settings_now.get("new_patients", {}) if isinstance(settings_now, dict) else {}
            scan_on = bool(np_cfg.get("scan_patients", False))
            st.write(
                f"**/patients scan (DVA/NDIS/TAC/WC new-patient signal):** "
                f"{'ON' if scan_on else 'OFF (default — Initial-type signal only)'}"
            )
            st.caption(
                "Toggle this in config/settings.yml → new_patients.scan_patients. "
                "Turning it on adds 20-50 API calls per refresh."
            )
            if st.button("Run /patients scan once (for this session only)",
                         key="patient_scan_once"):
                try:
                    from datetime import datetime as _dt
                    start_utc = _dt.fromisoformat(dr.start_iso_utc.replace("Z", "+00:00"))
                    end_utc = _dt.fromisoformat(dr.end_iso_utc.replace("Z", "+00:00"))
                    created_hits = 0
                    scanned = 0
                    max_pages = int(np_cfg.get("scan_patients_max_pages", 20))
                    for p in client.paginate(
                        "patients",
                        params={"q[]": [f"updated_at:>={dr.start_iso_utc}"]},
                        max_pages=max_pages,
                    ):
                        scanned += 1
                        c = p.get("created_at")
                        if not c:
                            continue
                        try:
                            c_dt = _dt.fromisoformat(c.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        if start_utc <= c_dt < end_utc:
                            created_hits += 1
                    st.success(
                        f"Scan complete — **{created_hits}** patients created "
                        f"in range out of {scanned} recently-updated records "
                        f"(capped at {max_pages * 100})."
                    )
                except Exception as e:
                    st.warning(f"Created-in-range scan failed: {e}")
        except Exception as e:
            st.error(f"Pipeline audit failed: {e}")


if __name__ == "__main__":
    main()
