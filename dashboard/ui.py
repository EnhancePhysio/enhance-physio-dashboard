"""Streamlit UI helpers — charts, KPI cards, tables."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.scoring import (
    CLINICAL_METRICS, NONCLINICAL_METRICS, METRIC_VALUE_COLS,
    ZONE_COLORS, ZONE_ACTIONS,
)


def kpi_card(label: str, value: str, sub: str = "") -> None:
    st.metric(label=label, value=value, delta=sub or None, delta_color="off")


def matrix_scatter(scored: pd.DataFrame, title: str = "Clinical × Non-clinical") -> go.Figure:
    if scored.empty:
        return go.Figure()
    df = scored.copy()
    df["clinical_score"] = df["clinical_axis"]
    df["nonclinical_score"] = df["nonclinical_axis"]

    # Replace NaN band values with the string "N/A" for hover display. We
    # have to do this via a separate display column because plotly's
    # hovertemplate can't conditionally format NaN.
    df["nps_band_display"] = df["nps_band"].apply(
        lambda v: "N/A" if pd.isna(v) else f"{int(v)}"
    )

    fig = go.Figure()

    # v26.3 — axis flip (option A): non-clinical is now the X-axis, clinical
    # the Y-axis. Zones keep their semantic meaning, but their screen
    # positions rotate accordingly:
    #   Red    (low clinical, low non-clinical)  → bottom-left
    #   Blue   (low clinical, high non-clinical) → bottom-right (was top-left)
    #   Orange (high clinical, low non-clinical) → top-left     (was bottom-right)
    #   Green  (high both)                       → top-right
    #   Gold   (top both)                        → top-right inner
    fig.add_shape(type="rect", x0=0, y0=0, x1=5, y1=5,
                  fillcolor=ZONE_COLORS["Red"], opacity=0.10, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=5, y0=0, x1=10, y1=5,
                  fillcolor=ZONE_COLORS["Blue"], opacity=0.10, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=5, x1=5, y1=10,
                  fillcolor=ZONE_COLORS["Orange"], opacity=0.10, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=5, y0=5, x1=10, y1=10,
                  fillcolor=ZONE_COLORS["Green"], opacity=0.10, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=7.5, y0=7.5, x1=10, y1=10,
                  fillcolor=ZONE_COLORS["Gold"], opacity=0.20, line_width=0, layer="below")

    # Markers by zone — non-clinical on X, clinical on Y.
    for zone, group in df.groupby("zone"):
        fig.add_trace(go.Scatter(
            x=group["nonclinical_score"],
            y=group["clinical_score"],
            mode="markers+text",
            text=group["label"],
            textposition="top center",
            marker=dict(size=16, color=ZONE_COLORS.get(zone, "#888"),
                        line=dict(color="white", width=2)),
            name=zone,
            customdata=group[[
                "label", "clinical_axis", "nonclinical_axis",
                "service_hours_band", "pva_band", "ppva_band",
                "cx_dna_combined_rate_band", "utilisation_band",
                "nps_band_display", "audit_pct_band", "notes_completion_band",
                "punctuality_within_15_band",
            ]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Clinical: %{customdata[1]:.1f} / 10<br>"
                "Non-clinical: %{customdata[2]:.1f} / 10<br>"
                "— Clinical breakdown —<br>"
                "Service Hours: %{customdata[3]}<br>"
                "PVA: %{customdata[4]}<br>"
                "PPVA: %{customdata[5]}<br>"
                "Cx/DNA: %{customdata[6]}<br>"
                "Utilisation: %{customdata[7]}<br>"
                "— Non-clinical breakdown —<br>"
                "NPS: %{customdata[8]}<br>"
                "Audit: %{customdata[9]}<br>"
                "Notes: %{customdata[10]}<br>"
                "Punctuality: %{customdata[11]}"
                "<extra></extra>"
            ),
        ))

    # Reference lines
    fig.add_hline(y=5, line_dash="dash", line_color="#999")
    fig.add_vline(x=5, line_dash="dash", line_color="#999")
    fig.add_hline(y=7.5, line_dash="dot", line_color="#D4A017")
    fig.add_vline(x=7.5, line_dash="dot", line_color="#D4A017")

    fig.update_layout(
        title=title,
        xaxis=dict(title="Non-clinical score (band avg, 0–10)", range=[0, 10], dtick=1),
        yaxis=dict(title="Clinical score (band avg, 0–10)", range=[0, 10], dtick=1),
        height=600,
        showlegend=True,
        legend=dict(title="Zone"),
    )
    return fig


def zone_summary_table(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    rows = []
    for zone in ["Red", "Orange", "Blue", "Green", "Gold"]:
        sub = scored[scored["zone"] == zone]
        rows.append({
            "Zone": zone,
            "Count": len(sub),
            "Practitioners": ", ".join(sub["label"].tolist()),
            "Typical action": ZONE_ACTIONS[zone],
        })
    return pd.DataFrame(rows)


def practitioner_detail(scored: pd.DataFrame, practitioner_label: str) -> None:
    row = scored[scored["label"] == practitioner_label]
    if row.empty:
        st.info("Select a practitioner to see the breakdown.")
        return
    r = row.iloc[0]
    st.subheader(f"{r['label']} — Zone: {r['zone']}")
    st.caption(ZONE_ACTIONS.get(r["zone"], ""))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Clinical axis**")
        for m in CLINICAL_METRICS:
            src = METRIC_VALUE_COLS[m]
            value = r.get(src, 0)
            band = int(r.get(f"{m}_band", 0))
            label = m.replace("_", " ").replace("cx dna combined rate", "Cx/DNA").title()
            if m == "cx_dna_combined_rate":
                label = "Cx / DNA"
            if m in ("cx_dna_combined_rate", "utilisation"):
                val_str = f"{value * 100:.1f}%"
            else:
                val_str = f"{value:.2f}"
            st.write(f"{label}: **{val_str}** → band **{band}/10**")
        st.write(f"**Clinical avg: {r['clinical_axis']:.1f}/10**")
    with col2:
        st.markdown("**Non-clinical axis**")
        for m in NONCLINICAL_METRICS:
            src = METRIC_VALUE_COLS[m]
            value = r.get(src, 0)
            band_raw = r.get(f"{m}_band", 0)
            label = {"nps": "NPS", "audit_pct": "Audit",
                     "notes_completion": "Notes (24 h)",
                     "punctuality_within_15": "Punctuality (<15 min)"}[m]
            # NPS N/A when no survey responses — show "N/A" rather than "0/10"
            if m == "nps" and (pd.isna(value) or pd.isna(band_raw)):
                st.write(f"{label}: **N/A** (no survey responses in this period)")
                continue
            if m == "nps":
                raw = r.get("nps_raw", value * 200 - 100)
                val_str = f"{raw:.0f} (rubric-scaled {value * 100:.0f}%)"
            else:
                val_str = f"{value * 100:.1f}%"
            band = int(band_raw) if not pd.isna(band_raw) else 0
            st.write(f"{label}: **{val_str}** → band **{band}/10**")
        n_applic = int(r.get("nonclinical_n", len(NONCLINICAL_METRICS)))
        denom = n_applic * 10
        sum_of_bands = r["nonclinical_axis"] * n_applic if n_applic else 0
        st.write(
            f"**Non-clinical avg: {r['nonclinical_axis']:.1f}/10** "
            f"(sum {sum_of_bands:.0f} / {denom} across {n_applic} applicable metrics)"
        )
