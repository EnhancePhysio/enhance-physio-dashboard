"""Rubric band lookup + matrix aggregation.

Bands are 1-10. Each metric has its own band table (see Section 7 of the design doc).
Cx/DNA is inverted (lower % = higher band).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class Band:
    """One band in a rubric. Bounds are expressed in the metric's natural units."""
    score: int
    low: float   # inclusive lower
    high: float  # exclusive upper (except top band = inclusive)


# -------------------------------------------------------------------
# Rubric tables
# (from Section 7 of the design doc, with gaps/typos fixed per Section 14.1)
# -------------------------------------------------------------------
SERVICE_HOURS_BANDS = [
    Band(1,  0.00,  4.11), Band(2,  4.11, 4.41), Band(3,  4.41, 4.71),
    Band(4,  4.71, 5.01), Band(5,  5.01, 5.31), Band(6,  5.31, 5.61),
    Band(7,  5.61, 5.91), Band(8,  5.91, 6.21), Band(9,  6.21, 6.51),
    Band(10, 6.51, float("inf")),
]

PVA_BANDS = [
    Band(1, 0.00, 3.00),  Band(2, 3.00, 5.00),  Band(3, 5.00, 7.00),
    Band(4, 7.00, 9.00),  Band(5, 9.00, 11.00), Band(6, 11.00, 13.00),
    Band(7, 13.00, 15.00),Band(8, 15.00, 17.00),Band(9, 17.00, 19.00),
    Band(10, 19.00, float("inf")),
]

PPVA_BANDS = [
    Band(1, 0.00, 2.00), Band(2, 2.00, 3.00), Band(3, 3.00, 4.00),
    Band(4, 4.00, 5.00), Band(5, 5.00, 6.00), Band(6, 6.00, 7.00),
    Band(7, 7.00, 8.00), Band(8, 8.00, 9.00), Band(9, 9.00, 10.00),
    Band(10, 10.00, float("inf")),
]

# Cx/DNA is INVERTED — lower % earns higher band.
# We store with 'low' = LOWER (fractional) pct and assign band such that
# >= 21% → band 1, <= 5% → band 10.
CX_DNA_BANDS_INVERTED = [
    # Treat as (minimum_allowed, maximum_allowed). Band 10 is the best (≤ 5%).
    Band(10, 0.00, 0.05),
    Band(9,  0.0501, 0.07),
    Band(8,  0.0701, 0.09),
    Band(7,  0.0901, 0.11),
    Band(6,  0.1101, 0.13),
    Band(5,  0.1301, 0.15),
    Band(4,  0.1501, 0.17),
    Band(3,  0.1701, 0.19),
    Band(2,  0.1901, 0.21),
    Band(1,  0.2101, float("inf")),
]

UTILISATION_BANDS = [
    Band(1, 0.00,  0.64),
    Band(2, 0.64,  0.68),
    Band(3, 0.68,  0.72),
    Band(4, 0.72,  0.76),
    Band(5, 0.76,  0.80),
    Band(6, 0.80,  0.84),
    Band(7, 0.84,  0.88),
    Band(8, 0.88,  0.92),
    Band(9, 0.92,  0.96),
    Band(10, 0.96, 1.01),
]

NPS_BANDS = [
    Band(1, 0.00,  0.64),
    Band(2, 0.64,  0.66),
    Band(3, 0.66,  0.68),
    Band(4, 0.68,  0.72),
    Band(5, 0.72,  0.76),
    Band(6, 0.76,  0.80),
    Band(7, 0.80,  0.84),
    Band(8, 0.84,  0.88),
    Band(9, 0.88,  0.92),
    Band(10, 0.92, float("inf")),
]

AUDIT_BANDS = [
    Band(1, 0.00,  0.82),
    Band(2, 0.82,  0.84),
    Band(3, 0.84,  0.86),
    Band(4, 0.86,  0.88),
    Band(5, 0.88,  0.90),
    Band(6, 0.90,  0.92),
    Band(7, 0.92,  0.94),
    Band(8, 0.94,  0.96),
    Band(9, 0.96,  0.98),
    Band(10, 0.98, float("inf")),
]

NOTES_BANDS = [
    Band(1, 0.00,  0.91),
    Band(2, 0.91,  0.92),
    Band(3, 0.92,  0.93),
    Band(4, 0.93,  0.94),
    Band(5, 0.94,  0.95),
    Band(6, 0.95,  0.96),
    Band(7, 0.96,  0.97),
    Band(8, 0.97,  0.98),
    Band(9, 0.98,  0.99),
    Band(10, 0.99, float("inf")),
]

PUNCTUALITY_BANDS = AUDIT_BANDS  # same thresholds per rubric sheet


RUBRIC = {
    "service_hours":        {"bands": SERVICE_HOURS_BANDS,    "inverted": False},
    "pva":                  {"bands": PVA_BANDS,              "inverted": False},
    "ppva":                 {"bands": PPVA_BANDS,             "inverted": False},
    "cx_dna_combined_rate": {"bands": CX_DNA_BANDS_INVERTED,  "inverted": True},
    "utilisation":          {"bands": UTILISATION_BANDS,      "inverted": False},
    "nps":                  {"bands": NPS_BANDS,              "inverted": False},
    "audit_pct":            {"bands": AUDIT_BANDS,            "inverted": False},
    "notes_completion":     {"bands": NOTES_BANDS,            "inverted": False},
    "punctuality_within_15":{"bands": PUNCTUALITY_BANDS,      "inverted": False},
}


CLINICAL_METRICS = ["service_hours", "pva", "ppva", "cx_dna_combined_rate", "utilisation"]
NONCLINICAL_METRICS = ["nps", "audit_pct", "notes_completion", "punctuality_within_15"]

# For Service Hours, use avg_hours_per_day rather than raw total
METRIC_VALUE_COLS = {
    "service_hours": "avg_hours_per_day",  # per design doc
    "pva": "pva",
    "ppva": "ppva",
    "cx_dna_combined_rate": "cx_dna_combined_rate",
    "utilisation": "utilisation",
    "nps": "nps",
    "audit_pct": "audit_pct",
    "notes_completion": "notes_completion",
    "punctuality_within_15": "punctuality_within_15",
}


def band_score(metric_key: str, value: float | None) -> float:
    """Return the 1-10 band score for a raw metric value.

    Missing value handling:
    * For NPS specifically, NaN/None → NaN (N/A). A practitioner with no
      survey responses shouldn't be scored on NPS; the non-clinical axis
      mean will skip NaN values, so their denominator drops from 40 to 30.
    * For every other metric, NaN/None → 0 (missing data still counts as
      a failure, e.g. no notes submitted means notes_completion = 0%).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        if metric_key == "nps":
            return float("nan")
        return 0
    spec = RUBRIC[metric_key]
    bands = spec["bands"]
    for b in bands:
        if b.low <= value < b.high:
            return b.score
    # Top-band catch
    top = max(bands, key=lambda b: b.score)
    if value >= top.low:
        return top.score
    return 0


def score_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Attach band-score columns for each metric in RUBRIC.

    The non-clinical axis is the mean of whichever band scores are
    **applicable** for that practitioner. In practice the only metric that
    can be N/A is NPS (when they received no survey responses), so for
    those rows the mean is taken over 3 metrics (denominator effectively
    30) instead of 4 (denominator 40).
    """
    out = wide.copy()
    # For NPS, preserve NaN through the source column so the band function
    # can emit NaN cleanly. For every other metric, missing → 0.0 is fine
    # and matches the previous behaviour.
    for metric, src_col in METRIC_VALUE_COLS.items():
        if src_col not in out.columns:
            out[src_col] = float("nan") if metric == "nps" else 0.0
        out[f"{metric}_band"] = out[src_col].apply(lambda v, m=metric: band_score(m, v))

    # Mean skips NaN by default — exactly the behaviour we want for N/A NPS.
    out["clinical_axis"] = out[[f"{m}_band" for m in CLINICAL_METRICS]].mean(axis=1)
    out["nonclinical_axis"] = out[[f"{m}_band" for m in NONCLINICAL_METRICS]].mean(axis=1)
    # How many non-clinical metrics actually contributed — useful in the
    # raw table so Matt can see "3 of 4" for NPS-N/A practitioners.
    out["nonclinical_n"] = out[[f"{m}_band" for m in NONCLINICAL_METRICS]].notna().sum(axis=1)
    out["clinical_pct"] = out["clinical_axis"] / 10.0
    out["nonclinical_pct"] = out["nonclinical_axis"] / 10.0
    out["zone"] = out.apply(
        lambda r: zone_for(r["clinical_pct"], r["nonclinical_pct"]), axis=1,
    )
    return out


ZoneName = Literal["Red", "Orange", "Blue", "Green", "Gold"]


def zone_for(clinical_pct: float, nonclinical_pct: float) -> ZoneName:
    """5-zone classification per Section 8.3."""
    if clinical_pct < 0.50 and nonclinical_pct < 0.50:
        return "Red"
    if clinical_pct >= 0.75 and nonclinical_pct >= 0.75:
        return "Gold"
    if clinical_pct >= 0.50 and nonclinical_pct < 0.50:
        return "Orange"
    if clinical_pct < 0.50 and nonclinical_pct >= 0.50:
        return "Blue"
    return "Green"


ZONE_COLORS = {
    "Red":    "#D64545",
    "Orange": "#E8943A",
    "Blue":   "#3B82F6",
    "Green":  "#22A36A",
    "Gold":   "#D4A017",
}


ZONE_ACTIONS = {
    "Red":    "Official warning / formal performance review",
    "Orange": "Compliance warning (1st offence, not official)",
    "Blue":   "Clinical training / mentoring plan",
    "Green":  "Maintain — standard check-ins",
    "Gold":   "Promotion candidate (if roles available)",
}
