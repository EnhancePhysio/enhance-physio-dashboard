"""Smoke tests for the scoring engine. Run with: python -m pytest tests/"""
import pandas as pd

from dashboard.scoring import band_score, score_table, zone_for


def test_service_hours_bands():
    assert band_score("service_hours", 3.0) == 1
    assert band_score("service_hours", 4.11) == 2
    assert band_score("service_hours", 5.00) == 4
    assert band_score("service_hours", 6.60) == 10


def test_pva_bands():
    assert band_score("pva", 2.5) == 1
    assert band_score("pva", 9.5) == 5
    assert band_score("pva", 25.0) == 10


def test_ppva_bands():
    assert band_score("ppva", 1.5) == 1
    assert band_score("ppva", 6.5) == 6
    assert band_score("ppva", 12.0) == 10


def test_cxdna_inverted():
    assert band_score("cx_dna_combined_rate", 0.03) == 10
    assert band_score("cx_dna_combined_rate", 0.10) == 7
    assert band_score("cx_dna_combined_rate", 0.25) == 1


def test_utilisation_bands():
    assert band_score("utilisation", 0.50) == 1
    assert band_score("utilisation", 0.78) == 5
    assert band_score("utilisation", 1.00) == 10


def test_audit_bands():
    assert band_score("audit_pct", 0.80) == 1
    assert band_score("audit_pct", 0.93) == 7
    assert band_score("audit_pct", 1.00) == 10


def test_notes_bands():
    assert band_score("notes_completion", 0.90) == 1
    assert band_score("notes_completion", 0.98) == 9
    assert band_score("notes_completion", 1.00) == 10


def test_zones():
    assert zone_for(0.3, 0.3) == "Red"
    assert zone_for(0.6, 0.3) == "Orange"
    assert zone_for(0.3, 0.6) == "Blue"
    assert zone_for(0.6, 0.6) == "Green"
    assert zone_for(0.8, 0.8) == "Gold"
    # Only one axis at ≥75% → Green, not Gold
    assert zone_for(0.9, 0.6) == "Green"
    assert zone_for(0.6, 0.9) == "Green"


def test_score_table_end_to_end():
    wide = pd.DataFrame([{
        "practitioner_id": 1, "label": "Alice",
        "avg_hours_per_day": 5.5, "pva": 8.0, "ppva": 4.5,
        "cx_dna_combined_rate": 0.06, "utilisation": 0.82,
        "nps": 0.80, "audit_pct": 0.95, "notes_completion": 0.98,
        "punctuality_within_15": 0.94,
    }])
    scored = score_table(wide)
    assert "clinical_axis" in scored.columns
    assert "nonclinical_axis" in scored.columns
    assert scored["zone"].iloc[0] in {"Red", "Orange", "Blue", "Green", "Gold"}


def test_score_table_handles_missing():
    wide = pd.DataFrame([{"practitioner_id": 1, "label": "B"}])
    scored = score_table(wide)
    # All metrics default to 0 → both axes = 1 each (band 1 across the board)
    assert scored["zone"].iloc[0] == "Red"
