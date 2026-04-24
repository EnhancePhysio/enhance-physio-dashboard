"""Smoke tests for metric calculators (no Cliniko needed)."""
import pandas as pd
from datetime import datetime, timedelta

from dashboard.metrics import (
    cx_dna_rates, pva, ppva, total_consults, _is_delivered,
)


def _make_appts():
    return pd.DataFrame([
        # Alice: 2 delivered, 1 cancelled, 1 DNA = 4 scheduled
        {"id": 1, "practitioner_id": 100, "patient_id": 1,
         "business_id": 10, "appointment_type_id": 200,
         "starts_at": pd.Timestamp("2026-04-01T09:00", tz="UTC"),
         "ends_at": pd.Timestamp("2026-04-01T09:30", tz="UTC"),
         "cancelled_at": pd.NaT, "did_not_arrive": False},
        {"id": 2, "practitioner_id": 100, "patient_id": 2,
         "business_id": 10, "appointment_type_id": 200,
         "starts_at": pd.Timestamp("2026-04-02T09:00", tz="UTC"),
         "ends_at": pd.Timestamp("2026-04-02T09:30", tz="UTC"),
         "cancelled_at": pd.NaT, "did_not_arrive": False},
        {"id": 3, "practitioner_id": 100, "patient_id": 3,
         "business_id": 10, "appointment_type_id": 200,
         "starts_at": pd.Timestamp("2026-04-03T09:00", tz="UTC"),
         "ends_at": pd.Timestamp("2026-04-03T09:30", tz="UTC"),
         "cancelled_at": pd.Timestamp("2026-04-02T12:00", tz="UTC"),
         "did_not_arrive": False},
        {"id": 4, "practitioner_id": 100, "patient_id": 4,
         "business_id": 10, "appointment_type_id": 200,
         "starts_at": pd.Timestamp("2026-04-04T09:00", tz="UTC"),
         "ends_at": pd.Timestamp("2026-04-04T09:30", tz="UTC"),
         "cancelled_at": pd.NaT, "did_not_arrive": True},
    ])


def test_delivered_flag():
    df = _make_appts()
    mask = _is_delivered(df)
    assert mask.tolist() == [True, True, False, False]


def test_total_consults():
    g = total_consults(_make_appts())
    assert g.loc[g["practitioner_id"] == 100, "consults_delivered"].iloc[0] == 2


def test_cx_dna_rates():
    g = cx_dna_rates(_make_appts())
    row = g[g["practitioner_id"] == 100].iloc[0]
    assert row["scheduled"] == 4
    assert row["cancelled"] == 1
    assert row["dna"] == 1
    assert row["cx_dna_combined_rate"] == 0.5


def test_pva_basic():
    # Two patients seen twice each vs one patient seen twice
    df = pd.DataFrame([
        {"id": i, "practitioner_id": 100, "patient_id": (i % 2) + 1,
         "business_id": 10, "appointment_type_id": 200,
         "starts_at": pd.Timestamp(f"2026-04-{i+1:02d}", tz="UTC"),
         "ends_at": pd.Timestamp(f"2026-04-{i+1:02d}", tz="UTC") + pd.Timedelta(minutes=30),
         "cancelled_at": pd.NaT, "did_not_arrive": False} for i in range(4)
    ])
    g = pva(df)
    # 4 appts / 2 patients = 2.0
    assert g.loc[g["practitioner_id"] == 100, "pva"].iloc[0] == 2.0


def test_ppva_includes_epc():
    appts = _make_appts()
    appts.loc[0, "appointment_type_id"] = 200  # Initial Private
    appts.loc[1, "appointment_type_id"] = 201  # Initial EPC
    types = pd.DataFrame([
        {"id": 200, "name": "Initial Private Senior Physio"},
        {"id": 201, "name": "Initial EPC Senior Physio"},
    ])
    g = ppva(appts, types)
    # 2 private consults (both initial) / 2 initial = 1.0
    row = g[g["practitioner_id"] == 100].iloc[0]
    assert row["ppva"] == 1.0
