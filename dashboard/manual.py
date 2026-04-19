"""Manual data ingestion: punctuality + NPS, with optional vision extraction."""
from __future__ import annotations

import base64
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.config import DATA_DIR, anthropic_api_key, load_settings


PUNCT_DIR = DATA_DIR / "punctuality"
NPS_DIR = DATA_DIR / "nps"
UPLOADS_DIR = DATA_DIR / "uploads"

PUNCT_DIR.mkdir(parents=True, exist_ok=True)
NPS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


PUNCTUALITY_COLUMNS = [
    "week_starting", "clinic", "practitioner", "day",
    "bucket_0_5", "bucket_6_10", "bucket_11_14", "bucket_15_plus", "status",
]

NPS_COLUMNS = [
    "week_starting", "clinic", "practitioner",
    "responses", "promoters", "passives", "detractors", "nps",
]


# -------------------------------------------------------------------
# Punctuality
# -------------------------------------------------------------------
def load_punctuality() -> pd.DataFrame:
    """Load every CSV in data/punctuality/ and concatenate."""
    files = sorted(PUNCT_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=PUNCTUALITY_COLUMNS)
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # Coerce numeric cols
    for col in ("bucket_0_5", "bucket_6_10", "bucket_11_14", "bucket_15_plus"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def punctuality_per_practitioner(df: pd.DataFrame, start: date, end: date,
                                   practitioner_name_to_id: dict[str, int] | None = None
                                   ) -> pd.DataFrame:
    """Aggregate to one row per practitioner: % seen within 15 min."""
    if df.empty:
        return pd.DataFrame(columns=["practitioner_id", "punctuality_within_15"])
    work = df.copy()
    work["week_starting"] = pd.to_datetime(work["week_starting"], errors="coerce").dt.date
    work = work[(work["week_starting"] >= start) & (work["week_starting"] <= end)]
    leave_vals = load_settings()["punctuality"].get("leave_status_values", [])
    work = work[~work.get("status", "").astype(str).str.lower().isin(leave_vals)]

    g = work.groupby("practitioner").agg(
        b05=("bucket_0_5", "sum"),
        b610=("bucket_6_10", "sum"),
        b1114=("bucket_11_14", "sum"),
        b15=("bucket_15_plus", "sum"),
    ).reset_index()
    g["total"] = g["b05"] + g["b610"] + g["b1114"] + g["b15"]
    g["punctuality_within_15"] = (g["b05"] + g["b610"] + g["b1114"]) / g["total"].where(g["total"] > 0, 1)

    if practitioner_name_to_id:
        g["practitioner_id"] = g["practitioner"].map(practitioner_name_to_id)
        g = g.dropna(subset=["practitioner_id"])
        g["practitioner_id"] = g["practitioner_id"].astype(str)
    else:
        g["practitioner_id"] = None
    return g[["practitioner_id", "practitioner", "punctuality_within_15",
              "b05", "b610", "b1114", "b15", "total"]]


def save_punctuality_csv(df: pd.DataFrame, week_starting: str, clinic: str) -> Path:
    safe_clinic = re.sub(r"[^a-zA-Z0-9_-]", "_", clinic)
    path = PUNCT_DIR / f"{week_starting}_{safe_clinic}.csv"
    df.to_csv(path, index=False)
    return path


# -------------------------------------------------------------------
# NPS
# -------------------------------------------------------------------
def load_nps() -> pd.DataFrame:
    files = sorted(NPS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=NPS_COLUMNS)
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    for col in ("responses", "promoters", "passives", "detractors", "nps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def nps_per_practitioner(df: pd.DataFrame, start: date, end: date,
                          practitioner_name_to_id: dict[str, int] | None = None
                          ) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["practitioner_id", "nps"])
    work = df.copy()
    work["week_starting"] = pd.to_datetime(work["week_starting"], errors="coerce").dt.date
    work = work[(work["week_starting"] >= start) & (work["week_starting"] <= end)]

    g = work.groupby("practitioner").agg(
        responses=("responses", "sum"),
        promoters=("promoters", "sum"),
        detractors=("detractors", "sum"),
    ).reset_index()
    # NPS scale is -100 to +100 — normalise to 0-1 for scoring (add 100, divide 200)
    g["nps_raw"] = ((g["promoters"] - g["detractors"]) / g["responses"].where(g["responses"] > 0, 1)) * 100
    g["nps"] = (g["nps_raw"] + 100) / 200.0  # map -100..100 → 0..1 for rubric lookup

    if practitioner_name_to_id:
        g["practitioner_id"] = g["practitioner"].map(practitioner_name_to_id)
        g = g.dropna(subset=["practitioner_id"])
        g["practitioner_id"] = g["practitioner_id"].astype(str)
    return g[["practitioner_id", "practitioner", "nps", "nps_raw", "responses"]]


# -------------------------------------------------------------------
# Vision extraction (optional; requires ANTHROPIC_API_KEY)
# -------------------------------------------------------------------
PUNCTUALITY_VISION_PROMPT = """You are reading a handwritten punctuality tally sheet from a physiotherapy clinic.
The sheet is organised as rows (practitioners) and columns (days of the week).
Each cell either contains:
  - the circled TOTAL patient count (large number) — use this when present, OR
  - four tally counts for buckets: 0-5 min, 6-10 min, 11-14 min, 15+ min, OR
  - a non-working-day marker: DD / DDAY / A/L / AL / S/L / N/A

Return JSON with this shape (no markdown, no commentary):
{
  "clinic": "<clinic name from sheet>",
  "week_starting": "YYYY-MM-DD",
  "rows": [
    {
      "practitioner": "<name>",
      "days": [
        {"day": "Monday", "bucket_0_5": 0, "bucket_6_10": 0,
         "bucket_11_14": 0, "bucket_15_plus": 0, "status": ""}
      ]
    }
  ]
}
Set status to one of: "off" (DD/DDAY), "leave" (A/L/AL), "sick" (S/L), "na", or "" for a normal working day.
For normal days, if you can read circled totals but not per-bucket counts, put the total in bucket_0_5 and set the others to 0 with a note field.
"""


def extract_punctuality_from_image(image_bytes: bytes, media_type: str = "image/jpeg") -> dict[str, Any]:
    """Call Claude Vision to parse a punctuality sheet image.

    Returns the parsed JSON structure. Raises RuntimeError if the Anthropic key
    is missing or the response isn't valid JSON.
    """
    key = anthropic_api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot use vision extraction. "
                           "Set it in .env or fall back to CSV entry.")
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from e

    client = anthropic.Anthropic(api_key=key)
    b64 = base64.standard_b64encode(image_bytes).decode()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64",
                                              "media_type": media_type, "data": b64}},
                {"type": "text", "text": PUNCTUALITY_VISION_PROMPT},
            ],
        }],
    )
    text = "".join(blk.text for blk in msg.content if getattr(blk, "type", "") == "text")
    # Strip any code fences the model might add
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Vision response was not valid JSON:\n{text[:500]}") from e


def vision_response_to_dataframe(response: dict[str, Any]) -> pd.DataFrame:
    rows = []
    clinic = response.get("clinic", "")
    week = response.get("week_starting", "")
    for prac in response.get("rows", []):
        name = prac.get("practitioner", "")
        for day in prac.get("days", []):
            rows.append({
                "week_starting": week,
                "clinic": clinic,
                "practitioner": name,
                "day": day.get("day", ""),
                "bucket_0_5": day.get("bucket_0_5", 0),
                "bucket_6_10": day.get("bucket_6_10", 0),
                "bucket_11_14": day.get("bucket_11_14", 0),
                "bucket_15_plus": day.get("bucket_15_plus", 0),
                "status": day.get("status", ""),
            })
    return pd.DataFrame(rows, columns=PUNCTUALITY_COLUMNS)
