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
# Fuzzy practitioner-name matching
# -------------------------------------------------------------------
_TITLE_PREFIX_RE = re.compile(r"^(dr|mr|mrs|ms|miss|prof|doctor)\.?\s+", re.IGNORECASE)


def _normalise_name(s: str) -> str:
    """Lower-case, strip titles, collapse punctuation + whitespace."""
    if not s:
        return ""
    s = str(s).strip()
    s = _TITLE_PREFIX_RE.sub("", s)
    s = re.sub(r"[^\w\s]", " ", s)  # punctuation → space
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _compact(s: str) -> str:
    """Whitespace-free form of a normalised name (so 'O Brien' == 'OBrien')."""
    return _normalise_name(s).replace(" ", "")


def build_practitioner_name_index(practs: pd.DataFrame | None,
                                    legacy_dict: dict[str, Any] | None = None,
                                    ) -> dict[str, str]:
    """Build a {normalised-name-variant: practitioner_id} lookup.

    Generates several forms per practitioner so CSV exports using any of
    the common conventions ("Jane Doe", "Doe, Jane", "Dr Jane Doe",
    "J. Doe", "Doe Jane", "O'Brien" vs "OBrien") all resolve to the same
    Cliniko ID. Also indexes a whitespace-free form to handle surnames
    with apostrophes or hyphens that the source system has stripped.
    """
    idx: dict[str, str] = {}

    def _add(raw: str, pid: str) -> None:
        k = _normalise_name(raw)
        if k:
            idx.setdefault(k, pid)
        c = _compact(raw)
        if c and c != k:
            idx.setdefault(c, pid)

    if practs is not None and not practs.empty:
        for _, r in practs.iterrows():
            pid = str(r["id"])
            fn = str(r.get("first_name") or "").strip()
            ln = str(r.get("last_name") or "").strip()
            label = str(r.get("label") or "").strip()
            _add(label, pid)
            if fn and ln:
                _add(f"{fn} {ln}", pid)
                _add(f"{ln} {fn}", pid)
                _add(f"{ln}, {fn}", pid)
                _add(f"{fn[0]} {ln}", pid)
                _add(f"{fn[0]}. {ln}", pid)
                _add(ln, pid)  # last-name-only fallback (low priority via setdefault)

    # Legacy dict (kept for backward compat): {label: id} — inject with lowest priority
    if legacy_dict:
        for name, pid in legacy_dict.items():
            _add(str(name), str(pid))

    return idx


def resolve_practitioner_id(name: str, idx: dict[str, str]) -> str | None:
    """Best-effort lookup: exact → swapped-order → initial+last → compact → last-only."""
    if not name or not idx:
        return None
    key = _normalise_name(name)
    compact = key.replace(" ", "")
    if not key:
        return None
    # Direct hit
    if key in idx:
        return idx[key]
    # Whitespace-free (handles "O Brien" → "obrien")
    if compact in idx:
        return idx[compact]
    parts = key.split()
    if len(parts) >= 2:
        swapped = " ".join(reversed(parts))
        if swapped in idx:
            return idx[swapped]
        if swapped.replace(" ", "") in idx:
            return idx[swapped.replace(" ", "")]
        initial_last = f"{parts[0][0]} {parts[-1]}"
        if initial_last in idx:
            return idx[initial_last]
        last_first = f"{parts[-1]} {parts[0]}"
        if last_first in idx:
            return idx[last_first]
        # Last-name-only (risky if multiple practitioners share a surname,
        # but build_practitioner_name_index uses setdefault so the first-
        # seen wins — good enough as a last resort).
        if parts[-1] in idx:
            return idx[parts[-1]]
    return None


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
                                   practitioner_name_to_id: dict[str, int] | None = None,
                                   practitioners: pd.DataFrame | None = None,
                                   ) -> pd.DataFrame:
    """Aggregate to one row per practitioner: % seen within 15 min.

    Joins to Cliniko practitioners using a fuzzy name resolver (handles
    "Last, First", title prefixes, initials, case/punctuation differences).
    Unmatched names are attached to the returned DataFrame as
    ``df.attrs['unmatched_names']`` so the UI can surface them.
    """
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

    idx = build_practitioner_name_index(practitioners, practitioner_name_to_id)
    g["practitioner_id"] = g["practitioner"].apply(lambda n: resolve_practitioner_id(n, idx))

    unmatched = g.loc[g["practitioner_id"].isna(), "practitioner"].dropna().unique().tolist()
    g = g.dropna(subset=["practitioner_id"]).copy()
    if not g.empty:
        g["practitioner_id"] = g["practitioner_id"].astype(str)
    out = g[["practitioner_id", "practitioner", "punctuality_within_15",
             "b05", "b610", "b1114", "b15", "total"]]
    out.attrs["unmatched_names"] = unmatched
    return out


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


def aggregate_nps_from_individual_scores(
    raw: pd.DataFrame,
    default_week_starting: str | None = None,
    default_clinic: str = "",
) -> pd.DataFrame:
    """Convert a CSV of individual NPS responses to the aggregated format.

    Accepts the messy CSV exports that review platforms ship (column names
    differ between GoPractice, AskNicely, Cliniqapps, Google Reviews, etc.).
    Auto-detects the practitioner and score columns case-insensitively so
    Matt doesn't need to pre-format each export.

    Standard NPS classification:
      • score 9-10 → promoter
      • score 7-8  → passive
      • score 0-6  → detractor
      • NPS (raw)  = (promoters - detractors) / total × 100     → [-100, +100]

    Returned columns match NPS_COLUMNS so the result can be concatenated with
    existing files in data/nps/ and loaded by load_nps() unchanged.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=NPS_COLUMNS)

    # -- Column autodetect (case-insensitive, substring) ----------------
    lower = {c.lower().strip(): c for c in raw.columns}

    def _find(patterns: list[str]) -> str | None:
        # Exact match first, then substring match
        for p in patterns:
            if p in lower:
                return lower[p]
        for p in patterns:
            for k, orig in lower.items():
                if p in k:
                    return orig
        return None

    prac_col = _find([
        "practitioner", "clinician", "therapist", "staff",
        "provider", "physio", "name of practitioner", "practitioner name",
    ])
    score_col = _find([
        "score", "nps", "rating", "recommend", "likelihood",
        "how likely", "nps score",
    ])
    date_col = _find(["date", "submitted", "created", "response date", "week_starting"])
    clinic_col = _find(["clinic", "business", "location", "site"])

    if prac_col is None or score_col is None:
        raise ValueError(
            "Couldn't find practitioner + score columns in CSV. "
            f"Columns detected: {list(raw.columns)}. "
            "Rename the practitioner column to 'practitioner' and the score "
            "column to 'score' (or 'nps' / 'rating') and try again."
        )

    work = raw.copy()
    work["_prac"] = work[prac_col].astype(str).str.strip()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=["_prac", "_score"])
    work = work[work["_prac"] != ""]
    if work.empty:
        return pd.DataFrame(columns=NPS_COLUMNS)

    # Bucket each response
    work["_promoter"] = (work["_score"] >= 9).astype(int)
    work["_passive"] = ((work["_score"] >= 7) & (work["_score"] <= 8)).astype(int)
    work["_detractor"] = (work["_score"] <= 6).astype(int)

    # Group by practitioner + optional week_starting (keeps history intact
    # if the export spans multiple weeks)
    if date_col:
        parsed = pd.to_datetime(work[date_col], errors="coerce")
        # Snap each response to the Monday of its week
        work["week_starting"] = (
            parsed - pd.to_timedelta(parsed.dt.weekday, unit="D")
        ).dt.strftime("%Y-%m-%d")
    else:
        work["week_starting"] = default_week_starting or ""

    work["clinic"] = (
        work[clinic_col].astype(str) if clinic_col else default_clinic
    )

    g = work.groupby(["week_starting", "clinic", "_prac"]).agg(
        responses=("_score", "count"),
        promoters=("_promoter", "sum"),
        passives=("_passive", "sum"),
        detractors=("_detractor", "sum"),
    ).reset_index().rename(columns={"_prac": "practitioner"})

    g["nps"] = (
        (g["promoters"] - g["detractors"]) / g["responses"].where(g["responses"] > 0, 1) * 100.0
    )
    return g[NPS_COLUMNS]


def save_nps_csv(df: pd.DataFrame, filename: str) -> Path:
    """Write an aggregated NPS frame into data/nps/ so load_nps() sees it."""
    if not filename.lower().endswith(".csv"):
        filename = f"{filename}.csv"
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    out = NPS_DIR / safe
    df.to_csv(out, index=False)
    return out


def nps_per_practitioner(df: pd.DataFrame, start: date, end: date,
                          practitioner_name_to_id: dict[str, int] | None = None,
                          practitioners: pd.DataFrame | None = None,
                          ) -> pd.DataFrame:
    """Aggregate NPS to one row per practitioner over [start, end].

    Joins to Cliniko practitioners using a fuzzy name resolver so CSV
    exports using any common naming convention match correctly.
    Unmatched names are attached to the returned DataFrame as
    ``df.attrs['unmatched_names']`` so the UI can surface them.
    """
    if df.empty:
        out = pd.DataFrame(columns=["practitioner_id", "practitioner", "nps", "nps_raw", "responses"])
        out.attrs["unmatched_names"] = []
        return out
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

    idx = build_practitioner_name_index(practitioners, practitioner_name_to_id)
    g["practitioner_id"] = g["practitioner"].apply(lambda n: resolve_practitioner_id(n, idx))

    unmatched = g.loc[g["practitioner_id"].isna(), "practitioner"].dropna().unique().tolist()
    g = g.dropna(subset=["practitioner_id"]).copy()
    if not g.empty:
        g["practitioner_id"] = g["practitioner_id"].astype(str)
    out = g[["practitioner_id", "practitioner", "nps", "nps_raw", "responses"]]
    out.attrs["unmatched_names"] = unmatched
    return out


# -------------------------------------------------------------------
# Vision extraction (optional; requires ANTHROPIC_API_KEY)
# -------------------------------------------------------------------
PUNCTUALITY_VISION_PROMPT = """You are reading a handwritten punctuality tally sheet from a physiotherapy clinic.
The sheet is organised as rows (practitioners) and columns (days of the week).
Each cell either contains:
  - the circled TOTAL patient count (large number) — use this when present, OR
  - four tally counts for buckets: 0-5 min, 6-10 min, 11-14 min, 15+ min, OR
  - a non-working-day marker: DD / DDAY / A/L / AL / S/L / N/A

CRITICAL: Your ENTIRE response must be a single JSON object and nothing else.
Do NOT narrate what you see. Do NOT say "I'll analyze...". Do NOT wrap in
markdown fences. Start with `{` and end with `}`. Any prose before or
after the JSON will crash the downstream parser.

Return JSON with this shape:
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


_CLAUDE_VISION_MEDIA_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _normalise_image_to_jpeg(image_bytes: bytes) -> tuple[bytes, str]:
    """Convert any uploaded image (including iPhone HEIC/HEIF) to JPEG bytes.

    Claude Vision only accepts jpeg/png/gif/webp. Rather than try to guess
    the input format from the browser-reported MIME type (which is
    frequently wrong — iPhone HEIC gets reported as image/heic or even
    application/octet-stream), we let Pillow sniff the format from the
    file signature and re-encode to JPEG. This also strips EXIF and
    shrinks massive iPhone originals (~5MB → ~500KB) before upload, which
    speeds up the vision call meaningfully.
    """
    from io import BytesIO
    from PIL import Image

    # PDF upload — render page 1 to a PIL image via pypdfium2, which ships
    # its own bundled binary and works on Streamlit Cloud without needing
    # poppler/ghostscript at the system level. We render at 200 DPI which
    # is plenty for Claude Vision to read handwritten tick boxes.
    if image_bytes[:5] == b"%PDF-":
        try:
            import pypdfium2 as pdfium  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "PDF upload detected, but the PDF renderer (pypdfium2) isn't "
                "installed. Check your requirements.txt contains "
                "'pypdfium2>=4.30.0' and reboot the app (Manage app → Reboot). "
                "Or screenshot the PDF page and upload the image instead."
            ) from e
        try:
            pdf = pdfium.PdfDocument(image_bytes)
            if len(pdf) == 0:
                raise RuntimeError("PDF has no pages.")
            page = pdf[0]
            pil_img = page.render(scale=200 / 72.0).to_pil()  # 200 DPI
            page.close()
            pdf.close()
        except Exception as e:
            raise RuntimeError(
                f"Couldn't render the PDF. Underlying error: "
                f"{type(e).__name__}: {e}. Try screenshotting the PDF page "
                "and uploading the image instead."
            ) from e
        # Re-enter the normal pipeline via an in-memory PNG round-trip so
        # all the downstream mode / resize / EXIF handling below still runs.
        from io import BytesIO as _BIO
        _png = _BIO()
        pil_img.save(_png, format="PNG")
        image_bytes = _png.getvalue()

    # Register HEIC/HEIF support if the plugin is available. Track whether
    # it actually loaded so we can give a precise error if a HEIC file
    # arrives and the plugin isn't installed.
    heic_available = False
    try:
        from pillow_heif import register_heif_opener  # type: ignore
        register_heif_opener()
        heic_available = True
    except ImportError:
        pass

    # File signature sniff — used only to produce a better error message.
    sig = image_bytes[:16]
    looks_heic = (
        len(sig) >= 12
        and sig[4:8] == b"ftyp"
        and sig[8:12] in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"hevm",
                           b"hevs", b"mif1", b"msf1")
    )

    try:
        img = Image.open(BytesIO(image_bytes))
        # Force decode so a lazy-open exception surfaces here, not later
        img.load()
    except Exception as e:
        if looks_heic and not heic_available:
            raise RuntimeError(
                "This looks like an iPhone HEIC photo but the HEIC decoder "
                "(pillow-heif) isn't installed in the deployed app. Check "
                "your requirements.txt contains 'pillow-heif>=0.16.0' and "
                "that Streamlit Cloud has reinstalled dependencies "
                "(Manage app → Reboot app)."
            ) from e
        raise RuntimeError(
            f"Couldn't read the uploaded file as an image. "
            f"Underlying error: {type(e).__name__}: {e}. "
            f"First bytes (hex): {sig.hex()[:32]}. "
            "Try saving it as JPG or PNG and uploading again."
        ) from e

    # JPEG doesn't support alpha channels — flatten onto white.
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        img_rgb = img.convert("RGBA") if img.mode != "RGBA" else img
        bg.paste(img_rgb, mask=img_rgb.split()[-1] if img_rgb.mode == "RGBA" else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Cap longest side at 2000px to keep payloads small without hurting OCR.
    max_edge = 2000
    if max(img.size) > max_edge:
        ratio = max_edge / float(max(img.size))
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    out = BytesIO()
    img.save(out, format="JPEG", quality=90, optimize=True)
    return out.getvalue(), "image/jpeg"


def extract_punctuality_from_image(image_bytes: bytes, media_type: str = "image/jpeg") -> dict[str, Any]:
    """Call Claude Vision to parse a punctuality sheet image.

    Normalises every upload to JPEG first (handles HEIC from iPhone,
    PNG, WEBP, BMP, and oversized originals). Returns the parsed JSON
    structure. Raises RuntimeError if the Anthropic key is missing or
    the response isn't valid JSON.
    """
    key = anthropic_api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — cannot use vision extraction. "
                           "Set it in .env or fall back to CSV entry.")
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic") from e

    # Always normalise. Even a legit image/jpeg from a phone is worth
    # passing through Pillow to strip EXIF and shrink to ≤2000px — the
    # vision call is much faster on a 500KB JPEG than a 5MB original.
    jpeg_bytes, norm_media_type = _normalise_image_to_jpeg(image_bytes)

    client = anthropic.Anthropic(api_key=key)
    b64 = base64.standard_b64encode(jpeg_bytes).decode()
    # v25.2 — force JSON-only output via an assistant prefill. Without this,
    # Claude Sonnet 4.6 sometimes preambles with analysis ("I'll carefully
    # analyze this tally sheet...") and runs out of max_tokens before
    # emitting valid JSON. Seeding the assistant turn with `{` means the
    # model must continue JSON from the very first token.
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,  # bumped for safety — a full 6-practitioner × 5-day sheet is ~1.5k tokens
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": norm_media_type, "data": b64}},
                    {"type": "text", "text": PUNCTUALITY_VISION_PROMPT},
                ],
            },
            {"role": "assistant", "content": "{"},
        ],
    )
    text = "".join(blk.text for blk in msg.content if getattr(blk, "type", "") == "text")
    # The assistant turn began with `{`, so the response continues from
    # the next character. Prepend it so we get a parseable whole.
    if not text.lstrip().startswith("{"):
        text = "{" + text
    # Strip any code fences the model might still try to add
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Defence in depth — extract the first `{...}` block by balancing
        # braces, in case the model wrapped its output in prose somehow.
        extracted = _extract_first_json_object(text)
        if extracted is not None:
            try:
                return json.loads(extracted)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(
            "Vision response was not valid JSON. First 500 chars of response "
            f"below — if this looks like an explanation of the sheet rather "
            f"than JSON, retry with a clearer image.\n\n{text[:500]}"
        ) from None


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first balanced `{...}` block from `text`.

    Handles simple cases where the model preambled or appended prose around
    a valid JSON object. Returns None if no balanced block is found.
    Not a full JSON parser — just brace-counting with string awareness.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


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
