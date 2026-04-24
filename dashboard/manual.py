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


# v26.2 — common first-name nicknames. Used bidirectionally: if the CSV
# has "Paddy" we try "patrick" against the Cliniko list, and vice-versa.
# Matt's staff write these variants on the punctuality sheets, so without
# this map the vision extraction produces valid rows that drop at the
# name-matching stage and score as 0. Lower-case keys.
NICKNAME_TO_FULL: dict[str, str] = {
    "paddy": "patrick", "pat": "patrick",
    "tori": "victoria", "vicki": "victoria", "vic": "victoria",
    "liz": "elizabeth", "beth": "elizabeth", "libby": "elizabeth",
    "bob": "robert", "rob": "robert", "robbie": "robert",
    "jim": "james", "jimmy": "james",
    "jack": "john", "johnny": "john",
    "charlie": "charles",
    "tony": "anthony",
    "mike": "michael",
    "dick": "richard", "rick": "richard", "rich": "richard",
    "bill": "william", "billy": "william", "will": "william",
    "nick": "nicholas",
    "ted": "edward", "ned": "edward", "eddie": "edward", "ed": "edward",
    "pete": "peter",
    "sam": "samuel", "sammy": "samuel",
    "jess": "jessica", "jessie": "jessica",
    "chris": "christopher",
    "alex": "alexander",
    "andy": "andrew", "drew": "andrew",
    "greg": "gregory",
    "tom": "thomas", "tommy": "thomas",
    "dan": "daniel", "danny": "daniel",
    "joe": "joseph",
    "dave": "david",
    "matt": "matthew", "matty": "matthew",
    "kate": "katherine", "katie": "katherine",
    "maggie": "margaret", "meg": "margaret",
    "jen": "jennifer", "jenny": "jennifer",
    "abby": "abigail",
    "nath": "nathan", "nat": "nathan",
    "seb": "sebastian", "sebby": "sebastian",
    "ben": "benjamin",
    "mitch": "mitchell",
    "mackie": "mackenzie",
}


def _first_name_candidates(first: str) -> set[str]:
    """Return all plausible canonical forms of a first-name token.

    Bidirectional — if we see "matt" we try "matthew"; if we see "matthew"
    we also try "matt" against the Cliniko list (in case the staff roster
    uses the nickname as the canonical label). Always includes `first`
    itself.
    """
    out = {first}
    if first in NICKNAME_TO_FULL:
        out.add(NICKNAME_TO_FULL[first])
    for nick, full in NICKNAME_TO_FULL.items():
        if full == first:
            out.add(nick)
    return out


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

    # v26.2 — also stash the per-practitioner (first, last, pid) rows on the
    # index so resolve_practitioner_id can do first-name-unique, nickname,
    # and "First L." matching without re-fetching the practitioners df.
    rows: list[tuple[str, str, str]] = []
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
            rows.append((_normalise_name(fn), _normalise_name(ln), pid))

    # Legacy dict (kept for backward compat): {label: id} — inject with lowest priority
    if legacy_dict:
        for name, pid in legacy_dict.items():
            _add(str(name), str(pid))

    # Use a key guaranteed not to collide with any real normalised name
    # (normalisation strips punctuation, so double-underscore is safe).
    idx["__rows__"] = rows  # type: ignore[assignment]
    return idx


def resolve_practitioner_id(name: str, idx: dict[str, str]) -> str | None:
    """Best-effort lookup: exact → swapped-order → initial+last → compact →
    last-only → (v26.2) nickname/first-name-unique → "First L." disambiguation.

    The last three fall-backs fire when the handwritten punctuality sheets
    use short forms ("Nath", "Paddy", "Tori", "Sarah C.") that the exact-
    match index above can't bridge. They only return a pid when the
    resolution is UNAMBIGUOUS — if two practitioners both match the short
    form, we return None rather than guess.
    """
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

    # v26.2 — nickname + first-name-unique + "First L." fall-backs.
    rows = idx.get("__rows__")  # type: ignore[assignment]
    if not rows:
        return None
    first = parts[0]
    first_cands = _first_name_candidates(first)

    def _first_matches(fn: str) -> bool:
        if not fn:
            return False
        # Exact match against any nickname candidate...
        if fn in first_cands:
            return True
        # ...or the Cliniko first name starts with what was written
        # ("nath" → "nathan", "seb" → "sebastian", "mitch" → "mitch wadley")
        if fn.startswith(first):
            return True
        # ...or what was written starts with the Cliniko first name (rare,
        # but covers "patricki" being extracted for "Patrick").
        if first.startswith(fn) and len(fn) >= 3:
            return True
        return False

    hits = [(pid, fn, ln) for (fn, ln, pid) in rows if _first_matches(fn)]  # type: ignore[misc]
    if len(hits) == 1:
        return hits[0][0]
    if len(hits) > 1 and len(parts) >= 2:
        # Disambiguate by last-name prefix. "Sarah C." → normalised "sarah c"
        # → parts=["sarah", "c"] → last-name must start with "c" → Clemm.
        # Also handles "Chris Ob" → Oberson (not Oats).
        second = parts[-1]
        narrowed = [(pid, fn, ln) for (pid, fn, ln) in hits
                    if ln and ln.startswith(second)]
        if len(narrowed) == 1:
            return narrowed[0][0]
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
    # Drop leave days AND any row the vision extractor marked as unreadable
    # (all-zero buckets on a working day would otherwise poison the ratio).
    excluded = set(leave_vals) | {"unreadable"}
    work = work[~work.get("status", "").astype(str).str.lower().isin(excluded)]

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
    """Write a punctuality CSV to data/punctuality/ and (if configured)
    push it to the GitHub repo so it survives Streamlit Cloud reboots.

    The return value is the local path; GitHub-commit status is surfaced
    separately via ``df.attrs['github_sync_result']`` on the returned
    dataframe's caller side. Callers that want to know sync status should
    inspect the logs or call ``save_punctuality_csv_with_sync_status``.
    """
    safe_clinic = re.sub(r"[^a-zA-Z0-9_-]", "_", clinic)
    path = PUNCT_DIR / f"{week_starting}_{safe_clinic}.csv"
    df.to_csv(path, index=False)
    # v26 — auto-commit to GitHub if configured. Silent no-op if not.
    try:
        from dashboard.github_persistence import save_file_to_github
        ok, msg = save_file_to_github(
            path, commit_msg=f"[data] punctuality {week_starting} {clinic}",
        )
        # Attach the result to a module-level cache so the UI can display
        # a toast/banner after the save button rerun.
        _last_github_sync["punctuality"] = (ok, msg)
    except Exception as e:
        _last_github_sync["punctuality"] = (False, f"sync skipped: {e}")
    return path


# v26 — last sync result per data kind. Module-level (not session-state)
# so it survives reruns without needing a st.* write from this file.
_last_github_sync: dict[str, tuple[bool, str]] = {}


def last_github_sync(kind: str) -> tuple[bool, str] | None:
    """Return the most recent GitHub sync result for 'punctuality' or
    'nps', or None if no save has been attempted this session."""
    return _last_github_sync.get(kind)


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
    """Write an aggregated NPS frame into data/nps/ so load_nps() sees it.

    v26 — also auto-commits to the configured GitHub repo so weekly NPS
    uploads survive Streamlit Cloud reboots.
    """
    if not filename.lower().endswith(".csv"):
        filename = f"{filename}.csv"
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    out = NPS_DIR / safe
    df.to_csv(out, index=False)
    try:
        from dashboard.github_persistence import save_file_to_github
        ok, msg = save_file_to_github(
            out, commit_msg=f"[data] nps {safe}",
        )
        _last_github_sync["nps"] = (ok, msg)
    except Exception as e:
        _last_github_sync["nps"] = (False, f"sync skipped: {e}")
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
Each working-day cell contains tally marks distributed across FOUR time buckets:
  - bucket_0_5:    patients seen within 0-5 minutes of their appointment
  - bucket_6_10:   patients seen 6-10 minutes late
  - bucket_11_14:  patients seen 11-14 minutes late
  - bucket_15_plus: patients seen 15+ minutes late (i.e. NOT punctual)
Off-day markers: DD / DDAY / A/L / AL / S/L / N/A.

Your job is to COUNT the tally marks in each bucket for each practitioner / day.
Count strokes carefully — a "|||| " five-bar-gate is 5, not 1. Grouped clusters
of short strokes each count as one mark. If a bucket is blank, it is 0.

CRITICAL accuracy rules:
1. NEVER dump a circled daily total into bucket_0_5. A circled total on its own
   (with no per-bucket breakdown visible) means the breakdown is unreadable —
   set ALL four buckets to 0 and set status to "unreadable". Dumping into
   bucket_0_5 would falsely report that practitioner as 100% on time.
2. bucket_0_5 is ONLY for patients seen 0-5 min after start time. It is NOT
   a catch-all.
3. If you can read SOME buckets but not others, do your best with the ones you
   can read and leave the rest at 0 (this is fine — tallies are cumulative
   across the week).
4. Only count tallies you are confident about. Better to leave a bucket at 0
   than to guess.

CRITICAL output format: Your ENTIRE response must be a single JSON object and
nothing else. Do NOT narrate what you see. Do NOT say "I'll analyze...". Do
NOT wrap in markdown fences. Start with `{` and end with `}`. Any prose
before or after the JSON will crash the downstream parser.

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
Set status to one of: "off" (DD/DDAY), "leave" (A/L/AL), "sick" (S/L), "na",
"unreadable" (cell has tallies but you can't resolve them into buckets), or
"" for a normal working day with readable tallies.
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
    # v26.0.1 — the previous prefill trick (seeding an assistant-turn with
    # `{`) is rejected by some Anthropic models with
    #   "This model does not support assistant message prefill."
    # so we drop it and rely on (a) a strongly-worded JSON-only prompt,
    # (b) a generous max_tokens budget, and (c) the brace-balanced extractor
    # below which can salvage a JSON object out of preambled prose.
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,  # enough for a full 6-practitioner × 5-day sheet + any preamble
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": norm_media_type, "data": b64}},
                    {"type": "text", "text": PUNCTUALITY_VISION_PROMPT},
                ],
            },
        ],
    )
    text = "".join(blk.text for blk in msg.content if getattr(blk, "type", "") == "text")
    # Strip any code fences the model might try to add
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    # First: try parsing the whole response as JSON (happy path — prompt was obeyed).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Defence in depth — the model may have preambled or appended prose.
    # Extract the first balanced `{...}` block and try that.
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
