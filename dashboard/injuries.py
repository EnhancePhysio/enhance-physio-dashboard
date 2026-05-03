"""Injury-area analytics for initial-consult treatment notes.

v26.8 — pulls every initial-consult treatment_note in the date range from
Cliniko, parses two signals out of the structured `content.sections`:

  1. The "Area of Injury" question (type=checkboxes) — preferred. Comes
     back as comma-separated text like "Knee, Ankle".
  2. The "History of Presenting Complaint" / "HoPC" paragraph — fallback
     when the practitioner skipped the checkboxes.

Both are run through a keyword classifier that maps to one of 12 broad
buckets (Neck/Head, Lower Back, Knee, etc — see settings.yml).

A patient with multiple ticked areas counts in EACH bucket, so the
totals reflect "things the clinic saw" rather than unique patient
counts. That matches Matt's reporting intent.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import timedelta
from typing import Any

import pandas as pd

from dashboard.cliniko import ClinikoClient
from dashboard.config import load_settings
from dashboard.date_ranges import DateRange


# Default category list — only used as a fallback if settings.yml is
# missing the `injuries.categories` block. Keep in sync with settings.yml.
_DEFAULT_CATEGORIES: dict[str, list[str]] = {
    "Neck/Head": ["neck", "cervical", "head", "headache", "migraine",
                   "TMJ", "jaw", "concussion", "whiplash", "vertigo", "BPPV"],
    "Upper Back / Trunk": ["thoracic", "upper back", "mid back", "rib",
                            "ribs", "trunk", "chest", "sternum",
                            "costochondritis", "intercostal"],
    "Lower Back": ["lumbar", "lower back", "low back", "LBP", "sacrum",
                    "sacroiliac", "SI joint", "sciatica", "disc",
                    "facet", "spondylolisthesis", "spondylolysis",
                    "pars defect"],
    "Hip / Groin": ["hip", "groin", "glute", "gluteal", "piriformis",
                     "ITB", "iliotibial", "psoas", "adductor",
                     "femoroacetabular", "FAI", "labral tear",
                     "trochanter", "trochanteric"],
    "Knee": ["knee", "kneecap", "patella", "patellar", "ACL", "MCL",
              "LCL", "PCL", "meniscus", "meniscal", "Osgood",
              "Schlatter", "chondromalacia", "jumpers knee",
              "runners knee", "patellofemoral"],
    "Ankle": ["ankle", "Achilles", "peroneal", "syndesmosis",
               "high ankle"],
    "Foot": ["foot", "feet", "plantar", "heel", "toe", "midfoot",
              "forefoot", "bunion", "hallux", "metatarsal", "Lisfranc",
              "Lis Franc", "Mortons", "Morton's", "sesamoid"],
    "Shoulder": ["shoulder", "AC joint", "ACJ", "acromioclavicular",
                  "rotator cuff", "supraspinatus", "infraspinatus",
                  "subscapularis", "teres minor", "biceps tendon",
                  "labrum", "SLAP", "glenohumeral", "frozen shoulder",
                  "subacromial", "scapula"],
    "Elbow": ["elbow", "tennis elbow", "golfers elbow", "golfer's elbow",
               "lateral epicondyle", "medial epicondyle",
               "epicondylitis", "olecranon", "ulnar nerve"],
    "Wrist/Hand": ["wrist", "hand", "finger", "thumb", "carpal",
                    "TFCC", "de Quervain", "DeQuervain", "trigger finger",
                    "mallet finger", "scaphoid", "Colles", "phalange",
                    "metacarpal"],
    "Soft Tissue": ["hamstring", "calf", "gastrocnemius", "soleus",
                     "quadriceps", "quad strain", "vastus"],
}


_INITIAL_TYPE_PATTERN = re.compile(
    r"\b(initial|new\s*patient|new\s*client|assessment|first\s*visit)\b",
    re.IGNORECASE,
)


def _settings() -> dict[str, Any]:
    return load_settings().get("injuries", {}) or {}


def _categories() -> dict[str, list[str]]:
    cats = _settings().get("categories")
    if isinstance(cats, dict) and cats:
        # Preserve YAML insertion order (Python 3.7+ dicts are ordered)
        return {str(k): [str(x) for x in v] for k, v in cats.items()}
    return _DEFAULT_CATEGORIES


def _area_question_names() -> list[str]:
    names = _settings().get("area_question_names",
                              ["Area of Injury", "Injury Area",
                               "Body Region", "Region", "Area"])
    return [str(n).lower() for n in names]


def _hopc_question_names() -> list[str]:
    names = _settings().get("presenting_complaint_names",
                              ["HoPC", "History of Presenting Complaint",
                               "Presenting Complaint", "Complaint history",
                               "Chief Complaint"])
    return [str(n).lower() for n in names]


def _compile_patterns(categories: dict[str, list[str]]
                       ) -> dict[str, list[re.Pattern[str]]]:
    """Compile each keyword into a case-insensitive word-boundary regex.

    We use \\b at both ends rather than a permissive suffix so "hip"
    doesn't accidentally match "hippopotamus" (unlikely in clinical
    notes but cheap to be safe). For plurals/compounds we list explicit
    forms in the keyword list (e.g. both "knee" and "kneecap").
    """
    out: dict[str, list[re.Pattern[str]]] = {}
    for cat, kws in categories.items():
        compiled: list[re.Pattern[str]] = []
        for kw in kws:
            kw_str = str(kw).strip()
            if not kw_str:
                continue
            # Word boundary at both ends; case-insensitive
            try:
                compiled.append(re.compile(r"\b" + re.escape(kw_str) + r"\b",
                                            re.IGNORECASE))
            except re.error:
                continue
        out[cat] = compiled
    return out


def classify_injury(text: str,
                     patterns_by_cat: dict[str, list[re.Pattern[str]]] | None = None
                     ) -> list[str]:
    """Return the list of injury categories matched in ``text``.

    Multiple categories can match — e.g. "right knee + left ankle" returns
    ["Knee", "Ankle"]. Returns ["Other"] only when text exists but no
    patterns match. Returns [] for empty/missing text (so callers can
    skip noting an injury at all).
    """
    if not text or not isinstance(text, str):
        return []
    if patterns_by_cat is None:
        patterns_by_cat = _compile_patterns(_categories())
    matches: list[str] = []
    for cat, patterns in patterns_by_cat.items():
        if any(p.search(text) for p in patterns):
            matches.append(cat)
    return matches if matches else ["Other"]


def extract_injury_text(note: dict[str, Any]) -> tuple[str, str]:
    """Pull (checkbox_text, hopc_text) from a Cliniko treatment_note.

    Returns whichever fields could be found, joined to plain text. If a
    section uses ``type=checkboxes`` we read the answer values; if it's
    ``type=paragraph`` we read the body.

    Cliniko's note JSON shape (observed):
      content.sections[*].questions[*]
        - .name   ("Area of Injury", "HoPC", etc)
        - .type   ("checkboxes" | "paragraph" | "text" | "radiobuttons" |
                    "date" | "bodycharts")
        - .answers OR .answer (depending on type) — strings, arrays, or
          dicts

    This is defensive against variation in Cliniko's payload across
    organisations and template versions — it tries multiple shapes
    before giving up.
    """
    if not isinstance(note, dict):
        return "", ""
    content = note.get("content")
    if not isinstance(content, dict):
        return "", ""
    sections = content.get("sections")
    if not isinstance(sections, list):
        return "", ""

    area_names = _area_question_names()
    hopc_names = _hopc_question_names()

    checkbox_bits: list[str] = []
    hopc_bits: list[str] = []

    for section in sections:
        if not isinstance(section, dict):
            continue
        questions = section.get("questions")
        if not isinstance(questions, list):
            continue
        for q in questions:
            if not isinstance(q, dict):
                continue
            qname = str(q.get("name") or "").strip()
            qname_lower = qname.lower()
            qtype = str(q.get("type") or "").lower()

            is_area = qname_lower in area_names
            is_hopc = qname_lower in hopc_names

            if not (is_area or is_hopc):
                continue

            # Try every plausible answer-bearing key. Cliniko's API
            # returns checkboxes as a list of selected option strings,
            # but some versions nest them under "selected", "value",
            # "answer", or even "options" with checked flags.
            ans_text = _stringify_answer(q)
            if not ans_text:
                continue

            if is_area:
                checkbox_bits.append(ans_text)
            elif is_hopc:
                hopc_bits.append(ans_text)

    return " ; ".join(checkbox_bits).strip(), " ; ".join(hopc_bits).strip()


_SELECT_KEYS = ("checked", "selected", "ticked", "is_selected", "is_checked")


def _is_item_selected(item: dict[str, Any]) -> bool | None:
    """True if a dict item has a truthy 'selected' key, False if explicitly
    falsy, None if no select key is present at all.

    The None case matters: if Cliniko gives us [{"label":"Knee"},
    {"label":"Ankle"}, ...] with NO select key, we cannot tell which
    are ticked vs not — so the safest move is to treat the answer as
    structurally ambiguous and let the caller decide whether to use it.
    """
    for k in _SELECT_KEYS:
        if k in item:
            return bool(item[k])
    return None


def _stringify_answer(question: dict[str, Any]) -> str:
    """Convert Cliniko's checkbox answer into a flat text string of
    ONLY the ticked options.

    v26.8.1 — earlier version was too permissive: when the answer was a
    list of dicts WITHOUT a select-state key, it included every label
    as if ticked. With Cliniko's checkbox payload that's every option
    in the template (~12 body parts), so every note matched every
    category. Now: if a list of dicts has NO recognisable select key,
    we treat the data as ambiguous and skip Pass A entirely (caller
    falls back to HoPC keyword scan).
    """
    # ---- "answers" or "answer" or "value" lists/strings/dicts ----
    for key in ("answer", "answers", "value", "values", "selected", "text"):
        val = question.get(key)
        if val is None:
            continue

        # Plain string: Cliniko's checkboxAnswers() helper renders
        # the ticked items as comma-separated text (we saw this in the
        # Cliniko UI HTML). Trust it as-is.
        if isinstance(val, str):
            if val.strip():
                return val

        # List form
        elif isinstance(val, (list, tuple)):
            # Two sub-shapes:
            #   (a) list of plain strings — Cliniko returns ticked items
            #       only. Trust as-is.
            #   (b) list of dicts {"label":..., "checked":bool}. Trust
            #       only items with explicit selected=True.
            string_items = [s for s in val if isinstance(s, str) and s.strip()]
            dict_items = [d for d in val if isinstance(d, dict)]

            if string_items and not dict_items:
                # Pure list of strings → ticked items only
                return ", ".join(string_items)

            if dict_items:
                selected_flags = [_is_item_selected(d) for d in dict_items]
                # Determinable: at least one item has a select-state key
                if any(s is not None for s in selected_flags):
                    chosen: list[str] = []
                    for d, sel in zip(dict_items, selected_flags):
                        if not sel:  # treat None as not-selected here
                            continue
                        lbl = d.get("label") or d.get("value") or d.get("name")
                        if isinstance(lbl, str) and lbl.strip():
                            chosen.append(lbl)
                    if chosen:
                        return ", ".join(chosen)
                    # Determinable but no items ticked → empty answer
                    return ""
                # No select keys present — ambiguous. Skip rather than
                # guess (falls through to HoPC).
                continue

        # Single dict
        elif isinstance(val, dict):
            label = val.get("label") or val.get("value") or val.get("text")
            if isinstance(label, str) and label.strip():
                return label

    # ---- "options" with embedded checked state ----
    options = question.get("options")
    if isinstance(options, list):
        chosen: list[str] = []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            if _is_item_selected(opt) is not True:
                continue
            lbl = opt.get("label") or opt.get("value") or opt.get("name")
            if isinstance(lbl, str) and lbl.strip():
                chosen.append(lbl)
        if chosen:
            return ", ".join(chosen)

    return ""


# v26.8.1 — sanity cap. A real patient ticking 6+ different body-part
# categories on one initial-consult is implausible (we only have ~11
# categories and most patients have 1–3 areas of complaint). If Pass A
# returns more than this many categories, the parser almost certainly
# leaked unticked options into the answer text — so we throw out Pass A
# and fall back to HoPC instead.
_MAX_PLAUSIBLE_CATEGORIES_PER_NOTE = 5


def extract_injuries_from_note(note: dict[str, Any],
                                 patterns_by_cat: dict[str, list[re.Pattern[str]]] | None = None
                                 ) -> list[str]:
    """End-to-end: note → list of injury categories.

    Pass A: structured checkbox answers (preferred, no false positives).
    Pass B: HoPC keyword scan (fallback, mild risk of false positives).
    Returns empty list when both signals are silent — caller should
    decide whether to log as "Other" or skip entirely.

    v26.8.1 — Pass A now has a sanity cap (see _MAX_PLAUSIBLE_CATEGORIES
    _PER_NOTE). If the structured answer text matches more than 5
    categories, treat it as a parser leak and fall through to HoPC.
    """
    checkbox_text, hopc_text = extract_injury_text(note)
    if patterns_by_cat is None:
        patterns_by_cat = _compile_patterns(_categories())

    # Pass A — checkbox answers go to classifier
    if checkbox_text:
        cats = classify_injury(checkbox_text, patterns_by_cat)
        # Sanity guard: too many categories on one note ≠ a real patient,
        # almost certainly a parser leak (every option included as if
        # ticked). Fall through to HoPC.
        if cats and cats != ["Other"] and len(cats) <= _MAX_PLAUSIBLE_CATEGORIES_PER_NOTE:
            return cats
        # Fall through to HoPC if the checkbox answer was something
        # unmappable, blank, or implausibly broad.

    # Pass B — keyword scan of HoPC paragraph
    if hopc_text:
        cats = classify_injury(hopc_text, patterns_by_cat)
        # Same sanity cap applies — an HoPC paragraph that mentions
        # 6+ body parts is almost certainly boilerplate / template
        # text, not a real patient's complaint.
        if cats and len(cats) <= _MAX_PLAUSIBLE_CATEGORIES_PER_NOTE:
            return cats

    return []


# -------------------------------------------------------------------
# Cliniko data fetch
# -------------------------------------------------------------------
def _fetch_treatment_notes_with_content(client: ClinikoClient,
                                         dr: DateRange) -> list[dict[str, Any]]:
    """Pull /treatment_notes for the date range, retaining the full
    content payload so downstream parsers can read structured questions.

    Different from metrics._fetch_treatment_notes_for_range, which
    discards content (it only needs created_at / finalized_at for
    notes_completion). We need the full payload here.
    """
    start_dt = pd.Timestamp(dr.start_iso_utc)
    end_dt = pd.Timestamp(dr.end_iso_utc)
    # Buffer ±3 days to catch notes finalised just before/after the
    # appt window — same as metrics.py does.
    q_start = (start_dt - pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    q_end = (end_dt + pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"q[]": [f"created_at:>={q_start}", f"created_at:<{q_end}"]}

    notes: list[dict[str, Any]] = []
    try:
        for n in client.paginate("treatment_notes", params=params):
            if not isinstance(n, dict):
                continue
            if n.get("archived_at"):
                # Skip archived/deleted notes
                continue
            notes.append(n)
    except Exception:
        # If the endpoint fails entirely, return what we have. The UI
        # will show 0 rows and the diagnostic banner will explain.
        return notes
    return notes


def _initial_appointment_ids(appts: pd.DataFrame,
                               appointment_types: pd.DataFrame | None = None
                               ) -> set[str]:
    """Return the set of appointment IDs whose type is an Initial / New
    Patient / Assessment. Mirrors the logic used by new_patients() in
    metrics.py — a Cliniko appointment is "initial" if its type name
    matches the configured pattern.

    If we can't get appointment_types from the caller, fall back to
    treating every appt as initial — caller will then over-count, but
    safer than under-counting (UI shows raw text on hover).
    """
    if appts is None or appts.empty:
        return set()
    if appointment_types is None or appointment_types.empty:
        # Best-effort: include every appt id; the parser still needs
        # to find a matching note before it counts.
        return set(appts.get("id", pd.Series(dtype=str)).astype(str))
    types = appointment_types.copy()
    types["_is_initial"] = types["name"].fillna("").apply(
        lambda n: bool(_INITIAL_TYPE_PATTERN.search(str(n)))
    )
    initial_type_ids = set(types.loc[types["_is_initial"], "id"].astype(str))
    if "appointment_type_id" not in appts.columns:
        return set()
    mask = appts["appointment_type_id"].astype(str).isin(initial_type_ids)
    return set(appts.loc[mask, "id"].astype(str)) if "id" in appts.columns else set()


def _appt_id_from_note(note: dict[str, Any]) -> str | None:
    """Best-effort extraction of the appointment id linked to a note.

    Mirrors the logic that's worked for notes_completion: try several
    candidate field names and link shapes before giving up.
    """
    for k in ("appointment_id", "individual_appointment_id",
               "booking_id", "event_id"):
        v = note.get(k)
        if v:
            return str(v)
    for k in ("appointment", "individual_appointment", "booking", "event"):
        v = note.get(k)
        if isinstance(v, dict):
            nid = v.get("id")
            if nid:
                return str(nid)
            # Fall through to embedded link
            url = (v.get("links") or {}).get("self") if isinstance(v.get("links"), dict) else None
            if isinstance(url, str):
                tail = url.rstrip("/").rsplit("/", 1)[-1]
                if tail:
                    return tail
    # Last-ditch: links.appointment URL
    links = note.get("links")
    if isinstance(links, dict):
        url = links.get("appointment")
        if isinstance(url, str):
            tail = url.rstrip("/").rsplit("/", 1)[-1]
            if tail:
                return tail
    return None


def injuries_breakdown(client: ClinikoClient, appts: pd.DataFrame,
                        dr: DateRange,
                        appointment_types: pd.DataFrame | None = None,
                        practitioners: pd.DataFrame | None = None,
                        ) -> pd.DataFrame:
    """Produce a long-format DataFrame:
      ``[appointment_id, practitioner_id, business_id, local_date, category]``

    One row per injury category per note. A note that ticks two areas
    yields two rows. A note with no readable injury yields zero rows.

    Use the result with .groupby() to slice by category, clinic, time,
    or practitioner — UI does this on the fly.
    """
    cols = ["appointment_id", "practitioner_id", "business_id",
            "local_date", "category"]
    if appts is None or appts.empty:
        return pd.DataFrame(columns=cols)

    initial_ids = _initial_appointment_ids(appts, appointment_types)
    if not initial_ids:
        return pd.DataFrame(columns=cols)

    notes = _fetch_treatment_notes_with_content(client, dr)
    if not notes:
        return pd.DataFrame(columns=cols)

    patterns = _compile_patterns(_categories())

    # Build appt_id → (practitioner_id, business_id, local_date) lookup
    appt_lookup: dict[str, dict[str, Any]] = {}
    if "id" in appts.columns and "starts_at" in appts.columns:
        for _, row in appts.iterrows():
            aid = str(row.get("id") or "")
            if not aid:
                continue
            starts = row.get("starts_at")
            local_date = None
            if pd.notna(starts):
                try:
                    local_date = pd.Timestamp(starts).date()
                except Exception:
                    local_date = None
            appt_lookup[aid] = {
                "practitioner_id": str(row.get("practitioner_id") or ""),
                "business_id": str(row.get("business_id") or ""),
                "local_date": local_date,
            }

    rows: list[dict[str, Any]] = []
    for note in notes:
        aid = _appt_id_from_note(note)
        if aid is None or aid not in initial_ids:
            continue
        cats = extract_injuries_from_note(note, patterns)
        if not cats:
            continue
        meta = appt_lookup.get(aid, {})
        for cat in cats:
            rows.append({
                "appointment_id": aid,
                "practitioner_id": meta.get("practitioner_id", ""),
                "business_id": meta.get("business_id", ""),
                "local_date": meta.get("local_date"),
                "category": cat,
            })

    return pd.DataFrame(rows, columns=cols)


# -------------------------------------------------------------------
# Aggregations for the UI
# -------------------------------------------------------------------
def total_by_category(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Bar-chart data: one row per category, sorted by count desc."""
    if breakdown is None or breakdown.empty:
        return pd.DataFrame(columns=["category", "count"])
    out = (breakdown.groupby("category", as_index=False)
                    .size()
                    .rename(columns={"size": "count"})
                    .sort_values("count", ascending=False))
    return out.reset_index(drop=True)


def by_category_and_clinic(breakdown: pd.DataFrame,
                             businesses: pd.DataFrame | None = None
                             ) -> pd.DataFrame:
    """Cross-tab: rows = category, columns = clinic name (or biz id),
    values = count. Includes a Total column."""
    if breakdown is None or breakdown.empty:
        return pd.DataFrame()
    name_map: dict[str, str] = {}
    if businesses is not None and not businesses.empty:
        for _, row in businesses.iterrows():
            name_map[str(row.get("id"))] = str(row.get("name") or row.get("id"))
    work = breakdown.copy()
    work["clinic"] = work["business_id"].map(name_map).fillna(work["business_id"])
    pivot = (work.pivot_table(index="category", columns="clinic",
                                values="appointment_id", aggfunc="count",
                                fill_value=0))
    pivot["Total"] = pivot.sum(axis=1)
    return pivot.sort_values("Total", ascending=False)


def by_category_monthly(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Time-series: one row per (month, category) for trend line.
    Months are zero-filled across the observed range so charts don't
    jump over months with no data of a given category."""
    if breakdown is None or breakdown.empty:
        return pd.DataFrame(columns=["month", "category", "count"])
    work = breakdown.copy()
    work["local_date"] = pd.to_datetime(work["local_date"], errors="coerce")
    work = work.dropna(subset=["local_date"])
    if work.empty:
        return pd.DataFrame(columns=["month", "category", "count"])
    work["month"] = work["local_date"].dt.to_period("M").dt.to_timestamp()
    out = (work.groupby(["month", "category"], as_index=False)
                .size()
                .rename(columns={"size": "count"}))
    return out.sort_values(["month", "category"]).reset_index(drop=True)


def by_category_and_practitioner(breakdown: pd.DataFrame,
                                    practitioners: pd.DataFrame | None = None
                                    ) -> pd.DataFrame:
    """Cross-tab: rows = practitioner, columns = category. For the
    optional drill-down."""
    if breakdown is None or breakdown.empty:
        return pd.DataFrame()
    name_map: dict[str, str] = {}
    if practitioners is not None and not practitioners.empty:
        for _, row in practitioners.iterrows():
            pid = str(row.get("id"))
            label = (row.get("label")
                      or row.get("display_name")
                      or row.get("name")
                      or pid)
            name_map[pid] = str(label)
    work = breakdown.copy()
    work["practitioner"] = (work["practitioner_id"]
                              .map(name_map)
                              .fillna(work["practitioner_id"]))
    pivot = work.pivot_table(index="practitioner", columns="category",
                              values="appointment_id", aggfunc="count",
                              fill_value=0)
    pivot["Total"] = pivot.sum(axis=1)
    return pivot.sort_values("Total", ascending=False)
