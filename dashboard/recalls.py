"""v26.5 — Cliniko Recalls ingestion.

Cliniko doesn't expose patient recalls via their public API (the
``/patient_recalls`` endpoint 404s on the Enhance Physio plan — per Cliniko
support it's a UI-only feature on this tier). So we ingest them via the
browser-bookmarklet workflow:

    1. Matt opens the Recalls report in Cliniko.
    2. Runs the one-click bookmarklet in his browser.
    3. Script auto-clicks "Load more" until the list is exhausted, then
       downloads ``cliniko_recalls.csv``.
    4. He uploads that CSV on the Manual tab.
    5. We parse the CSV, match each row to a Cliniko ``patient_id`` (by
       mobile number, then name), and persist the resulting id-set.
    6. Audit Check 4 consults that id-set: a patient passes the check if
       they either have an upcoming appointment OR appear in the recalls
       set. Falls through to a fail otherwise.

The CSV shape the bookmarklet produces (v1):
    col0 — Recall on        e.g. "1 Apr 2026"
    col1 — Type             e.g. "Discharge - 12 month follow up"
    col2 — Patient          e.g. "Cameron Smith"
    col3 — Last practitioner e.g. "Jackson Casey"
    col4 — Mobile #         e.g. "0427 383 319"
    col5 — Recalled?        e.g. "Recalled 1 Apr 2026 - ES" (or blank)

We only need cols 2 + 4 to match to a patient_id. The rest is retained
for human-readable audit trails.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from dashboard.config import DATA_DIR


RECALLS_DIR = DATA_DIR / "recalls"
RECALLS_DIR.mkdir(parents=True, exist_ok=True)


RECALL_COLUMNS = [
    "recall_on",
    "recall_type",
    "patient_name",
    "last_practitioner",
    "mobile",
    "recalled_note",
]


# -------------------------------------------------------------------
# Normalisation helpers
# -------------------------------------------------------------------
_MOBILE_DIGIT_RE = re.compile(r"\D+")
_MULTISPACE_RE = re.compile(r"\s+")
_PAREN_RE = re.compile(r"\(.*?\)")


def _normalise_mobile(raw: Any) -> str:
    """Return a digits-only form of an AU mobile, or '' if we can't make
    one. Strips country code 61, removes leading 0 ambiguity by always
    coercing to 10-digit 04xxxxxxxx form.
    """
    if raw is None:
        return ""
    digits = _MOBILE_DIGIT_RE.sub("", str(raw))
    if not digits:
        return ""
    # International prefix 61 → strip and re-add leading 0
    if digits.startswith("61") and len(digits) >= 11:
        digits = "0" + digits[2:]
    # Anything that starts with 4 and is 9 digits long is missing the 0
    if len(digits) == 9 and digits.startswith("4"):
        digits = "0" + digits
    return digits


def _normalise_name(raw: Any) -> str:
    """Lower-case, strip parenthetical aliases ("(Thomas)", "(Dad)"),
    remove punctuation, collapse whitespace. Used for matching and for
    dedup keys — NOT for display.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    s = _PAREN_RE.sub("", s)
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = _MULTISPACE_RE.sub(" ", s)
    return s.strip().lower()


def _split_first_last(name: str) -> tuple[str, str]:
    """Best-effort split of a full-name into (first, last). Handles
    hyphenated / multi-word surnames by treating the LAST whitespace-
    separated token as surname. "Shanae Bohr-Howell" → ("shanae", "bohr-howell")."""
    norm = _normalise_name(name)
    if not norm:
        return ("", "")
    parts = norm.split()
    if len(parts) == 1:
        return (parts[0], "")
    return (parts[0], parts[-1])


# -------------------------------------------------------------------
# CSV parsing
# -------------------------------------------------------------------
def parse_recalls_csv(file_or_path: Any) -> pd.DataFrame:
    """Load the bookmarklet CSV into a normalised DataFrame.

    Tolerates both headerless CSVs (the raw bookmarklet output) and CSVs
    where the user has added a header row. Returns a DataFrame with the
    canonical ``RECALL_COLUMNS`` plus derived ``mobile_norm`` and
    ``name_norm`` columns ready for matching.
    """
    # pandas read_csv with no header — we detect a header row ourselves
    df = pd.read_csv(file_or_path, header=None, dtype=str).fillna("")
    if df.empty:
        return pd.DataFrame(columns=RECALL_COLUMNS + ["mobile_norm", "name_norm"])

    # Detect a user-added header row (first cell says "Recall on" or similar)
    first_cell = str(df.iat[0, 0]).strip().lower()
    if first_cell in {"recall on", "recall_on", "date", "recall date"}:
        df = df.iloc[1:].reset_index(drop=True)

    # Pad to 6 cols (some rows might be short)
    while df.shape[1] < 6:
        df[df.shape[1]] = ""
    df = df.iloc[:, :6]
    df.columns = RECALL_COLUMNS

    # Strip whitespace across string cells
    for c in RECALL_COLUMNS:
        df[c] = df[c].astype(str).str.strip()

    df["mobile_norm"] = df["mobile"].apply(_normalise_mobile)
    df["name_norm"] = df["patient_name"].apply(_normalise_name)

    # Drop rows that have neither a name nor a mobile — can't match those
    df = df[(df["mobile_norm"].str.len() > 0) | (df["name_norm"].str.len() > 0)]
    return df.reset_index(drop=True)


# -------------------------------------------------------------------
# Patient matching
# -------------------------------------------------------------------
def match_recalls_to_patient_ids(
    recalls_df: pd.DataFrame,
    patients: pd.DataFrame,
) -> tuple[set[str], pd.DataFrame]:
    """Resolve each recall row to a Cliniko ``patient_id``.

    Matching strategy (in priority order):
      1. **Mobile number** — exact match on normalised digits. Primary
         match because two patients named "Sarah Hamilton" exist in the
         same clinic and the mobile disambiguates them. Very high
         precision.
      2. **First + last name** — case-insensitive full name match, but
         ONLY when unique across the patient list. Used when the mobile
         in the recall row doesn't match any patient (e.g. Matt's staff
         updated the mobile after the recall was set) or when the patient
         has no mobile on file.

    Note: we deliberately do NOT fall back to last-name-only matching —
    when one patient exists with surname "Smith" and Matt has a recall
    for a different "Jennifer Smith" not yet in our patient list, that
    fallback silently resolves Jennifer onto the wrong patient_id and
    the audit passes Check 4 for a patient it shouldn't have. Better to
    leave those rows unmatched and surface them to the UI.

    Returns
    -------
    (ids, diagnostics)
        ids          : set of patient_id strings for audit Check 4.
        diagnostics  : DataFrame echoing each input row + the resolved
                       patient_id (or blank) + a ``match_method`` column.
                       The UI uses this to show Matt which rows didn't
                       resolve so he can fix them in Cliniko.
    """
    if recalls_df.empty or patients is None or patients.empty:
        empty = recalls_df.copy() if not recalls_df.empty else pd.DataFrame(columns=RECALL_COLUMNS)
        empty["patient_id"] = ""
        empty["match_method"] = "no_data"
        return set(), empty

    p = patients.copy()
    # Cliniko mobile can be in several fields across API versions
    mobile_cols = [c for c in ("mobile_phone_number", "mobile_phone", "phone_mobile") if c in p.columns]
    p["_mobile_norm"] = ""
    for c in mobile_cols:
        fallback = p[c].astype(str).apply(_normalise_mobile)
        p["_mobile_norm"] = p["_mobile_norm"].where(p["_mobile_norm"].str.len() > 0, fallback)

    p["_first"] = p.get("first_name", pd.Series([""] * len(p))).astype(str).apply(_normalise_name)
    p["_last"] = p.get("last_name", pd.Series([""] * len(p))).astype(str).apply(_normalise_name)
    p["_id"] = p["id"].astype(str)

    # Index by mobile (may have dups if two patients share a mobile — keep all)
    mobile_index: dict[str, list[str]] = {}
    for _, r in p.iterrows():
        m = r["_mobile_norm"]
        if m:
            mobile_index.setdefault(m, []).append(r["_id"])

    # Index by (first, last) full-name key
    name_index: dict[tuple[str, str], list[str]] = {}
    for _, r in p.iterrows():
        f, l = r["_first"], r["_last"]
        if f and l:
            name_index.setdefault((f, l), []).append(r["_id"])

    ids: set[str] = set()
    diag_rows = []
    for _, r in recalls_df.iterrows():
        pid = ""
        method = "unmatched"
        mob = r.get("mobile_norm", "")
        first, last = _split_first_last(r.get("patient_name", ""))

        # 1. Mobile
        if mob and mob in mobile_index:
            candidates = mobile_index[mob]
            if len(candidates) == 1:
                pid, method = candidates[0], "mobile"
            else:
                # Multiple patients share this mobile — use name to disambiguate
                narrowed = [c for c in candidates
                             if (first, last) in name_index
                             and c in name_index.get((first, last), [])]
                if len(narrowed) == 1:
                    pid, method = narrowed[0], "mobile+name"

        # 2. Full name (unique only)
        if not pid and first and last:
            key = (first, last)
            cands = name_index.get(key, [])
            if len(cands) == 1:
                pid, method = cands[0], "name"

        if pid:
            ids.add(pid)
        diag_rows.append({
            "patient_name": r.get("patient_name", ""),
            "mobile": r.get("mobile", ""),
            "recall_on": r.get("recall_on", ""),
            "recall_type": r.get("recall_type", ""),
            "patient_id": pid,
            "match_method": method,
        })

    return ids, pd.DataFrame(diag_rows)


# -------------------------------------------------------------------
# Persistence
# -------------------------------------------------------------------
_last_github_sync: dict[str, tuple[bool, str]] = {}


def save_recalls_csv(df: pd.DataFrame, label: str = "recalls") -> Path:
    """Persist the normalised recalls DataFrame to data/recalls/<label>.csv
    and (if configured) push it to the GitHub repo for cross-reboot
    survival. Same pattern as save_punctuality_csv.
    """
    safe_label = re.sub(r"[^a-zA-Z0-9_-]", "_", label)
    path = RECALLS_DIR / f"{safe_label}.csv"
    # Only save the canonical columns — keep mobile_norm / name_norm off disk
    # since they're cheap to recompute on load and just bloat the checked-in
    # file with duplicate data.
    to_write = df[[c for c in RECALL_COLUMNS if c in df.columns]].copy()
    to_write.to_csv(path, index=False)
    try:
        from dashboard.github_persistence import save_file_to_github
        ok, msg = save_file_to_github(
            path, commit_msg=f"[data] recalls {label}",
        )
        _last_github_sync["recalls"] = (ok, msg)
    except Exception as e:
        _last_github_sync["recalls"] = (False, f"sync skipped: {e}")
    return path


def last_github_sync(kind: str = "recalls") -> tuple[bool, str] | None:
    return _last_github_sync.get(kind)


def load_recalls() -> pd.DataFrame:
    """Concatenate every CSV in data/recalls/. Returns normalised form
    (canonical cols + mobile_norm + name_norm)."""
    files = sorted(RECALLS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=RECALL_COLUMNS + ["mobile_norm", "name_norm"])
    frames = []
    for f in files:
        try:
            frames.append(parse_recalls_csv(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=RECALL_COLUMNS + ["mobile_norm", "name_norm"])
    df = pd.concat(frames, ignore_index=True)
    # De-dupe on (name_norm, mobile_norm, recall_on) — if Matt uploads twice
    df = df.drop_duplicates(subset=["name_norm", "mobile_norm", "recall_on"])
    return df.reset_index(drop=True)


def build_recall_patient_id_set(
    patients: pd.DataFrame,
    session_df: pd.DataFrame | None = None,
) -> tuple[set[str], pd.DataFrame]:
    """Resolve the effective recall patient_id set for the current audit,
    when a bulk patients DataFrame is available (fast path used by some
    callers). Most audit callsites use ``build_recall_lookup`` instead —
    no bulk patient fetch required.

    Prefers ``session_df`` (the just-uploaded frame) over anything on disk
    so a re-upload takes effect immediately. Falls back to ``load_recalls()``.
    """
    src = _coerce_source(session_df)
    return match_recalls_to_patient_ids(src, patients)


def _coerce_source(session_df: pd.DataFrame | None) -> pd.DataFrame:
    if session_df is not None and not session_df.empty:
        src = session_df.copy()
        if "mobile_norm" not in src.columns:
            src["mobile_norm"] = src.get("mobile", pd.Series([""] * len(src))).apply(_normalise_mobile)
        if "name_norm" not in src.columns:
            src["name_norm"] = src.get("patient_name", pd.Series([""] * len(src))).apply(_normalise_name)
        return src
    return load_recalls()


def build_recall_lookup(
    session_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a lookup object audit_patient can check per-patient without
    needing a bulk patients DataFrame.

    Returns a dict with:
        ``mobiles``  : set[str]  — normalised 10-digit AU mobiles
        ``names``    : set[tuple[str, str]] — (first_norm, last_norm) pairs
        ``source``   : str — "session" or "disk" or "empty"
        ``row_count``: int — rows seen in the recalls CSV
    Audit code checks ``patient.mobile in mobiles OR (first, last) in names``.
    Pre-building these sets once per audit run is O(n) on the CSV and the
    per-patient check is O(1) — much cheaper than calling the matcher for
    every audited patient.
    """
    src = _coerce_source(session_df)
    if src.empty:
        return {"mobiles": set(), "names": set(), "source": "empty", "row_count": 0}
    mobiles: set[str] = set(m for m in src["mobile_norm"].tolist() if m)
    names: set[tuple[str, str]] = set()
    for nm in src["name_norm"].tolist():
        if not nm:
            continue
        first, last = _split_first_last(nm)
        if first and last:
            names.add((first, last))
    source = "session" if (session_df is not None and not session_df.empty) else "disk"
    return {
        "mobiles": mobiles,
        "names": names,
        "source": source,
        "row_count": int(len(src)),
    }


def patient_is_in_recalls(patient: dict[str, Any],
                           lookup: dict[str, Any] | None) -> bool:
    """Return True iff this Cliniko patient matches a row in the recall
    CSV, based on mobile number OR (first, last) name.

    Pure function — no I/O. Called from audit.audit_patient once per
    patient under review.
    """
    if not lookup:
        return False
    mobiles = lookup.get("mobiles") or set()
    names = lookup.get("names") or set()
    # Mobile — try the usual Cliniko fields
    for field in ("mobile_phone_number", "mobile_phone", "phone_mobile"):
        m = _normalise_mobile(patient.get(field))
        if m and m in mobiles:
            return True
    # Name — first + last
    first = _normalise_name(patient.get("first_name", ""))
    last = _normalise_name(patient.get("last_name", ""))
    if first and last and (first, last) in names:
        return True
    return False
