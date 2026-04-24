# Enhance Physio — Performance Reporting Dashboard

Local-run Python / Streamlit dashboard that pulls real-time data from Cliniko
and presents a 12-metric practitioner performance view with a Clinical ×
Non-clinical scoring matrix.

See `Enhance Physio Dashboard - Design Doc.docx` for full design context.
This README covers how to run it.

## Prerequisites

- Python 3.11+
- A Cliniko API key (Cliniko → Settings → My Info → API keys). Any key shard
  works; the dashboard infers it from the key suffix (`...-au1`, etc).
- Optional: Anthropic API key (for vision extraction of punctuality sheets
  and NPS screenshots). Not required if you enter that data via CSV.

## First-run setup

```bash
# 1. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate      # (Windows: .venv\Scripts\activate)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
cp .env.example .env
# Now open .env and paste in your CLINIKO_API_KEY. Optionally set
# ANTHROPIC_API_KEY if you want vision extraction.

# 4. Run the dashboard
streamlit run dashboard/app.py
```

The dashboard opens at http://localhost:8501.

## What you'll see

- **Overview tab** — KPI cards, the Clinical × Non-clinical matrix plot,
  a zone summary table, and a per-practitioner detail drawer.
- **Manual data tab** — upload punctuality photos (vision extraction) or NPS
  CSV exports; save to the local `data/` folder.

## Filters

- Date range: last 7/30/90 days, last month, last quarter, YTD, current AU
  financial year (July start), or custom.
- Clinic (Cliniko "businesses") — multi-select.
- Practitioner — multi-select.

All dates interpreted in Australia/Sydney (overridable in `.env` via `TZ`).

## Running the audit

The audit calls Cliniko repeatedly (roughly 5 calls per patient), so it runs
on demand rather than automatically. Inside the Overview tab, expand **Audit
(run on demand...)** and click **Run audit for this filter**. The results
get cached for the current filter selection until you change filters or
restart Streamlit.

## Manual data files

```
data/punctuality/         <- CSVs, one per clinic × week
data/nps/                 <- CSVs with NPS weekly/monthly exports
data/uploads/             <- Archived photo/screenshot uploads
```

CSV schemas:

```
# punctuality
week_starting,clinic,practitioner,day,bucket_0_5,bucket_6_10,bucket_11_14,bucket_15_plus,status
2026-04-06,Albury,Alice,Monday,12,3,1,0,
...
# status can be: off | leave | sick | na | (blank for a normal day)

# nps
week_starting,clinic,practitioner,responses,promoters,passives,detractors,nps
2026-04-01,Albury,Alice,15,12,2,1,73
```

## RAP-exempt practitioners

Edit `config/rap_exempt_practitioners.yml` to add/remove new grads. Changes
take effect on the next dashboard run (or click Streamlit's "Rerun" in the
top-right menu).

## Configuration

`config/settings.yml` has all the tunable thresholds and keyword lists
(utilisation keywords, PPVA patterns, audit patterns, punctuality buckets,
matrix zone thresholds). Edit, save, rerun — no code changes needed.

## Troubleshooting

**"CLINIKO_API_KEY is not set"** — you haven't created `.env` yet. Copy
`.env.example` to `.env` and paste your key.

**401 on first request** — check for trailing whitespace in the key.
Verify the shard suffix matches your Cliniko account region.

**Dashboard feels slow on big date ranges** — the first call warms
Cliniko's pagination; Streamlit caches results for 5 minutes. If you need
to force a refresh, use Streamlit's "Clear cache" from the top-right menu.

**Audit seems to hang** — check the progress bar. The audit issues ~5 calls
per patient at 200/min; 150 patients takes roughly 4 minutes.

## File structure

```
enhance-physio-dashboard/
├── dashboard/
│   ├── app.py              # Streamlit entry
│   ├── cliniko.py          # API client (auth, rate limit, pagination)
│   ├── metrics.py          # Metric calculators (4.1 – 4.10)
│   ├── audit.py            # Audit engine (4.11)
│   ├── scoring.py          # Rubric → bands → matrix
│   ├── manual.py           # CSV + vision extraction
│   ├── ui.py               # Plotly matrix + practitioner drawer
│   ├── date_ranges.py      # Presets (AU FY, YTD, etc.)
│   ├── reference_data.py   # Practitioners / businesses / types
│   └── config.py           # .env + YAML loaders
├── config/
│   ├── settings.yml                  # Tunable thresholds & keywords
│   └── rap_exempt_practitioners.yml  # New-grad exempt list
├── data/
│   ├── punctuality/        # Manual-entry CSVs
│   ├── nps/                # Cliniqapps exports
│   └── uploads/            # Archived uploads
├── .env.example
├── requirements.txt
└── README.md
```

## Security

- `.env` is git-ignored; never commit it.
- The dashboard runs locally. Data only leaves your machine when calling
  Cliniko (required) and optionally Anthropic (only if you use vision
  extraction).
- No Cliniqapps credentials are accepted or stored — see design doc
  Section 10.2 for the reasoning.

## Next steps (after prototype validation)

Phase 2 — hosted with read-only manager access. Phase 3 — optional
white-label multi-tenant SaaS. Cost estimates and roadmap in the design
doc Section 12.
