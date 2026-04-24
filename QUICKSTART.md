# Quick start (1 page)

Install Python 3.11, then in a terminal:

```bash
cd enhance-physio-dashboard
cp .env.example .env     # paste your Cliniko API key into .env
./run.sh                 # Mac / Linux
# or on Windows:
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt && streamlit run dashboard/app.py
```

Dashboard opens at http://localhost:8501.

## First thing to check

1. Dropdowns in the sidebar populate (clinics, practitioners). If not —
   open the browser console or Streamlit log; most likely cause is a bad
   API key or a wrong shard.
2. Pick "Last 30 days" → the matrix plot should show markers.
3. To run the audit, expand the "Audit" panel in the Overview tab and
   click "Run audit for this filter". Takes ~1 min per 15 patients.

## Tuning

- Utilisation keywords, PPVA patterns, audit patterns, punctuality
  buckets, matrix zones → `config/settings.yml`
- RAP-exempt practitioners → `config/rap_exempt_practitioners.yml`
- Change the dashboard's timezone → `.env` `TZ=Australia/Sydney`

## Where data lives

- Cached Cliniko responses → in-memory only (Streamlit cache, 5 min TTL)
- Manual punctuality entries → `data/punctuality/*.csv`
- Manual NPS entries → `data/nps/*.csv`
- Original uploaded photos/screenshots → `data/uploads/`

Nothing syncs back to Cliniko. The dashboard is read-only against the API.
