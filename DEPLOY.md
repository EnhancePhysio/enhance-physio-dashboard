# Deploy the dashboard to a URL

These steps take about 15 minutes if you've never used GitHub before, 5 minutes if you have. You'll end up with a URL like:

```
https://enhance-physio-dashboard.streamlit.app
```

...that only you (and anyone with the password) can access.

## Why Streamlit Community Cloud

Free, handles HTTPS, pulls updates automatically from GitHub, has a built-in secrets vault so your Cliniko API key never goes in the code. Other options (Render, Fly.io) work too, but this is the lowest-friction path.

## What you need before you start

A GitHub account and a Streamlit Cloud account. Both are free and take 2 minutes each to create. Sign up with the same email if you can — keeps things tidy.

- GitHub sign-up: https://github.com/signup
- Streamlit Cloud sign-up: https://share.streamlit.io (click "Continue with GitHub" — it piggy-backs on the account above)

You'll also need your **Cliniko API key**. If you don't have one handy: Cliniko → Settings (bottom-left) → My Info → API keys → "Add API key". Copy it immediately — Cliniko only shows it once.

## Step 1 — Put the code on GitHub

1. Log into GitHub.
2. Top-right corner → "+" icon → **New repository**.
3. Repository name: `enhance-physio-dashboard`
4. Set to **Private** (important — this is your code).
5. Tick **"Add a README file"** so the repo isn't empty.
6. Click **Create repository**.

Now upload the dashboard folder contents:

7. On the new repo page, click **"Add file"** → **"Upload files"**.
8. In Finder, open the `enhance-physio-dashboard` folder I gave you. Select **everything inside it** (not the folder itself — the files and subfolders inside), drag onto the GitHub upload area.
9. Scroll down. In "Commit changes" leave the default message. Click **Commit changes**.
10. Wait ~30 seconds for files to upload. Refresh the page; you should see `dashboard/`, `config/`, `data/`, `.env.example`, `requirements.txt`, etc.

**Important**: double-check you did NOT accidentally upload a file called `.env` (no `.example` suffix). If you did, delete it from GitHub now. Your API key doesn't belong in the repo.

## Step 2 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **Continue with GitHub** and authorise.
3. On the dashboard that appears, click **Create app** (or "New app").
4. Choose **"Deploy a public app from GitHub"**.
5. Fill in:
   - **Repository**: `your-username/enhance-physio-dashboard`
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
   - **App URL**: pick something memorable (default is fine)
6. Click **Advanced settings** (or "Edit secrets" after deploy — either works):
   - In the **Secrets** text box, paste this, replacing the placeholders:

     ```toml
     CLINIKO_API_KEY = "paste-your-cliniko-key-here"
     CLINIKO_SHARD = "au1"
     CLINIKO_USER_AGENT = "EnhancePhysioDashboard (matt@enhance.physio)"
     DASHBOARD_PASSWORD = "pick-a-strong-password"
     TZ = "Australia/Sydney"
     ```

   - Leave `ANTHROPIC_API_KEY = ""` unless you want vision extraction for punctuality photos.
7. Click **Deploy**.

Streamlit Cloud will now:
- Install Python 3.11
- Install the packages in `requirements.txt` (takes 1-2 minutes)
- Start the app

You'll see a log streaming. When it's ready, the page auto-redirects to your app URL.

## Step 3 — First test

1. Visit your URL (e.g. `https://enhance-physio-dashboard.streamlit.app`).
2. Enter the password you set in `DASHBOARD_PASSWORD`.
3. You should see the dashboard load. Left sidebar will populate with your clinics + practitioners within a few seconds as it calls Cliniko.
4. Pick "Last 30 days". The matrix plot should populate.

## Troubleshooting

**"CLINIKO_API_KEY is not set"** — the secret didn't save. Open app → bottom-right `⋮` menu → **Settings → Secrets** and confirm the values.

**App won't start / keeps erroring** — click **Manage app** (bottom-right of the running app) → view logs. Most likely cause is a typo in `secrets.toml`. TOML is fussy: string values must be in double quotes.

**Slow on first load** — normal. Streamlit Cloud sleeps apps with no traffic; the first request wakes it back up (~30 seconds).

**"This app has gone over its free resource allowance"** — the free tier caps at ~1 GB RAM. With <30 practitioners you should never hit this. If you do, upgrade the app's tier in Streamlit Cloud settings (~US$20/mo) or move to Render/Fly.

## Updating the dashboard

When I (or you) change the code, push the update to GitHub and Streamlit Cloud auto-redeploys within 1-2 minutes. No manual step needed.

If you want me to push fixes directly, I'd need you to invite me to the GitHub repo as a collaborator (Settings → Collaborators → Add people → my GitHub handle). Otherwise just download the updated files from Claude, drag them into GitHub's web UI like Step 1, commit.

## Sharing with your team

Phase 1: the password gate is the access control. Share the URL and the password with anyone who needs access. You can rotate the password any time by editing the secret in Streamlit Cloud.

Phase 2 (later): we add real per-user login so managers get their own account scoped to their clinic. That's a code change, not a hosting change.

## Turning it off

To pause/delete: Streamlit Cloud dashboard → your app → `⋮` → **Delete**. Your GitHub repo stays; only the running URL is torn down.

## Cost

Streamlit Community Cloud: **free** for this volume.

Cliniko API: **included in your existing subscription** (no extra charge).

Anthropic API (only if you turn on vision extraction): ~US$0.50/month for weekly punctuality sheets across 3 clinics.

Total: basically free unless you use vision extraction.
