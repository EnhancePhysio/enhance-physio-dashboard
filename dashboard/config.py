"""Configuration loading: .env + settings.yml + exempt list."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"


_env_loaded = False
_settings_cache: dict[str, Any] | None = None
_rap_exempt_cache: list[dict[str, Any]] | None = None


def _ensure_env() -> None:
    """Load secrets from either Streamlit Cloud (st.secrets) or local .env file.

    Streamlit Cloud puts them in st.secrets; local runs use .env.
    If both are missing for a key, the downstream call raises a clear error.
    """
    global _env_loaded
    if _env_loaded:
        return
    # Try Streamlit secrets first — exists on Streamlit Cloud and optionally local
    try:
        import streamlit as st  # type: ignore
        try:
            for k in ("CLINIKO_API_KEY", "CLINIKO_SHARD", "CLINIKO_USER_AGENT",
                      "ANTHROPIC_API_KEY", "TZ", "DASHBOARD_PASSWORD"):
                if k in st.secrets and not os.environ.get(k):
                    os.environ[k] = str(st.secrets[k])
        except Exception:
            pass  # no secrets.toml available
    except ImportError:
        pass  # not running under Streamlit
    # Then load .env for local runs
    load_dotenv(ROOT / ".env")
    _env_loaded = True


def dashboard_password() -> str | None:
    """Optional password gate. Returns None to disable."""
    _ensure_env()
    pw = os.environ.get("DASHBOARD_PASSWORD", "").strip()
    return pw or None


def load_settings() -> dict[str, Any]:
    """Load config/settings.yml (cached)."""
    global _settings_cache
    if _settings_cache is None:
        with open(CONFIG_DIR / "settings.yml", "r", encoding="utf-8") as f:
            _settings_cache = yaml.safe_load(f)
    return _settings_cache


def load_rap_exempt() -> list[dict[str, Any]]:
    """Load RAP-exempt practitioners list (cached)."""
    global _rap_exempt_cache
    if _rap_exempt_cache is None:
        path = CONFIG_DIR / "rap_exempt_practitioners.yml"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _rap_exempt_cache = data.get("exempt", []) or []
        else:
            _rap_exempt_cache = []
    return _rap_exempt_cache


def cliniko_api_key() -> str:
    _ensure_env()
    key = os.environ.get("CLINIKO_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "CLINIKO_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    return key


def cliniko_shard() -> str:
    _ensure_env()
    shard = os.environ.get("CLINIKO_SHARD", "").strip()
    if shard:
        return shard
    # Infer from key suffix: "<token>-<shard>"
    key = cliniko_api_key()
    if "-" in key:
        return key.rsplit("-", 1)[-1]
    return "au1"


def cliniko_user_agent() -> str:
    _ensure_env()
    return os.environ.get(
        "CLINIKO_USER_AGENT",
        "EnhancePhysioDashboard (contact@example.com)",
    ).strip().strip("'\"")


def anthropic_api_key() -> str | None:
    _ensure_env()
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key or None


def timezone_name() -> str:
    _ensure_env()
    tz = os.environ.get("TZ", "").strip()
    if tz:
        return tz
    return load_settings().get("timezone", "Australia/Sydney")
