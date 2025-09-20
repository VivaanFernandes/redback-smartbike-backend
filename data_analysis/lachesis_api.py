"""
Lachesis API helper
Centralised HTTP client used by modelling/exploratory/visualize modules.

Usage:
    from data_analysis.lachesis_api import analyze

    result = analyze({
        "metric": "heart_rate_prediction",
        "features": [[5.2], [6.3], [7.1]],
        "target": [120, 135, 150]
    })
    if result:
        print(result["rmse"], result.get("predictions"))
"""

from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, Optional

import requests

# --- Configuration ---
# These can be overridden by environment variables on the server.
LACHESIS_URL = os.getenv("LACHESIS_URL", "https://lachesis.example.com/api/v1/analyze")
LACHESIS_TOKEN = os.getenv("LACHESIS_TOKEN", "")  # if your API uses bearer auth
TIMEOUT_SEC = float(os.getenv("LACHESIS_TIMEOUT", "12"))
RETRIES = int(os.getenv("LACHESIS_RETRIES", "2"))


def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if LACHESIS_TOKEN:
        h["Authorization"] = f"Bearer {LACHESIS_TOKEN}"
    return h


def analyze(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Send a JSON payload to Lachesis and return parsed JSON on success.
    Returns None on any failure (callers should gracefully fall back to local analysis).
    Retries a few times on transient errors.
    """
    data = json.dumps(payload)
    for attempt in range(RETRIES + 1):
        try:
            resp = requests.post(LACHESIS_URL, data=data, headers=_headers(), timeout=TIMEOUT_SEC)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            # Log and retry backoff
            print(f"[lachesis] request failed (attempt {attempt+1}/{RETRIES+1}): {e}")
            if attempt < RETRIES:
                time.sleep(0.7 * (attempt + 1))
            else:
                return None
