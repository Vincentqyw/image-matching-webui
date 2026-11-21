"""Anonymous per-matcher usage counting for the hosted WebUI.

Disabled unless MATCH_TELEMETRY_URL is set. See telemetry-worker/README.md.
"""

import json
import os
import threading
import urllib.request

from ..hloc import logger

_URL = os.getenv("MATCH_TELEMETRY_URL")
_TOKEN = os.getenv("MATCH_TELEMETRY_TOKEN")
assert (_URL is None) == (
    _TOKEN is None
), "MATCH_TELEMETRY_URL and MATCH_TELEMETRY_TOKEN must be set together"


def _post(url: str, token: str, key: str) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps({"key": key}).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        urllib.request.urlopen(request, timeout=10).close()
    except Exception:  # telemetry must never break matching
        logger.warning("Telemetry post failed", exc_info=True)


def record(key: str) -> None:
    """Count one completed match run for the matcher_zoo entry ``key``."""
    if _URL is None:
        return
    assert _TOKEN is not None  # guaranteed by the module-level assert
    threading.Thread(target=_post, args=(_URL, _TOKEN, key), daemon=True).start()
