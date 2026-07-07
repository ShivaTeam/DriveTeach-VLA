"""Safe JSON loading with repair fallback for VLM outputs."""

import json
from typing import Optional


def safe_json_loads(text: str, repair: bool = True) -> Optional[dict]:
    """Try json.loads, fall back to json_repair if enabled."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if repair:
            try:
                import json_repair as jr
                return jr.loads(text)
            except Exception:
                return None
        return None


def extract_clean_json(text: str, repair: bool = True) -> Optional[dict]:
    """Extract and parse JSON from VLM raw output.

    Tries direct json.loads first, falls back to extracting {..} substring.
    """
    if not text:
        return None

    obj = safe_json_loads(text.strip(), repair=repair)
    if obj is not None:
        return obj

    # Fallback: find { to } and try again
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        return safe_json_loads(text[start:end + 1], repair=repair)
    return None
