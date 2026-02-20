import json
import re
from typing import Any, Dict, Optional


def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model response")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response")
    return json.loads(m.group(0))


def safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return int(s)
    except Exception:
        return None
