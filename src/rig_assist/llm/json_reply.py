"""Reading a JSON object out of an LLM reply (models often wrap it in a ```json fence)."""

import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """The reply as a dict; ValueError if it isn't a JSON object (a fence around it is fine)."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    try:
        data = json.loads(fenced.group(1) if fenced else stripped)
    except json.JSONDecodeError as error:
        raise ValueError(f"not JSON ({error.msg})") from error
    if not isinstance(data, dict):
        raise ValueError("not a JSON object")
    return data
