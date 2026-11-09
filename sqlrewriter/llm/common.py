"""Helpers shared by LLM providers: response cache and output parsing."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from .base import LLMError


class ResponseCache:
    """Content-addressed on-disk cache of raw LLM responses.

    Re-running an experiment with identical prompts costs nothing and yields
    exactly the same candidates, which keeps runs reproducible.
    """

    def __init__(self, directory: Path, enabled: bool = True):
        self.directory = directory
        self.enabled = enabled
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        p = self.directory / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
        return None

    def put(self, key: str, value: dict) -> None:
        if self.enabled:
            (self.directory / f"{key}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False),
                                                        encoding="utf-8")


def parse_candidates(text: str) -> list[dict]:
    """Extract ``{"candidates": [...]}`` from a model response (tolerant of fences / prose)."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    data = None
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                data = None
    if data is None:
        raise LLMError("model output is not valid JSON")
    items = data.get("candidates") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise LLMError("model output has no 'candidates' list")
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("sql"), str) and it["sql"].strip():
            out.append({"strategy": str(it.get("strategy", "")).strip(),
                        "rationale": str(it.get("rationale", "")).strip(),
                        "sql": it["sql"].strip()})
    return out
