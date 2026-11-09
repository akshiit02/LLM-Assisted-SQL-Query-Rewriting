"""Local open-weight model rewriter via an Ollama server (no API key needed).

Uses Ollama's native ``/api/chat`` endpoint with a JSON-schema ``format`` so
the reply is structured.  Start a server with e.g. ``ollama run qwen2.5-coder``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..config import LLMSettings, PROJECT_ROOT
from ..models import Candidate
from .base import LLMError, RepairItem, RewriteRequest, Rewriter
from .common import ResponseCache, parse_candidates
from .prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT, build_repair_prompt, build_rewrite_prompt


class OllamaRewriter(Rewriter):
    name = "ollama"
    source = "llm"

    def __init__(self, settings: LLMSettings):
        super().__init__()
        self.settings = settings
        self.model = settings.model
        self.url = settings.ollama_url.rstrip("/") + "/api/chat"
        cache_dir = Path(settings.cache_dir)
        self.cache = ResponseCache(cache_dir if cache_dir.is_absolute() else PROJECT_ROOT / cache_dir,
                                   settings.use_cache)

    @property
    def supports_repair(self) -> bool:
        return True

    def _call(self, user_prompt: str) -> str:
        key = self.cache.key(self.name, self.model, SYSTEM_PROMPT, user_prompt)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached["text"]
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.settings.request_timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMError(f"cannot reach Ollama at {self.url}: {exc}") from exc
        finally:
            self.stats.calls += 1
            self.stats.seconds += time.perf_counter() - t0
        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"Ollama returned no content: {data.get('error', data)}")
        self.stats.input_tokens += int(data.get("prompt_eval_count", 0) or 0)
        self.stats.output_tokens += int(data.get("eval_count", 0) or 0)
        self.cache.put(key, {"model": self.model, "text": text})
        return text

    def generate(self, request: RewriteRequest) -> list[Candidate]:
        items = parse_candidates(self._call(build_rewrite_prompt(request)))
        return [
            Candidate(id=f"L{i}", query_id=request.query.id, sql=it["sql"], source="llm",
                      strategy=it["strategy"], rationale=it["rationale"], generator=f"ollama:{self.model}")
            for i, it in enumerate(items[: max(1, request.num_candidates)], start=1)
        ]

    def repair(self, request: RewriteRequest, items: list[RepairItem], round_no: int) -> list[Candidate]:
        if not items:
            return []
        fixed = parse_candidates(self._call(build_repair_prompt(request, items)))
        return [
            Candidate(id=f"{item.candidate.id}R{round_no}", query_id=request.query.id, sql=it["sql"],
                      source="repair", strategy=it["strategy"] or item.candidate.strategy,
                      rationale=it["rationale"], parent_id=item.candidate.id, generator=f"ollama:{self.model}")
            for item, it in zip(items, fixed)
        ]
