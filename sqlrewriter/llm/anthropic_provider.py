"""Claude (Anthropic API) query rewriter.

* Structured outputs (``output_config.format`` JSON schema) guarantee a
  parseable ``{"candidates": [...]}`` response.
* Adaptive thinking is on by default for current Claude models; ``effort``
  controls how hard the model thinks.
* Server-side refusal fallback (``fallbacks: "default"``) is enabled so a
  policy decline is retried on Anthropic's recommended fallback model.
* Streaming is used because rewrites of large queries plus reasoning can be long.

Credentials are resolved by the SDK: ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or
an ``ant auth login`` profile.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..config import LLMSettings, PROJECT_ROOT
from ..models import Candidate
from .base import LLMError, RepairItem, RewriteRequest, Rewriter
from .common import ResponseCache, parse_candidates
from .prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT, build_repair_prompt, build_rewrite_prompt

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicRewriter(Rewriter):
    name = "anthropic"
    source = "llm"

    def __init__(self, settings: LLMSettings, fallbacks: bool = True):
        super().__init__()
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The 'anthropic' package is required: pip install anthropic") from exc
        self._anthropic = anthropic
        self.settings = settings
        self.model = settings.model
        self.fallbacks = fallbacks
        self.client = anthropic.Anthropic(timeout=settings.request_timeout_s, max_retries=3)
        cache_dir = Path(settings.cache_dir)
        self.cache = ResponseCache(cache_dir if cache_dir.is_absolute() else PROJECT_ROOT / cache_dir,
                                   settings.use_cache)

    @property
    def supports_repair(self) -> bool:
        return True

    # ------------------------------------------------------------------ API call
    def _call(self, user_prompt: str) -> str:
        key = self.cache.key(self.name, self.model, self.settings.effort, SYSTEM_PROMPT, user_prompt)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached["text"]

        a = self._anthropic
        kwargs = dict(
            model=self.model,
            max_tokens=self.settings.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={
                "effort": self.settings.effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
        )
        if self.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["extra_body"] = {"fallbacks": "default"}

        t0 = time.perf_counter()
        try:
            with self.client.beta.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()
        except TypeError as exc:
            if "authentication" not in str(exc).lower():
                raise
            raise LLMError("no Anthropic credentials found – set ANTHROPIC_API_KEY (see .env.example) "
                           "or run `ant auth login`") from exc
        except a.NotFoundError as exc:
            raise LLMError(f"model {self.model!r} not found: {exc}") from exc
        except a.AuthenticationError as exc:
            raise LLMError("Anthropic authentication failed – set ANTHROPIC_API_KEY or run `ant auth login`") from exc
        except a.RateLimitError as exc:
            raise LLMError(f"rate limited by the Anthropic API: {exc}") from exc
        except a.BadRequestError as exc:
            raise LLMError(f"request rejected by the Anthropic API: {exc}") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc}") from exc
        except a.APIConnectionError as exc:
            raise LLMError(f"cannot reach the Anthropic API: {exc}") from exc
        finally:
            self.stats.calls += 1
            self.stats.seconds += time.perf_counter() - t0

        usage = getattr(message, "usage", None)
        if usage is not None:
            self.stats.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.stats.output_tokens += getattr(usage, "output_tokens", 0) or 0

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMError(f"model declined the request (refusal: {getattr(details, 'category', None)})")
        if message.stop_reason == "max_tokens":
            raise LLMError("model output was truncated (max_tokens reached) – increase llm.max_tokens")

        text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise LLMError("model returned no text content")
        self.cache.put(key, {"model": getattr(message, "model", self.model), "text": text})
        return text

    # ------------------------------------------------------------------ Rewriter API
    def generate(self, request: RewriteRequest) -> list[Candidate]:
        items = parse_candidates(self._call(build_rewrite_prompt(request)))
        return [
            Candidate(id=f"L{i}", query_id=request.query.id, sql=it["sql"], source="llm",
                      strategy=it["strategy"], rationale=it["rationale"], generator=f"anthropic:{self.model}")
            for i, it in enumerate(items[: max(1, request.num_candidates)], start=1)
        ]

    def repair(self, request: RewriteRequest, items: list[RepairItem], round_no: int) -> list[Candidate]:
        if not items:
            return []
        fixed = parse_candidates(self._call(build_repair_prompt(request, items)))
        out = []
        for item, it in zip(items, fixed):
            out.append(Candidate(id=f"{item.candidate.id}R{round_no}", query_id=request.query.id, sql=it["sql"],
                                 source="repair", strategy=it["strategy"] or item.candidate.strategy,
                                 rationale=it["rationale"], parent_id=item.candidate.id,
                                 generator=f"anthropic:{self.model}"))
        return out
