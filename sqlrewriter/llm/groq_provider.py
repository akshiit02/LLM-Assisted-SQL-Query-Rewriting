"""Groq query rewriter (open-weight models served on Groq's LPU cloud).

* Structured outputs: models that support strict JSON-schema decoding
  (``openai/gpt-oss-*``, ``qwen/qwen3.8-27b``) get ``response_format`` type
  ``json_schema`` with ``strict: true``; every other model uses JSON-object
  mode.  If a model rejects the schema mode the provider falls back to
  JSON-object mode once and remembers that.
* ``llm.effort`` maps to ``reasoning_effort`` on the gpt-oss reasoning models.
* Free-tier friendly: a 429 waits for the server's ``retry-after`` and tries
  again (tokens-per-minute limits).  A wait longer than
  ``llm.rate_limit_max_wait_s`` (a daily limit) fails the call with a clear
  message; responses received so far stay in the cache, so a later re-run
  resumes where this one stopped.

Credentials: GROQ_API_KEY (environment or .env).  Get a key at
https://console.groq.com/keys.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..config import LLMSettings, PROJECT_ROOT
from ..models import Candidate
from .base import LLMError, RepairItem, RewriteRequest, Rewriter
from .common import ResponseCache, parse_candidates
from .prompts import OUTPUT_SCHEMA, SYSTEM_PROMPT, build_repair_prompt, build_rewrite_prompt

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

# Models with strict JSON-schema structured outputs on Groq (console.groq.com/docs/structured-outputs)
STRICT_SCHEMA_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"}

_EFFORT_TO_REASONING = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


class GroqRewriter(Rewriter):
    name = "groq"
    source = "llm"

    def __init__(self, settings: LLMSettings):
        super().__init__()
        try:
            import groq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The 'groq' package is required: pip install groq") from exc
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set – create a key at https://console.groq.com/keys and put "
                               "GROQ_API_KEY=... in .env (see .env.example)")
        self._groq = groq
        self.settings = settings
        self.model = settings.model if settings.model and not settings.model.startswith("claude") \
            else DEFAULT_GROQ_MODEL
        self.client = groq.Groq(timeout=settings.request_timeout_s, max_retries=2)
        self.response_mode = "json_schema" if self.model in STRICT_SCHEMA_MODELS else "json_object"
        self.exhausted: str | None = None  # set when a daily limit is hit: later calls fail without a request
        cache_dir = Path(settings.cache_dir)
        self.cache = ResponseCache(cache_dir if cache_dir.is_absolute() else PROJECT_ROOT / cache_dir,
                                   settings.use_cache)

    @property
    def supports_repair(self) -> bool:
        return True

    # ------------------------------------------------------------------ request
    def _request_kwargs(self, user_prompt: str) -> dict:
        if self.response_mode == "json_schema":
            response_format = {"type": "json_schema",
                               "json_schema": {"name": "rewrite_candidates", "strict": True, "schema": OUTPUT_SCHEMA}}
        else:
            response_format = {"type": "json_object"}
        kwargs = dict(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}],
            response_format=response_format,
            max_completion_tokens=self.settings.max_tokens,
        )
        if self.model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = _EFFORT_TO_REASONING.get(self.settings.effort, "medium")
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        return kwargs

    def _wait_for_rate_limit(self, exc, attempt: int) -> None:
        retry_after = None
        try:
            retry_after = float(exc.response.headers.get("retry-after"))
        except (AttributeError, TypeError, ValueError):
            pass
        wait = retry_after if retry_after is not None else min(60.0, 5.0 * 2 ** attempt)
        if wait > self.settings.rate_limit_max_wait_s:
            self.exhausted = (f"Groq rate limit reached and resets in {wait:.0f}s (likely the daily token/request "
                              f"limit of your tier) – re-run later; responses received so far are cached")
            raise LLMError(f"{self.exhausted}. Details: {exc}") from exc
        self.stats.notes.append(f"rate limited, waited {wait:.0f}s")
        time.sleep(wait + 0.5)

    def _complete(self, user_prompt: str):
        g = self._groq
        rate_attempts = 0
        json_retries = 0
        while True:
            t0 = time.perf_counter()
            rate_limited = None
            try:
                return self.client.chat.completions.create(**self._request_kwargs(user_prompt))
            except g.RateLimitError as exc:
                if rate_attempts >= self.settings.rate_limit_retries:
                    raise LLMError(f"still rate limited by Groq after {rate_attempts} retries: {exc}") from exc
                rate_limited = exc
            except g.AuthenticationError as exc:
                raise LLMError("Groq authentication failed – check GROQ_API_KEY") from exc
            except g.NotFoundError as exc:
                raise LLMError(f"Groq model {self.model!r} not found (see console.groq.com/docs/models): "
                               f"{exc}") from exc
            except g.BadRequestError as exc:
                text = str(exc).lower()
                if self.response_mode == "json_schema" and ("response_format" in text or "json_schema" in text
                                                            or "does not support" in text):
                    self.response_mode = "json_object"  # model without schema decoding
                    self.stats.notes.append("json_schema unsupported, switched to json_object mode")
                    continue
                if ("json_validate_failed" in text or "does not match the expected schema" in text) \
                        and json_retries < 1:
                    json_retries += 1
                    continue
                raise LLMError(f"request rejected by the Groq API: {exc}") from exc
            except g.APIStatusError as exc:
                if exc.status_code == 413:
                    raise LLMError("request too large for your Groq tier's tokens-per-minute limit – lower "
                                   "llm.max_tokens or llm.num_candidates, or set include_plan_in_prompt: false") \
                        from exc
                raise LLMError(f"Groq API error {exc.status_code}: {exc}") from exc
            except g.APIConnectionError as exc:
                raise LLMError(f"cannot reach the Groq API: {exc}") from exc
            finally:
                self.stats.calls += 1
                self.stats.seconds += time.perf_counter() - t0
            if rate_limited is not None:
                self._wait_for_rate_limit(rate_limited, rate_attempts)
                rate_attempts += 1

    def _call(self, user_prompt: str) -> str:
        key = self.cache.key(self.name, self.model, self.settings.effort, self.settings.temperature,
                             SYSTEM_PROMPT, user_prompt)
        cached = self.cache.get(key)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached["text"]

        if self.exhausted:
            raise LLMError(self.exhausted)
        completion = self._complete(user_prompt)
        usage = getattr(completion, "usage", None)
        if usage is not None:
            self.stats.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.stats.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        if not completion.choices:
            raise LLMError("Groq returned no choices")
        choice = completion.choices[0]
        if choice.finish_reason == "length":
            raise LLMError("model output was truncated (max_tokens reached) – increase llm.max_tokens")
        text = choice.message.content or ""
        if not text.strip():
            raise LLMError("Groq returned no text content")
        self.cache.put(key, {"model": getattr(completion, "model", self.model), "text": text})
        return text

    # ------------------------------------------------------------------ Rewriter API
    def generate(self, request: RewriteRequest) -> list[Candidate]:
        items = parse_candidates(self._call(build_rewrite_prompt(request)))
        return [
            Candidate(id=f"L{i}", query_id=request.query.id, sql=it["sql"], source="llm",
                      strategy=it["strategy"], rationale=it["rationale"], generator=f"groq:{self.model}")
            for i, it in enumerate(items[: max(1, request.num_candidates)], start=1)
        ]

    def repair(self, request: RewriteRequest, items: list[RepairItem], round_no: int) -> list[Candidate]:
        if not items:
            return []
        fixed = parse_candidates(self._call(build_repair_prompt(request, items)))
        return [
            Candidate(id=f"{item.candidate.id}R{round_no}", query_id=request.query.id, sql=it["sql"],
                      source="repair", strategy=it["strategy"] or item.candidate.strategy,
                      rationale=it["rationale"], parent_id=item.candidate.id, generator=f"groq:{self.model}")
            for item, it in zip(items, fixed)
        ]
