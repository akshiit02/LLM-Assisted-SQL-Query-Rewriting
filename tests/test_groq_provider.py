"""Groq provider: request shape, parsing, caching, rate-limit handling (mocked client, no network)."""

import json
from types import SimpleNamespace

import httpx
import pytest

groq = pytest.importorskip("groq")

from sqlrewriter.context import SchemaContext  # noqa: E402
from sqlrewriter.llm import create_rewriter, groq_provider  # noqa: E402
from sqlrewriter.llm.base import LLMError, RepairItem, RewriteRequest  # noqa: E402
from sqlrewriter.models import Candidate, WorkloadQuery  # noqa: E402

REPLY = json.dumps({"candidates": [
    {"strategy": "exists", "rationale": "semi-join", "sql": "SELECT 1"},
    {"strategy": "join", "rationale": "join", "sql": "SELECT 2"},
]})


def _completion(text=REPLY, finish_reason="stop"):
    return SimpleNamespace(model="openai/gpt-oss-120b", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
                           choices=[SimpleNamespace(finish_reason=finish_reason,
                                                    message=SimpleNamespace(content=text))])


def _rate_limit(retry_after: str):
    response = httpx.Response(429, headers={"retry-after": retry_after},
                              request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return groq.RateLimitError("rate limit exceeded", response=response, body=None)


class FakeGroq:
    """Stands in for groq.Groq; replies are popped from ``script`` (exceptions are raised)."""

    script: list = []
    calls: list = []

    def __init__(self, **kw):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        FakeGroq.calls.append(kwargs)
        item = FakeGroq.script.pop(0) if FakeGroq.script else _completion()
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def make_rewriter(monkeypatch, settings, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(groq, "Groq", FakeGroq)
    monkeypatch.setattr(groq_provider.time, "sleep", lambda s: None)
    FakeGroq.script, FakeGroq.calls = [], []

    def make(**overrides):
        llm = settings.llm.__class__(**{**settings.llm.__dict__, "provider": "groq", "model": "openai/gpt-oss-120b",
                                        "cache_dir": str(tmp_path / "cache"), "effort": "xhigh", **overrides})
        return create_rewriter(llm)
    return make


def _request(n=3):
    return RewriteRequest(query=WorkloadQuery("T", "basic", "t", "SELECT 1"), sql="SELECT 1", dialect="sqlite",
                          dialect_label="SQLite", context=SchemaContext("sqlite", "sqlite", 0.1, "fk", "ctx", {}),
                          num_candidates=n)


def test_request_shape_parsing_and_cache(make_rewriter):
    rw = make_rewriter()
    assert isinstance(rw, groq_provider.GroqRewriter)
    cands = rw.generate(_request())
    assert [c.sql for c in cands] == ["SELECT 1", "SELECT 2"]
    assert cands[0].source == "llm" and cands[0].generator == "groq:openai/gpt-oss-120b"
    sent = FakeGroq.calls[0]
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["reasoning_effort"] == "high"  # xhigh maps to Groq's highest level
    assert sent["messages"][0]["role"] == "system" and "SELECT 1" in sent["messages"][1]["content"]
    assert rw.stats.input_tokens == 100 and rw.stats.output_tokens == 50
    rw.generate(_request())  # identical prompt -> served from the response cache
    assert len(FakeGroq.calls) == 1 and rw.stats.cache_hits == 1


def test_non_schema_model_uses_json_object_and_claude_default_is_replaced(make_rewriter):
    rw = make_rewriter(model="llama-3.3-70b-versatile")
    rw.generate(_request())
    assert FakeGroq.calls[0]["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in FakeGroq.calls[0]
    assert make_rewriter(model="claude-opus-5-5").model == groq_provider.DEFAULT_GROQ_MODEL


def test_waits_out_per_minute_rate_limit(make_rewriter):
    rw = make_rewriter()
    FakeGroq.script = [_rate_limit("2"), _rate_limit("3"), _completion()]
    assert len(rw.generate(_request())) == 2
    assert len(FakeGroq.calls) == 3
    assert rw.stats.notes == ["rate limited, waited 2s", "rate limited, waited 3s"]


def test_daily_limit_fails_fast_and_stops_further_requests(make_rewriter):
    rw = make_rewriter()
    FakeGroq.script = [_rate_limit("7200")]
    with pytest.raises(LLMError, match="daily"):
        rw.generate(_request())
    with pytest.raises(LLMError, match="daily"):
        rw.generate(_request(2))  # different prompt, but no new request is sent
    assert len(FakeGroq.calls) == 1


def test_truncated_output_is_an_error(make_rewriter):
    rw = make_rewriter()
    FakeGroq.script = [_completion(text='{"candidates": [', finish_reason="length")]
    with pytest.raises(LLMError, match="max_tokens"):
        rw.generate(_request())


def test_repair_keeps_lineage(make_rewriter):
    rw = make_rewriter()
    failed = Candidate(id="L2", query_id="T", sql="SELECT x", source="llm", strategy="join")
    fixed = rw.repair(_request(), [RepairItem(failed, "row count differs")], round_no=1)
    assert fixed[0].id == "L2R1" and fixed[0].parent_id == "L2" and fixed[0].source == "repair"
    assert "row count differs" in FakeGroq.calls[0]["messages"][1]["content"]


def test_missing_key_is_reported(monkeypatch, settings):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm = settings.llm.__class__(**{**settings.llm.__dict__, "provider": "groq"})
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        create_rewriter(llm)
