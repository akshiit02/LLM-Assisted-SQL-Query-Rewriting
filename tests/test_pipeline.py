import json
from types import SimpleNamespace

from sqlrewriter.experiment import ExperimentRunner
from sqlrewriter.llm.base import RepairItem, RewriteRequest, Rewriter
from sqlrewriter.llm.common import parse_candidates
from sqlrewriter.models import Candidate, ExperimentCondition, WorkloadQuery
from sqlrewriter.pipeline import RewritePipeline
from sqlrewriter.report import build_report

from .conftest import SCALE

ORIGINAL = ("SELECT c.customer_id, c.name FROM customers c WHERE c.customer_id IN "
            "(SELECT o.customer_id FROM orders o WHERE o.total_amount > 500) ORDER BY c.customer_id")


class FakeRewriter(Rewriter):
    """Returns fixed candidates covering every verdict path; repairs the non-equivalent one."""

    name = "fake"

    def __init__(self):
        super().__init__()
        self.repair_calls = []

    @property
    def supports_repair(self):
        return True

    def generate(self, request: RewriteRequest):
        sqls = {
            "equivalent": ("SELECT c.customer_id, c.name FROM customers c WHERE EXISTS (SELECT 1 FROM orders o "
                           "WHERE o.customer_id = c.customer_id AND o.total_amount > 500) ORDER BY c.customer_id"),
            "join_dupes": ("SELECT c.customer_id, c.name FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
                           "WHERE o.total_amount > 500 ORDER BY c.customer_id"),
            "bad_column": "SELECT c.customer_id, c.nam FROM customers c",
            "unsafe": "DELETE FROM customers",
            "wrong_order": ("SELECT c.customer_id, c.name FROM customers c WHERE c.customer_id IN (SELECT o.customer_id "
                            "FROM orders o WHERE o.total_amount > 500) ORDER BY c.customer_id DESC"),
        }
        return [Candidate(id=f"L{i}", query_id=request.query.id, sql=s, source="llm", strategy=k,
                          generator="fake") for i, (k, s) in enumerate(sqls.items(), 1)]

    def repair(self, request, items: list[RepairItem], round_no: int):
        self.repair_calls.append(items)
        fixed = ("SELECT DISTINCT c.customer_id, c.name FROM customers c JOIN orders o ON o.customer_id = c.customer_id "
                 "WHERE o.total_amount > 500 ORDER BY c.customer_id")
        return [Candidate(id=f"{it.candidate.id}R{round_no}", query_id=request.query.id, sql=fixed, source="repair",
                          strategy="fixed", parent_id=it.candidate.id, generator="fake") for it in items]


def test_pipeline_verdicts_and_repair(settings, backend):
    rw = FakeRewriter()
    pipe = RewritePipeline(settings, backend, [rw])
    q = WorkloadQuery(id="T1", complexity="intermediate", title="test", sql=ORIGINAL)
    res = pipe.evaluate(q, ExperimentCondition(SCALE, "fk_indexes", "sqlite"))
    verdicts = {o.candidate.strategy: o.verdict for o in res.outcomes}
    assert verdicts["equivalent"] in ("improved", "neutral", "regressed")
    assert verdicts["join_dupes"] == "rejected_nonequivalent"
    assert verdicts["bad_column"] == "rejected_unexecutable"
    assert verdicts["unsafe"] == "rejected_unsafe"
    assert verdicts["wrong_order"] == "rejected_nonequivalent"
    # repair round received the non-equivalent and unexecutable candidates (not the unsafe one)
    assert len(rw.repair_calls) == 1
    repaired_ids = {it.candidate.id for it in rw.repair_calls[0]}
    assert repaired_ids == {"L2", "L3", "L5"}
    repaired = [o for o in res.outcomes if o.candidate.source == "repair"]
    assert repaired and all(o.verdict in ("improved", "neutral", "regressed") for o in repaired)
    assert "rows" in pipe.describe_failure(next(o for o in res.outcomes if o.candidate.id == "L2"))


def test_parse_candidates_tolerates_fences():
    text = '```json\n{"candidates": [{"strategy": "s", "rationale": "r", "sql": "SELECT 1"}, {"sql": ""}]}\n```'
    assert parse_candidates(text) == [{"strategy": "s", "rationale": "r", "sql": "SELECT 1"}]


def test_anthropic_request_shape(monkeypatch, settings):
    """The Claude provider sends structured-output + fallback parameters and parses the reply."""
    from sqlrewriter.context import SchemaContext
    from sqlrewriter.llm import anthropic_provider

    captured = {}
    reply = json.dumps({"candidates": [{"strategy": "exists", "rationale": "semi-join", "sql": "SELECT 1"}]})

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_final_message(self):
            return SimpleNamespace(stop_reason="end_turn", model="claude-opus-5-5",
                                   usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                                   content=[SimpleNamespace(type="text", text=reply)])

    class FakeClient:
        def __init__(self, **kw):
            self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

        def _stream(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    llm = settings.llm.__class__(**{**settings.llm.__dict__, "provider": "anthropic", "use_cache": False})
    rw = anthropic_provider.AnthropicRewriter(llm)
    req = RewriteRequest(query=WorkloadQuery("T", "basic", "t", "SELECT 1"), sql="SELECT 1", dialect="sqlite",
                         dialect_label="SQLite", context=SchemaContext("sqlite", "sqlite", 0.1, "fk", "ctx", {}))
    cands = rw.generate(req)
    assert cands[0].sql == "SELECT 1" and cands[0].source == "llm"
    assert captured["model"] == "claude-opus-5-5"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["betas"] == ["server-side-fallback-2026-07-01"]
    assert captured["extra_body"] == {"fallbacks": "default"}


def test_experiment_and_report_end_to_end(settings):
    s = settings
    s.experiment.queries = ["Q01", "Q06", "Q11"]
    s.experiment.name = "pytest"
    run_dir = ExperimentRunner(s, log=lambda m: None).run()
    report = build_report(run_dir)
    assert report.exists() and "Research questions" in report.read_text(encoding="utf-8")
    for f in ("results.json", "outcomes.csv", "runs.csv", "summary.json", "report.md"):
        assert (run_dir / f).exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["rq_answers"]) == {"RQ1", "RQ2", "RQ3", "RQ4", "RQ5"}
