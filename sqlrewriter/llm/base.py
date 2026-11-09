"""Rewriter interface shared by LLM providers and the rule-based baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..context import SchemaContext
from ..models import Candidate, WorkloadQuery


class LLMError(Exception):
    """Generation failed (API error, refusal, malformed output ...)."""


@dataclass
class RewriteRequest:
    query: WorkloadQuery
    sql: str  # original query in the backend dialect
    dialect: str  # sqlglot dialect name
    dialect_label: str  # human readable, e.g. "Oracle SQL"
    context: SchemaContext
    original_plan: Optional[str] = None
    num_candidates: int = 3


@dataclass
class RepairItem:
    candidate: Candidate
    problem: str  # validation error / equivalence diff description


@dataclass
class GenerationStats:
    calls: int = 0
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    notes: list[str] = field(default_factory=list)


class Rewriter(ABC):
    name: str = "rewriter"
    source: str = "llm"

    def __init__(self) -> None:
        self.stats = GenerationStats()

    @abstractmethod
    def generate(self, request: RewriteRequest) -> list[Candidate]: ...

    def repair(self, request: RewriteRequest, items: list[RepairItem], round_no: int) -> list[Candidate]:
        """Ask for corrected versions of failed candidates (optional capability)."""
        return []

    @property
    def supports_repair(self) -> bool:
        return False
