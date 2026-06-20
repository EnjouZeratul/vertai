"""Evaluation / reviewer scenario (S3 generalize).

Generalizes the previous ``Reviewer`` into an LLM-as-judge
:class:`Evaluation` abstraction that depends on
:class:`~vertai.core.provider.LLMProvider` (not the legacy ``LLMEngine``
single-prompt API). :class:`Reviewer` is the concrete criteria-based judge.

The provider is obtained via :func:`~vertai.core.provider.create_provider` (or
injection); generation uses ``provider.generate([ChatMessage])`` and reads
``result.content``. Input is sanitized against prompt injection (English +
Chinese patterns).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from vertai.core.llm import LLMEngine
from vertai.core.provider import (
    ChatMessage,
    LLMConfig,
    LLMProvider,
    create_provider,
)

__all__ = [
    "ReviewResult",
    "ReviewerConfig",
    "Evaluation",
    "Reviewer",
]


@dataclass
class ReviewResult:
    """A single evaluation result."""

    score: int
    criteria_scores: dict[str, int]
    comments: str
    suggestions: list[str]
    details: dict[str, Any] = field(default_factory=dict)


class ReviewerConfig(BaseModel):
    """Reviewer configuration."""

    max_score: int = Field(default=100, ge=1, le=1000, description="max score")
    model: Optional[str] = Field(default=None, description="model name")
    template: Optional[str] = Field(default=None, description="review template")
    max_submission_length: int = Field(default=10000, ge=1, description="max submission length")
    max_reference_length: int = Field(default=5000, ge=1, description="max reference length")

    model_config = {"extra": "forbid"}


# Prompt-injection patterns sanitized out of submissions / references. Covers
# English and Chinese injection attempts. Compiled with re.IGNORECASE (inline
# ``(?i)`` is illegal mid-expression on Python 3.11+).
_INJECTION_PATTERNS = [
    r"system\s*:",
    r"assistant\s*:",
    r"user\s*:",
    r"ignore\s+(previous|all|prior)\s*(instructions?|prompts?)",
    r"forget\s+(everything|all)",
    r"disregard\s+",
    r"忽略(之前|上面|前面|上述)(的)?(指令|提示|规则)",
    r"忘记(之前|所有)(的)?(指令|内容)",
    r"你现在(扮演|是)",
    r"无视(之前|上述)(的)?(指令|规则)",
    r"<<<\s*.*?\s*>>>",
    r"<\|.*?\|>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_REDACTION = "[removed]"


class Evaluation(ABC):
    """Generalized LLM-as-judge evaluation abstraction.

    Subclasses implement :meth:`evaluate`. The LLM is accessed through the
    :class:`LLMProvider` abstraction: pass ``provider=`` directly, or pass
    ``llm=`` an :class:`~vertai.core.llm.LLMEngine` (its ``.provider`` is
    extracted). With neither, a provider is built from
    :func:`~vertai.core.provider.create_provider` (default Ollama config).
    """

    def __init__(
        self,
        *,
        config: ReviewerConfig | None = None,
        provider: LLMProvider | None = None,
        llm: LLMEngine | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.config = config or ReviewerConfig()
        self._provider: LLMProvider | None = provider or (
            llm.provider if isinstance(llm, LLMEngine) else None
        )
        self._llm_engine: LLMEngine | None = llm
        self._llm_config: LLMConfig | None = llm_config

    @abstractmethod
    def evaluate(
        self, submission: str, reference: Optional[str] = None
    ) -> ReviewResult:
        """Evaluate ``submission`` (optionally against ``reference``)."""

    def get_provider(self) -> LLMProvider:
        """Return the LLM provider (injected, or built from config)."""
        if self._provider is not None:
            return self._provider
        if self._llm_engine is not None:
            return self._llm_engine.provider
        model = self.config.model
        config = LLMConfig(model=model) if model else (self._llm_config or LLMConfig())
        return create_provider(config)

    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Redact prompt-injection patterns and strip control characters."""
        cleaned = _INJECTION_RE.sub(_REDACTION, text)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
        return cleaned.strip()


class Reviewer(Evaluation):
    """Criteria-based LLM-as-judge reviewer.

    Args:
        criteria: review criteria; must be non-empty.
        config: optional :class:`ReviewerConfig`.
        provider: optional :class:`LLMProvider` (preferred injection point).
        llm: optional :class:`LLMEngine` (its ``.provider`` is used).
        llm_config: optional :class:`LLMConfig` for the default provider.

    Raises:
        ValueError: if ``criteria`` is empty.

    Example:
        reviewer = Reviewer(criteria=["accuracy", "completeness"], provider=prov)
        result = reviewer.evaluate(submission)
    """

    def __init__(
        self,
        criteria: list[str],
        config: ReviewerConfig | None = None,
        provider: LLMProvider | None = None,
        llm: LLMEngine | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        super().__init__(
            config=config, provider=provider, llm=llm, llm_config=llm_config
        )
        if not criteria:
            raise ValueError("criteria must not be empty")
        self.criteria = list(criteria)

    @property
    def max_score(self) -> int:
        return self.config.max_score

    @property
    def model(self) -> Optional[str]:
        return self.config.model

    @property
    def template(self) -> Optional[str]:
        return self.config.template

    def evaluate(
        self, submission: str, reference: Optional[str] = None
    ) -> ReviewResult:
        """Evaluate ``submission`` (optionally against ``reference``).

        Raises ``ValueError`` if the submission is empty or exceeds the length
        limit.
        """
        if not submission or not submission.strip():
            raise ValueError("submission must not be empty")

        validated_submission = self._validate_input(
            submission, self.config.max_submission_length, "submission"
        )
        validated_reference: Optional[str] = None
        if reference:
            validated_reference = self._validate_input(
                reference, self.config.max_reference_length, "reference"
            )

        provider = self.get_provider()
        prompt = self._build_prompt(validated_submission, validated_reference)
        result = provider.generate([ChatMessage(role="user", content=prompt)])
        return self._parse_response(result.content)

    def _validate_input(
        self, text: str, max_length: int, field_name: str
    ) -> str:
        if len(text) > max_length:
            raise ValueError(
                f"{field_name} exceeds max length limit ({max_length} characters)"
            )
        return self._sanitize_input(text)

    def _build_prompt(self, submission: str, reference: Optional[str]) -> str:
        criteria_text = ", ".join(self.criteria)
        prompt_parts = [
            f"Review the submission against these criteria: {criteria_text}",
            "",
            "Criteria detail:",
        ]
        per_criterion = self.max_score // max(len(self.criteria), 1)
        for i, criterion in enumerate(self.criteria, 1):
            prompt_parts.append(
                f"{i}. {criterion} (max {per_criterion} points)"
            )
        prompt_parts.extend(["", "Submission:", submission])
        if reference:
            prompt_parts.extend(["", "Reference:", reference])
        prompt_parts.extend(
            [
                "",
                "Output the review result as JSON:",
                "{",
                '  "score": <int total>,',
                '  "criteria_scores": {"<criterion>": <int>},',
                '  "comments": "<overall comment>",',
                '  "suggestions": ["<suggestion>", ...]',
                "}",
            ]
        )
        return "\n".join(prompt_parts)

    def _parse_response(self, response: str) -> ReviewResult:
        """Parse the LLM JSON response into a :class:`ReviewResult`."""
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return ReviewResult(
                score=0,
                criteria_scores={c: 0 for c in self.criteria},
                comments="unable to parse review result",
                suggestions=["please resubmit the review"],
                details={"raw_response": response},
            )
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ReviewResult(
                score=0,
                criteria_scores={c: 0 for c in self.criteria},
                comments="review result format error",
                suggestions=["please resubmit the review"],
                details={"raw_response": response},
            )

        criteria_scores: dict[str, int] = {
            str(k): int(v) for k, v in dict(data.get("criteria_scores", {})).items()
        }
        for criterion in self.criteria:
            if criterion not in criteria_scores:
                criteria_scores[criterion] = 0
        raw_score = data.get("score", 0)
        try:
            score_int = int(raw_score)
        except (TypeError, ValueError):
            score_int = 0
        return ReviewResult(
            score=min(max(score_int, 0), self.max_score),
            criteria_scores=criteria_scores,
            comments=str(data.get("comments", "")),
            suggestions=list(data.get("suggestions", [])),
            details={"raw_response": response},
        )
