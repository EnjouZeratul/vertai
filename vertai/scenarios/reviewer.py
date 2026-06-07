"""AI Agent SDK - 批阅评审模块"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from vertai.core.llm import LLMConfig, LLMEngine


@dataclass
class ReviewResult:
    """评审结果"""

    score: int
    criteria_scores: dict[str, int]
    comments: str
    suggestions: list[str]
    details: dict[str, Any] = field(default_factory=dict)


class ReviewerConfig(BaseModel):
    """评审器配置

    示例:
        # 使用默认配置
        config = ReviewerConfig()

        # 自定义配置
        config = ReviewerConfig(
            max_score=50,
            model="llama3.2",
        )
    """

    max_score: int = Field(default=100, ge=1, le=1000, description="最高分数")
    model: Optional[str] = Field(default=None, description="模型名称")
    template: Optional[str] = Field(default=None, description="评审模板")
    max_submission_length: int = Field(default=10000, ge=1, description="提交内容最大长度")
    max_reference_length: int = Field(default=5000, ge=1, description="参考答案最大长度")

    model_config = {"extra": "forbid"}


class Reviewer:
    """批阅评审器

    支持自定义评审标准，输出评分、批语和建议。

    Args:
        criteria: 评审标准列表，不能为空
        config: 评审器配置，可选
        llm: LLM引擎实例，可选

    Raises:
        ValueError: 当 criteria 为空时

    使用示例:
        reviewer = Reviewer(
            criteria=["准确性", "完整性", "格式规范"],
        )
        result = reviewer.evaluate(submission)
    """

    def __init__(
        self,
        criteria: list[str],
        config: Optional[ReviewerConfig] = None,
        llm: Optional[LLMEngine] = None,
    ):
        if not criteria:
            raise ValueError("评审标准不能为空")

        self.criteria = criteria
        self.config = config or ReviewerConfig()
        self._llm = llm

    @property
    def max_score(self) -> int:
        """最高分数"""
        return self.config.max_score

    @property
    def model(self) -> Optional[str]:
        """模型名称"""
        return self.config.model

    @property
    def template(self) -> Optional[str]:
        """评审模板"""
        return self.config.template

    def evaluate(self, submission: str, reference: Optional[str] = None) -> ReviewResult:
        """评审提交内容

        Args:
            submission: 待评审的内容
            reference: 参考答案，可选

        Returns:
            ReviewResult: 评审结果

        Raises:
            ValueError: 当提交内容为空或超过长度限制时
        """
        if not submission or not submission.strip():
            raise ValueError("提交内容不能为空")

        validated_submission = self._validate_input(submission, self.config.max_submission_length, "提交内容")
        validated_reference = None
        if reference:
            validated_reference = self._validate_input(reference, self.config.max_reference_length, "参考答案")

        llm = self._get_llm()
        prompt = self._build_prompt(validated_submission, validated_reference)
        response = llm.generate(prompt)
        return self._parse_response(response.content)

    def _validate_input(self, text: str, max_length: int, field_name: str) -> str:
        """验证并清理输入内容，防止提示词注入

        Args:
            text: 输入文本
            max_length: 最大长度限制
            field_name: 字段名称，用于错误提示

        Returns:
            str: 验证后的文本

        Raises:
            ValueError: 当内容超过长度限制时
        """
        if len(text) > max_length:
            raise ValueError(f"{field_name}超过最大长度限制 ({max_length} 字符)")

        cleaned = self._sanitize_input(text)
        return cleaned

    def _sanitize_input(self, text: str) -> str:
        """清理输入内容，移除潜在的提示词注入

        Args:
            text: 原始文本

        Returns:
            str: 清理后的文本
        """
        dangerous_patterns = [
            r'(?i)system\s*:',
            r'(?i)assistant\s*:',
            r'(?i)user\s*:',
            r'(?i)ignore\s+(previous|all)\s*(instructions|prompts)',
            r'(?i)forget\s+(everything|all)',
            r'(?i)disregard\s+',
            r'<<<\s*.*?\s*>>>',
            r'<\|.*?\|>',
        ]

        cleaned = text
        for pattern in dangerous_patterns:
            cleaned = re.sub(pattern, '[已移除]', cleaned)

        return cleaned.strip()

    def _get_llm(self) -> LLMEngine:
        """获取LLM引擎"""
        if self._llm:
            return self._llm
        config = LLMConfig(model=self.config.model) if self.config.model else LLMConfig()
        return LLMEngine(config=config)

    def _build_prompt(self, submission: str, reference: Optional[str]) -> str:
        """构建评审提示词

        Args:
            submission: 已验证的提交内容
            reference: 已验证的参考答案，可选

        Returns:
            str: 构建的提示词
        """
        criteria_text = "、".join(self.criteria)

        prompt_parts = [
            f"请根据以下评审标准对提交内容进行评审：{criteria_text}",
            "",
            "评审标准详情：",
        ]

        for i, criterion in enumerate(self.criteria, 1):
            prompt_parts.append(f"{i}. {criterion}（满分{self.max_score // len(self.criteria)}分）")

        prompt_parts.extend([
            "",
            "待评审内容：",
            submission,
        ])

        if reference:
            prompt_parts.extend([
                "",
                "参考答案：",
                reference,
            ])

        prompt_parts.extend([
            "",
            "请按以下JSON格式输出评审结果：",
            "{",
            '  "score": 总分（整数）,',
            '  "criteria_scores": {"标准名": 分数},',
            '  "comments": "整体评价",',
            '  "suggestions": ["改进建议1", "改进建议2"]',
            "}",
        ])

        return "\n".join(prompt_parts)

    def _parse_response(self, response: str) -> ReviewResult:
        """解析LLM响应

        Args:
            response: LLM返回的原始响应

        Returns:
            ReviewResult: 解析后的评审结果
        """
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return ReviewResult(
                score=0,
                criteria_scores={c: 0 for c in self.criteria},
                comments="无法解析评审结果",
                suggestions=["请重新提交评审"],
                details={"raw_response": response},
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return ReviewResult(
                score=0,
                criteria_scores={c: 0 for c in self.criteria},
                comments="评审结果格式错误",
                suggestions=["请重新提交评审"],
                details={"raw_response": response},
            )

        criteria_scores = data.get("criteria_scores", {})
        for criterion in self.criteria:
            if criterion not in criteria_scores:
                criteria_scores[criterion] = 0

        return ReviewResult(
            score=min(max(data.get("score", 0), 0), self.max_score),
            criteria_scores=criteria_scores,
            comments=data.get("comments", ""),
            suggestions=data.get("suggestions", []),
            details={"raw_response": response},
        )
