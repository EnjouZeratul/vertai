"""AI Agent SDK - 知识库问答模块

企业知识库问答功能，支持本地文档的索引、检索和答案生成。
提供来源追溯和置信度评分。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vertai.core.llm import LLMEngine
from vertai.core.vector import (
    Document,
    SearchResult,
    VectorConfig,
    VectorEngine,
    InMemoryVectorStore,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceReference:
    """来源引用"""

    content: str
    source: str
    page: int | None = None
    paragraph: int | None = None
    relevance_score: float = 0.0


@dataclass
class AnswerResult:
    """问答结果"""

    question: str
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# 提示词注入防护：限制输入长度和危险模式
_MAX_QUESTION_LENGTH = 1000
_MAX_CONTEXT_LENGTH_SANITY = 10000
_DANGEROUS_PATTERNS = [
    r"ignore\s+(previous|all|above)\s+(instructions|prompts|rules)",
    r"forget\s+(everything|all|previous)",
    r"you\s+are\s+now",
    r"new\s+instructions?",
    r"disregard\s+(all|previous)",
    r"override\s+(previous|default|system)",
]


def _sanitize_input(text: str) -> str:
    """清理用户输入，防止提示词注入

    Args:
        text: 原始输入文本

    Returns:
        清理后的文本

    Raises:
        ValueError: 如果输入包含潜在注入模式或过长
    """
    if len(text) > _MAX_QUESTION_LENGTH:
        raise ValueError(f"输入过长，最大允许 {_MAX_QUESTION_LENGTH} 字符")

    text_lower = text.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, text_lower):
            raise ValueError("输入包含不允许的内容")

    # 移除控制字符
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned.strip()


def _sanitize_context(text: str) -> str:
    """清理上下文内容，防止通过文档注入

    Args:
        text: 上下文文本

    Returns:
        清理后的文本
    """
    # 限制长度作为安全边界
    if len(text) > _MAX_CONTEXT_LENGTH_SANITY:
        text = text[:_MAX_CONTEXT_LENGTH_SANITY]

    # 移除控制字符
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return cleaned


def _get_env_int(key: str, default: int) -> int:
    """从环境变量获取整数配置"""
    value = os.environ.get(key)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def _get_env_float(key: str, default: float) -> float:
    """从环境变量获取浮点数配置"""
    value = os.environ.get(key)
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return default


def _get_env_list(key: str, default: list[str]) -> list[str]:
    """从环境变量获取列表配置（逗号分隔）"""
    value = os.environ.get(key)
    if value:
        return [v.strip() for v in value.split(",") if v.strip()]
    return default


@dataclass
class KnowledgeQAConfig:
    """知识库问答配置

    支持环境变量配置：
    - VERTAI_CHUNK_SIZE: 文档分块大小
    - VERTAI_CHUNK_OVERLAP: 分块重叠大小
    - VERTAI_TOP_K: 检索返回数量
    - VERTAI_MIN_CONFIDENCE: 最小置信度阈值
    - VERTAI_MAX_CONTEXT_LENGTH: 最大上下文长度
    - VERTAI_SUPPORTED_FORMATS: 支持的文档格式（逗号分隔）
    """

    chunk_size: int = field(default_factory=lambda: _get_env_int("VERTAI_CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _get_env_int("VERTAI_CHUNK_OVERLAP", 50))
    top_k: int = field(default_factory=lambda: _get_env_int("VERTAI_TOP_K", 5))
    min_confidence: float = field(default_factory=lambda: _get_env_float("VERTAI_MIN_CONFIDENCE", 0.3))
    max_context_length: int = field(default_factory=lambda: _get_env_int("VERTAI_MAX_CONTEXT_LENGTH", 4000))
    supported_formats: list[str] = field(default_factory=lambda: _get_env_list("VERTAI_SUPPORTED_FORMATS", ["txt", "md", "json"]))


class DocumentLoader:
    """文档加载器

    支持多种文档格式的解析和分块。
    """

    def __init__(self, config: KnowledgeQAConfig | None = None):
        self.config = config or KnowledgeQAConfig()

    def load_directory(self, directory: str | Path) -> list[Document]:
        """加载目录下所有文档

        Args:
            directory: 文档目录路径

        Returns:
            文档列表
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")

        documents = []
        for ext in self.config.supported_formats:
            for file_path in directory.glob(f"**/*.{ext}"):
                try:
                    docs = self.load_file(file_path)
                    documents.extend(docs)
                    logger.info(f"已加载: {file_path}")
                except (OSError, ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"加载失败 {file_path}: {e}")

        return documents

    def load_file(self, file_path: str | Path) -> list[Document]:
        """加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            文档片段列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = file_path.suffix.lower().lstrip(".")
        content = file_path.read_text(encoding="utf-8")

        if suffix == "json":
            return self._parse_json(content, str(file_path))
        else:
            return self._chunk_text(content, str(file_path))

    def _parse_json(self, content: str, source: str) -> list[Document]:
        """解析 JSON 文件"""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}")

        documents = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text = item.get("content") or item.get("text") or str(item)
                    metadata = {k: v for k, v in item.items() if k not in ["content", "text"]}
                    metadata["source"] = source
                    metadata["index"] = i
                    documents.append(Document(content=text, metadata=metadata))
        elif isinstance(data, dict):
            text = data.get("content") or data.get("text") or str(data)
            metadata = {k: v for k, v in data.items() if k not in ["content", "text"]}
            metadata["source"] = source
            documents.append(Document(content=text, metadata=metadata))

        return documents

    def _create_chunk(self, content: str, source: str, index: int) -> Document:
        """创建文档块"""
        return Document(
            content=content.strip(),
            metadata={"source": source, "chunk_index": index},
        )

    def _split_into_paragraphs(self, content: str) -> list[str]:
        """将内容拆分为段落"""
        return [p.strip() for p in content.split("\n\n") if p.strip()]

    def _chunk_text(self, content: str, source: str) -> list[Document]:
        """文本分块"""
        chunks = []
        paragraphs = self._split_into_paragraphs(content)

        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.config.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, source, chunk_index))
                    chunk_index += 1
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, source, chunk_index))

        return chunks


class KnowledgeQA:
    """知识库问答系统

    支持本地文档的索引、检索和答案生成。

    使用示例:
        qa = KnowledgeQA("./docs")
        result = qa.ask("问题")
        print(result.answer)
        for source in result.sources:
            print(f"来源: {source.source}")
    """

    def __init__(
        self,
        docs_path: str | Path | None = None,
        config: KnowledgeQAConfig | None = None,
        vector_config: VectorConfig | None = None,
        llm: LLMEngine | None = None,
    ):
        """初始化知识库问答

        Args:
            docs_path: 文档目录路径
            config: 问答配置
            vector_config: 向量引擎配置
            llm: LLM引擎实例
        """
        self.config = config or KnowledgeQAConfig()
        self.vector_config = vector_config or VectorConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            top_k=self.config.top_k,
        )

        self._vector_engine = VectorEngine(
            config=self.vector_config,
            store_type="memory",
        )
        self._llm = llm
        self._loader = DocumentLoader(self.config)
        self._indexed = False
        self._documents: list[Document] = []

        if docs_path:
            self.index(docs_path)

    def index(self, docs_path: str | Path) -> int:
        """索引文档目录

        Args:
            docs_path: 文档目录路径

        Returns:
            索引的文档数量
        """
        docs_path = Path(docs_path)
        documents = self._loader.load_directory(docs_path)
        self._documents = documents

        if documents:
            self._vector_engine.index_documents(documents)
            self._indexed = True
            logger.info(f"已索引 {len(documents)} 个文档片段")

        return len(documents)

    def add_documents(self, documents: list[Document]) -> int:
        """添加文档

        Args:
            documents: 文档列表

        Returns:
            添加的文档数量
        """
        self._documents.extend(documents)
        self._vector_engine.index_documents(documents)
        self._indexed = True
        return len(documents)

    def ask(self, question: str) -> AnswerResult:
        """提问

        Args:
            question: 用户问题

        Returns:
            AnswerResult: 包含答案、来源和置信度

        Raises:
            ValueError: 问题为空或包含不允许的内容
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        # 安全验证：防止提示词注入
        question = _sanitize_input(question)

        if not self._indexed:
            return AnswerResult(
                question=question,
                answer="知识库尚未索引任何文档，请先添加文档。",
                confidence=0.0,
            )

        # 检索相关文档
        search_results = self._vector_engine.search(
            question,
            top_k=self.config.top_k,
        )

        if not search_results:
            return AnswerResult(
                question=question,
                answer="抱歉，未找到与问题相关的内容。",
                confidence=0.0,
            )

        # 过滤低相关性结果
        relevant_results = [
            r for r in search_results
            if r.score >= self.config.min_confidence
        ]

        if not relevant_results:
            return AnswerResult(
                question=question,
                answer="抱歉，未找到足够相关的内容来回答问题。",
                confidence=0.0,
            )

        # 构建上下文
        context = self._build_context(relevant_results)

        # 生成答案
        answer = self._generate_answer(question, context)

        # 计算置信度
        confidence = self._calculate_confidence(relevant_results)

        # 构建来源引用
        sources = self._build_sources(relevant_results)

        return AnswerResult(
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            metadata={
                "context_length": len(context),
                "num_sources": len(sources),
            }
        )

    def _build_context(self, results: list[SearchResult]) -> str:
        """构建上下文"""
        context_parts = []
        total_length = 0

        for result in results:
            content = result.document.content
            if total_length + len(content) <= self.config.max_context_length:
                context_parts.append(content)
                total_length += len(content)
            else:
                remaining = self.config.max_context_length - total_length
                if remaining > 100:
                    context_parts.append(content[:remaining] + "...")
                break

        return "\n\n---\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """生成答案

        Args:
            question: 已清理的用户问题
            context: 已清理的上下文内容

        Returns:
            生成的答案
        """
        llm = self._get_llm()

        # 清理上下文防止注入
        safe_context = _sanitize_context(context)

        # 使用明确的分隔符，降低注入风险
        prompt = (
            "请基于以下参考资料回答问题。如果资料中没有相关信息，请明确说明。\n\n"
            "---参考资料开始---\n"
            f"{safe_context}\n"
            "---参考资料结束---\n\n"
            "---用户问题开始---\n"
            f"{question}\n"
            "---用户问题结束---\n\n"
            "请给出准确、简洁的回答，仅基于上述参考资料，并指出信息来源："
        )

        response = llm.generate(prompt)
        return self._parse_answer(response)

    def _parse_answer(self, response: str) -> str:
        """解析答案"""
        # 移除可能的 JSON 包装
        if response.startswith("{") and response.endswith("}"):
            try:
                data = json.loads(response)
                return data.get("answer", response)
            except json.JSONDecodeError:
                pass

        return response.strip()

    def _get_llm(self) -> LLMEngine:
        """获取 LLM 引擎"""
        if self._llm:
            return self._llm
        return LLMEngine(model="local")

    def _calculate_confidence(self, results: list[SearchResult]) -> float:
        """计算置信度"""
        if not results:
            return 0.0

        # 基于检索分数计算
        scores = [r.score for r in results[:3]]
        avg_score = sum(scores) / len(scores)

        # 考虑来源数量
        source_factor = min(len(results) / 3, 1.0)

        return round(avg_score * source_factor, 2)

    def _build_sources(self, results: list[SearchResult]) -> list[SourceReference]:
        """构建来源引用"""
        sources = []
        seen = set()

        for result in results:
            doc = result.document
            source_key = f"{doc.metadata.get('source', 'unknown')}:{doc.doc_id}"

            if source_key in seen:
                continue
            seen.add(source_key)

            sources.append(SourceReference(
                content=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                source=doc.metadata.get("source", "未知来源"),
                page=doc.metadata.get("page"),
                paragraph=doc.metadata.get("paragraph"),
                relevance_score=round(result.score, 3),
            ))

        return sources

    def count_documents(self) -> int:
        """获取索引文档数量"""
        return self._vector_engine.count()

    def clear(self) -> None:
        """清空知识库"""
        if self._documents:
            doc_ids = [doc.doc_id for doc in self._documents]
            self._vector_engine.delete_documents(doc_ids)
        self._documents = []
        self._indexed = False
