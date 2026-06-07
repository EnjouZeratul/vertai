"""
LLM 引擎模块 - 本地模型支持与多模型管理

核心功能:
- Ollama 自动检测与连接
- Anthropic/OpenAI 兼容 API 支持
- 流式与非流式输出
- 多模型切换
- 容错处理与友好提示
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Iterator,
    Optional,
    Union,
)

import httpx
from pydantic import BaseModel, ConfigDict, Field

# Anthropic API 版本常量
ANTHROPIC_API_VERSION = "2023-06-01"


class ModelProvider(str, Enum):
    """模型提供商枚举"""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """模型状态枚举"""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class ModelInfo:
    """模型信息"""

    name: str
    provider: ModelProvider
    status: ModelStatus = ModelStatus.AVAILABLE
    size: Optional[str] = None
    parameters: Optional[str] = None
    quantization: Optional[str] = None
    modified_at: Optional[str] = None


class LLMConfig(BaseModel):
    """LLM 配置

    示例:
        # 使用默认本地模型 (Ollama)
        config = LLMConfig()

        # 使用 Anthropic 兼容 API (如 DeepSeek)
        config = LLMConfig(
            provider=ModelProvider.ANTHROPIC,
            base_url="https://api.deepseek.com/anthropic",
            api_key="sk-xxx",
            model="claude-3-5-sonnet-20241022",
        )

        # 使用 DeepSeek 专用配置
        config = LLMConfig(
            provider=ModelProvider.DEEPSEEK,
            base_url="https://api.deepseek.com/anthropic",
            api_key="sk-xxx",
            model="deepseek-v4-flash",
        )

        # 高级参数
        config = LLMConfig(
            model="llama3.2",
            temperature=0.7,
            max_tokens=2048,
        )
    """

    model: str = Field(default="llama3.2", description="模型名称")
    provider: ModelProvider = Field(default=ModelProvider.OLLAMA, description="模型提供商")
    base_url: str = Field(default="http://localhost:11434", description="API 基础地址")
    api_key: Optional[str] = Field(default=None, description="API 密钥")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="生成温度")
    max_tokens: int = Field(default=4096, ge=1, description="最大生成 token 数")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 采样")
    top_k: int = Field(default=40, ge=0, description="Top-k 采样")
    repeat_penalty: float = Field(default=1.1, ge=1.0, description="重复惩罚")
    seed: Optional[int] = Field(default=None, description="随机种子")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    timeout: float = Field(default=120.0, ge=1.0, description="请求超时时间(秒)")

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    def __init__(self, **data: Any) -> None:
        # 支持从环境变量读取 API Key
        if "api_key" not in data:
            env_key = os.environ.get("VERTAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if env_key:
                data["api_key"] = env_key
        super().__init__(**data)

    @staticmethod
    def _validate_model_name(model_name: str) -> str:
        """验证模型名称，防止注入攻击"""
        if not re.match(r'^[a-zA-Z0-9._-]+$', model_name):
            raise ValueError(
                "模型名称包含非法字符。"
                "只允许字母、数字、点(.)、下划线(_)、连字符(-)。"
            )
        return model_name

    def model_post_init(self, __context: Any) -> None:
        """初始化后验证模型名称"""
        self._validate_model_name(self.model)

    def is_anthropic_compatible(self) -> bool:
        """检查是否使用 Anthropic 兼容 API"""
        return self.provider in (
            ModelProvider.ANTHROPIC,
            ModelProvider.DEEPSEEK,
            ModelProvider.OPENAI,
        )


class ChatMessage(BaseModel):
    """聊天消息"""

    role: str = Field(description="角色: system/user/assistant")
    content: str = Field(description="消息内容")

    model_config = ConfigDict(frozen=True)


class GenerateResult(BaseModel):
    """生成结果"""

    content: str = Field(description="生成的文本内容")
    model: str = Field(description="使用的模型名称")
    prompt_tokens: int = Field(default=0, description="输入 token 数")
    completion_tokens: int = Field(default=0, description="输出 token 数")
    total_tokens: int = Field(default=0, description="总 token 数")
    finish_reason: str = Field(default="stop", description="结束原因")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据")

    model_config = ConfigDict(frozen=True)


class OllamaDetector:
    """Ollama 服务检测器

    自动检测本地 Ollama 服务是否运行，并列出可用模型。

    示例:
        detector = OllamaDetector()

        # 检查服务是否运行
        if detector.is_running():
            print("Ollama is running!")

        # 获取可用模型列表
        models = detector.list_models()
        for model in models:
            print(f"- {model.name}")
    """

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """获取或创建 HTTP 客户端"""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def is_running(self) -> bool:
        """检查 Ollama 服务是否运行"""
        try:
            response = self._get_client().get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def list_models(self) -> list[ModelInfo]:
        """获取可用模型列表"""
        if not self.is_running():
            return []

        try:
            response = self._get_client().get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for item in data.get("models", []):
                details = item.get("details", {})
                models.append(
                    ModelInfo(
                        name=item.get("name", "unknown"),
                        provider=ModelProvider.OLLAMA,
                        status=ModelStatus.AVAILABLE,
                        size=item.get("size"),
                        modified_at=item.get("modified_at"),
                        parameters=details.get("parameter_size") if details else None,
                        quantization=details.get("quantization_level") if details else None,
                    )
                )
            return models
        except (httpx.HTTPError, json.JSONDecodeError):
            return []

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """获取指定模型的详细信息"""
        models = self.list_models()
        for model in models:
            if model.name == model_name or model.name.startswith(f"{model_name}:"):
                return model
        return None

    def pull_model(self, model_name: str, stream: bool = False) -> Union[bool, Iterator[dict]]:
        """拉取模型

        Args:
            model_name: 模型名称
            stream: 是否流式返回进度

        Returns:
            stream=False 时返回是否成功
            stream=True 时返回进度迭代器
        """
        if stream:
            return self._pull_model_stream(model_name)

        try:
            response = self._get_client().post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=300.0,
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def _pull_model_stream(self, model_name: str) -> Iterator[dict]:
        """流式拉取模型"""
        with self._get_client().stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"name": model_name, "stream": True},
            timeout=300.0,
        ) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OllamaDetector":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class LLMEngine:
    """LLM 引擎 - 本地模型支持与多模型管理

    核心功能:
    - 自动检测本地 Ollama 服务
    - 支持 Anthropic 兼容 API (DeepSeek 等)
    - 流式与非流式生成
    - 多模型切换
    - 容错处理

    示例:
        # 最简单的用法 - 自动检测 Ollama
        engine = LLMEngine()
        result = engine.generate("你好!")
        print(result.content)

        # 使用 DeepSeek API
        config = LLMConfig(
            provider=ModelProvider.DEEPSEEK,
            base_url="https://api.deepseek.com/anthropic",
            api_key="sk-xxx",
            model="deepseek-v4-flash",
        )
        engine = LLMEngine(config)
        result = engine.generate("你好!")

        # 流式输出
        for chunk in engine.stream("讲个故事"):
            print(chunk, end="", flush=True)

        # 多轮对话
        messages = [
            ChatMessage(role="user", content="你好"),
            ChatMessage(role="assistant", content="你好!有什么可以帮你的?"),
            ChatMessage(role="user", content="介绍一下你自己"),
        ]
        result = engine.chat(messages)

        # 切换模型
        engine.switch_model("mistral")

        # 自定义配置
        config = LLMConfig(
            model="llama3.2",
            temperature=0.5,
            max_tokens=1024,
        )
        engine = LLMEngine(config)
    """

    # 模型名称安全字符：字母、数字、点、下划线、连字符
    _MODEL_NAME_PATTERN = r'^[a-zA-Z0-9._-]+$'

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._detector = OllamaDetector(self.config.base_url)
        self._client: Optional[httpx.Client] = None
        self._current_model = self._validate_model_name(self.config.model)
        self._model_cache: dict[str, ModelInfo] = {}

    def _get_client(self) -> httpx.Client:
        """获取或创建 HTTP 客户端"""
        if self._client is None:
            self._client = httpx.Client(timeout=self.config.timeout)
        return self._client

    def _get_anthropic_headers(self) -> dict[str, str]:
        """获取 Anthropic 兼容 API 请求头"""
        return {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key or "",
            "anthropic-version": ANTHROPIC_API_VERSION,
        }

    def _build_anthropic_payload(
        self,
        messages: list[dict[str, str]],
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """构建 Anthropic 兼容 API 请求体"""
        payload: dict[str, Any] = {
            "model": self._current_model,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": messages,
            "stream": stream,
        }

        # 系统提示词 (Anthropic 格式使用独立的 system 字段)
        system = system_prompt or self.config.system_prompt
        if system:
            payload["system"] = system

        # Anthropic 支持的参数
        if "temperature" in kwargs or self.config.temperature != 0.7:
            payload["temperature"] = kwargs.get("temperature", self.config.temperature)
        if "top_p" in kwargs or self.config.top_p != 0.9:
            payload["top_p"] = kwargs.get("top_p", self.config.top_p)
        if "top_k" in kwargs or self.config.top_k != 40:
            payload["top_k"] = kwargs.get("top_k", self.config.top_k)

        return payload

    def _parse_anthropic_response(self, data: dict[str, Any]) -> GenerateResult:
        """解析 Anthropic 兼容 API 响应

        DeepSeek/Anthropic 返回的内容块格式:
        - {"type": "thinking", "thinking": "..."}  # 思考过程
        - {"type": "text", "text": "..."}          # 实际输出文本

        默认只返回 text 块内容，thinking 内容存入 metadata。
        """
        content_blocks = data.get("content", [])
        text_content = ""
        thinking_content = ""

        for block in content_blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                text_content += block.get("text", "")
            elif block_type == "thinking":
                # DeepSeek 的思考过程
                thinking_content += block.get("thinking", "")

        usage = data.get("usage", {})

        return GenerateResult(
            content=text_content,
            model=data.get("model", self._current_model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            finish_reason=data.get("stop_reason", "stop"),
            metadata={
                "id": data.get("id"),
                "type": data.get("type"),
                "thinking": thinking_content,  # 存储思考过程
            },
        )

    def _validate_model_name(self, model_name: str) -> str:
        """验证模型名称，防止注入攻击

        Args:
            model_name: 模型名称

        Returns:
            验证后的模型名称

        Raises:
            ValueError: 模型名称包含非法字符
        """
        if not re.match(self._MODEL_NAME_PATTERN, model_name):
            raise ValueError(
                f"模型名称包含非法字符。"
                f"只允许字母、数字、点(.)、下划线(_)、连字符(-)。"
            )
        return model_name

    @property
    def current_model(self) -> str:
        """当前使用的模型名称"""
        return self._current_model

    @property
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self._detector.is_running()

    def list_models(self) -> list[ModelInfo]:
        """列出所有可用模型"""
        return self._detector.list_models()

    def switch_model(self, model_name: str) -> bool:
        """切换模型

        Args:
            model_name: 目标模型名称

        Returns:
            是否切换成功
        """
        # 先验证模型名称
        validated_name = self._validate_model_name(model_name)

        if not self.is_available:
            raise RuntimeError(
                "Ollama 服务未运行。请先启动 Ollama:\n"
                "  1. 安装: https://ollama.ai\n"
                "  2. 运行: ollama serve\n"
                "  3. 拉取模型: ollama pull <模型名>"
            )

        model_info = self._detector.get_model_info(validated_name)
        if model_info is None:
            available = [m.name for m in self.list_models()]
            if available:
                raise ValueError(
                    f"请求的模型不存在。\n"
                    f"可用模型: {', '.join(available)}\n"
                    f"请使用 'ollama pull' 拉取新模型。"
                )
            else:
                raise ValueError(
                    "请求的模型不存在，且本地无可用模型。\n"
                    "请使用 'ollama pull' 拉取模型。"
                )

        self._current_model = validated_name
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerateResult:
        """生成文本（非流式）

        Args:
            prompt: 输入提示词
            system_prompt: 系统提示词（可选，覆盖默认配置）
            **kwargs: 其他生成参数

        Returns:
            生成结果
        """
        self._ensure_available()

        # Anthropic 兼容 API 路径
        if self.config.is_anthropic_compatible():
            return self._generate_anthropic(prompt, system_prompt, **kwargs)

        # Ollama 路径
        payload = self._build_payload(prompt, system_prompt, stream=False, **kwargs)

        try:
            response = self._get_client().post(
                f"{self.config.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return GenerateResult(
                content=data.get("response", ""),
                model=data.get("model", self._current_model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                finish_reason="stop" if data.get("done") else "length",
                metadata={
                    "context": data.get("context"),
                    "created_at": data.get("created_at"),
                },
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"生成请求失败: {e.response.status_code} - {e.response.text}") from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 Ollama 服务 ({self.config.base_url})\n"
                f"请确保 Ollama 正在运行: ollama serve"
            ) from e

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> GenerateResult:
        """使用 Anthropic 兼容 API 生成文本"""
        messages = [{"role": "user", "content": prompt}]
        payload = self._build_anthropic_payload(messages, system_prompt, stream=False, **kwargs)

        try:
            response = self._get_client().post(
                f"{self.config.base_url}/v1/messages",
                json=payload,
                headers=self._get_anthropic_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_anthropic_response(data)
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            raise RuntimeError(
                f"Anthropic API 请求失败: {e.response.status_code}\n"
                f"响应: {error_body}"
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 API 服务 ({self.config.base_url})\n"
                f"请检查网络连接和 API 地址配置"
            ) from e

    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式生成文本

        Args:
            prompt: 输入提示词
            system_prompt: 系统提示词（可选）
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        self._ensure_available()

        # Anthropic 兼容 API 路径
        if self.config.is_anthropic_compatible():
            yield from self._stream_anthropic(prompt, system_prompt, **kwargs)
            return

        # Ollama 路径
        payload = self._build_payload(prompt, system_prompt, stream=True, **kwargs)

        try:
            with self._get_client().stream(
                "POST",
                f"{self.config.base_url}/api/generate",
                json=payload,
            ) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 Ollama 服务 ({self.config.base_url})\n"
                f"请确保 Ollama 正在运行: ollama serve"
            ) from e

    def _stream_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """使用 Anthropic 兼容 API 流式生成文本

        DeepSeek 流式事件:
        - content_block_start: 标记块类型 (thinking/text)
        - content_block_delta: 增量内容 (thinking_delta/text_delta)
        """
        messages = [{"role": "user", "content": prompt}]
        payload = self._build_anthropic_payload(messages, system_prompt, stream=True, **kwargs)

        try:
            with self._get_client().stream(
                "POST",
                f"{self.config.base_url}/v1/messages",
                json=payload,
                headers=self._get_anthropic_headers(),
            ) as response:
                current_block_type = None  # 跟踪当前块类型

                for line in response.iter_lines():
                    if line:
                        # 处理 SSE 事件
                        if line.startswith("event:"):
                            # 可以根据事件类型做处理，但主要通过 data 判断
                            continue
                        elif line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                event_type = data.get("type", "")

                                if event_type == "content_block_start":
                                    # 记录当前块类型
                                    block = data.get("content_block", {})
                                    current_block_type = block.get("type", "")

                                elif event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    delta_type = delta.get("type", "")

                                    # 只输出 text_delta，跳过 thinking_delta
                                    if delta_type == "text_delta":
                                        yield delta.get("text", "")

                                elif event_type == "message_stop":
                                    break

                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 API 服务 ({self.config.base_url})\n"
                f"请检查网络连接和 API 地址配置"
            ) from e

    def chat(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> GenerateResult:
        """多轮对话

        Args:
            messages: 对话消息列表
            **kwargs: 其他生成参数

        Returns:
            生成结果
        """
        self._ensure_available()

        # Anthropic 兼容 API 路径
        if self.config.is_anthropic_compatible():
            return self._chat_anthropic(messages, **kwargs)

        # Ollama 路径
        payload = {
            "model": self._current_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
            },
        }

        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed

        try:
            response = self._get_client().post(
                f"{self.config.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            message = data.get("message", {})

            return GenerateResult(
                content=message.get("content", ""),
                model=data.get("model", self._current_model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                finish_reason="stop",
                metadata={
                    "done": data.get("done"),
                },
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"对话请求失败: {e.response.status_code}") from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 Ollama 服务 ({self.config.base_url})\n"
                f"请确保 Ollama 正在运行: ollama serve"
            ) from e

    def _chat_anthropic(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> GenerateResult:
        """使用 Anthropic 兼容 API 进行多轮对话"""
        msg_list = [{"role": m.role, "content": m.content} for m in messages]
        payload = self._build_anthropic_payload(msg_list, stream=False, **kwargs)

        try:
            response = self._get_client().post(
                f"{self.config.base_url}/v1/messages",
                json=payload,
                headers=self._get_anthropic_headers(),
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_anthropic_response(data)
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            raise RuntimeError(
                f"Anthropic API 请求失败: {e.response.status_code}\n"
                f"响应: {error_body}"
            ) from e
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 API 服务 ({self.config.base_url})\n"
                f"请检查网络连接和 API 地址配置"
            ) from e

    def chat_stream(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Iterator[str]:
        """流式多轮对话

        Args:
            messages: 对话消息列表
            **kwargs: 其他生成参数

        Yields:
            生成的文本片段
        """
        self._ensure_available()

        # Anthropic 兼容 API 路径
        if self.config.is_anthropic_compatible():
            yield from self._chat_stream_anthropic(messages, **kwargs)
            return

        # Ollama 路径
        payload = {
            "model": self._current_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
            },
        }

        try:
            with self._get_client().stream(
                "POST",
                f"{self.config.base_url}/api/chat",
                json=payload,
            ) as response:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            message = data.get("message", {})
                            if "content" in message:
                                yield message["content"]
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 Ollama 服务 ({self.config.base_url})\n"
                f"请确保 Ollama 正在运行: ollama serve"
            ) from e

    def _chat_stream_anthropic(
        self,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> Iterator[str]:
        """使用 Anthropic 兼容 API 进行流式多轮对话"""
        msg_list = [{"role": m.role, "content": m.content} for m in messages]
        payload = self._build_anthropic_payload(msg_list, stream=True, **kwargs)

        try:
            with self._get_client().stream(
                "POST",
                f"{self.config.base_url}/v1/messages",
                json=payload,
                headers=self._get_anthropic_headers(),
            ) as response:
                for line in response.iter_lines():
                    if line:
                        # 处理 SSE 事件
                        if line.startswith("event:"):
                            continue
                        elif line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                event_type = data.get("type", "")

                                if event_type == "content_block_start":
                                    # 块开始，可用于跟踪类型
                                    continue
                                elif event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    delta_type = delta.get("type", "")
                                    # 只输出 text_delta，跳过 thinking_delta
                                    if delta_type == "text_delta":
                                        yield delta.get("text", "")
                                elif event_type == "message_stop":
                                    break
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError as e:
            raise RuntimeError(
                f"无法连接到 API 服务 ({self.config.base_url})\n"
                f"请检查网络连接和 API 地址配置"
            ) from e

    def embeddings(self, text: Union[str, list[str]]) -> list[list[float]]:
        """获取文本嵌入向量

        支持 Ollama 和云端 API（OpenAI/DeepSeek 等）。

        Args:
            text: 单个文本或文本列表

        Returns:
            嵌入向量列表
        """
        self._ensure_available()

        texts = [text] if isinstance(text, str) else text

        # Anthropic 兼容 API 路径
        if self.config.is_anthropic_compatible():
            return self._embeddings_anthropic(texts)

        # Ollama 路径
        results = []
        for t in texts:
            response = self._get_client().post(
                f"{self.config.base_url}/api/embeddings",
                json={"model": self._current_model, "prompt": t},
            )
            response.raise_for_status()
            data = response.json()
            results.append(data.get("embedding", []))

        return results

    def _embeddings_anthropic(self, texts: list[str]) -> list[list[float]]:
        """使用云端 API 获取嵌入向量（OpenAI 格式）"""
        results = []
        for t in texts:
            response = self._get_client().post(
                f"{self.config.base_url}/v1/embeddings",
                json={"input": t, "model": self._current_model},
                headers=self._get_anthropic_headers(),
            )
            response.raise_for_status()
            data = response.json()
            # OpenAI 格式: {"data": [{"embedding": [...]}]}
            embedding_data = data.get("data", [])
            if embedding_data:
                results.append(embedding_data[0].get("embedding", []))
            else:
                results.append([])

        return results

    def _build_payload(
        self,
        prompt: str,
        system_prompt: Optional[str],
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """构建请求载荷"""
        payload: dict[str, Any] = {
            "model": self._current_model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "repeat_penalty": kwargs.get("repeat_penalty", self.config.repeat_penalty),
            },
        }

        # 系统提示词
        system = system_prompt or self.config.system_prompt
        if system:
            payload["system"] = system

        # 随机种子
        if self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed

        return payload

    def _ensure_available(self) -> None:
        """确保服务可用"""
        # Anthropic 兼容 API 只需要验证 API Key
        if self.config.is_anthropic_compatible():
            if not self.config.api_key:
                raise RuntimeError(
                    "Anthropic 兼容 API 需要配置 API 密钥。\n\n"
                    "请通过以下方式之一提供密钥:\n"
                    "  1. 环境变量: VERTAI_API_KEY 或 ANTHROPIC_API_KEY\n"
                    "  2. 配置参数: LLMConfig(api_key='sk-xxx')\n"
                )
            return

        # Ollama 需要检测服务是否运行
        if not self.is_available:
            raise RuntimeError(
                "Ollama 服务未运行。\n\n"
                "请按以下步骤操作:\n"
                "  1. 安装 Ollama: https://ollama.ai\n"
                "  2. 启动服务: ollama serve\n"
                "  3. 拉取模型: ollama pull <模型名>\n\n"
                "请检查服务地址配置。"
            )

    def close(self) -> None:
        """关闭引擎"""
        if self._client is not None:
            self._client.close()
            self._client = None
        self._detector.close()

    def __enter__(self) -> "LLMEngine":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
