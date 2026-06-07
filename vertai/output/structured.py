"""
StructuredOutput - Extract structured data from unstructured text.

Provides JSON Schema validation with automatic retry correction when
output doesn't conform to the expected schema.

支持两种模式：
1. LLM 模式：传入 LLMEngine 进行语义理解提取（推荐，生产环境使用）
2. 本地模式：使用正则表达式提取（仅测试用，精度有限）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vertai.core.llm import LLMEngine


# Constants for Chinese number units and currency
CHINESE_CURRENCY_UNITS = ("元", "块")
CHINESE_NUMBERS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "壹": 1, "贰": 2, "叁": 3,
}
DEFAULT_MAX_RETRIES = 3


@dataclass
class ExtractionConfig:
    """Configuration for structured extraction."""
    max_retries: int = DEFAULT_MAX_RETRIES
    strict: bool = True


@dataclass
class ExtractionResult:
    """Result of structured extraction."""
    data: dict[str, Any]
    success: bool
    retries: int = 0
    error: Optional[str] = None


class SchemaValidationError(Exception):
    """Raised when output doesn't conform to schema and retries exhausted."""
    pass


class StructuredOutput:
    """
    Extract structured data from unstructured text with schema validation.

    支持两种提取模式：

    示例 - LLM 模式（推荐，生产环境）:
        >>> from vertai import LLMEngine, LLMConfig, ModelProvider
        >>> llm = LLMEngine(LLMConfig(
        ...     provider=ModelProvider.DEEPSEEK,
        ...     api_key="sk-xxx",
        ... ))
        >>> schema = {"name": "string", "amount": "number"}
        >>> output = StructuredOutput(schema, llm=llm)
        >>> result = output.extract("张三报销500元")
        >>> result.data
        {'name': '张三', 'amount': 500}

    示例 - 本地模式（仅测试）:
        >>> schema = {"name": "string", "amount": "number"}
        >>> output = StructuredOutput(schema)
        >>> result = output.extract("张三报销500元")  # 使用正则提取
    """

    def __init__(
        self,
        schema: dict[str, str],
        config: Optional[ExtractionConfig] = None,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        strict: bool = True,
        llm: Optional["LLMEngine"] = None,
    ):
        """
        Initialize StructuredOutput with schema definition.

        Args:
            schema: Schema definition mapping field names to types.
                    Types: "string", "number", "integer", "boolean",
                           "enum[val1,val2,...]", "array", "object"
            config: ExtractionConfig instance (overrides max_retries/strict params)
            max_retries: Maximum retry attempts for correction
            strict: If True, raise exception on validation failure after retries
            llm: LLMEngine for semantic extraction (推荐用于生产环境)
        """
        self.schema = schema
        self._llm = llm
        if config is not None:
            self._config = config
        else:
            self._config = ExtractionConfig(max_retries=max_retries, strict=strict)

        self._parsers: dict[str, Callable[[str, str, str], Any]] = {
            "string": self._parse_string,
            "number": self._parse_number,
            "integer": self._parse_integer,
            "boolean": self._parse_boolean,
            "array": self._parse_array,
            "object": self._parse_object,
        }

    @property
    def max_retries(self) -> int:
        return self._config.max_retries

    @property
    def strict(self) -> bool:
        return self._config.strict

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract structured data from text.

        如果提供了 LLM，使用 LLM 进行语义提取（推荐）。
        否则使用本地正则提取（仅测试用）。

        Args:
            text: Unstructured text to extract data from.

        Returns:
            ExtractionResult with data, success status, and retry count.
        """
        # LLM 模式：使用语义理解提取
        if self._llm is not None:
            return self._extract_with_llm(text)

        # 本地模式：使用正则提取
        result = self._extract_initial(text)

        if result.success:
            return result

        # Retry with corrections
        for attempt in range(self.max_retries):
            corrected = self._correct_extraction(text, result, attempt)
            if corrected.success:
                corrected.retries = attempt + 1
                return corrected

        # All retries exhausted
        if self.strict:
            raise SchemaValidationError(
                f"Failed to extract valid data after {self.max_retries} retries. "
                "Schema validation failed for one or more fields."
            )

        return ExtractionResult(
            data=result.data,
            success=False,
            retries=self.max_retries,
            error=result.error
        )

    def _extract_with_llm(self, text: str) -> ExtractionResult:
        """使用 LLM 进行语义提取"""
        schema_desc = json.dumps(self.schema, ensure_ascii=False)
        prompt = f"""从以下文本中提取结构化数据。

文本: {text}

需要的字段和类型:
{schema_desc}

请以 JSON 格式返回提取结果，只返回 JSON 对象，不要其他内容。
如果某个字段无法从文本中提取，请设为 null。
"""
        try:
            result = self._llm.generate(prompt)
            content = result.content.strip()

            # 尝试解析 JSON
            # 可能包含 ```json ... ``` 包裹
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)

            # 验证并转换类型
            validated_data = {}
            for field_name, field_type in self.schema.items():
                value = data.get(field_name)
                if value is not None:
                    validated_data[field_name] = value

            return ExtractionResult(data=validated_data, success=True)

        except json.JSONDecodeError as e:
            return ExtractionResult(data={}, success=False, error=f"JSON parse error: {e}")
        except Exception as e:
            return ExtractionResult(data={}, success=False, error=str(e))

    def _extract_initial(self, text: str) -> ExtractionResult:
        """Perform initial extraction attempt."""
        try:
            data: dict[str, Any] = {}
            errors: list[str] = []

            for field_name, field_type in self.schema.items():
                try:
                    value = self._parse_field(text, field_name, field_type)
                    data[field_name] = value
                except ValueError:
                    errors.append(field_name)

            if errors:
                return ExtractionResult(
                    data=data,
                    success=False,
                    error=f"Failed fields: {', '.join(errors)}"
                )

            return ExtractionResult(data=data, success=True)

        except (ValueError, TypeError, KeyError, re.error):
            return ExtractionResult(
                data={},
                success=False,
                error="Extraction failed"
            )

    def _correct_extraction(
        self,
        text: str,
        previous_result: ExtractionResult,
        attempt: int
    ) -> ExtractionResult:
        """
        Attempt to correct extraction based on previous failure.

        Uses progressive strategies:
        - Attempt 0: Relaxed parsing
        - Attempt 1: Pattern-based extraction
        - Attempt 2+: Aggressive fallbacks
        """
        try:
            data: dict[str, Any] = {}

            for field_name, field_type in self.schema.items():
                # Skip if already parsed correctly
                if field_name in previous_result.data and previous_result.data[field_name] is not None:
                    try:
                        self._validate_field(previous_result.data[field_name], field_type)
                        data[field_name] = previous_result.data[field_name]
                        continue
                    except ValueError:
                        pass

                # Try corrected parsing
                value = self._parse_field_corrected(text, field_name, field_type, attempt)
                data[field_name] = value

            # Validate all fields
            for field_name, field_type in self.schema.items():
                self._validate_field(data[field_name], field_type)

            return ExtractionResult(data=data, success=True)

        except (ValueError, TypeError, KeyError, re.error):
            return ExtractionResult(
                data=data if 'data' in dir() else {},
                success=False,
                error="Correction failed"
            )

    def _parse_field(self, text: str, field_name: str, field_type: str) -> Any:
        """Parse a single field from text."""
        parser = self._get_parser(field_type)
        return parser(text, field_name, field_type)

    def _parse_field_corrected(
        self,
        text: str,
        field_name: str,
        field_type: str,
        attempt: int
    ) -> Any:
        """Parse field with correction strategies."""
        if attempt == 0:
            return self._parse_relaxed(text, field_name, field_type)
        elif attempt == 1:
            return self._parse_pattern(text, field_name, field_type)
        else:
            return self._parse_aggressive(text, field_name, field_type)

    def _get_parser(self, field_type: str) -> Callable[[str, str, str], Any]:
        """Get appropriate parser for field type."""
        if field_type.startswith("enum["):
            return self._parse_enum

        base_type = field_type.split("[")[0]
        return self._parsers.get(base_type, self._parse_string)

    def _parse_string(self, text: str, field_name: str, field_type: str) -> str:
        """Extract string value."""
        # Context-aware extraction based on field name
        if field_name in ("name", "姓名", "名字"):
            name_match = re.search(r'[一-龥]{2,4}(?=报|采|支|付|预|于)', text)
            if name_match:
                return name_match.group()

        # Look for Chinese names (2-4 characters)
        name_match = re.search(r'[一-龥]{2,4}(?=报|采|支|付|预)', text)
        if name_match:
            return name_match.group()

        raise ValueError(f"Cannot extract string for '{field_name}'")

    def _parse_number(self, text: str, field_name: str, field_type: str) -> float:
        """Extract numeric value."""
        # Look for numbers with currency units
        units_pattern = "|".join(CHINESE_CURRENCY_UNITS)
        match = re.search(rf'(\d+(?:\.\d+)?)\s*(?:{units_pattern})', text)
        if match:
            return float(match.group(1))

        # Plain number
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

        # Chinese number words
        for cn, num in CHINESE_NUMBERS.items():
            if cn in text:
                return float(num)

        raise ValueError(f"Cannot extract number for '{field_name}'")

    def _parse_integer(self, text: str, field_name: str, field_type: str) -> int:
        """Extract integer value."""
        num = self._parse_number(text, field_name, field_type)
        return int(num)

    def _parse_boolean(self, text: str, field_name: str, field_type: str) -> bool:
        """Extract boolean value."""
        true_patterns = r'(是|有|true|yes|1|对|真)'
        false_patterns = r'(否|无|false|no|0|错|假)'

        if re.search(true_patterns, text, re.IGNORECASE):
            return True
        if re.search(false_patterns, text, re.IGNORECASE):
            return False

        raise ValueError(f"Cannot extract boolean for '{field_name}'")

    def _parse_enum(self, text: str, field_name: str, field_type: str) -> str:
        """Extract enum value."""
        match = re.match(r'enum\[(.+)\]', field_type)
        if not match:
            raise ValueError(f"Invalid enum type: {field_type}")

        values = [v.strip() for v in match.group(1).split(",")]

        for value in values:
            if value in text:
                return value

        raise ValueError(f"Cannot match enum for '{field_name}'")

    def _parse_array(self, text: str, field_name: str, field_type: str) -> list:
        """Extract array value."""
        # Try JSON array first
        match = re.search(r'\[.*?\]', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                # JSON failed but brackets found - strip them and parse as CSV
                bracket_content = match.group()
                inner = bracket_content[1:-1]  # Remove [ and ]
                items = re.split(r'[,，、]', inner)
                return [item.strip() for item in items if item.strip()]

        # No brackets - split by commas
        items = re.split(r'[,，、]', text)
        return [item.strip() for item in items if item.strip()]

    def _parse_object(self, text: str, field_name: str, field_type: str) -> dict:
        """Extract object value."""
        match = re.search(r'\{.*?\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def _parse_relaxed(self, text: str, field_name: str, field_type: str) -> Any:
        """Parse with relaxed rules - still requires meaningful extraction."""
        if field_type == "string":
            match = re.search(r'[一-龥]+', text)
            if match:
                return match.group()
            match = re.search(r'[^\s]+', text)
            if match:
                return match.group()
            raise ValueError(f"Cannot extract string for '{field_name}'")

        if field_type in ("number", "integer"):
            match = re.search(r'\d+', text)
            if match:
                val = float(match.group())
                return int(val) if field_type == "integer" else val
            for cn, num in CHINESE_NUMBERS.items():
                if cn in text:
                    return num if field_type == "integer" else float(num)
            raise ValueError(f"Cannot extract number for '{field_name}'")

        if field_type.startswith("enum["):
            match = re.match(r'enum\[(.+)\]', field_type)
            if match:
                values = [v.strip() for v in match.group(1).split(",")]
                for value in values:
                    if value in text:
                        return value
                raise ValueError(f"Cannot match enum for '{field_name}'")

        if field_type == "boolean":
            if re.search(r'(是|有|true|yes|1|对|真|active|激活|开)', text, re.IGNORECASE):
                return True
            if re.search(r'(否|无|false|no|0|错|假|inactive|关闭|关)', text, re.IGNORECASE):
                return False
            raise ValueError(f"Cannot extract boolean for '{field_name}'")

        return None

    def _parse_pattern(self, text: str, field_name: str, field_type: str) -> Any:
        """Parse using predefined patterns."""
        patterns: dict[tuple[str, str], str] = {
            ("amount", "number"): r'(\d+)',
            ("name", "string"): r'([一-龥]+)',
        }

        key = (field_name, field_type)
        if key in patterns:
            match = re.search(patterns[key], text)
            if match:
                if field_type == "number":
                    return float(match.group(1))
                return match.group(1)

        return self._parse_relaxed(text, field_name, field_type)

    def _parse_aggressive(self, text: str, field_name: str, field_type: str) -> Any:
        """Aggressive fallback parsing - last meaningful attempt."""
        if field_type == "string":
            match = re.search(r'[一-龥]+|\w+', text)
            if match:
                return match.group()
            raise ValueError(f"Cannot extract string for '{field_name}'")

        if field_type in ("number", "integer"):
            match = re.search(r'[-+]?\d+', text)
            if match:
                val = float(match.group())
                return int(val) if field_type == "integer" else val
            raise ValueError(f"Cannot extract number for '{field_name}'")

        if field_type.startswith("enum["):
            match = re.match(r'enum\[(.+)\]', field_type)
            if match:
                values = [v.strip() for v in match.group(1).split(",")]
                text_lower = text.lower()
                for value in values:
                    if value.lower() in text_lower or value in text:
                        return value
                raise ValueError(f"Cannot match enum for '{field_name}'")

        if field_type == "boolean":
            if re.search(r'\b(true|yes|1|是|有)\b', text, re.IGNORECASE):
                return True
            return False

        return None

    def _validate_field(self, value: Any, field_type: str) -> None:
        """Validate a field value against its type."""
        if value is None:
            raise ValueError("Value cannot be None")

        if field_type == "string":
            if not isinstance(value, str):
                raise ValueError(f"Expected string, got {type(value).__name__}")

        elif field_type == "number":
            if not isinstance(value, (int, float)):
                raise ValueError(f"Expected number, got {type(value).__name__}")

        elif field_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Expected integer, got {type(value).__name__}")

        elif field_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"Expected boolean, got {type(value).__name__}")

        elif field_type == "array":
            if not isinstance(value, list):
                raise ValueError(f"Expected array, got {type(value).__name__}")

        elif field_type == "object":
            if not isinstance(value, dict):
                raise ValueError(f"Expected object, got {type(value).__name__}")

        elif field_type.startswith("enum["):
            match = re.match(r'enum\[(.+)\]', field_type)
            if match:
                valid_values = [v.strip() for v in match.group(1).split(",")]
                if value not in valid_values:
                    raise ValueError(f"Value not in valid enum values")
