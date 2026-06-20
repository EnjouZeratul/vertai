"""StructuredOutput - extract structured data from unstructured text.

Provides JSON-Schema-style validation with automatic retry correction when the
extracted output does not conform to the expected schema.

Two extraction modes are supported:

1. LLM mode: pass an ``LLMEngine`` for semantic understanding extraction
   (recommended for production use). The LLM output is schema-validated, with
   retry on validation failure.
2. Local mode: pure regex extraction. This is only suitable for testing and
   simple inputs; precision is limited.

The local ``string`` parser is intentionally *generic* (returns the first
meaningful token). A small number of field names (``name`` / ``姓名`` /
``名字``) trigger a documented *domain-example heuristic* that recognises the
common Chinese reimbursement/purchase phrasing ``<name>报销|采购|...``. This
heuristic is not part of the general contract; rely on the LLM mode for
robust extraction.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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

# Field names that trigger the documented Chinese-name domain-example
# heuristic. This is NOT a generic contract — it only recognises the
# "<name>报销|采购|支付|预|于" reimbursement/purchase phrasing. Generic string
# extraction is used for every other field name.
_CHINESE_NAME_FIELD_HINTS = frozenset({"name", "姓名", "名字", "buyer"})
_CHINESE_NAME_CONTEXT_PATTERN = re.compile(r"[一-龥]{2,4}(?=报|采|支|付|预|于)")


@dataclass
class ExtractionConfig:
    """Configuration for structured extraction."""
    max_retries: int = DEFAULT_MAX_RETRIES
    strict: bool = True


@dataclass
class ExtractionResult:
    """Result of structured extraction.

    Attributes:
        data: Extracted key-value mapping (empty on failure).
        success: Whether schema validation passed for all fields.
        retries: Number of correction retries performed.
        error: Optional human-readable error description.
    """
    data: dict[str, Any]
    success: bool
    retries: int = 0
    error: Optional[str] = None


class SchemaValidationError(Exception):
    """Raised when output doesn't conform to schema and retries exhausted."""
    pass


class StructuredOutput:
    """Extract structured data from unstructured text with schema validation.

    Examples - LLM mode (recommended for production):

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

    Examples - local mode (testing only):

        >>> schema = {"amount": "number"}
        >>> output = StructuredOutput(schema)
        >>> result = output.extract("金额500元")  # regex extraction
    """

    def __init__(
        self,
        schema: dict[str, str],
        config: Optional[ExtractionConfig] = None,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        strict: bool = True,
        llm: Optional["LLMEngine"] = None,
    ) -> None:
        """Initialize StructuredOutput with schema definition.

        Args:
            schema: Schema definition mapping field names to types.
                    Types: "string", "number", "integer", "boolean",
                           "enum[val1,val2,...]", "array", "object"
            config: ExtractionConfig instance (overrides max_retries/strict).
            max_retries: Maximum retry attempts for correction.
            strict: If True, raise on validation failure after retries.
            llm: LLMEngine for semantic extraction (recommended for production).
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
        """Extract structured data from text.

        Uses the LLM for semantic extraction when one was provided; otherwise
        falls back to local regex extraction (testing only).

        Args:
            text: Unstructured text to extract data from.

        Returns:
            ExtractionResult with data, success status, and retry count.

        Raises:
            SchemaValidationError: In strict mode when validation fails after
                all retries.
        """
        # LLM mode: semantic extraction with schema validation + retry.
        if self._llm is not None:
            return self._extract_with_llm(text)

        # Local mode: regex extraction.
        result = self._extract_initial(text)

        if result.success:
            return result

        # Retry with corrections.
        for attempt in range(self.max_retries):
            corrected = self._correct_extraction(text, result, attempt)
            if corrected.success:
                corrected.retries = attempt + 1
                return corrected

        # All retries exhausted.
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
        """Extract using the LLM, then schema-validate with retry.

        The raw LLM JSON is validated field-by-field via ``_validate_field``.
        Invalid fields are dropped and the LLM is re-prompted with the
        validation errors, up to ``max_retries`` times.
        """
        llm = self._llm
        assert llm is not None  # narrowed by caller

        schema_desc = json.dumps(self.schema, ensure_ascii=False)
        prompt = (
            "Extract structured data from the following text.\n\n"
            f"Text: {text}\n\n"
            f"Required fields and types:\n{schema_desc}\n\n"
            "Return only a JSON object. If a field cannot be extracted, set it "
            "to null."
        )

        last_error: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                result = llm.generate(prompt)
                content = result.content.strip()

                # Strip optional ```json ... ``` fences.
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                data: dict[str, Any] = json.loads(content)
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                continue
            except Exception as e:  # network / provider error
                last_error = str(e)
                continue

            # Schema-validate every field. Collect failures so we can
            # re-prompt the LLM with actionable feedback.
            validation_errors: list[str] = []
            validated_data: dict[str, Any] = {}
            for field_name, field_type in self.schema.items():
                value = data.get(field_name)
                if value is None:
                    validation_errors.append(
                        f"{field_name}: missing or null"
                    )
                    continue
                try:
                    self._validate_field(value, field_type)
                    validated_data[field_name] = value
                except ValueError as e:
                    validation_errors.append(f"{field_name}: {e}")

            if not validation_errors:
                return ExtractionResult(
                    data=validated_data,
                    success=True,
                    retries=attempt,
                )

            last_error = "; ".join(validation_errors)
            prompt = (
                "Extract structured data from the following text.\n\n"
                f"Text: {text}\n\n"
                f"Required fields and types:\n{schema_desc}\n\n"
                "Return only a JSON object. If a field cannot be extracted, "
                "set it to null.\n\n"
                f"Your previous answer had these validation errors:\n"
                f"{last_error}\n\nPlease correct them."
            )

        # Exhausted retries.
        if self.strict:
            raise SchemaValidationError(
                f"LLM extraction failed schema validation after "
                f"{self.max_retries} retries: {last_error}"
            )
        return ExtractionResult(
            data={},
            success=False,
            retries=self.max_retries,
            error=last_error,
        )

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
        """Attempt to correct extraction based on previous failure.

        Progressive strategies:
        - Attempt 0: Relaxed parsing
        - Attempt 1: Pattern-based extraction
        - Attempt 2+: Aggressive fallbacks
        """
        # Pre-initialise so the except branch always has a defined binding
        # (replaces the previous ``data if 'data' in dir() else {}`` introspection).
        data: dict[str, Any] = {}
        try:
            for field_name, field_type in self.schema.items():
                # Skip if already parsed correctly.
                if field_name in previous_result.data and previous_result.data[field_name] is not None:
                    try:
                        self._validate_field(previous_result.data[field_name], field_type)
                        data[field_name] = previous_result.data[field_name]
                        continue
                    except ValueError:
                        pass

                # Try corrected parsing.
                value = self._parse_field_corrected(text, field_name, field_type, attempt)
                data[field_name] = value

            # Validate all fields.
            for field_name, field_type in self.schema.items():
                self._validate_field(data[field_name], field_type)

            return ExtractionResult(data=data, success=True)

        except (ValueError, TypeError, KeyError, re.error):
            return ExtractionResult(
                data=data,
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
        """Extract a string value (generic).

        For the small set of field names in ``_CHINESE_NAME_FIELD_HINTS`` we
        apply a documented domain-example heuristic that recognises the common
        Chinese ``<name>报销|采购|支付|预|于`` reimbursement/purchase phrasing.
        For all other field names we return the first non-whitespace token,
        which is a deliberately weak default — use the LLM mode for robust
        extraction.
        """
        # Domain-example heuristic (NOT part of the general contract).
        if field_name in _CHINESE_NAME_FIELD_HINTS:
            name_match = _CHINESE_NAME_CONTEXT_PATTERN.search(text)
            if name_match:
                return name_match.group()

        # Generic fallback: first non-whitespace token (works for both CJK
        # runs and ASCII words).
        match = re.search(r"\S+", text)
        if match:
            return match.group()

        raise ValueError(f"Cannot extract string for '{field_name}'")

    def _parse_number(self, text: str, field_name: str, field_type: str) -> float:
        """Extract numeric value."""
        # Look for numbers with currency units.
        units_pattern = "|".join(CHINESE_CURRENCY_UNITS)
        match = re.search(rf'(\d+(?:\.\d+)?)\s*(?:{units_pattern})', text)
        if match:
            return float(match.group(1))

        # Plain number.
        match = re.search(r'(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

        # Chinese number words.
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

    def _parse_array(self, text: str, field_name: str, field_type: str) -> list[Any]:
        """Extract array value."""
        # Try JSON array first.
        match = re.search(r'\[.*?\]', text)
        if match:
            try:
                parsed: Any = json.loads(match.group())
                return list(parsed)
            except json.JSONDecodeError:
                # JSON failed but brackets found - strip them and parse as CSV.
                bracket_content = match.group()
                inner = bracket_content[1:-1]  # Remove [ and ]
                items = re.split(r'[,，、]', inner)
                return [item.strip() for item in items if item.strip()]

        # No brackets - split by commas.
        items = re.split(r'[,，、]', text)
        return [item.strip() for item in items if item.strip()]

    def _parse_object(self, text: str, field_name: str, field_type: str) -> dict[str, Any]:
        """Extract object value."""
        match = re.search(r'\{.*?\}', text)
        if match:
            try:
                parsed_obj: Any = json.loads(match.group())
                return dict(parsed_obj)
            except (json.JSONDecodeError, ValueError):
                pass

        return {}

    def _parse_relaxed(self, text: str, field_name: str, field_type: str) -> Any:
        """Parse with relaxed rules - still requires meaningful extraction."""
        if field_type == "string":
            # Domain-example heuristic for the name field family.
            if field_name in _CHINESE_NAME_FIELD_HINTS:
                match = _CHINESE_NAME_CONTEXT_PATTERN.search(text)
                if match:
                    return match.group()
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
            # Domain-example heuristic for the name field family.
            if field_name in _CHINESE_NAME_FIELD_HINTS:
                match = _CHINESE_NAME_CONTEXT_PATTERN.search(text)
                if match:
                    return match.group()
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
                    raise ValueError("Value not in valid enum values")
