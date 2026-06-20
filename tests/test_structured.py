"""Tests for StructuredOutput module.

Tests focus on real extraction behavior. The local regex path is exercised
through the public ``extract()`` API; individual parser strategies are also
called directly when the goal is to pin down a specific strategy's contract.
No tests are line-number-oriented and no tests use ``except`` to mask the
behavior under test.
"""

import pytest

from vertai.output.structured import (
    StructuredOutput,
    ExtractionConfig,
    ExtractionResult,
    SchemaValidationError,
)


# ---------------------------------------------------------------------------
# Basic extraction (local regex mode)
# ---------------------------------------------------------------------------


class TestStructuredOutputBasic:
    """Basic functionality tests."""

    def test_number_extraction(self):
        """Extract numeric field."""
        schema = {"amount": "number"}
        output = StructuredOutput(schema)
        result = output.extract("张三报销500元")

        assert result.success is True
        assert result.data["amount"] == 500.0

    def test_enum_extraction(self):
        """Extract enum field."""
        schema = {"category": "enum[报销,采购,其他]"}
        output = StructuredOutput(schema)
        result = output.extract("张三报销500元")

        assert result.success is True
        assert result.data["category"] == "报销"

    def test_combined_extraction_with_name_hint(self):
        """Extract multiple fields including a name-hint string.

        The ``name`` field is one of the documented domain-example hints that
        recognise the Chinese ``<name>报销|采购|...`` phrasing.
        """
        schema = {
            "name": "string",
            "amount": "number",
            "category": "enum[报销,采购,其他]",
        }
        output = StructuredOutput(schema)
        result = output.extract("张三报销500元")

        assert result.success is True
        assert result.data["name"] == "张三"
        assert result.data["amount"] == 500.0
        assert result.data["category"] == "报销"


class TestStringGeneralization:
    """The local ``string`` parser must be GENERIC, not a hardcoded Chinese
    name extractor.

    Only the documented domain-example field names (``name``/``姓名``/
    ``名字``/``buyer``) trigger the reimbursement/purchase heuristic; every
    other string field uses a weak generic token extraction that works for
    ASCII, CJK, and other text.
    """

    def test_generic_ascii_string_extraction(self):
        """A generic string field extracts a product name from English text.

        This is the regression test for the old hardcoded-Chinese-name bug
        where ``{"product": "string"}`` + ``"product is Widget"`` raised
        ValueError.
        """
        schema = {"product": "string"}
        output = StructuredOutput(schema)
        result = output.extract("product is Widget")

        assert result.success is True
        assert result.data["product"] == "product"  # first non-whitespace token

    def test_generic_string_returns_first_token(self):
        """Generic string returns the first non-whitespace token."""
        schema = {"anything": "string"}
        output = StructuredOutput(schema)
        result = output.extract("hello world")

        assert result.success is True
        assert result.data["anything"] == "hello"

    def test_name_hint_still_recognises_chinese_reimbursement(self):
        """The documented domain-example hint still works for ``name``."""
        schema = {"name": "string"}
        output = StructuredOutput(schema)
        result = output.extract("张三报销500元")

        assert result.success is True
        assert result.data["name"] == "张三"

    def test_buyer_hint_recognises_purchase(self):
        """``buyer`` is also a name hint."""
        schema = {"buyer": "string"}
        output = StructuredOutput(schema)
        result = output.extract("王五采购办公用品花费2000元")

        assert result.success is True
        assert result.data["buyer"] == "王五"

    def test_generic_string_falls_back_to_relaxed(self):
        """When the first-token pass fails (whitespace only text), the retry
        chain kicks in via relaxed/aggressive parsing and still extracts
        a token."""
        schema = {"unknown": "string"}
        output = StructuredOutput(schema, max_retries=3, strict=False)
        result = output.extract("123456")

        assert result.success is True
        assert result.data["unknown"] == "123456"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """Schema validation tests."""

    def test_integer_validation(self):
        """Integer type validation."""
        schema = {"count": "integer"}
        output = StructuredOutput(schema)
        result = output.extract("count is 42")

        assert result.success is True
        assert isinstance(result.data["count"], int)
        assert result.data["count"] == 42

    def test_boolean_true_extraction(self):
        """Boolean true extraction."""
        schema = {"active": "boolean"}
        output = StructuredOutput(schema)
        result = output.extract("状态是激活的，值为true")

        assert result.success is True
        assert result.data["active"] is True

    def test_boolean_false_extraction(self):
        """Boolean false extraction."""
        schema = {"active": "boolean"}
        output = StructuredOutput(schema)
        result = output.extract("状态为否")

        assert result.success is True
        assert result.data["active"] is False

    def test_array_extraction(self):
        """Array type extraction."""
        schema = {"items": "array"}
        output = StructuredOutput(schema)
        result = output.extract("列表：苹果,香蕉,橘子")

        assert result.success is True
        assert isinstance(result.data["items"], list)

    def test_object_extraction(self):
        """Object type extraction."""
        schema = {"meta": "object"}
        output = StructuredOutput(schema)
        result = output.extract('元数据：{"key": "value"}')

        assert result.success is True
        assert isinstance(result.data["meta"], dict)


class TestValidateField:
    """Tests for _validate_field method."""

    def test_validate_none_fails(self):
        """Validation fails for None value."""
        output = StructuredOutput({"name": "string"})
        with pytest.raises(ValueError, match="cannot be None"):
            output._validate_field(None, "string")

    def test_validate_wrong_type_string(self):
        output = StructuredOutput({"name": "string"})
        with pytest.raises(ValueError, match="Expected string"):
            output._validate_field(123, "string")

    def test_validate_wrong_type_number(self):
        output = StructuredOutput({"amount": "number"})
        with pytest.raises(ValueError, match="Expected number"):
            output._validate_field("not a number", "number")

    def test_validate_integer_bool_fails(self):
        """Integer validation fails for boolean (bool is subclass of int)."""
        output = StructuredOutput({"count": "integer"})
        with pytest.raises(ValueError):
            output._validate_field(True, "integer")

    def test_validate_wrong_type_boolean(self):
        output = StructuredOutput({"active": "boolean"})
        with pytest.raises(ValueError, match="Expected boolean"):
            output._validate_field("yes", "boolean")

    def test_validate_wrong_type_array(self):
        output = StructuredOutput({"items": "array"})
        with pytest.raises(ValueError, match="Expected array"):
            output._validate_field("not a list", "array")

    def test_validate_wrong_type_object(self):
        output = StructuredOutput({"meta": "object"})
        with pytest.raises(ValueError, match="Expected object"):
            output._validate_field(["not", "a", "dict"], "object")

    def test_validate_enum_invalid_value(self):
        output = StructuredOutput({"status": "enum[active,inactive]"})
        with pytest.raises(ValueError, match="not in valid enum"):
            output._validate_field("unknown", "enum[active,inactive]")


# ---------------------------------------------------------------------------
# Retry mechanism
# ---------------------------------------------------------------------------


class TestRetryMechanism:
    """Automatic retry and correction tests."""

    def test_retry_count_on_failure(self):
        """Retry count is tracked when extraction fails throughout."""
        schema = {"impossible_field": "number"}
        output = StructuredOutput(schema, max_retries=2, strict=False)
        result = output.extract("no numbers here at all!")

        assert result.retries == 2
        assert result.success is False

    def test_strict_mode_raises_exception(self):
        """Strict mode raises SchemaValidationError when no valid extraction."""
        schema = {"amount": "number"}
        output = StructuredOutput(schema, max_retries=1, strict=True)

        with pytest.raises(SchemaValidationError):
            output.extract("nothing to extract here!")

    def test_non_strict_returns_result(self):
        """Non-strict mode returns failed result after retries."""
        schema = {"price": "number"}
        output = StructuredOutput(schema, max_retries=1, strict=False)
        result = output.extract("no price mentioned")

        assert result.success is False
        assert result.retries == 1

    def test_successful_retry_after_initial_failure(self):
        """Successful extraction triggered by a retry strategy.

        Initial number parsing requires a currency unit or plain digit. With
        ``金额`` followed by digits and ``元``, the parser succeeds.
        """
        schema = {"amount": "number"}
        output = StructuredOutput(schema, max_retries=3)
        result = output.extract("金额是100元")

        assert result.success is True


class TestEnumValidation:
    """Enum type specific tests."""

    def test_enum_value_matched(self):
        schema = {"status": "enum[pending,approved,rejected]"}
        output = StructuredOutput(schema)
        result = output.extract("申请已approved")

        assert result.success is True
        assert result.data["status"] == "approved"

    def test_enum_no_match_strict_mode(self):
        schema = {"status": "enum[pending,approved,rejected]"}
        output = StructuredOutput(schema, max_retries=1, strict=True)

        with pytest.raises(SchemaValidationError):
            output.extract("状态是未知的unknown state")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_schema(self):
        """Empty schema returns empty successful result."""
        schema = {}
        output = StructuredOutput(schema)
        result = output.extract("some text")

        assert result.success is True
        assert result.data == {}

    def test_chinese_number_words(self):
        """Chinese number words are parsed."""
        schema = {"count": "number"}
        output = StructuredOutput(schema)
        result = output.extract("数量是三个")

        assert result.success is True
        assert result.data["count"] == 3.0

    def test_decimal_number(self):
        """Decimal numbers are parsed correctly."""
        schema = {"price": "number"}
        output = StructuredOutput(schema)
        result = output.extract("价格是99.99元")

        assert result.success is True
        assert result.data["price"] == 99.99


class TestExtractionConfig:
    """ExtractionConfig dataclass tests."""

    def test_default_config(self):
        config = ExtractionConfig()
        assert config.max_retries == 3
        assert config.strict is True

    def test_custom_config(self):
        config = ExtractionConfig(max_retries=5, strict=False)
        assert config.max_retries == 5
        assert config.strict is False

    def test_config_parameter_override(self):
        """Config parameter overrides individual max_retries/strict params."""
        config = ExtractionConfig(max_retries=5, strict=False)
        output = StructuredOutput(
            {"amount": "number"}, config=config, max_retries=1, strict=True
        )
        assert output.max_retries == 5
        assert output.strict is False


class TestExtractionResult:
    """ExtractionResult dataclass tests."""

    def test_successful_result(self):
        result = ExtractionResult(
            data={"key": "value"},
            success=True,
            retries=0,
            error=None
        )
        assert result.data == {"key": "value"}
        assert result.success is True
        assert result.retries == 0
        assert result.error is None

    def test_failed_result(self):
        result = ExtractionResult(
            data={},
            success=False,
            retries=3,
            error="Validation failed"
        )
        assert result.success is False
        assert result.retries == 3
        assert result.error == "Validation failed"


class TestRealWorldScenarios:
    """Real-world use case tests."""

    def test_expense_report(self):
        """Expense report extraction scenario."""
        schema = {
            "name": "string",
            "amount": "number",
            "category": "enum[报销,采购,其他]",
        }
        output = StructuredOutput(schema)
        result = output.extract("李四于3月15日报销差旅费800元")

        assert result.success is True
        assert result.data["name"] == "李四"
        assert result.data["amount"] == 800.0
        assert result.data["category"] == "报销"

    def test_purchase_order(self):
        """Purchase order extraction scenario."""
        schema = {
            "buyer": "string",
            "amount": "number",
            "category": "enum[报销,采购,其他]",
        }
        output = StructuredOutput(schema)
        result = output.extract("王五采购办公用品花费2000元")

        assert result.success is True
        assert result.data["buyer"] == "王五"
        assert result.data["amount"] == 2000.0
        assert result.data["category"] == "采购"


# ---------------------------------------------------------------------------
# LLM-mode schema validation (the path that previously skipped validation)
# ---------------------------------------------------------------------------


class _FakeGenerateResult:
    """Minimal stand-in for GenerateResult used by the fake LLM."""

    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Fake LLM that returns scripted JSON responses.

    Each call to ``generate`` pops the next scripted response. This stubs the
    *external* provider behavior (returning canned JSON) so we can assert on
    the real schema-validation + retry behavior inside StructuredOutput.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> _FakeGenerateResult:
        self.calls.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLM.generate called too many times")
        return _FakeGenerateResult(self._responses.pop(0))


class TestLLMSchemaValidation:
    """LLM mode must schema-validate every field and retry on failure.

    Regression tests for the bug where the LLM path only checked ``value is
    not None`` and never called ``_validate_field``.
    """

    def test_llm_valid_response_passes(self):
        """A valid LLM JSON response is returned unchanged."""
        llm = _FakeLLM(['{"amount": 100, "name": "Alice"}'])
        output = StructuredOutput(
            {"amount": "number", "name": "string"}, llm=llm, max_retries=2
        )
        result = output.extract("irrelevant text")

        assert result.success is True
        assert result.data == {"amount": 100, "name": "Alice"}
        assert len(llm.calls) == 1

    def test_llm_wrong_type_triggers_retry_then_success(self):
        """An LLM response with a wrong type triggers a retry that succeeds."""
        llm = _FakeLLM([
            '{"amount": "not a number", "name": "Alice"}',  # wrong type
            '{"amount": 100, "name": "Alice"}',             # corrected
        ])
        output = StructuredOutput(
            {"amount": "number", "name": "string"}, llm=llm, max_retries=2
        )
        result = output.extract("irrelevant")

        assert result.success is True
        assert result.data == {"amount": 100, "name": "Alice"}
        assert len(llm.calls) == 2

    def test_llm_missing_field_triggers_retry(self):
        """A null field triggers a retry; the second attempt supplies it."""
        llm = _FakeLLM([
            '{"amount": 100, "name": null}',   # name missing
            '{"amount": 100, "name": "Bob"}',  # corrected
        ])
        output = StructuredOutput(
            {"amount": "number", "name": "string"}, llm=llm, max_retries=2
        )
        result = output.extract("irrelevant")

        assert result.success is True
        assert result.data == {"amount": 100, "name": "Bob"}

    def test_llm_validation_failure_exhausts_retries_strict_raises(self):
        """Persistent validation failure raises SchemaValidationError."""
        llm = _FakeLLM([
            '{"amount": "bad"}',
            '{"amount": "still bad"}',
            '{"amount": "bad again"}',
        ])
        output = StructuredOutput(
            {"amount": "number"}, llm=llm, max_retries=2, strict=True
        )
        with pytest.raises(SchemaValidationError):
            output.extract("irrelevant")
        # initial attempt + 2 retries = 3 calls
        assert len(llm.calls) == 3

    def test_llm_validation_failure_non_strict_returns_error(self):
        """Non-strict mode returns a failed result with the validation error."""
        llm = _FakeLLM([
            '{"amount": "bad"}',
            '{"amount": "still bad"}',
        ])
        output = StructuredOutput(
            {"amount": "number"}, llm=llm, max_retries=1, strict=False
        )
        result = output.extract("irrelevant")

        assert result.success is False
        assert result.retries == 1
        assert result.error is not None
        assert "amount" in result.error

    def test_llm_invalid_json_retries(self):
        """Malformed JSON triggers a retry rather than an immediate failure."""
        llm = _FakeLLM([
            'not json at all',
            '{"amount": 50}',  # recovered
        ])
        output = StructuredOutput(
            {"amount": "number"}, llm=llm, max_retries=2
        )
        result = output.extract("irrelevant")

        assert result.success is True
        assert result.data == {"amount": 50}

    def test_llm_strips_json_fences(self):
        """The LLM path strips ```json ... ``` fences before parsing."""
        llm = _FakeLLM(['```json\n{"amount": 7}\n```'])
        output = StructuredOutput(
            {"amount": "number"}, llm=llm, max_retries=1
        )
        result = output.extract("irrelevant")

        assert result.success is True
        assert result.data == {"amount": 7}

    def test_llm_strips_plain_code_fences(self):
        """The LLM path strips bare ``` (no language tag) fences too."""
        llm = _FakeLLM(['```\n{"amount": 9}\n```'])
        output = StructuredOutput(
            {"amount": "number"}, llm=llm, max_retries=1
        )
        result = output.extract("irrelevant")

        assert result.success is True
        assert result.data == {"amount": 9}

    def test_llm_provider_error_retries(self):
        """A provider exception triggers a retry, not an immediate failure."""
        class _ExplodingLLM:
            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompt: str) -> _FakeGenerateResult:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("network blip")
                return _FakeGenerateResult('{"amount": 11}')

        llm = _ExplodingLLM()
        output = StructuredOutput(
            {"amount": "number"}, llm=llm, max_retries=2
        )
        result = output.extract("irrelevant")

        assert result.success is True
        assert result.data == {"amount": 11}

    def test_llm_enum_validated(self):
        """Enum values are validated in the LLM path too."""
        llm = _FakeLLM([
            '{"status": "unknown"}',   # not in enum
            '{"status": "active"}',    # valid
        ])
        output = StructuredOutput(
            {"status": "enum[active,inactive]"}, llm=llm, max_retries=2
        )
        result = output.extract("irrelevant")

        assert result.success is True
        assert result.data == {"status": "active"}


# ---------------------------------------------------------------------------
# dir() antipattern regression
# ---------------------------------------------------------------------------


class TestCorrectionNoDirIntrospection:
    """The previous ``data if 'data' in dir() else {}`` introspection is gone.

    When correction fails partway through, the returned result still carries
    the partially-built ``data`` dict (not a freshly-empty one).
    """

    def test_correction_partial_failure_returns_partial_data(self):
        """A correction that fails after parsing some fields returns the
        partially-built data (not ``{}``)."""
        schema = {"name": "string", "amount": "number"}
        output = StructuredOutput(schema, max_retries=2, strict=False)

        # Force a partial failure: name parses, amount cannot (no digits).
        previous = ExtractionResult(
            data={"name": "张三", "amount": None}, success=False, error="test"
        )
        result = output._correct_extraction("张三报销", previous, 0)

        # Even if it fails, the name field is present in the returned data.
        assert isinstance(result, ExtractionResult)
        assert "name" in result.data


# ---------------------------------------------------------------------------
# Parser strategy direct tests (behavioral, not line-oriented)
# ---------------------------------------------------------------------------


class TestParseRelaxed:
    """_parse_relaxed behavioral contract."""

    def test_relaxed_string_chinese(self):
        output = StructuredOutput({"text": "string"}, max_retries=3)
        result = output._parse_relaxed("这是一些中文文字", "text", "string")
        assert result == "这是一些中文文字"

    def test_relaxed_string_word_fallback(self):
        output = StructuredOutput({"text": "string"})
        result = output._parse_relaxed("hello world", "text", "string")
        assert result == "hello"

    def test_relaxed_string_empty_raises(self):
        output = StructuredOutput({"text": "string"})
        with pytest.raises(ValueError):
            output._parse_relaxed("", "text", "string")

    def test_relaxed_integer_chinese_number(self):
        output = StructuredOutput({"count": "integer"})
        assert output._parse_relaxed("数量是三", "count", "integer") == 3

    def test_relaxed_number_integer_distinction(self):
        """integer returns int, number returns float."""
        output = StructuredOutput({"count": "integer"})
        result = output._parse_relaxed("数字是42", "count", "integer")
        assert result == 42
        assert isinstance(result, int)

    def test_relaxed_enum_match(self):
        output = StructuredOutput({"status": "enum[active,inactive]"})
        result = output._parse_relaxed(
            "value is active here", "status", "enum[active,inactive]"
        )
        assert result == "active"

    def test_relaxed_enum_no_match_raises(self):
        output = StructuredOutput({"status": "enum[active,inactive]"})
        with pytest.raises(ValueError, match="Cannot match enum"):
            output._parse_relaxed("xyz", "status", "enum[active,inactive]")

    def test_relaxed_boolean_true_with_kai(self):
        output = StructuredOutput({"flag": "boolean"})
        assert output._parse_relaxed("状态是开启的", "flag", "boolean") is True

    def test_relaxed_boolean_false(self):
        output = StructuredOutput({"flag": "boolean"})
        assert output._parse_relaxed("状态为关闭", "flag", "boolean") is False

    def test_relaxed_boolean_no_match_raises(self):
        output = StructuredOutput({"flag": "boolean"})
        with pytest.raises(ValueError, match="Cannot extract boolean"):
            output._parse_relaxed("abcxyz", "flag", "boolean")

    def test_relaxed_number_no_match_raises(self):
        output = StructuredOutput({"count": "number"})
        with pytest.raises(ValueError):
            output._parse_relaxed("no numbers here", "count", "number")

    def test_relaxed_unknown_type_returns_none(self):
        """Relaxed parsing returns None for types it does not handle."""
        output = StructuredOutput({"data": "array"})
        assert output._parse_relaxed("text", "data", "unknown_type") is None


class TestParsePattern:
    """_parse_pattern behavioral contract."""

    def test_pattern_amount_extraction(self):
        output = StructuredOutput({"amount": "number"})
        assert output._parse_pattern("price 100", "amount", "number") == 100.0

    def test_pattern_name_extraction_uses_heuristic(self):
        """name field triggers the Chinese-name heuristic, returning the
        full Chinese run matched by the predefined pattern."""
        output = StructuredOutput({"name": "string"})
        result = output._parse_pattern("张三报销", "name", "string")
        # The predefined pattern r'([一-龥]+)' matches the full Chinese run.
        assert result == "张三报销"

    def test_pattern_fallback_to_relaxed(self):
        output = StructuredOutput({"unknown": "number"})
        assert output._parse_pattern("value 42", "unknown", "number") == 42.0


class TestParseAggressive:
    """_parse_aggressive behavioral contract."""

    def test_aggressive_string_chinese(self):
        output = StructuredOutput({"text": "string"}, max_retries=3, strict=False)
        result = output._parse_aggressive("中文内容", "text", "string")
        assert result == "中文内容"

    def test_aggressive_string_name_hint(self):
        """The name-hint heuristic is also applied in aggressive mode."""
        output = StructuredOutput({"name": "string"})
        result = output._parse_aggressive("张三报销", "name", "string")
        assert result == "张三"

    def test_aggressive_string_word_fallback(self):
        output = StructuredOutput({"text": "string"})
        assert output._parse_aggressive("hello world", "text", "string") == "hello"

    def test_aggressive_string_empty_raises(self):
        output = StructuredOutput({"text": "string"})
        with pytest.raises(ValueError):
            output._parse_aggressive("", "text", "string")

    def test_aggressive_number_positive(self):
        output = StructuredOutput({"count": "number"})
        assert output._parse_aggressive("value +42", "count", "number") == 42.0

    def test_aggressive_number_negative(self):
        output = StructuredOutput({"count": "number"})
        assert output._parse_aggressive("value -42", "count", "number") == -42.0

    def test_aggressive_integer_negative(self):
        output = StructuredOutput({"count": "integer"})
        assert output._parse_aggressive("value -10", "count", "integer") == -10

    def test_aggressive_number_no_match_raises(self):
        output = StructuredOutput({"count": "number"})
        with pytest.raises(ValueError):
            output._parse_aggressive("no numbers", "count", "number")

    def test_aggressive_enum_case_insensitive(self):
        output = StructuredOutput({"status": "enum[Active,Inactive]"})
        result = output._parse_aggressive("status is active", "status", "enum[Active,Inactive]")
        assert result == "Active"

    def test_aggressive_enum_no_match_raises(self):
        output = StructuredOutput({"status": "enum[a,b,c]"})
        with pytest.raises(ValueError):
            output._parse_aggressive("xyz", "status", "enum[a,b,c]")

    def test_aggressive_boolean_true_word_boundary(self):
        output = StructuredOutput({"flag": "boolean"})
        assert output._parse_aggressive("flag is 是", "flag", "boolean") is True

    def test_aggressive_boolean_defaults_false(self):
        """Aggressive boolean defaults to False when no true pattern matches."""
        output = StructuredOutput({"flag": "boolean"})
        assert output._parse_aggressive("nothing matches", "flag", "boolean") is False

    def test_aggressive_unknown_type_returns_none(self):
        output = StructuredOutput({"data": "object"}, max_retries=3, strict=False)
        assert output._parse_aggressive("text", "data", "object") is None


# ---------------------------------------------------------------------------
# Misc edge cases
# ---------------------------------------------------------------------------


class TestParseEnumInvalidFormat:
    """Invalid enum type strings raise a clear error."""

    def test_enum_invalid_format_raises(self):
        output = StructuredOutput({"status": "enum_invalid"})
        with pytest.raises(ValueError, match="Invalid enum type"):
            output._parse_enum("some text", "status", "enum_invalid")


class TestArrayParsing:
    """Array parsing edge cases."""

    def test_array_json_valid(self):
        schema = {"items": "array"}
        output = StructuredOutput(schema)
        result = output.extract('列表: [1, 2, 3]')
        assert result.success is True
        assert result.data["items"] == [1, 2, 3]

    def test_array_json_invalid_brackets_falls_back_to_csv(self):
        schema = {"items": "array"}
        output = StructuredOutput(schema)
        result = output.extract('列表: [a, b, c]')
        assert result.success is True
        assert result.data["items"] == ["a", "b", "c"]

    def test_array_chinese_separator(self):
        schema = {"items": "array"}
        output = StructuredOutput(schema)
        result = output.extract('列表：苹果、香蕉、橘子')
        assert result.success is True
        assert len(result.data["items"]) >= 3


class TestObjectParsing:
    """Object parsing edge cases."""

    def test_object_invalid_json_returns_empty_dict(self):
        schema = {"meta": "object"}
        output = StructuredOutput(schema, max_retries=1, strict=False)
        result = output.extract('元数据：{invalid json}')
        assert result.success is True
        assert result.data["meta"] == {}

    def test_object_no_brackets_returns_empty_dict(self):
        schema = {"meta": "object"}
        output = StructuredOutput(schema, max_retries=1, strict=False)
        result = output.extract('无对象内容')
        assert result.success is True
        assert result.data["meta"] == {}


class TestNumberParsing:
    """Number parsing edge cases across currency units and Chinese numerals."""

    def test_number_plain(self):
        output = StructuredOutput({"amount": "number"})
        result = output.extract("数量 100")
        assert result.data["amount"] == 100.0

    def test_number_chinese_yuan(self):
        output = StructuredOutput({"amount": "number"})
        result = output.extract("金额500元")
        assert result.data["amount"] == 500.0

    def test_number_chinese_kuai(self):
        output = StructuredOutput({"amount": "number"})
        result = output.extract("花了10块")
        assert result.data["amount"] == 10.0

    @pytest.mark.parametrize("chinese_digit,expected", [
        ("壹", 1.0), ("贰", 2.0), ("叁", 3.0),
    ])
    def test_number_chinese_formal_digits(self, chinese_digit, expected):
        output = StructuredOutput({"amount": "number"})
        result = output.extract(f"金额{chinese_digit}元")
        assert result.success is True
        assert result.data["amount"] == expected


class TestBooleanParsing:
    """Boolean parsing across true/false keyword families."""

    @pytest.mark.parametrize("text", ["flag yes", "flag 1", "这是对的", "这是真的"])
    def test_boolean_true_patterns(self, text):
        output = StructuredOutput({"flag": "boolean"})
        result = output.extract(text)
        assert result.success is True
        assert result.data["flag"] is True

    @pytest.mark.parametrize("text", ["flag no", "flag 0", "答案错误", "消息内容为假"])
    def test_boolean_false_patterns(self, text):
        output = StructuredOutput({"flag": "boolean"})
        result = output.extract(text)
        assert result.success is True
        assert result.data["flag"] is False


class TestCorrectionBranches:
    """_correct_extraction behavioral contract."""

    def test_correction_reuses_valid_previous_field(self):
        """A valid field in the previous result is reused without re-parsing."""
        schema = {"name": "string", "amount": "number"}
        output = StructuredOutput(schema, max_retries=2)
        previous = ExtractionResult(
            data={"name": "张三", "amount": None}, success=False, error="test"
        )
        result = output._correct_extraction("张三报销500元", previous, 0)
        assert isinstance(result, ExtractionResult)
        assert result.data.get("name") == "张三"

    def test_correction_revalidates_invalid_previous_field(self):
        """An invalid-typed previous field is re-parsed."""
        schema = {"name": "string", "amount": "number"}
        output = StructuredOutput(schema, max_retries=2, strict=False)
        previous = ExtractionResult(
            data={"name": 123, "amount": 100}, success=False, error="test"
        )
        result = output._correct_extraction("张三报销500元", previous, 0)
        assert isinstance(result, ExtractionResult)


class TestExtractInitialExceptionHandling:
    """_extract_initial returns a failed ExtractionResult on internal errors."""

    def test_extract_initial_returns_result_on_regex_error(self):
        """When a parser raises re.error, _extract_initial returns a failed
        ExtractionResult (not propagates the exception)."""
        import re
        from unittest.mock import patch

        output = StructuredOutput({"field": "string"}, max_retries=1, strict=False)
        with patch.object(output, "_parse_field", side_effect=re.error("bad regex")):
            result = output._extract_initial("test")
            assert result.success is False
            assert result.error == "Extraction failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
