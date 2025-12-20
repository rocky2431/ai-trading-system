"""JSON Schema Validation for LLM Structured Output.

This module provides schema-based validation for LLM-generated content,
ensuring structured outputs meet expected formats (RD-Agent pattern).

Key Features:
1. JSON schema definitions for factor generation, evaluation, etc.
2. Validation with detailed error messages
3. Auto-repair for common formatting issues
4. Retry hint generation for LLM refinement

Design Goals:
- Structured output success rate >95%
- Clear error messages for LLM self-correction
- Support for multiple output formats
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    import jsonschema
    from jsonschema import Draft7Validator, ValidationError
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    Draft7Validator = None
    ValidationError = None


class OutputType(Enum):
    """Supported LLM output types with associated schemas."""

    FACTOR_EXPRESSION = "factor_expression"  # Qlib expression output
    FACTOR_CODE = "factor_code"  # Python factor code
    EVALUATION_RESULT = "evaluation_result"  # Factor evaluation results
    HYPOTHESIS = "hypothesis"  # Research hypothesis
    ANALYSIS = "analysis"  # General analysis output
    EVALUATION_INSIGHTS = "evaluation_insights"  # LLM insights for evaluation agent
    STRATEGY_RECOMMENDATIONS = "strategy_recommendations"  # LLM strategy advice
    RISK_ANALYSIS = "risk_analysis"  # LLM risk analysis output
    HYPOTHESES_LIST = "hypotheses_list"  # List output for hypothesis generator
    FEEDBACK_ANALYSIS = "feedback_analysis"  # LLM feedback for failed hypotheses


# JSON Schema Definitions
SCHEMAS: dict[OutputType, dict[str, Any]] = {
    OutputType.FACTOR_EXPRESSION: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["expression"],
        "properties": {
            "expression": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": "Qlib expression string",
            },
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100,
                "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
                "description": "Factor name (valid Python identifier)",
            },
            "description": {
                "type": "string",
                "maxLength": 500,
                "description": "Brief description of the factor",
            },
            "family": {
                "type": "string",
                "enum": [
                    "momentum", "value", "volatility", "quality",
                    "sentiment", "liquidity", "funding", "open_interest",
                    "liquidation", "orderbook", "onchain"
                ],
                "description": "Factor family category",
            },
        },
        "additionalProperties": True,
    },

    OutputType.FACTOR_CODE: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["code"],
        "properties": {
            "code": {
                "type": "string",
                "minLength": 10,
                "maxLength": 10000,
                "description": "Python factor code",
            },
            "name": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100,
                "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
            },
            "description": {
                "type": "string",
                "maxLength": 500,
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required Python packages",
            },
        },
        "additionalProperties": True,
    },

    OutputType.EVALUATION_RESULT: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["sharpe_ratio"],
        "properties": {
            "sharpe_ratio": {
                "type": "number",
                "description": "Sharpe ratio of the factor",
            },
            "ic_mean": {
                "type": "number",
                "description": "Mean Information Coefficient",
            },
            "ir": {
                "type": "number",
                "description": "Information Ratio",
            },
            "max_drawdown": {
                "type": "number",
                "maximum": 0,
                "description": "Maximum drawdown (negative value)",
            },
            "win_rate": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Win rate (0-1)",
            },
            "passed": {
                "type": "boolean",
                "description": "Whether factor passed evaluation threshold",
            },
        },
        "additionalProperties": True,
    },

    OutputType.HYPOTHESIS: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["hypothesis", "rationale"],
        "properties": {
            "hypothesis": {
                "type": "string",
                "minLength": 10,
                "maxLength": 1000,
                "description": "The hypothesis statement",
            },
            "rationale": {
                "type": "string",
                "minLength": 10,
                "maxLength": 2000,
                "description": "Reasoning behind the hypothesis",
            },
            "expected_outcome": {
                "type": "string",
                "maxLength": 500,
                "description": "Expected outcome if hypothesis is true",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence level (0-1)",
            },
        },
        "additionalProperties": True,
    },

    OutputType.ANALYSIS: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["summary"],
        "properties": {
            "summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": "Analysis summary",
            },
            "findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key findings",
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recommendations based on analysis",
            },
        },
        "additionalProperties": True,
    },

    OutputType.EVALUATION_INSIGHTS: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["analysis", "insights", "recommendations"],
        "properties": {
            "analysis": {"type": "string", "minLength": 1, "maxLength": 2000},
            "insights": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "recommendations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "additionalProperties": True,
    },

    OutputType.STRATEGY_RECOMMENDATIONS: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["assessment", "risks", "recommendations"],
        "properties": {
            "assessment": {"type": "string", "minLength": 1, "maxLength": 2000},
            "risks": {"type": "array", "items": {"type": "string"}},
            "recommendations": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    },

    OutputType.RISK_ANALYSIS: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": [
            "summary",
            "key_risks",
            "mitigation_actions",
            "position_recommendations",
            "hedging_strategies",
        ],
        "properties": {
            "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
            "key_risks": {"type": "array", "items": {"type": "string"}},
            "mitigation_actions": {"type": "array", "items": {"type": "string"}},
            "position_recommendations": {"type": "string", "maxLength": 2000},
            "hedging_strategies": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "additionalProperties": True,
    },

    OutputType.HYPOTHESES_LIST: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "required": ["name", "description", "family", "rationale"],
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 200},
                "description": {"type": "string", "minLength": 1, "maxLength": 2000},
                "family": {"type": "string"},
                "rationale": {"type": "string", "minLength": 1, "maxLength": 2000},
                "expected_ic": {"type": "number"},
                "expected_direction": {"type": "string"},
            },
            "additionalProperties": True,
        },
    },

    OutputType.FEEDBACK_ANALYSIS: {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["feedback", "suggestions"],
        "properties": {
            "feedback": {"type": "string", "minLength": 1, "maxLength": 5000},
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    },
}


@dataclass
class SchemaValidationResult:
    """Result of JSON schema validation."""

    is_valid: bool
    data: Optional[dict[str, Any] | list[Any]] = None
    error_message: Optional[str] = None
    error_path: Optional[str] = None
    repair_hint: Optional[str] = None

    # Statistics
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "data": self.data,
            "error_message": self.error_message,
            "error_path": self.error_path,
            "repair_hint": self.repair_hint,
            "validation_errors": self.validation_errors,
        }


class JSONSchemaValidator:
    """JSON Schema validator for LLM outputs.

    Provides:
    1. Schema validation with detailed error messages
    2. Auto-repair for common formatting issues
    3. Retry hints for LLM self-correction

    Example:
        validator = JSONSchemaValidator()

        # Validate factor expression output
        result = validator.validate(
            '{"expression": "Mean($close, 20)", "name": "ma_20"}',
            OutputType.FACTOR_EXPRESSION
        )

        if not result.is_valid:
            print(f"Validation failed: {result.error_message}")
            print(f"Repair hint: {result.repair_hint}")
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize validator.

        Args:
            strict_mode: If True, reject additional properties not in schema.
        """
        self.strict_mode = strict_mode
        self._validators: dict[OutputType, Any] = {}

        if HAS_JSONSCHEMA:
            for output_type, schema in SCHEMAS.items():
                if strict_mode:
                    schema = {**schema, "additionalProperties": False}
                self._validators[output_type] = Draft7Validator(schema)

    def validate(
        self,
        content: str | dict[str, Any] | list[Any],
        output_type: OutputType,
        auto_repair: bool = True,
    ) -> SchemaValidationResult:
        """Validate content against schema.

        Args:
            content: JSON string or dict to validate
            output_type: Expected output type
            auto_repair: Attempt to repair common issues

        Returns:
            SchemaValidationResult with validation status
        """
        if not HAS_JSONSCHEMA:
            # Fallback: basic validation without jsonschema library
            return self._validate_basic(content, output_type)

        # Parse JSON if string
        if isinstance(content, str):
            parsed_result = self._parse_json(content, auto_repair)
            if not parsed_result.is_valid:
                return parsed_result
            data = parsed_result.data
        else:
            data = content

        # Get validator
        validator = self._validators.get(output_type)
        if validator is None:
            return SchemaValidationResult(
                is_valid=False,
                error_message=f"Unknown output type: {output_type}",
            )

        # Validate against schema
        errors = list(validator.iter_errors(data))

        if not errors:
            return SchemaValidationResult(is_valid=True, data=data)

        # Collect all errors
        validation_errors = []
        for error in errors:
            path = ".".join(str(p) for p in error.absolute_path) or "root"
            validation_errors.append(f"{path}: {error.message}")

        # Primary error
        primary_error = errors[0]
        error_path = ".".join(str(p) for p in primary_error.absolute_path) or "root"

        # Generate repair hint
        repair_hint = self._generate_repair_hint(primary_error, output_type)

        return SchemaValidationResult(
            is_valid=False,
            data=data,
            error_message=primary_error.message,
            error_path=error_path,
            repair_hint=repair_hint,
            validation_errors=validation_errors,
        )

    def _parse_json(
        self,
        content: str,
        auto_repair: bool = True,
    ) -> SchemaValidationResult:
        """Parse JSON string with optional auto-repair.

        Args:
            content: JSON string to parse
            auto_repair: Attempt to repair common issues

        Returns:
            SchemaValidationResult with parsed data or error
        """
        content = content.strip()

        # Direct parse attempt
        try:
            data = json.loads(content)
            return SchemaValidationResult(is_valid=True, data=data)
        except json.JSONDecodeError as e:
            if not auto_repair:
                return SchemaValidationResult(
                    is_valid=False,
                    error_message=f"JSON parse error: {e}",
                    repair_hint="Ensure valid JSON format with proper quotes and brackets.",
                )

        # Auto-repair attempts
        repaired_content = self._attempt_json_repair(content)

        try:
            data = json.loads(repaired_content)
            return SchemaValidationResult(is_valid=True, data=data)
        except json.JSONDecodeError as e:
            return SchemaValidationResult(
                is_valid=False,
                error_message=f"JSON parse error after repair attempt: {e}",
                repair_hint=(
                    "Output must be valid JSON. Common fixes:\n"
                    "1. Use double quotes for strings\n"
                    "2. No trailing commas\n"
                    "3. Escape special characters in strings\n"
                    "4. Use null instead of None"
                ),
            )

    def _attempt_json_repair(self, content: str) -> str:
        """Attempt to repair common JSON issues.

        Args:
            content: Malformed JSON string

        Returns:
            Repaired JSON string
        """
        repaired = content

        # Remove markdown code blocks
        repaired = re.sub(r"```(?:json)?\s*\n?", "", repaired)
        repaired = re.sub(r"\n?```", "", repaired)

        # Extract JSON from mixed content (object or array)
        json_match = re.search(r"\{[\s\S]*\}", repaired)
        if not json_match:
            json_match = re.search(r"\[[\s\S]*\]", repaired)
        if json_match:
            repaired = json_match.group(0)

        # Replace single quotes with double quotes (careful with apostrophes)
        # Only replace quotes that look like JSON string delimiters
        repaired = re.sub(r"(?<=[{,\[:\s])'", '"', repaired)
        repaired = re.sub(r"'(?=[,}\]:\s])", '"', repaired)

        # Replace Python None with null
        repaired = re.sub(r"\bNone\b", "null", repaired)

        # Replace Python True/False with true/false
        repaired = re.sub(r"\bTrue\b", "true", repaired)
        repaired = re.sub(r"\bFalse\b", "false", repaired)

        # Remove trailing commas
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

        return repaired.strip()

    def _generate_repair_hint(
        self,
        error: Any,
        output_type: OutputType,
    ) -> str:
        """Generate repair hint for LLM based on validation error.

        Args:
            error: jsonschema ValidationError
            output_type: Expected output type

        Returns:
            Human-readable repair hint
        """
        schema = SCHEMAS.get(output_type, {})
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Build context-specific hints
        hints = []

        if "required" in error.schema_path or "is a required property" in error.message:
            hints.append(f"Missing required field. Required fields: {required}")

        if "type" in error.schema_path:
            expected_type = error.schema.get("type", "unknown")
            hints.append(f"Wrong type. Expected: {expected_type}")

        if "minLength" in error.schema_path:
            hints.append("Value too short. Provide more content.")

        if "maxLength" in error.schema_path:
            hints.append("Value too long. Keep it concise.")

        if "pattern" in error.schema_path:
            pattern = error.schema.get("pattern", "")
            hints.append(f"Value doesn't match pattern: {pattern}")

        if "enum" in error.schema_path:
            allowed = error.schema.get("enum", [])
            hints.append(f"Invalid value. Allowed values: {allowed}")

        # Default hint with schema reference
        if not hints:
            hints.append(f"Validation error at {error.absolute_path}: {error.message}")

        # Add example of correct format
        example = self._get_example_output(output_type)
        if example:
            hints.append(f"Example format:\n{example}")

        return "\n".join(hints)

    def _get_example_output(self, output_type: OutputType) -> str:
        """Get example output for a given type.

        Args:
            output_type: Output type to get example for

        Returns:
            Example JSON string
        """
        examples = {
            OutputType.FACTOR_EXPRESSION: json.dumps({
                "expression": "Mean($close, 20)",
                "name": "ma_20",
                "description": "20-period moving average",
                "family": "momentum"
            }, indent=2),
            OutputType.FACTOR_CODE: json.dumps({
                "code": "def factor(df):\n    return df['close'].rolling(20).mean()",
                "name": "rolling_mean_20",
                "description": "Rolling mean factor"
            }, indent=2),
            OutputType.EVALUATION_RESULT: json.dumps({
                "sharpe_ratio": 1.5,
                "ic_mean": 0.05,
                "ir": 1.2,
                "max_drawdown": -0.15,
                "win_rate": 0.55,
                "passed": True
            }, indent=2),
            OutputType.HYPOTHESIS: json.dumps({
                "hypothesis": "Funding rate mean reversion predicts returns",
                "rationale": "High funding rates indicate crowded positioning",
                "expected_outcome": "Positive Sharpe when funding extreme",
                "confidence": 0.7
            }, indent=2),
            OutputType.ANALYSIS: json.dumps({
                "summary": "Factor shows strong momentum characteristics",
                "findings": ["Positive IC", "Low turnover", "Stable returns"],
                "recommendations": ["Increase allocation", "Monitor drawdown"]
            }, indent=2),
            OutputType.EVALUATION_INSIGHTS: json.dumps({
                "analysis": "Factor passed thresholds with stable IC and acceptable drawdown.",
                "insights": ["IC is consistently positive", "Low turnover suggests robustness"],
                "recommendations": ["Test alternative lookbacks", "Add volatility filter"],
            }, indent=2),
            OutputType.STRATEGY_RECOMMENDATIONS: json.dumps({
                "assessment": "Combination is diversified with balanced exposure.",
                "risks": ["Correlation between factors may rise in stress regimes"],
                "recommendations": ["Cap single-factor weight", "Monitor turnover and costs"],
            }, indent=2),
            OutputType.RISK_ANALYSIS: json.dumps({
                "summary": "Risk is acceptable but drawdown risk is elevated.",
                "key_risks": ["High max drawdown", "Concentration risk"],
                "mitigation_actions": ["Reduce max position size", "Add stop-loss rules"],
                "position_recommendations": "Reduce exposure by 20% and cap per-asset weight at 5%.",
                "hedging_strategies": ["Delta hedge via BTC/ETH futures"],
                "confidence": 0.7,
            }, indent=2),
            OutputType.HYPOTHESES_LIST: json.dumps([
                {
                    "name": "Funding Rate Mean Reversion",
                    "description": "Extreme funding rates revert and predict next-period returns.",
                    "family": "funding",
                    "rationale": "Crowded positioning reflected in funding is mean-reverting.",
                    "expected_ic": 0.03,
                    "expected_direction": "long_short",
                }
            ], indent=2),
            OutputType.FEEDBACK_ANALYSIS: json.dumps({
                "feedback": "IC/IR are below thresholds; the signal may be noisy in this regime.",
                "suggestions": ["Try longer lookback", "Add volatility filter", "Use rank normalization"],
            }, indent=2),
        }
        return examples.get(output_type, "")

    def _validate_basic(
        self,
        content: str | dict[str, Any] | list[Any],
        output_type: OutputType,
    ) -> SchemaValidationResult:
        """Basic validation without jsonschema library.

        Args:
            content: Content to validate
            output_type: Expected output type

        Returns:
            SchemaValidationResult
        """
        # Parse if string
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                return SchemaValidationResult(
                    is_valid=False,
                    error_message=f"Invalid JSON: {e}",
                )
        else:
            data = content

        schema = SCHEMAS.get(output_type, {})
        expected_type = schema.get("type", "object")

        if expected_type == "array":
            if not isinstance(data, list):
                return SchemaValidationResult(
                    is_valid=False,
                    error_message="Expected JSON array (list)",
                )
            return SchemaValidationResult(is_valid=True, data=data)

        if not isinstance(data, dict):
            return SchemaValidationResult(
                is_valid=False,
                error_message="Expected JSON object (dict)",
            )

        # Check required fields for object schemas
        required = schema.get("required", [])
        missing = [f for f in required if f not in data]

        if missing:
            return SchemaValidationResult(
                is_valid=False,
                error_message=f"Missing required fields: {missing}",
                repair_hint=f"Include these fields: {required}",
            )

        return SchemaValidationResult(is_valid=True, data=data)

    def get_schema(self, output_type: OutputType) -> dict[str, Any]:
        """Get schema for output type.

        Args:
            output_type: Output type

        Returns:
            JSON schema dict
        """
        return SCHEMAS.get(output_type, {}).copy()

    def get_required_fields(self, output_type: OutputType) -> list[str]:
        """Get required fields for output type.

        Args:
            output_type: Output type

        Returns:
            List of required field names
        """
        schema = SCHEMAS.get(output_type, {})
        return schema.get("required", [])


# Convenience function for direct validation
def validate_json_output(
    content: str | dict[str, Any] | list[Any],
    output_type: OutputType,
    auto_repair: bool = True,
) -> SchemaValidationResult:
    """Validate LLM output against schema.

    Args:
        content: Content to validate
        output_type: Expected output type
        auto_repair: Attempt to repair common issues

    Returns:
        SchemaValidationResult
    """
    validator = JSONSchemaValidator()
    return validator.validate(content, output_type, auto_repair)


# Convenience function to extract JSON from LLM response
def extract_json_from_response(
    response: str,
    output_type: Optional[OutputType] = None,
) -> tuple[Optional[dict[str, Any] | list[Any]], Optional[str]]:
    """Extract and optionally validate JSON from LLM response.

    Args:
        response: Raw LLM response text
        output_type: Optional output type for validation

    Returns:
        Tuple of (parsed_data, error_message)
    """
    validator = JSONSchemaValidator()
    result = validator._parse_json(response, auto_repair=True)

    if not result.is_valid:
        return None, result.error_message

    if output_type is not None:
        validation = validator.validate(result.data, output_type, auto_repair=False)
        if not validation.is_valid:
            return None, validation.error_message

    return result.data, None
