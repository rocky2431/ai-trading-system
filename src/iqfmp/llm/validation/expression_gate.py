"""Expression Syntax Gate for Qlib Expressions.

This module validates LLM-generated Qlib expressions before execution,
ensuring they are safe and syntactically correct.

Checks performed:
1. Bracket balance
2. Forbidden keywords (code injection prevention)
3. Maximum nesting depth
4. Field validation (only allowed fields)
5. Operator validation (only allowed operators)
6. Maximum length

Supports:
- Traditional OHLCV fields
- Crypto-specific fields (funding_rate, open_interest, etc.)
- Orderbook fields
- On-chain fields
- Pre-computed technical indicators
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FieldSet(Enum):
    """Predefined field sets for different market types."""

    BASIC = "basic"  # OHLCV only
    CRYPTO = "crypto"  # OHLCV + crypto-specific
    FULL = "full"  # All available fields


class FieldRegistry:
    """Registry of allowed fields for Qlib expressions.

    Provides predefined field sets for different market types and
    allows custom field registration.
    """

    # Standard OHLCV fields
    BASIC_FIELDS = {"open", "high", "low", "close", "volume"}

    # Crypto-specific fields
    CRYPTO_FIELDS = {
        "funding_rate",
        "funding_rate_predicted",
        "open_interest",
        "open_interest_change",
        "basis",
        "premium",
        "mark_price",
        "index_price",
        "liquidation_volume",
        "long_ratio",
        "short_ratio",
    }

    # Orderbook fields
    ORDERBOOK_FIELDS = {
        "bid_volume",
        "ask_volume",
        "spread",
        "bid_ask_imbalance",
        "depth_imbalance",
    }

    # On-chain fields
    ONCHAIN_FIELDS = {
        "whale_flow",
        "exchange_reserve",
        "active_addresses",
        "transaction_count",
        "nvt_ratio",
    }

    # Technical indicator fields (pre-computed)
    INDICATOR_FIELDS = {
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_lower",
        "atr_14",
        "ema_12",
        "ema_26",
        "sma_20",
    }

    # Return fields
    RETURN_FIELDS = {
        "return_1d",
        "return_5d",
        "return_10d",
        "return_20d",
    }

    @classmethod
    def get_fields(cls, field_set: FieldSet) -> set[str]:
        """Get fields for a specific field set.

        Args:
            field_set: The field set to retrieve.

        Returns:
            Set of allowed field names (without $ prefix).
        """
        if field_set == FieldSet.BASIC:
            return cls.BASIC_FIELDS.copy()
        elif field_set == FieldSet.CRYPTO:
            return (
                cls.BASIC_FIELDS
                | cls.CRYPTO_FIELDS
                | cls.ORDERBOOK_FIELDS
                | cls.ONCHAIN_FIELDS
                | cls.INDICATOR_FIELDS
                | cls.RETURN_FIELDS
            )
        elif field_set == FieldSet.FULL:
            return (
                cls.BASIC_FIELDS
                | cls.CRYPTO_FIELDS
                | cls.ORDERBOOK_FIELDS
                | cls.ONCHAIN_FIELDS
                | cls.INDICATOR_FIELDS
                | cls.RETURN_FIELDS
            )
        else:
            return cls.BASIC_FIELDS.copy()

    @classmethod
    def get_all_fields(cls) -> set[str]:
        """Get all registered fields."""
        return cls.get_fields(FieldSet.FULL)


@dataclass
class ExpressionValidationResult:
    """Result of expression validation."""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    # Metadata about the expression
    used_fields: list[str] = field(default_factory=list)
    used_operators: list[str] = field(default_factory=list)
    nesting_depth: int = 0
    expression_length: int = 0


class ExpressionGate:
    """Expression syntax gate for validating Qlib expressions.

    Prevents:
    - Code injection attacks
    - Invalid field usage
    - Invalid operator usage
    - Malformed expressions

    Supports multiple field sets:
    - BASIC: Standard OHLCV fields
    - CRYPTO: OHLCV + crypto-specific fields (funding_rate, open_interest, etc.)
    - FULL: All available fields including orderbook and on-chain

    Example:
        # For crypto markets with extended fields
        gate = ExpressionGate(field_set=FieldSet.CRYPTO)
        result = gate.validate("Mean($funding_rate, 8)")

        # For basic OHLCV only
        gate = ExpressionGate(field_set=FieldSet.BASIC)
        result = gate.validate("EMA($close, 12) - EMA($close, 26)")

        # With custom allowed fields
        gate = ExpressionGate()
        result = gate.validate(
            "EMA($close, 12) - EMA($close, 26)",
            allowed_fields=["open", "high", "low", "close", "volume"]
        )
        if not result.is_valid:
            print(f"Invalid: {result.error_message}")
    """

    # Allowed Qlib operators
    ALLOWED_OPERATORS = {
        # Rolling window operators
        "Ref", "Mean", "Std", "Var", "Sum", "Max", "Min", "Med",
        "Count", "Quantile", "Kurt", "Skew", "Mad",
        # Delta operators
        "Delta",
        # Moving averages
        "EMA", "WMA",
        # Pair operators
        "Corr", "Cov", "Resi", "Slope", "Rsquare",
        # Element-wise operators
        "Abs", "Log", "Sign", "Sqrt", "Power", "Exp",
        # Cross-sectional operators
        "Rank", "Scale", "CSRank", "CSZScore",
        # Technical indicators
        "RSI", "MACD",
        # Conditional
        "If",
        # Time-series
        "Ts_Rank", "Ts_Max", "Ts_Min", "Ts_Argmax", "Ts_Argmin",
    }

    # Forbidden keywords (code injection prevention)
    FORBIDDEN_KEYWORDS = {
        # Python keywords
        "import", "exec", "eval", "compile", "open(",
        "__", "lambda", "class", "def ", "return",
        "global", "local", "nonlocal",
        # System access
        "os.", "sys.", "subprocess", "shutil",
        "pathlib", "builtins",
        # Network
        "socket", "requests", "urllib", "http",
        # File operations
        "read(", "write(", "file(",
        # Dangerous builtins
        "getattr", "setattr", "delattr", "hasattr",
        "globals", "locals", "vars",
    }

    # Default configuration
    MAX_LENGTH = 2000
    MAX_DEPTH = 10
    MAX_OPERATORS = 20

    def __init__(
        self,
        field_set: FieldSet = FieldSet.BASIC,
        max_length: int = MAX_LENGTH,
        max_depth: int = MAX_DEPTH,
        max_operators: int = MAX_OPERATORS,
    ):
        """Initialize expression gate.

        Args:
            field_set: Predefined field set to use (BASIC, CRYPTO, or FULL).
                      Defaults to BASIC (OHLCV only).
            max_length: Maximum expression length in characters.
            max_depth: Maximum nesting depth.
            max_operators: Maximum number of operators allowed.
        """
        self.field_set = field_set
        self.default_fields = list(FieldRegistry.get_fields(field_set))
        self.max_length = max_length
        self.max_depth = max_depth
        self.max_operators = max_operators

    def validate(
        self,
        expression: str,
        allowed_fields: Optional[list[str]] = None,
    ) -> ExpressionValidationResult:
        """Validate a Qlib expression.

        Args:
            expression: The expression to validate.
            allowed_fields: List of allowed field names (without $ prefix).
                           If None, uses the default fields from the configured field_set.

        Returns:
            ExpressionValidationResult with validation status and details.
        """
        if allowed_fields is None:
            allowed_fields = self.default_fields

        result = ExpressionValidationResult(
            is_valid=True,
            expression_length=len(expression) if expression else 0,
        )

        # Empty expression check
        if not expression or not expression.strip():
            result.is_valid = False
            result.error_message = "Expression is empty"
            return result

        expression = expression.strip()
        result.expression_length = len(expression)

        # Extract comment (everything after #)
        expression_without_comment = expression.split('#')[0].strip()
        if not expression_without_comment:
            result.is_valid = False
            result.error_message = "Expression contains only a comment"
            return result

        # Length check
        if len(expression_without_comment) > self.max_length:
            result.is_valid = False
            result.error_message = f"Expression too long ({len(expression_without_comment)} > {self.max_length} chars)"
            return result

        # Forbidden keywords check
        expr_lower = expression_without_comment.lower()
        for kw in self.FORBIDDEN_KEYWORDS:
            if kw.lower() in expr_lower:
                result.is_valid = False
                result.error_message = f"Forbidden keyword detected: '{kw}'"
                return result

        # Bracket balance check
        if not self._check_brackets(expression_without_comment):
            result.is_valid = False
            result.error_message = "Unbalanced brackets in expression"
            return result

        # Nesting depth check
        depth = self._get_max_depth(expression_without_comment)
        result.nesting_depth = depth
        if depth > self.max_depth:
            result.is_valid = False
            result.error_message = f"Expression too deeply nested ({depth} > {self.max_depth})"
            return result

        # Extract and validate fields
        used_fields = self._extract_fields(expression_without_comment)
        result.used_fields = used_fields
        invalid_fields = [f for f in used_fields if f not in allowed_fields]
        if invalid_fields:
            result.is_valid = False
            result.error_message = (
                f"Invalid fields used: {invalid_fields}. "
                f"Allowed fields: {allowed_fields}"
            )
            return result

        # Extract and validate operators
        used_operators = self._extract_operators(expression_without_comment)
        result.used_operators = used_operators
        invalid_operators = [op for op in used_operators if op not in self.ALLOWED_OPERATORS]
        if invalid_operators:
            result.is_valid = False
            result.error_message = f"Invalid operators: {invalid_operators}"
            return result

        # Basic shape check: an expression must reference at least one field/operator/number.
        # This prevents plain natural language text from being treated as a valid expression.
        has_number = re.search(r"\d", expression_without_comment) is not None
        if not used_fields and not used_operators and not has_number:
            result.is_valid = False
            result.error_message = (
                "Expression must reference at least one $field, operator call (e.g., Mean/Ref), "
                "or numeric literal"
            )
            return result

        # Check operator count
        if len(used_operators) > self.max_operators:
            result.is_valid = False
            result.error_message = (
                f"Too many operators ({len(used_operators)} > {self.max_operators})"
            )
            return result

        # Add warnings for potential issues
        if depth > 5:
            result.warnings.append(f"High nesting depth ({depth}), may impact performance")
        if len(used_operators) > 10:
            result.warnings.append(f"Many operators ({len(used_operators)}), expression may be complex")

        return result

    def _check_brackets(self, expr: str) -> bool:
        """Check if brackets are balanced.

        Args:
            expr: Expression to check.

        Returns:
            True if brackets are balanced.
        """
        stack = []
        bracket_pairs = {')': '(', ']': '[', '}': '{'}

        for char in expr:
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack:
                    return False
                if stack[-1] != bracket_pairs[char]:
                    return False
                stack.pop()

        return len(stack) == 0

    def _get_max_depth(self, expr: str) -> int:
        """Calculate maximum nesting depth.

        Args:
            expr: Expression to analyze.

        Returns:
            Maximum nesting depth.
        """
        max_depth = 0
        current_depth = 0

        for char in expr:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1

        return max_depth

    def _extract_fields(self, expr: str) -> list[str]:
        """Extract field names from expression.

        Fields are prefixed with $ (e.g., $close, $volume).

        Args:
            expr: Expression to analyze.

        Returns:
            List of unique field names (without $ prefix).
        """
        # Match $field_name pattern
        pattern = r'\$([a-zA-Z_][a-zA-Z0-9_]*)'
        matches = re.findall(pattern, expr)
        return list(set(matches))

    def _extract_operators(self, expr: str) -> list[str]:
        """Extract operator names from expression.

        Operators are capitalized function names followed by (.

        Args:
            expr: Expression to analyze.

        Returns:
            List of unique operator names.
        """
        # Match Operator( pattern (capitalized)
        pattern = r'([A-Z][a-zA-Z_]*)\s*\('
        matches = re.findall(pattern, expr)
        return list(set(matches))

    def quick_check(self, expression: str) -> tuple[bool, Optional[str]]:
        """Quick validation without detailed results.

        Args:
            expression: Expression to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        result = self.validate(expression)
        return result.is_valid, result.error_message

    def sanitize(self, expression: str) -> str:
        """Attempt to sanitize an expression by removing comments and extra whitespace.

        Args:
            expression: Expression to sanitize.

        Returns:
            Sanitized expression.
        """
        if not expression:
            return ""

        # Remove comments
        expr = expression.split('#')[0].strip()

        # Normalize whitespace
        expr = ' '.join(expr.split())

        return expr


# Convenience function for quick validation
def validate_expression(
    expression: str,
    allowed_fields: Optional[list[str]] = None,
) -> tuple[bool, Optional[str]]:
    """Validate a Qlib expression.

    Args:
        expression: Expression to validate.
        allowed_fields: Allowed field names.

    Returns:
        Tuple of (is_valid, error_message).
    """
    gate = ExpressionGate()
    result = gate.validate(expression, allowed_fields)
    return result.is_valid, result.error_message
