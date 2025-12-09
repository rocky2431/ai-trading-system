"""Factor Generation Agent for IQFMP.

Implements natural language to Qlib-compatible factor code generation
with AST security checking integration.

Six-dimensional coverage:
1. Functional: Factor generation, prompt rendering, code extraction
2. Boundary: Edge cases for inputs
3. Exception: Error handling for LLM and security failures
4. Performance: Generation time optimization
5. Security: AST checker integration
6. Compatibility: Different factor families
"""

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol


class FactorGenerationError(Exception):
    """Base error for factor generation failures."""

    pass


class SecurityViolationError(FactorGenerationError):
    """Raised when generated code fails security checks."""

    pass


class InvalidFactorError(FactorGenerationError):
    """Raised when generated factor code is invalid."""

    pass


class FieldConstraintViolationError(FactorGenerationError):
    """Raised when generated code uses disallowed fields for the factor family."""

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        """Initialize with violation details.

        Args:
            message: Error message
            violations: List of disallowed field names that were used
        """
        super().__init__(message)
        self.violations = violations or []


class FactorFamily(Enum):
    """Factor family definitions with allowed fields."""

    MOMENTUM = "momentum"
    VALUE = "value"
    VOLATILITY = "volatility"
    QUALITY = "quality"
    SENTIMENT = "sentiment"
    LIQUIDITY = "liquidity"

    def get_allowed_fields(self) -> list[str]:
        """Get allowed data fields for this factor family."""
        base_fields = ["open", "high", "low", "close", "volume"]

        family_specific = {
            FactorFamily.MOMENTUM: [
                "close",
                "returns",
                "price_change",
                "momentum",
                "roc",
            ],
            FactorFamily.VALUE: [
                "close",
                "book_value",
                "earnings",
                "pe_ratio",
                "pb_ratio",
            ],
            FactorFamily.VOLATILITY: [
                "high",
                "low",
                "close",
                "returns",
                "std",
                "atr",
            ],
            FactorFamily.QUALITY: [
                "close",
                "volume",
                "roe",
                "roa",
                "profit_margin",
            ],
            FactorFamily.SENTIMENT: [
                "close",
                "volume",
                "sentiment_score",
                "news_count",
            ],
            FactorFamily.LIQUIDITY: [
                "close",
                "volume",
                "bid_ask_spread",
                "turnover",
                "amihud",
            ],
        }

        return list(set(base_fields + family_specific.get(self, [])))

    def get_field_descriptions(self) -> dict[str, str]:
        """Get descriptions for each allowed field.

        Returns:
            Dictionary mapping field names to descriptions
        """
        base_descriptions = {
            "open": "Opening price of the period",
            "high": "Highest price during the period",
            "low": "Lowest price during the period",
            "close": "Closing price of the period",
            "volume": "Trading volume during the period",
        }

        family_descriptions = {
            FactorFamily.MOMENTUM: {
                "returns": "Price returns over a period",
                "price_change": "Absolute price change",
                "momentum": "Price momentum indicator",
                "roc": "Rate of change indicator",
            },
            FactorFamily.VALUE: {
                "book_value": "Book value per share",
                "earnings": "Earnings per share",
                "pe_ratio": "Price to earnings ratio",
                "pb_ratio": "Price to book ratio",
            },
            FactorFamily.VOLATILITY: {
                "returns": "Price returns for volatility calculation",
                "std": "Standard deviation of returns",
                "atr": "Average true range",
            },
            FactorFamily.QUALITY: {
                "roe": "Return on equity",
                "roa": "Return on assets",
                "profit_margin": "Profit margin ratio",
            },
            FactorFamily.SENTIMENT: {
                "sentiment_score": "Aggregate sentiment score",
                "news_count": "Number of news articles",
            },
            FactorFamily.LIQUIDITY: {
                "bid_ask_spread": "Bid-ask spread",
                "turnover": "Trading turnover rate",
                "amihud": "Amihud illiquidity measure",
            },
        }

        result = base_descriptions.copy()
        result.update(family_descriptions.get(self, {}))
        return result


@dataclass
class GeneratedFactor:
    """Generated factor with metadata."""

    name: str
    description: str
    code: str
    family: FactorFamily
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if the factor code is valid Python."""
        try:
            ast.parse(self.code)
            return True
        except SyntaxError:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize factor to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "family": self.family.value,
            "metadata": self.metadata,
        }


@dataclass
class FieldValidationResult:
    """Result of field constraint validation."""

    is_valid: bool
    used_fields: set[str]
    allowed_fields: set[str]
    violations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "used_fields": list(self.used_fields),
            "allowed_fields": list(self.allowed_fields),
            "violations": self.violations,
        }


class FactorPromptTemplate:
    """Template for generating factor prompts."""

    def __init__(self) -> None:
        """Initialize prompt template."""
        self._system_prompt = self._build_system_prompt()
        self._examples = self._build_examples()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for factor generation."""
        return """You are an expert quantitative factor developer specializing in Qlib-compatible factors.

Your task is to generate Python factor code based on user requirements.

Guidelines:
1. Generate clean, well-documented Python code
2. Use pandas and numpy for calculations
3. Ensure the factor function takes a DataFrame and returns a Series
4. Follow Qlib naming conventions
5. Include docstrings explaining the factor logic
6. Handle edge cases (NaN values, empty data)

Output format:
- Return ONLY the Python code in a ```python code block
- The function should be named descriptively
- Include type hints where appropriate"""

    def _build_examples(self) -> str:
        """Build example factors for few-shot learning."""
        return '''
Example 1 - Momentum Factor:
```python
def momentum_20d(df):
    """20-day momentum factor based on price returns.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with momentum values
    """
    returns = df['close'].pct_change(20)
    return returns.fillna(0)
```

Example 2 - Volatility Factor:
```python
def volatility_20d(df):
    """20-day rolling volatility factor.

    Args:
        df: DataFrame with 'close' column

    Returns:
        Series with volatility values
    """
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=20).std()
    return volatility.fillna(0)
```
'''

    def get_system_prompt(self) -> str:
        """Get the system prompt for LLM."""
        return self._system_prompt

    def render(
        self,
        user_request: str,
        factor_family: Optional[FactorFamily] = None,
        include_examples: bool = False,
        include_field_constraints: bool = False,
    ) -> str:
        """Render the full prompt for factor generation.

        Args:
            user_request: User's natural language request
            factor_family: Optional factor family constraint
            include_examples: Whether to include few-shot examples
            include_field_constraints: Whether to include strict field constraints

        Returns:
            Rendered prompt string
        """
        parts = [f"User Request: {user_request}"]

        if factor_family:
            allowed_fields = factor_family.get_allowed_fields()
            parts.append(
                f"\nFactor Family: {factor_family.value}"
                f"\nAllowed Fields: {', '.join(sorted(allowed_fields))}"
            )

            if include_field_constraints:
                field_descriptions = factor_family.get_field_descriptions()
                constraint_text = (
                    "\n\n**IMPORTANT FIELD CONSTRAINTS:**"
                    "\nYou MUST only use the allowed fields listed above."
                    "\nDo NOT use any fields outside this list."
                    "\n\nField Descriptions:"
                )
                for field_name in sorted(allowed_fields):
                    desc = field_descriptions.get(field_name, "Data field")
                    constraint_text += f"\n- {field_name}: {desc}"
                parts.append(constraint_text)

        if include_examples:
            parts.append(f"\n{self._examples}")

        parts.append(
            "\nGenerate the Python factor code following the guidelines above."
        )

        return "\n".join(parts)


class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider interface."""

    async def complete(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Complete a prompt."""
        ...


@dataclass
class FactorGenerationConfig:
    """Configuration for factor generation agent."""

    name: str
    security_check_enabled: bool = True
    field_constraint_enabled: bool = True
    max_retries: int = 3
    timeout_seconds: float = 30.0
    include_examples: bool = True


class ASTSecurityChecker:
    """AST-based security checker for generated code."""

    DANGEROUS_MODULES = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "telnetlib",
        "smtplib",
        "pickle",
        "marshal",
        "shelve",
    }

    DANGEROUS_BUILTINS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }

    DANGEROUS_METHODS = {
        "system",
        "popen",
        "spawn",
        "call",
        "run",
        "Popen",
    }

    def check(self, code: str) -> tuple[bool, list[str]]:
        """Check code for security violations.

        Args:
            code: Python code to check

        Returns:
            Tuple of (is_safe, list of violation messages)
        """
        violations: list[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error: {e}"]

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module in self.DANGEROUS_MODULES:
                        violations.append(f"Dangerous import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split(".")[0]
                    if module in self.DANGEROUS_MODULES:
                        violations.append(f"Dangerous import from: {node.module}")

            # Check function calls
            elif isinstance(node, ast.Call):
                func_name = self._get_func_name(node)
                if func_name in self.DANGEROUS_BUILTINS:
                    violations.append(f"Dangerous builtin: {func_name}")
                elif func_name in self.DANGEROUS_METHODS:
                    violations.append(f"Dangerous method: {func_name}")

                # Check for attribute calls like os.system
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in self.DANGEROUS_METHODS:
                        violations.append(f"Dangerous method call: {node.func.attr}")

        is_safe = len(violations) == 0
        return is_safe, violations

    def _get_func_name(self, node: ast.Call) -> str:
        """Extract function name from Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""


class FactorFieldValidator:
    """Validator for factor field constraints.

    Extracts field names from generated code and validates
    them against the allowed fields for a factor family.
    """

    # Patterns to extract field names from code
    # df['field'], df["field"], df.field, df.loc[:, 'field']
    BRACKET_SINGLE_PATTERN = re.compile(r"df\s*\[\s*'([a-zA-Z_][a-zA-Z0-9_]*)'\s*\]")
    BRACKET_DOUBLE_PATTERN = re.compile(r'df\s*\[\s*"([a-zA-Z_][a-zA-Z0-9_]*)"\s*\]')
    DOT_PATTERN = re.compile(r"df\.([a-zA-Z_][a-zA-Z0-9_]*)")
    LOC_SINGLE_PATTERN = re.compile(r"df\.loc\s*\[\s*[^,]*,\s*'([a-zA-Z_][a-zA-Z0-9_]*)'\s*\]")
    LOC_DOUBLE_PATTERN = re.compile(r'df\.loc\s*\[\s*[^,]*,\s*"([a-zA-Z_][a-zA-Z0-9_]*)"\s*\]')
    LIST_SINGLE_PATTERN = re.compile(r"'([a-zA-Z_][a-zA-Z0-9_]*)'")
    LIST_DOUBLE_PATTERN = re.compile(r'"([a-zA-Z_][a-zA-Z0-9_]*)"')

    # Common pandas attributes to exclude from field extraction
    PANDAS_ATTRS = {
        "pct_change", "rolling", "mean", "std", "sum", "min", "max",
        "shift", "diff", "cumsum", "cumprod", "fillna", "dropna",
        "copy", "head", "tail", "iloc", "loc", "values", "index",
        "columns", "shape", "dtypes", "apply", "map", "transform",
        "groupby", "resample", "ewm", "expanding", "rank", "abs",
        "clip", "corr", "cov", "count", "describe", "quantile",
    }

    def extract_fields(self, code: str) -> set[str]:
        """Extract field names used in the code.

        Args:
            code: Python code to analyze

        Returns:
            Set of field names found in the code
        """
        fields: set[str] = set()

        # Extract from bracket notation with single quotes
        fields.update(self.BRACKET_SINGLE_PATTERN.findall(code))

        # Extract from bracket notation with double quotes
        fields.update(self.BRACKET_DOUBLE_PATTERN.findall(code))

        # Extract from dot notation (filter out pandas methods)
        dot_matches = self.DOT_PATTERN.findall(code)
        for match in dot_matches:
            if match not in self.PANDAS_ATTRS:
                fields.add(match)

        # Extract from loc accessor
        fields.update(self.LOC_SINGLE_PATTERN.findall(code))
        fields.update(self.LOC_DOUBLE_PATTERN.findall(code))

        # Try to extract fields from list patterns like ['close', 'volume']
        # Look for list-like patterns used with df[]
        list_pattern = re.compile(r"df\s*\[\s*\[([^\]]+)\]\s*\]")
        list_matches = list_pattern.findall(code)
        for match in list_matches:
            fields.update(self.LIST_SINGLE_PATTERN.findall(match))
            fields.update(self.LIST_DOUBLE_PATTERN.findall(match))

        return fields

    def validate(self, code: str, family: FactorFamily) -> FieldValidationResult:
        """Validate field usage against factor family constraints.

        Args:
            code: Python code to validate
            family: Factor family to validate against

        Returns:
            Validation result with details
        """
        used_fields = self.extract_fields(code)
        allowed_fields = set(family.get_allowed_fields())

        violations = [f for f in used_fields if f not in allowed_fields]

        return FieldValidationResult(
            is_valid=len(violations) == 0,
            used_fields=used_fields,
            allowed_fields=allowed_fields,
            violations=violations,
        )


class FactorGenerationAgent:
    """Agent for generating Qlib-compatible factors from natural language."""

    def __init__(
        self,
        config: FactorGenerationConfig,
        llm_provider: LLMProviderProtocol,
    ) -> None:
        """Initialize the factor generation agent.

        Args:
            config: Agent configuration
            llm_provider: LLM provider for code generation
        """
        self.config = config
        self.llm_provider = llm_provider
        self.template = FactorPromptTemplate()
        self.security_checker = ASTSecurityChecker()
        self.field_validator = FactorFieldValidator()

    async def generate(
        self,
        user_request: str,
        factor_family: Optional[FactorFamily] = None,
    ) -> GeneratedFactor:
        """Generate a factor from natural language description.

        Args:
            user_request: Natural language description of the factor
            factor_family: Optional factor family constraint

        Returns:
            Generated factor with code and metadata

        Raises:
            ValueError: If request is empty
            FactorGenerationError: If LLM call fails
            InvalidFactorError: If no valid code is generated
            SecurityViolationError: If code fails security checks
            FieldConstraintViolationError: If code uses disallowed fields
        """
        # Validate input
        if not user_request or not user_request.strip():
            raise ValueError("Request cannot be empty")

        # Build prompt with field constraints if enabled
        prompt = self.template.render(
            user_request=user_request,
            factor_family=factor_family,
            include_examples=self.config.include_examples,
            include_field_constraints=self.config.field_constraint_enabled,
        )
        system_prompt = self.template.get_system_prompt()

        # Call LLM
        try:
            response = await self.llm_provider.complete(
                prompt=prompt,
                system_prompt=system_prompt,
            )
            raw_content = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            raise FactorGenerationError(f"LLM call failed: {e}") from e

        # Extract code from response
        code = self._extract_code(raw_content)
        if not code:
            raise InvalidFactorError("No code found in LLM response")

        # Security check
        if self.config.security_check_enabled:
            is_safe, violations = self.security_checker.check(code)
            if not is_safe:
                raise SecurityViolationError(
                    f"Security violations: {', '.join(violations)}"
                )

        # Field constraint validation
        if self.config.field_constraint_enabled and factor_family:
            validation_result = self.field_validator.validate(code, factor_family)
            if not validation_result.is_valid:
                raise FieldConstraintViolationError(
                    f"Field constraint violations for {factor_family.value} family: "
                    f"{', '.join(validation_result.violations)}. "
                    f"Allowed fields: {', '.join(sorted(validation_result.allowed_fields))}",
                    violations=validation_result.violations,
                )

        # Extract factor name and description
        name = self._extract_factor_name(code)
        description = self._extract_description(code, user_request)

        # Determine family
        family = factor_family or self._infer_family(user_request)

        return GeneratedFactor(
            name=name,
            description=description,
            code=code,
            family=family,
            metadata={
                "user_request": user_request,
                "security_checked": self.config.security_check_enabled,
            },
        )

    def _extract_code(self, content: str) -> str:
        """Extract Python code from LLM response.

        Args:
            content: Raw LLM response content

        Returns:
            Extracted Python code or empty string
        """
        # Try to extract from markdown code blocks
        pattern = r"```(?:python)?\s*\n?(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Try to find function definition directly
        if "def " in content and "return " in content:
            # Find the function definition
            lines = content.split("\n")
            code_lines: list[str] = []
            in_function = False

            for line in lines:
                if line.strip().startswith("def "):
                    in_function = True
                if in_function:
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines).strip()

        return ""

    def _extract_factor_name(self, code: str) -> str:
        """Extract factor name from code.

        Args:
            code: Python code

        Returns:
            Factor function name
        """
        match = re.search(r"def\s+(\w+)\s*\(", code)
        if match:
            return match.group(1)
        return "generated_factor"

    def _extract_description(self, code: str, user_request: str) -> str:
        """Extract description from code docstring or user request.

        Args:
            code: Python code
            user_request: Original user request

        Returns:
            Factor description
        """
        # Try to extract docstring
        match = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if match:
            return match.group(1).strip().split("\n")[0]

        match = re.search(r"'''(.*?)'''", code, re.DOTALL)
        if match:
            return match.group(1).strip().split("\n")[0]

        return user_request[:100]

    def _infer_family(self, user_request: str) -> FactorFamily:
        """Infer factor family from user request.

        Args:
            user_request: Natural language request

        Returns:
            Inferred factor family
        """
        request_lower = user_request.lower()

        keywords = {
            FactorFamily.MOMENTUM: ["momentum", "return", "trend", "roc", "price change"],
            FactorFamily.VALUE: ["value", "pe", "pb", "book", "earnings", "fundamental"],
            FactorFamily.VOLATILITY: ["volatility", "vol", "std", "variance", "atr"],
            FactorFamily.QUALITY: ["quality", "roe", "roa", "profit", "margin"],
            FactorFamily.SENTIMENT: ["sentiment", "news", "social", "opinion"],
            FactorFamily.LIQUIDITY: ["liquidity", "volume", "turnover", "spread", "amihud"],
        }

        for family, words in keywords.items():
            for word in words:
                if word in request_lower:
                    return family

        return FactorFamily.MOMENTUM  # Default
