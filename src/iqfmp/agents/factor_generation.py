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
    ) -> str:
        """Render the full prompt for factor generation.

        Args:
            user_request: User's natural language request
            factor_family: Optional factor family constraint
            include_examples: Whether to include few-shot examples

        Returns:
            Rendered prompt string
        """
        parts = [f"User Request: {user_request}"]

        if factor_family:
            allowed_fields = factor_family.get_allowed_fields()
            parts.append(
                f"\nFactor Family: {factor_family.value}"
                f"\nAllowed Fields: {', '.join(allowed_fields)}"
            )

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
        """
        # Validate input
        if not user_request or not user_request.strip():
            raise ValueError("Request cannot be empty")

        # Build prompt
        prompt = self.template.render(
            user_request=user_request,
            factor_family=factor_family,
            include_examples=self.config.include_examples,
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
