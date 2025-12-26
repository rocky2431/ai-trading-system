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

from iqfmp.llm.validation import (
    ExpressionGate,
    ExpressionValidationResult,
    FieldSet,
)
from iqfmp.core.security import ASTSecurityChecker
from iqfmp.agents.field_capability import (
    DataSourceType,
    DynamicCapability,
    FieldRegistry,
    OperatorCatalog,
    TechnicalIndicatorCatalog,
    INDICATOR_CATALOG,
    create_default_capability,
    create_capability_for_sources,
    validate_expression_fields,
    generate_field_error_feedback,
)


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
    """Factor family definitions with allowed fields.

    Extended for cryptocurrency markets with crypto-specific families.
    """

    # Traditional factor families
    MOMENTUM = "momentum"
    VALUE = "value"
    VOLATILITY = "volatility"
    QUALITY = "quality"
    SENTIMENT = "sentiment"
    LIQUIDITY = "liquidity"

    # Crypto-specific factor families
    FUNDING = "funding"  # Perpetual futures funding rate factors
    OPEN_INTEREST = "open_interest"  # Open interest dynamics
    LIQUIDATION = "liquidation"  # Liquidation-based factors
    ORDERBOOK = "orderbook"  # Market microstructure factors
    ONCHAIN = "onchain"  # On-chain metrics (for major coins)

    def get_allowed_fields(self) -> list[str]:
        """Get allowed data fields for this factor family.

        Includes both traditional and crypto-specific fields.
        """
        # Core OHLCV fields available to all families
        base_fields = ["open", "high", "low", "close", "volume", "quote_volume"]

        family_specific = {
            # Traditional factor families
            FactorFamily.MOMENTUM: [
                "close",
                "returns",
                "price_change",
                "momentum",
                "roc",
                "funding_rate",  # Crypto: funding momentum
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
                "long_short_ratio",  # Crypto: positioning sentiment
                "taker_buy_ratio",  # Crypto: aggressive buying
            ],
            FactorFamily.LIQUIDITY: [
                "close",
                "volume",
                "bid_ask_spread",
                "turnover",
                "amihud",
                "spread",  # Crypto: bid-ask spread
                "depth_imbalance",  # Crypto: orderbook imbalance
            ],
            # Crypto-specific factor families
            FactorFamily.FUNDING: [
                "close",
                "funding_rate",
                "funding_rate_predicted",
                "mark_price",
                "index_price",
            ],
            FactorFamily.OPEN_INTEREST: [
                "close",
                "volume",
                "open_interest",
                "open_interest_change",
            ],
            FactorFamily.LIQUIDATION: [
                "close",
                "volume",
                "liquidation_long",
                "liquidation_short",
                "liquidation_total",
                "open_interest",
            ],
            FactorFamily.ORDERBOOK: [
                "close",
                "bid_price",
                "ask_price",
                "bid_volume",
                "ask_volume",
                "spread",
                "depth_imbalance",
            ],
            FactorFamily.ONCHAIN: [
                "close",
                "volume",
                "exchange_inflow",
                "exchange_outflow",
                "exchange_netflow",
                "whale_transactions",
                "active_addresses",
            ],
        }

        return list(set(base_fields + family_specific.get(self, [])))

    def get_field_descriptions(self) -> dict[str, str]:
        """Get descriptions for each allowed field.

        Returns:
            Dictionary mapping field names to descriptions

        Includes crypto-specific fields with detailed descriptions.
        """
        base_descriptions = {
            "open": "Opening price of the period",
            "high": "Highest price during the period",
            "low": "Lowest price during the period",
            "close": "Closing price of the period",
            "volume": "Trading volume in base currency",
            "quote_volume": "Trading volume in quote currency (USDT) - better for cross-asset comparison",
        }

        family_descriptions = {
            # Traditional factor families
            FactorFamily.MOMENTUM: {
                "returns": "Price returns over a period",
                "price_change": "Absolute price change",
                "momentum": "Price momentum indicator",
                "roc": "Rate of change indicator",
                "funding_rate": "Perpetual funding rate (positive = longs pay shorts)",
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
                "long_short_ratio": "Ratio of long to short accounts (>1 = more longs)",
                "taker_buy_ratio": "Taker buy volume / (buy + sell), >0.5 = net buying",
            },
            FactorFamily.LIQUIDITY: {
                "bid_ask_spread": "Bid-ask spread",
                "turnover": "Trading turnover rate",
                "amihud": "Amihud illiquidity measure",
                "spread": "Current bid-ask spread as percentage",
                "depth_imbalance": "Orderbook imbalance: (bid - ask) / (bid + ask)",
            },
            # Crypto-specific factor families
            FactorFamily.FUNDING: {
                "funding_rate": "8-hour funding rate (positive = longs pay shorts)",
                "funding_rate_predicted": "Predicted next funding rate",
                "mark_price": "Mark price used for liquidation calculations",
                "index_price": "Underlying spot index price",
            },
            FactorFamily.OPEN_INTEREST: {
                "open_interest": "Total open interest in USD",
                "open_interest_change": "Change in open interest over period",
            },
            FactorFamily.LIQUIDATION: {
                "liquidation_long": "Long position liquidation volume",
                "liquidation_short": "Short position liquidation volume",
                "liquidation_total": "Total liquidation volume",
                "open_interest": "Open interest for context",
            },
            FactorFamily.ORDERBOOK: {
                "bid_price": "Best bid price",
                "ask_price": "Best ask price",
                "bid_volume": "Total bid depth near best bid",
                "ask_volume": "Total ask depth near best ask",
                "spread": "Bid-ask spread as percentage",
                "depth_imbalance": "Orderbook imbalance: (bid - ask) / (bid + ask)",
            },
            FactorFamily.ONCHAIN: {
                "exchange_inflow": "Tokens flowing into exchanges (potential sell pressure)",
                "exchange_outflow": "Tokens flowing out of exchanges (accumulation)",
                "exchange_netflow": "Net token flow to exchanges",
                "whale_transactions": "Count of large transactions (>$100k)",
                "active_addresses": "Number of active addresses on-chain",
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
    """Template for generating factor prompts.

    This class now delegates to the modular crypto-optimized prompts
    in iqfmp.llm.prompts for comprehensive few-shot examples and
    crypto-specific context.

    For direct access to the enhanced prompts, use:
        from iqfmp.llm.prompts import FactorGenerationPrompt

    B3 Feature: Dynamic field capability injection
    - Automatically detects available data sources
    - Injects field constraints into prompts
    - Provides operator reference for LLM
    """

    def __init__(
        self,
        use_crypto_prompts: bool = True,
        data_sources: list[DataSourceType] | None = None,
    ) -> None:
        """Initialize prompt template.

        Args:
            use_crypto_prompts: If True, use crypto-optimized prompts from
                iqfmp.llm.prompts module. If False, use legacy prompts.
            data_sources: List of data sources to enable. If None, uses
                OHLCV + DERIVATIVES by default.
        """
        self._use_crypto = use_crypto_prompts
        self._crypto_prompt = None

        if use_crypto_prompts:
            try:
                from iqfmp.llm.prompts import FactorGenerationPrompt
                self._crypto_prompt = FactorGenerationPrompt()
            except ImportError:
                self._use_crypto = False

        # B3: Initialize dynamic capability with configured data sources
        if data_sources is None:
            self._capability = create_default_capability()
        else:
            self._capability = create_capability_for_sources(data_sources)

        # Legacy prompts as fallback
        self._system_prompt = self._build_system_prompt()
        self._examples = self._build_examples()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for factor generation.

        Returns crypto-optimized prompt if available, otherwise legacy.
        B3: Now uses dynamic field capability injection.
        """
        if self._crypto_prompt:
            return self._crypto_prompt.get_system_prompt()

        # B3: Build system prompt with dynamic field constraints
        base_prompt = """You are an expert quantitative factor developer specializing in cryptocurrency markets.

Your task is to generate **Qlib expression** factors that implement the user's hypothesis.

"""
        # Inject dynamic field constraints and operators from capability
        capability_context = self._capability.generate_full_context()

        additional_instructions = """
## CRITICAL REQUIREMENT

You MUST implement ALL indicators the user mentions (WR, SSL, MACD, Zigzag, Bollinger, etc.).
Research each indicator's formula and translate it to Qlib expression syntax.
The system will provide feedback if your implementation is incomplete.

## Output Format

Return ONLY a single Qlib expression. No Python code, no markdown.

You may add a brief comment after the expression starting with #"""

        return base_prompt + capability_context + additional_instructions

    def _build_examples(self) -> str:
        """Build example factors for few-shot learning."""
        if self._crypto_prompt:
            # Get examples from crypto prompt and format them
            examples = self._crypto_prompt.get_examples()
            parts = []
            for i, ex in enumerate(examples, 1):
                if ex.get("role") == "assistant":
                    parts.append(f"Example {i}:\n{ex.get('content', '')}")
            return "\n\n".join(parts[:3])  # Limit to 3 examples

        # Legacy examples (fallback) - now uses Qlib expressions
        return '''
Example 1 - Momentum:
Ref($close, -20) / $close - 1
# 20-period momentum

Example 2 - Mean Reversion:
($close - Mean($close, 20)) / Std($close, 20)
# Z-score mean reversion

Example 3 - Volatility Regime:
Std($close, 20) / Std($close, 60)
# Short-term vs long-term volatility

Example 4 - Volume Surge:
$volume / Mean($volume, 20)
# Volume ratio relative to average

Example 5 - RSI:
RSI($close, 14)
# 14-period RSI
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
        include_capability_context: bool = True,
    ) -> str:
        """Render the full prompt for factor generation.

        Args:
            user_request: User's natural language request
            factor_family: Optional factor family constraint
            include_examples: Whether to include few-shot examples
            include_field_constraints: Whether to include strict field constraints
            include_capability_context: Whether to include dynamic capability context (B3)

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

        # B3: Include dynamic capability context
        if include_capability_context:
            parts.append(f"\n{self._capability.generate_full_context()}")

        if include_examples:
            parts.append(f"\n{self._examples}")

        parts.append(
            "\nGenerate the Qlib expression following the guidelines above."
        )

        return "\n".join(parts)

    def get_capability(self) -> DynamicCapability:
        """Get the dynamic capability instance.

        Returns:
            DynamicCapability instance for field validation
        """
        return self._capability

    def enable_data_source(self, source: DataSourceType) -> None:
        """Enable an additional data source.

        Args:
            source: Data source type to enable
        """
        self._capability.field_registry.enable_source(source)
        # Rebuild system prompt with new fields
        self._system_prompt = self._build_system_prompt()

    def add_custom_field(
        self,
        name: str,
        description: str,
        data_type: str = "float",
    ) -> None:
        """Add a custom field to the capability.

        Args:
            name: Field name (with or without $ prefix)
            description: Field description
            data_type: Data type (float, int, bool)
        """
        self._capability.add_custom_field(name, description, data_type)
        # Rebuild system prompt with new fields
        self._system_prompt = self._build_system_prompt()


class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider interface."""

    async def complete(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Complete a prompt."""
        ...


@dataclass
class FactorGenerationConfig:
    """Configuration for factor generation agent.

    Attributes:
        name: Agent name for logging
        model: LLM model to use (defaults from AgentModelRegistry)
        temperature: Sampling temperature (lower = more deterministic)
        n_candidates: Number of candidates to generate and rerank
        candidate_seed: Optional seed for candidate generation
        rerank_by_indicator_coverage: Prefer candidates that satisfy requested indicators
        max_refine_rounds: Optional indicator-driven refinement rounds
        tool_context_enabled: Enable safe read-only tool context injection
        tool_similar_factors_limit: Max similar factors to include
        security_check_enabled: Enable AST security checking
        field_constraint_enabled: Enable field constraint validation
        max_retries: Maximum LLM call retries
        timeout_seconds: LLM call timeout
        include_examples: Include few-shot examples in prompt
        vector_dedup_enabled: Enable vector-based duplicate detection (Phase 2 fix)
        vector_dedup_threshold: Similarity threshold for duplicate detection (0-1)
    """

    name: str
    model: Optional[str] = None  # ModelType.value, None = use registry default
    temperature: Optional[float] = None  # None = use registry default
    n_candidates: int = 3
    candidate_seed: Optional[int] = None
    rerank_by_indicator_coverage: bool = True
    max_refine_rounds: int = 1
    tool_context_enabled: bool = False
    tool_similar_factors_limit: int = 5
    security_check_enabled: bool = True
    field_constraint_enabled: bool = True
    max_retries: int = 3
    timeout_seconds: float = 30.0
    include_examples: bool = True
    # Phase 2: Vector-based duplicate detection
    vector_dedup_enabled: bool = True  # ON by default to prevent "memory loss"
    vector_dedup_threshold: float = 0.85  # 85% similarity = duplicate
    # 严格模式：如果 vector_dedup_enabled=True 但 Qdrant 不可用，抛出异常而非静默降级
    # 生产环境必须设置为 True，只有测试环境可以设置为 False
    vector_strict_mode: bool = True


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
        import logging
        self._logger = logging.getLogger(__name__)

        self.config = config
        self.llm_provider = llm_provider
        self.template = FactorPromptTemplate()
        self.security_checker = ASTSecurityChecker()
        self.field_validator = FactorFieldValidator()
        # Use CRYPTO field set for cryptocurrency trading system
        self.expression_gate = ExpressionGate(field_set=FieldSet.CRYPTO)

        # Phase 2: Initialize vector store for duplicate detection
        # 严格模式（默认）：如果 Qdrant 不可用，抛出异常而非静默降级
        self._vector_searcher = None
        self._vector_store = None
        if self.config.vector_dedup_enabled:
            try:
                from iqfmp.vector import (
                    SimilaritySearcher,
                    FactorVectorStore,
                    QdrantUnavailableError,
                    QdrantConfig,
                )

                # 根据严格模式决定是否允许 Mock
                qdrant_config = QdrantConfig(
                    allow_mock=not self.config.vector_strict_mode
                )

                self._vector_searcher = SimilaritySearcher(
                    similarity_threshold=self.config.vector_dedup_threshold,
                    qdrant_config=qdrant_config,
                )
                self._vector_store = FactorVectorStore(qdrant_config=qdrant_config)
                self._logger.info(
                    f"Vector dedup enabled: threshold={self.config.vector_dedup_threshold}, "
                    f"strict_mode={self.config.vector_strict_mode}"
                )
            except ImportError as e:
                # qdrant-client 未安装
                if self.config.vector_strict_mode:
                    raise RuntimeError(
                        "Vector dedup is enabled but qdrant-client is not installed. "
                        "Install with: pip install qdrant-client. "
                        "Or set vector_strict_mode=False for testing without persistence."
                    ) from e
                else:
                    self._logger.warning(
                        f"Vector dedup unavailable ({e}). "
                        "Agent will generate factors without memory. "
                        "Set vector_strict_mode=True in production!"
                    )
            except Exception as e:
                # 其他错误（如 Qdrant 服务不可用）
                if self.config.vector_strict_mode:
                    raise RuntimeError(
                        f"Vector dedup is enabled but Qdrant is unavailable: {e}. "
                        "Please start Qdrant service or set vector_strict_mode=False for testing."
                    ) from e
                else:
                    self._logger.warning(
                        f"Vector dedup unavailable ({e}). "
                        "Agent will generate factors without memory. "
                        "Set vector_strict_mode=True in production!"
                    )

    def _check_duplicate_factor(
        self,
        user_request: str,
        factor_family: Optional[FactorFamily] = None,
    ) -> Optional["GeneratedFactor"]:
        """Check if a similar factor already exists in vector store.

        Phase 2 Anti-Memory-Loss: Before generating a new factor, check if
        we've already generated something similar. This prevents the Agent
        from "reinventing the wheel" every time.

        Args:
            user_request: Natural language description of the factor
            factor_family: Optional factor family constraint

        Returns:
            GeneratedFactor if duplicate found, None otherwise
        """
        if not self._vector_searcher:
            return None

        try:
            # Search for similar factors based on the hypothesis/request
            is_duplicate, similar = self._vector_searcher.check_duplicate(
                code="",  # We don't have code yet, search by hypothesis
                name="",
                hypothesis=user_request,
                threshold=self.config.vector_dedup_threshold,
            )

            if is_duplicate and similar:
                self._logger.info(
                    f"🔄 DUPLICATE DETECTED: Found similar factor '{similar.name}' "
                    f"(score={similar.score:.3f} >= {self.config.vector_dedup_threshold}). "
                    "Returning cached factor instead of regenerating."
                )

                # Return the existing factor as GeneratedFactor
                return GeneratedFactor(
                    code=similar.code,
                    name=similar.name,
                    family=similar.family,
                    hypothesis=similar.hypothesis,
                    is_python=False,  # Assume Qlib expression
                    metadata={
                        "from_cache": True,
                        "cache_score": similar.score,
                        "original_id": similar.factor_id,
                        **similar.metadata,
                    },
                )

            return None

        except Exception as e:
            self._logger.warning(f"Duplicate check failed: {e}. Proceeding with generation.")
            return None

    def _store_generated_factor(
        self,
        factor: "GeneratedFactor",
        user_request: str,
    ) -> None:
        """Store generated factor in vector store for future dedup.

        Args:
            factor: The generated factor to store
            user_request: Original user request (hypothesis)
        """
        if not self._vector_store:
            return

        try:
            import uuid
            factor_id = str(uuid.uuid4())

            self._vector_store.add_factor(
                factor_id=factor_id,
                name=factor.name,
                code=factor.code,
                hypothesis=user_request,
                family=factor.family,
                metadata={
                    "is_python": factor.is_python,
                    **(factor.metadata or {}),
                },
            )

            self._logger.info(
                f"📦 Stored factor '{factor.name}' in vector store (id={factor_id[:8]}...)"
            )

        except Exception as e:
            self._logger.warning(f"Failed to store factor in vector store: {e}")

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

        # ================================================================
        # Phase 2: Check for duplicate factor BEFORE LLM generation
        # This prevents the Agent from "reinventing the wheel"
        # ================================================================
        cached_factor = self._check_duplicate_factor(user_request, factor_family)
        if cached_factor is not None:
            return cached_factor

        # Build prompt with field constraints if enabled
        prompt = self.template.render(
            user_request=user_request,
            factor_family=factor_family,
            include_examples=self.config.include_examples,
            include_field_constraints=self.config.field_constraint_enabled,
        )

        # Optional safe tool context (read-only): fields + similar factors
        if self.config.tool_context_enabled:
            try:
                from iqfmp.llm.tools.read_only import list_available_fields, search_similar_factors

                fields = list_available_fields(field_set=FieldSet.CRYPTO, include_prefix=True)
                similar = search_similar_factors(
                    hypothesis=user_request,
                    limit=max(0, self.config.tool_similar_factors_limit),
                )

                tool_context = [
                    "## Read-only Tool Context",
                    "",
                    f"- Available fields ({len(fields)}): {', '.join(fields[:60])}"
                    + (" ..." if len(fields) > 60 else ""),
                ]
                if similar:
                    tool_context.append("- Similar existing factors (avoid duplicates):")
                    for item in similar:
                        tool_context.append(
                            f"  - {item.get('name','')} (score={item.get('score',0):.3f}, family={item.get('family','')})"
                        )
                prompt = f"{prompt}\n\n" + "\n".join(tool_context)
            except Exception:
                # Tool context is best-effort; never break generation.
                pass
        default_system_prompt = self.template.get_system_prompt()

        # Resolve agent-specific model configuration
        from iqfmp.agents.model_config import get_agent_full_config

        model_id, config_temperature, custom_system_prompt = get_agent_full_config(
            "factor_generation"
        )

        model = self.config.model or model_id
        temperature = self.config.temperature or config_temperature
        system_prompt = (
            custom_system_prompt if custom_system_prompt else default_system_prompt
        )

        n_candidates = max(1, self.config.n_candidates)

        # Generate candidates (prefer provider-native multi-candidate API when available)
        raw_candidates: list[str] = []
        try:
            provider_any: Any = self.llm_provider
            generate_candidates = getattr(provider_any, "generate_candidates", None)
            if n_candidates > 1 and callable(generate_candidates):
                import inspect

                if not inspect.iscoroutinefunction(generate_candidates):
                    generate_candidates = None

            if n_candidates > 1 and generate_candidates is not None:
                raw_candidates = await generate_candidates(
                    prompt=prompt,
                    n_candidates=n_candidates,
                    model=model,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    seed=self.config.candidate_seed,
                )
            else:
                for _ in range(n_candidates):
                    response = await self.llm_provider.complete(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=model,
                        temperature=temperature,
                    )
                    raw_content = (
                        response.content if hasattr(response, "content") else str(response)
                    )
                    raw_candidates.append(raw_content)
        except Exception as e:
            raise FactorGenerationError(f"LLM call failed: {e}") from e

        if not raw_candidates:
            raise InvalidFactorError("No candidates returned by LLM provider")

        # Score & select best candidate
        from iqfmp.agents.indicator_intelligence import analyze_indicator_coverage, generate_missing_indicator_feedback

        best_code: Optional[str] = None
        best_is_python = False
        best_score = float("-inf")
        best_analysis = None
        security_violations: list[str] = []
        field_constraint_violations: list[str] = []

        for raw_content in raw_candidates:
            code = self._extract_code(raw_content)
            if not code:
                continue

            is_python_code = (
                code.strip().startswith("def ")
                or code.strip().startswith("import ")
                or code.strip().startswith("from ")
                or "\ndef " in code
            )

            # Security check - only for Python code (Qlib expressions are safe by design)
            if self.config.security_check_enabled and is_python_code:
                check_result = self.security_checker.check(code)
                if not check_result.is_safe:
                    # Extract message strings from SecurityViolation objects
                    security_violations.extend([v.message for v in check_result.violations])
                    continue

            # Validate Qlib expression syntax using ExpressionGate
            if not is_python_code:
                expr_result = self._validate_qlib_expression(code)
                if not expr_result.is_valid:
                    continue

            # Field constraint validation (skip for Qlib expressions - they use $ prefix)
            if self.config.field_constraint_enabled and factor_family and is_python_code:
                validation_result = self.field_validator.validate(code, factor_family)
                if not validation_result.is_valid:
                    field_constraint_violations.extend(validation_result.violations)
                    continue

            analysis = analyze_indicator_coverage(user_request, code)
            score = 0.0
            if self.config.rerank_by_indicator_coverage:
                score += analysis.completion_rate
            # Prefer Qlib expressions in this repo's evaluation pipeline
            if not is_python_code:
                score += 0.05

            if score > best_score:
                best_score = score
                best_code = code
                best_is_python = is_python_code
                best_analysis = analysis

        if not best_code:
            if field_constraint_violations and factor_family is not None:
                raise FieldConstraintViolationError(
                    f"Field constraint violations for {factor_family.value} family: "
                    f"{', '.join(sorted(set(field_constraint_violations)))}. "
                    f"Allowed fields: {', '.join(sorted(factor_family.get_allowed_fields()))}",
                    violations=sorted(set(field_constraint_violations)),
                )
            if security_violations:
                unique = ", ".join(sorted(set(security_violations)))
                raise SecurityViolationError(f"Security violations: {unique}")
            raise InvalidFactorError("No code found in LLM response")

        # Optional: indicator-driven refinement loop
        refined_code = best_code
        refined_is_python = best_is_python
        refined_analysis = best_analysis

        for _ in range(max(0, self.config.max_refine_rounds)):
            if refined_analysis is None or refined_analysis.is_complete:
                break

            feedback = generate_missing_indicator_feedback(refined_analysis)
            if not feedback:
                break

            refine_prompt = (
                f"{prompt}\n\n"
                f"## Previous Expression\n{refined_code}\n\n"
                f"{feedback}\n\n"
                "Return ONLY the improved Qlib expression (no explanation)."
            )

            try:
                response = await self.llm_provider.complete(
                    prompt=refine_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                )
            except Exception:
                break

            raw_content = response.content if hasattr(response, "content") else str(response)
            candidate_code = self._extract_code(raw_content)
            if not candidate_code:
                continue

            candidate_is_python = (
                candidate_code.strip().startswith("def ")
                or candidate_code.strip().startswith("import ")
                or candidate_code.strip().startswith("from ")
                or "\ndef " in candidate_code
            )
            if candidate_is_python:
                # We only refine Qlib expressions to match expression-only pipeline.
                continue

            expr_result = self._validate_qlib_expression(candidate_code)
            if not expr_result.is_valid:
                continue

            candidate_analysis = analyze_indicator_coverage(user_request, candidate_code)
            if candidate_analysis.completion_rate >= refined_analysis.completion_rate:
                refined_code = candidate_code
                refined_is_python = candidate_is_python
                refined_analysis = candidate_analysis

        # Extract factor name and description
        name = self._extract_factor_name(refined_code)
        description = self._extract_description(refined_code, user_request)

        # Determine family
        family = factor_family or self._infer_family(user_request)

        metadata: dict[str, Any] = {
            "user_request": user_request,
            "security_checked": self.config.security_check_enabled,
            "is_python_code": refined_is_python,
            "n_candidates": n_candidates,
            "rerank_by_indicator_coverage": self.config.rerank_by_indicator_coverage,
        }
        if refined_analysis is not None:
            metadata.update(
                {
                    "requested_indicators": sorted(refined_analysis.requested),
                    "detected_indicators": sorted(refined_analysis.found),
                    "missing_indicators": sorted(refined_analysis.missing),
                    "indicator_completion_rate": refined_analysis.completion_rate,
                }
            )

        generated_factor = GeneratedFactor(
            name=name,
            description=description,
            code=refined_code,
            family=family,
            metadata=metadata,
        )

        # ================================================================
        # Phase 2: Store generated factor for future dedup
        # This builds the Agent's "memory" of generated factors
        # ================================================================
        self._store_generated_factor(generated_factor, user_request)

        return generated_factor

    def _extract_code(self, content: str) -> str:
        """Extract Qlib expression or Python code from LLM response.

        Args:
            content: Raw LLM response content

        Returns:
            Extracted Qlib expression or Python code
        """
        if not content:
            return ""

        content = content.strip()

        # First, try to extract Qlib expression (new format)
        # Remove markdown code blocks if present
        pattern = r"```(?:python|qlib)?\s*\n?(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            content = matches[0].strip()

        # Split into lines and process
        lines = content.split("\n")

        # Check if it's a Python function (old format) - this should fail validation later
        if "def " in content and "return " in content:
            code_lines: list[str] = []
            in_function = False
            for line in lines:
                if line.strip().startswith("def "):
                    in_function = True
                if in_function:
                    code_lines.append(line)
            if code_lines:
                return "\n".join(code_lines).strip()

        # Extract Qlib expression (new format)
        # Skip comment lines and empty lines, take first non-comment line
        expression_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                # Stop at first comment after expression
                if expression_lines:
                    break
                continue
            expression_lines.append(stripped)

        if expression_lines:
            # Join expression lines (in case of multi-line expressions)
            expression = " ".join(expression_lines)
            # Clean up any extra whitespace
            expression = re.sub(r'\s+', ' ', expression).strip()
            # Heuristic: if it doesn't look like a Qlib expression at all, treat as "no code".
            # This avoids accepting plain natural-language refusals as expressions.
            has_operator = re.search(r"[A-Z][a-zA-Z_]*\s*\(", expression) is not None
            has_number = re.search(r"\d", expression) is not None
            if "$" not in expression and not has_operator and not has_number:
                return ""
            return expression

        # Fallback: return the entire content if it looks like a Qlib expression
        if "$" in content or any(op in content for op in ["Mean(", "Std(", "Ref(", "Sum(", "RSI(", "MACD("]):
            # Extract just the first line that looks like an expression
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and ("$" in stripped or "(" in stripped):
                    return stripped
            return content.split("\n")[0].strip()

        return ""

    def _extract_factor_name(self, code: str) -> str:
        """Extract factor name from code or generate based on expression.

        Args:
            code: Qlib expression or Python code

        Returns:
            Factor name
        """
        # Check for Python function definition (legacy format)
        match = re.search(r"def\s+(\w+)\s*\(", code)
        if match:
            return match.group(1)

        # For Qlib expressions, generate name based on content
        code_lower = code.lower()
        if "rsi(" in code_lower:
            return "rsi_factor"
        elif "macd(" in code_lower:
            return "macd_factor"
        elif "ema(" in code_lower or "wma(" in code_lower:
            return "ma_factor"
        elif "std(" in code_lower:
            if "mean(" in code_lower:
                return "zscore_factor"
            return "volatility_factor"
        elif "ref(" in code_lower:
            return "momentum_factor"
        elif "corr(" in code_lower:
            return "correlation_factor"
        elif "rank(" in code_lower:
            return "rank_factor"
        elif "$volume" in code_lower:
            return "volume_factor"
        elif "max(" in code_lower or "min(" in code_lower:
            return "range_factor"

        return "qlib_factor"

    def _extract_description(self, code: str, user_request: str) -> str:
        """Extract description from code docstring/comment or user request.

        Args:
            code: Qlib expression or Python code
            user_request: Original user request

        Returns:
            Factor description
        """
        # Try to extract Python docstring (legacy format)
        match = re.search(r'"""(.*?)"""', code, re.DOTALL)
        if match:
            return match.group(1).strip().split("\n")[0]

        match = re.search(r"'''(.*?)'''", code, re.DOTALL)
        if match:
            return match.group(1).strip().split("\n")[0]

        # Try to extract comment from Qlib expression
        # Format: expression\n# comment
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped[1:].strip()[:100]

        return user_request[:100]

    def _validate_qlib_expression(
        self,
        expression: str,
        allowed_fields: Optional[list[str]] = None,
    ) -> ExpressionValidationResult:
        """Validate Qlib expression syntax using ExpressionGate.

        Args:
            expression: Qlib expression string
            allowed_fields: List of allowed field names (without $ prefix).
                           If None, defaults to OHLCV fields.

        Returns:
            ExpressionValidationResult with detailed validation status.
        """
        return self.expression_gate.validate(expression, allowed_fields)

    def _is_valid_qlib_expression(
        self,
        expression: str,
        allowed_fields: Optional[list[str]] = None,
    ) -> bool:
        """Quick check if expression is valid.

        Args:
            expression: Qlib expression string
            allowed_fields: List of allowed field names

        Returns:
            True if valid, False otherwise
        """
        result = self._validate_qlib_expression(expression, allowed_fields)
        return result.is_valid

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

    async def refine(
        self,
        original_code: str,
        error_message: str,
        user_request: str,
        factor_family: Optional[FactorFamily] = None,
    ) -> GeneratedFactor:
        """Refine a failed factor based on error feedback.

        This implements the error feedback loop: when a factor fails to compute
        or evaluate, this method takes the error information and generates
        an improved version.

        Args:
            original_code: The original Qlib expression that failed
            error_message: The error message or feedback explaining why it failed
            user_request: Original user request for context
            factor_family: Optional factor family constraint

        Returns:
            Refined GeneratedFactor with improved code

        Raises:
            FactorGenerationError: If refinement fails after max retries
        """
        # Build refinement prompt with error feedback
        refinement_prompt = f"""## Original Factor Expression (FAILED)

{original_code}

## Error / Feedback

{error_message}

## Original Request

{user_request}

## Task

The above factor expression failed. Analyze the error and generate an IMPROVED Qlib expression.

**IMPORTANT RULES:**
1. You can ONLY use these 5 fields: $open, $high, $low, $close, $volume
2. DO NOT use fields like $returns, $quote_volume, $funding_rate - they don't exist!
3. If you need returns, calculate: `$close / Ref($close, -1) - 1`
4. Return ONLY a single Qlib expression, no Python code

**Common Fixes:**
- If "$returns" error: replace with `$close / Ref($close, -1) - 1`
- If "$quote_volume" error: replace with `$volume`
- If "Ts" operator error: Ts is not a valid operator, use proper time-series operators
- If syntax error: check parentheses balance and operator syntax

Return the corrected expression:"""

        system_prompt = """You are an expert quantitative factor developer.
Your task is to fix a failed Qlib expression based on error feedback.

Available Fields (ONLY these 5):
- $open, $high, $low, $close, $volume

Available Operators:
- Ref($field, -N) - Reference N periods ago
- Mean($field, N) - Rolling mean
- Std($field, N) - Rolling standard deviation
- Sum($field, N) - Rolling sum
- Max($field, N), Min($field, N) - Rolling max/min
- Delta($field, N) - Change over N periods
- Rank($field) - Cross-sectional rank
- Abs(), Log(), Sign() - Math operations
- Corr($f1, $f2, N), Cov($f1, $f2, N) - Rolling correlation/covariance
- EMA($field, N), WMA($field, N) - Moving averages
- RSI($field, N) - Relative Strength Index
- MACD($field, fast, slow, signal) - MACD histogram

Arithmetic: +, -, *, /, >, <

Output: ONLY return the corrected Qlib expression. No explanation needed."""

        try:
            # Get model configuration
            from iqfmp.agents.model_config import get_agent_full_config
            model_id, config_temperature, custom_system_prompt = get_agent_full_config(
                "factor_generation"
            )

            model = self.config.model or model_id
            temperature = self.config.temperature or config_temperature

            response = await self.llm_provider.complete(
                prompt=refinement_prompt,
                system_prompt=custom_system_prompt if custom_system_prompt else system_prompt,
                model=model,
                temperature=temperature,
            )
            raw_content = response.content if hasattr(response, "content") else str(response)
            print(f"DEBUG Refinement Response: {raw_content[:500] if raw_content else 'EMPTY'}")

        except Exception as e:
            raise FactorGenerationError(f"LLM refinement call failed: {e}") from e

        # Extract the refined code
        code = self._extract_code(raw_content)
        if not code:
            raise InvalidFactorError("No refined code found in LLM response")

        # Validate Qlib expression using ExpressionGate
        expr_result = self._validate_qlib_expression(code)
        if not expr_result.is_valid:
            raise InvalidFactorError(
                f"Invalid refined Qlib expression: {expr_result.error_message}. "
                f"Expression: {code[:100]}..."
            )

        # Extract metadata
        name = self._extract_factor_name(code)
        description = self._extract_description(code, user_request)
        family = factor_family or self._infer_family(user_request)

        return GeneratedFactor(
            name=name,
            description=description,
            code=code,
            family=family,
            metadata={
                "user_request": user_request,
                "original_code": original_code,
                "error_message": error_message,
                "is_refined": True,
                "security_checked": False,  # Qlib expressions are safe
            },
        )
