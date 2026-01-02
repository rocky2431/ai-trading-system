"""Dynamic field capability injection for factor generation.

This module provides intelligent detection and injection of available data fields
into prompts, enabling the LLM to generate expressions using only valid fields.

B3 Task: Dynamic field capability injection
- Auto-detect available data sources
- Provide field constraints to prevent invalid expressions
- Generate context for prompt injection

B5 Task: Technical Indicator Catalog (指标/算子字典)
- Complete technical indicator library with Qlib formulas
- Searchable catalog by name, alias, or category
- Formula generation with parameter substitution
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================================
# Data Source Types
# ============================================================================


class DataSourceType(Enum):
    """Types of data sources available in the system."""

    OHLCV = "ohlcv"  # Basic price/volume data
    DERIVATIVES = "derivatives"  # Calculated indicators
    ORDERBOOK = "orderbook"  # Order book depth data
    ONCHAIN = "onchain"  # Blockchain metrics
    SENTIMENT = "sentiment"  # Social/news sentiment


# ============================================================================
# Field Definitions
# ============================================================================


@dataclass
class FieldDefinition:
    """Definition of a data field with metadata."""

    name: str
    description: str
    data_type: str  # float, int, bool
    source: DataSourceType
    qlib_compatible: bool = True
    example_values: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)


# Core OHLCV fields - always available
CORE_FIELDS: dict[str, FieldDefinition] = {
    "$open": FieldDefinition(
        name="$open",
        description="Opening price of the period",
        data_type="float",
        source=DataSourceType.OHLCV,
        example_values=["45000.0", "0.5"],
    ),
    "$high": FieldDefinition(
        name="$high",
        description="Highest price during the period",
        data_type="float",
        source=DataSourceType.OHLCV,
        example_values=["46000.0", "0.52"],
    ),
    "$low": FieldDefinition(
        name="$low",
        description="Lowest price during the period",
        data_type="float",
        source=DataSourceType.OHLCV,
        example_values=["44000.0", "0.48"],
    ),
    "$close": FieldDefinition(
        name="$close",
        description="Closing price of the period",
        data_type="float",
        source=DataSourceType.OHLCV,
        example_values=["45500.0", "0.51"],
    ),
    "$volume": FieldDefinition(
        name="$volume",
        description="Trading volume in base currency",
        data_type="float",
        source=DataSourceType.OHLCV,
        example_values=["1000000", "5000000"],
    ),
}

# Derivative fields - commonly calculated
DERIVATIVE_FIELDS: dict[str, FieldDefinition] = {
    "$vwap": FieldDefinition(
        name="$vwap",
        description="Volume-weighted average price",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["45200.0"],
    ),
    "$returns": FieldDefinition(
        name="$returns",
        description="Simple returns (close/close[-1] - 1)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.02", "-0.01"],
    ),
    "$log_returns": FieldDefinition(
        name="$log_returns",
        description="Logarithmic returns",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.0198", "-0.0101"],
    ),
    "$volatility": FieldDefinition(
        name="$volatility",
        description="Rolling volatility (std of returns)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.05", "0.12"],
    ),
    "$turnover": FieldDefinition(
        name="$turnover",
        description="Trading turnover (volume * price)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["45000000000"],
    ),
}

# Order book fields - exchange-specific
ORDERBOOK_FIELDS: dict[str, FieldDefinition] = {
    "$bid_price": FieldDefinition(
        name="$bid_price",
        description="Best bid price",
        data_type="float",
        source=DataSourceType.ORDERBOOK,
        example_values=["45499.0"],
    ),
    "$ask_price": FieldDefinition(
        name="$ask_price",
        description="Best ask price",
        data_type="float",
        source=DataSourceType.ORDERBOOK,
        example_values=["45501.0"],
    ),
    "$bid_volume": FieldDefinition(
        name="$bid_volume",
        description="Volume at best bid",
        data_type="float",
        source=DataSourceType.ORDERBOOK,
        example_values=["100.5"],
    ),
    "$ask_volume": FieldDefinition(
        name="$ask_volume",
        description="Volume at best ask",
        data_type="float",
        source=DataSourceType.ORDERBOOK,
        example_values=["85.2"],
    ),
    "$spread": FieldDefinition(
        name="$spread",
        description="Bid-ask spread",
        data_type="float",
        source=DataSourceType.ORDERBOOK,
        example_values=["2.0", "0.0001"],
    ),
    "$depth_imbalance": FieldDefinition(
        name="$depth_imbalance",
        description="Order book imbalance ratio",
        data_type="float",
        source=DataSourceType.ORDERBOOK,
        example_values=["0.15", "-0.2"],
    ),
}

# On-chain fields - blockchain specific
ONCHAIN_FIELDS: dict[str, FieldDefinition] = {
    "$active_addresses": FieldDefinition(
        name="$active_addresses",
        description="Number of active addresses",
        data_type="int",
        source=DataSourceType.ONCHAIN,
        example_values=["500000", "1000000"],
    ),
    "$transaction_count": FieldDefinition(
        name="$transaction_count",
        description="Number of transactions",
        data_type="int",
        source=DataSourceType.ONCHAIN,
        example_values=["300000", "500000"],
    ),
    "$hash_rate": FieldDefinition(
        name="$hash_rate",
        description="Network hash rate (PoW chains)",
        data_type="float",
        source=DataSourceType.ONCHAIN,
        example_values=["400e18"],
    ),
    "$exchange_inflow": FieldDefinition(
        name="$exchange_inflow",
        description="Tokens flowing into exchanges",
        data_type="float",
        source=DataSourceType.ONCHAIN,
        example_values=["10000", "50000"],
    ),
    "$exchange_outflow": FieldDefinition(
        name="$exchange_outflow",
        description="Tokens flowing out of exchanges",
        data_type="float",
        source=DataSourceType.ONCHAIN,
        example_values=["8000", "45000"],
    ),
}

# Sentiment fields
SENTIMENT_FIELDS: dict[str, FieldDefinition] = {
    "$fear_greed_index": FieldDefinition(
        name="$fear_greed_index",
        description="Crypto Fear & Greed Index (0-100)",
        data_type="int",
        source=DataSourceType.SENTIMENT,
        example_values=["25", "75"],
    ),
    "$social_volume": FieldDefinition(
        name="$social_volume",
        description="Social media mention volume",
        data_type="int",
        source=DataSourceType.SENTIMENT,
        example_values=["50000", "200000"],
    ),
    "$sentiment_score": FieldDefinition(
        name="$sentiment_score",
        description="Aggregated sentiment score (-1 to 1)",
        data_type="float",
        source=DataSourceType.SENTIMENT,
        example_values=["0.3", "-0.5"],
    ),
}


# ============================================================================
# C9: Crypto Perpetuals Derivative Fields (Phase 3)
# ============================================================================

# Crypto perpetuals/futures specific fields - from UnifiedMarketDataProvider
CRYPTO_PERPETUAL_FIELDS: dict[str, FieldDefinition] = {
    # Core derivative fields
    "$funding_rate": FieldDefinition(
        name="$funding_rate",
        description="Perpetual funding rate (positive = longs pay shorts)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.0001", "-0.0005", "0.001"],
        constraints={"typical_range": (-0.01, 0.01)},
    ),
    "$open_interest": FieldDefinition(
        name="$open_interest",
        description="Total open interest in contracts",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["50000", "150000"],
    ),
    "$open_interest_change": FieldDefinition(
        name="$open_interest_change",
        description="Open interest percentage change",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.05", "-0.03"],
    ),
    "$long_short_ratio": FieldDefinition(
        name="$long_short_ratio",
        description="Long/short position ratio (>1 = more longs)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["1.2", "0.8", "1.5"],
        constraints={"typical_range": (0.5, 2.0)},
    ),
    "$liquidation_long": FieldDefinition(
        name="$liquidation_long",
        description="Long position liquidations (USD value)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["1000000", "5000000"],
    ),
    "$liquidation_short": FieldDefinition(
        name="$liquidation_short",
        description="Short position liquidations (USD value)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["800000", "4000000"],
    ),
    "$liquidation_total": FieldDefinition(
        name="$liquidation_total",
        description="Total liquidations (long + short, USD value)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["1800000", "9000000"],
    ),
    # C5: Derived funding features
    "$funding_ma_8h": FieldDefinition(
        name="$funding_ma_8h",
        description="8-hour moving average of funding rate",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.0001", "-0.0002"],
    ),
    "$funding_ma_24h": FieldDefinition(
        name="$funding_ma_24h",
        description="24-hour moving average of funding rate",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.00015", "-0.00015"],
    ),
    "$funding_momentum": FieldDefinition(
        name="$funding_momentum",
        description="Funding rate momentum (change over 24h)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.0001", "-0.0003"],
    ),
    "$funding_zscore": FieldDefinition(
        name="$funding_zscore",
        description="Funding rate z-score vs 30-day mean",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["1.5", "-2.0", "0.5"],
        constraints={"typical_range": (-3.0, 3.0)},
    ),
    "$funding_extreme": FieldDefinition(
        name="$funding_extreme",
        description="Extreme funding flag (|z-score| > 2)",
        data_type="bool",
        source=DataSourceType.DERIVATIVES,
        example_values=["true", "false"],
    ),
    "$funding_annualized": FieldDefinition(
        name="$funding_annualized",
        description="Annualized funding rate (funding_rate * 3 * 365)",
        data_type="float",
        source=DataSourceType.DERIVATIVES,
        example_values=["0.109", "-0.219"],
    ),
}


# ============================================================================
# Qlib Operator Catalog
# ============================================================================


@dataclass
class OperatorDefinition:
    """Definition of a Qlib operator."""

    name: str
    description: str
    syntax: str
    category: str
    parameters: list[str]
    example: str
    notes: str = ""


# Time series operators
TIME_SERIES_OPERATORS: dict[str, OperatorDefinition] = {
    "Ref": OperatorDefinition(
        name="Ref",
        description="Reference value at N periods ago",
        syntax="Ref($field, N)",
        category="time_series",
        parameters=["field: data field", "N: periods back (positive integer)"],
        example="Ref($close, 5)  # Close price 5 periods ago",
    ),
    "Mean": OperatorDefinition(
        name="Mean",
        description="Rolling mean over N periods",
        syntax="Mean($field, N)",
        category="time_series",
        parameters=["field: data field", "N: window size"],
        example="Mean($close, 20)  # 20-period moving average",
    ),
    "Std": OperatorDefinition(
        name="Std",
        description="Rolling standard deviation over N periods",
        syntax="Std($field, N)",
        category="time_series",
        parameters=["field: data field", "N: window size"],
        example="Std($returns, 20)  # 20-period volatility",
    ),
    "Sum": OperatorDefinition(
        name="Sum",
        description="Rolling sum over N periods",
        syntax="Sum($field, N)",
        category="time_series",
        parameters=["field: data field", "N: window size"],
        example="Sum($volume, 5)  # 5-period cumulative volume",
    ),
    "Max": OperatorDefinition(
        name="Max",
        description="Rolling maximum over N periods",
        syntax="Max($field, N)",
        category="time_series",
        parameters=["field: data field", "N: window size"],
        example="Max($high, 20)  # 20-period high",
    ),
    "Min": OperatorDefinition(
        name="Min",
        description="Rolling minimum over N periods",
        syntax="Min($field, N)",
        category="time_series",
        parameters=["field: data field", "N: window size"],
        example="Min($low, 20)  # 20-period low",
    ),
    "Delta": OperatorDefinition(
        name="Delta",
        description="Difference from N periods ago",
        syntax="Delta($field, N)",
        category="time_series",
        parameters=["field: data field", "N: periods back"],
        example="Delta($close, 1)  # Price change from yesterday",
    ),
    "Rank": OperatorDefinition(
        name="Rank",
        description="Cross-sectional rank (0-1)",
        syntax="Rank($field)",
        category="time_series",
        parameters=["field: data field"],
        example="Rank($returns)  # Relative ranking of returns",
    ),
    "TsRank": OperatorDefinition(
        name="TsRank",
        description="Time-series rank over N periods",
        syntax="TsRank($field, N)",
        category="time_series",
        parameters=["field: data field", "N: window size"],
        example="TsRank($close, 20)  # Rank of current close in 20-day range",
    ),
    "Corr": OperatorDefinition(
        name="Corr",
        description="Rolling correlation between two fields",
        syntax="Corr($field1, $field2, N)",
        category="time_series",
        parameters=["field1: first field", "field2: second field", "N: window size"],
        example="Corr($close, $volume, 20)  # Price-volume correlation",
    ),
    "Cov": OperatorDefinition(
        name="Cov",
        description="Rolling covariance between two fields",
        syntax="Cov($field1, $field2, N)",
        category="time_series",
        parameters=["field1: first field", "field2: second field", "N: window size"],
        example="Cov($returns, $volume, 20)",
    ),
}

# Technical indicators
TECHNICAL_OPERATORS: dict[str, OperatorDefinition] = {
    "EMA": OperatorDefinition(
        name="EMA",
        description="Exponential moving average",
        syntax="EMA($field, N)",
        category="technical",
        parameters=["field: data field", "N: span (smoothing period)"],
        example="EMA($close, 12)  # 12-period EMA",
    ),
    "WMA": OperatorDefinition(
        name="WMA",
        description="Weighted moving average",
        syntax="WMA($field, N)",
        category="technical",
        parameters=["field: data field", "N: window size"],
        example="WMA($close, 10)  # 10-period WMA",
    ),
    "RSI": OperatorDefinition(
        name="RSI",
        description="Relative Strength Index",
        syntax="RSI($field, N)",
        category="technical",
        parameters=["field: price field", "N: period (typically 14)"],
        example="RSI($close, 14)  # 14-period RSI",
        notes="Use qlib_crypto.ta_indicators.RSI for proper implementation",
    ),
    "MACD": OperatorDefinition(
        name="MACD",
        description="Moving Average Convergence Divergence",
        syntax="EMA($close, 12) - EMA($close, 26)",
        category="technical",
        parameters=["fast: short EMA period", "slow: long EMA period"],
        example="EMA($close, 12) - EMA($close, 26)  # MACD line",
        notes="Construct using EMA operators",
    ),
    "Bollinger": OperatorDefinition(
        name="Bollinger",
        description="Bollinger Band position",
        syntax="($close - Mean($close, N)) / Std($close, N)",
        category="technical",
        parameters=["N: period (typically 20)"],
        example="($close - Mean($close, 20)) / Std($close, 20)",
        notes="Normalized position within bands",
    ),
}

# Math operators
MATH_OPERATORS: dict[str, OperatorDefinition] = {
    "Abs": OperatorDefinition(
        name="Abs",
        description="Absolute value",
        syntax="Abs($field)",
        category="math",
        parameters=["field: numeric field"],
        example="Abs($returns)  # Absolute returns",
    ),
    "Sign": OperatorDefinition(
        name="Sign",
        description="Sign of value (-1, 0, 1)",
        syntax="Sign($field)",
        category="math",
        parameters=["field: numeric field"],
        example="Sign($returns)  # Direction of movement",
    ),
    "Log": OperatorDefinition(
        name="Log",
        description="Natural logarithm",
        syntax="Log($field)",
        category="math",
        parameters=["field: positive numeric field"],
        example="Log($volume)  # Log volume",
    ),
    "Power": OperatorDefinition(
        name="Power",
        description="Raise to power",
        syntax="Power($field, N)",
        category="math",
        parameters=["field: numeric field", "N: exponent"],
        example="Power($returns, 2)  # Squared returns",
    ),
    "Sqrt": OperatorDefinition(
        name="Sqrt",
        description="Square root (Power with 0.5)",
        syntax="Power($field, 0.5)",
        category="math",
        parameters=["field: non-negative numeric field"],
        example="Power($volume, 0.5)  # Square root of volume",
    ),
}

# Conditional/Logic operators
LOGIC_OPERATORS: dict[str, OperatorDefinition] = {
    "If": OperatorDefinition(
        name="If",
        description="Conditional expression",
        syntax="If(condition, value_if_true, value_if_false)",
        category="logic",
        parameters=["condition: boolean expression", "true_val", "false_val"],
        example="If($returns > 0, $volume, 0)  # Volume only on up days",
    ),
    "And": OperatorDefinition(
        name="And",
        description="Logical AND",
        syntax="And(cond1, cond2)",
        category="logic",
        parameters=["cond1: first condition", "cond2: second condition"],
        example="And($returns > 0, $volume > Mean($volume, 20))",
    ),
    "Or": OperatorDefinition(
        name="Or",
        description="Logical OR",
        syntax="Or(cond1, cond2)",
        category="logic",
        parameters=["cond1: first condition", "cond2: second condition"],
        example="Or($high == Max($high, 20), $low == Min($low, 20))",
    ),
    "Greater": OperatorDefinition(
        name="Greater",
        description="Greater than comparison",
        syntax="$field1 > $field2",
        category="logic",
        parameters=["field1", "field2"],
        example="$close > Mean($close, 20)  # Above 20-day MA",
    ),
    "Less": OperatorDefinition(
        name="Less",
        description="Less than comparison",
        syntax="$field1 < $field2",
        category="logic",
        parameters=["field1", "field2"],
        example="$close < Mean($close, 20)  # Below 20-day MA",
    ),
}


# ============================================================================
# Field Registry
# ============================================================================


class FieldRegistry:
    """Registry of available data fields with capability detection."""

    def __init__(self) -> None:
        """Initialize with default field sets."""
        self._fields: dict[str, FieldDefinition] = {}
        self._enabled_sources: set[DataSourceType] = set()

        # Core fields always available
        self._register_fields(CORE_FIELDS)
        self._enabled_sources.add(DataSourceType.OHLCV)

    def _register_fields(self, fields: dict[str, FieldDefinition]) -> None:
        """Register a set of fields."""
        self._fields.update(fields)

    def enable_source(
        self,
        source: DataSourceType,
        include_crypto_perpetuals: bool = True,
    ) -> None:
        """Enable a data source and register its fields.

        Args:
            source: Data source type to enable.
            include_crypto_perpetuals: If True and source is DERIVATIVES,
                also include crypto perpetuals fields (C9).
        """
        if source in self._enabled_sources:
            return

        source_fields = {
            DataSourceType.DERIVATIVES: DERIVATIVE_FIELDS,
            DataSourceType.ORDERBOOK: ORDERBOOK_FIELDS,
            DataSourceType.ONCHAIN: ONCHAIN_FIELDS,
            DataSourceType.SENTIMENT: SENTIMENT_FIELDS,
        }

        if source in source_fields:
            self._register_fields(source_fields[source])
            self._enabled_sources.add(source)

            # C9: Include crypto perpetuals fields with derivatives
            if source == DataSourceType.DERIVATIVES and include_crypto_perpetuals:
                self._register_fields(CRYPTO_PERPETUAL_FIELDS)

    def disable_source(self, source: DataSourceType) -> None:
        """Disable a data source and unregister its fields."""
        if source == DataSourceType.OHLCV:
            return  # Core fields cannot be disabled

        if source not in self._enabled_sources:
            return

        source_fields = {
            DataSourceType.DERIVATIVES: DERIVATIVE_FIELDS,
            DataSourceType.ORDERBOOK: ORDERBOOK_FIELDS,
            DataSourceType.ONCHAIN: ONCHAIN_FIELDS,
            DataSourceType.SENTIMENT: SENTIMENT_FIELDS,
        }

        if source in source_fields:
            for field_name in source_fields[source]:
                self._fields.pop(field_name, None)
            self._enabled_sources.discard(source)

    def get_available_fields(self) -> list[str]:
        """Get list of available field names."""
        return list(self._fields.keys())

    def get_field(self, name: str) -> FieldDefinition | None:
        """Get field definition by name."""
        return self._fields.get(name)

    def get_fields_by_source(self, source: DataSourceType) -> list[FieldDefinition]:
        """Get all fields from a specific source."""
        return [f for f in self._fields.values() if f.source == source]

    def is_valid_field(self, name: str) -> bool:
        """Check if a field name is valid."""
        return name in self._fields

    def get_enabled_sources(self) -> set[DataSourceType]:
        """Get set of enabled data sources."""
        return self._enabled_sources.copy()


# ============================================================================
# Operator Catalog
# ============================================================================


class OperatorCatalog:
    """Catalog of available Qlib operators."""

    def __init__(self) -> None:
        """Initialize with all operator categories."""
        self._operators: dict[str, OperatorDefinition] = {}
        self._register_all()

    def _register_all(self) -> None:
        """Register all operator categories."""
        self._operators.update(TIME_SERIES_OPERATORS)
        self._operators.update(TECHNICAL_OPERATORS)
        self._operators.update(MATH_OPERATORS)
        self._operators.update(LOGIC_OPERATORS)

    def get_operator(self, name: str) -> OperatorDefinition | None:
        """Get operator by name."""
        return self._operators.get(name)

    def get_operators_by_category(self, category: str) -> list[OperatorDefinition]:
        """Get all operators in a category."""
        return [op for op in self._operators.values() if op.category == category]

    def get_all_operators(self) -> list[OperatorDefinition]:
        """Get all operators."""
        return list(self._operators.values())

    def get_categories(self) -> list[str]:
        """Get list of operator categories."""
        return list({op.category for op in self._operators.values()})


# ============================================================================
# Dynamic Capability Injection
# ============================================================================


@dataclass
class DynamicCapability:
    """Dynamic capability context for prompt injection."""

    field_registry: FieldRegistry
    operator_catalog: OperatorCatalog
    custom_fields: dict[str, FieldDefinition] = field(default_factory=dict)

    def generate_fields_section(self) -> str:
        """Generate the available fields section for prompts."""
        lines = ["## Available Data Fields", ""]

        # Group by source
        for source in sorted(self.field_registry.get_enabled_sources(), key=lambda x: x.value):
            source_fields = self.field_registry.get_fields_by_source(source)
            if not source_fields:
                continue

            lines.append(f"### {source.value.upper()} Fields")
            for field_def in source_fields:
                lines.append(f"- `{field_def.name}`: {field_def.description}")
            lines.append("")

        # Custom fields
        if self.custom_fields:
            lines.append("### Custom Fields")
            for name, field_def in self.custom_fields.items():
                lines.append(f"- `{name}`: {field_def.description}")
            lines.append("")

        return "\n".join(lines)

    def generate_operators_section(self, categories: list[str] | None = None) -> str:
        """Generate the operators reference section for prompts."""
        lines = ["## Qlib Expression Operators", ""]

        all_categories = categories or self.operator_catalog.get_categories()

        for category in sorted(all_categories):
            operators = self.operator_catalog.get_operators_by_category(category)
            if not operators:
                continue

            lines.append(f"### {category.replace('_', ' ').title()}")
            for op in operators:
                lines.append(f"- `{op.syntax}`: {op.description}")
                lines.append(f"  Example: `{op.example}`")
            lines.append("")

        return "\n".join(lines)

    def generate_field_constraints(self) -> str:
        """Generate field usage constraints for prompts."""
        valid_fields = self.field_registry.get_available_fields()
        field_list = ", ".join(f"`{f}`" for f in sorted(valid_fields))

        return f"""## Field Constraints

**ONLY use these {len(valid_fields)} fields**: {field_list}

Any expression using fields outside this list will be rejected.
"""

    def generate_full_context(self) -> str:
        """Generate complete capability context for injection."""
        sections = [
            self.generate_field_constraints(),
            self.generate_fields_section(),
            self.generate_operators_section(),
        ]
        return "\n".join(sections)

    def add_custom_field(
        self,
        name: str,
        description: str,
        data_type: str = "float",
        source: DataSourceType = DataSourceType.DERIVATIVES,
    ) -> None:
        """Add a custom field to the registry."""
        if not name.startswith("$"):
            name = f"${name}"

        self.custom_fields[name] = FieldDefinition(
            name=name,
            description=description,
            data_type=data_type,
            source=source,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_default_capability() -> DynamicCapability:
    """Create capability with default fields (OHLCV + derivatives)."""
    registry = FieldRegistry()
    registry.enable_source(DataSourceType.DERIVATIVES)

    return DynamicCapability(
        field_registry=registry,
        operator_catalog=OperatorCatalog(),
    )


def create_full_capability() -> DynamicCapability:
    """Create capability with all available fields."""
    registry = FieldRegistry()
    for source in DataSourceType:
        registry.enable_source(source)

    return DynamicCapability(
        field_registry=registry,
        operator_catalog=OperatorCatalog(),
    )


def create_capability_for_sources(sources: list[DataSourceType]) -> DynamicCapability:
    """Create capability for specific data sources."""
    registry = FieldRegistry()
    for source in sources:
        registry.enable_source(source)

    return DynamicCapability(
        field_registry=registry,
        operator_catalog=OperatorCatalog(),
    )


async def detect_available_sources_from_db(
    instrument: str,
    session: Any = None,  # AsyncSession
) -> list[DataSourceType]:
    """Detect available data sources from database for an instrument.

    This function queries the database to determine which data sources
    have data available for the given instrument.

    Args:
        instrument: The instrument symbol (e.g., "BTCUSDT")
        session: Optional async database session

    Returns:
        List of available DataSourceType values
    """
    import logging
    from sqlalchemy import select, func

    logger = logging.getLogger(__name__)

    # Core OHLCV and derivatives always available
    available = [DataSourceType.OHLCV, DataSourceType.DERIVATIVES]

    if session is None:
        return available

    # Check for order book data availability
    try:
        from iqfmp.db.models import OrderBookSnapshotORM

        result = await session.execute(
            select(func.count())
            .select_from(OrderBookSnapshotORM)
            .where(OrderBookSnapshotORM.symbol == instrument)
            .limit(1)
        )
        if (result.scalar() or 0) > 0:
            available.append(DataSourceType.ORDERBOOK)
    except Exception as e:
        logger.warning(f"Failed to check OrderBook availability for {instrument}: {e}")

    # OnChain and Sentiment data sources are not yet implemented in the database schema.
    # When these models are added, extend this function to check their availability.

    return available


def create_capability_context_for_prompt(
    sources: list[DataSourceType] | None = None,
    include_operators: bool = True,
    operator_categories: list[str] | None = None,
) -> str:
    """Create a complete capability context string for prompt injection.

    Args:
        sources: Data sources to include (default: OHLCV + DERIVATIVES)
        include_operators: Whether to include operator reference
        operator_categories: Specific operator categories to include

    Returns:
        Formatted string for prompt injection
    """
    if sources is None:
        sources = [DataSourceType.OHLCV, DataSourceType.DERIVATIVES]

    capability = create_capability_for_sources(sources)

    if include_operators:
        return capability.generate_full_context()
    else:
        return "\n".join([
            capability.generate_field_constraints(),
            capability.generate_fields_section(),
        ])


# ============================================================================
# Expression Validation Integration
# ============================================================================


def validate_expression_fields(expression: str, capability: DynamicCapability) -> tuple[bool, list[str]]:
    """Validate that an expression only uses available fields.

    Args:
        expression: Qlib expression string
        capability: DynamicCapability with available fields

    Returns:
        Tuple of (is_valid, list of invalid field names)
    """
    import re

    # Find all field references in the expression
    field_pattern = r'\$[a-z_]+'
    found_fields = set(re.findall(field_pattern, expression.lower()))

    # Get valid fields
    valid_fields = set(f.lower() for f in capability.field_registry.get_available_fields())

    # Find invalid fields
    invalid_fields = [f for f in found_fields if f not in valid_fields]

    return len(invalid_fields) == 0, invalid_fields


def generate_field_error_feedback(invalid_fields: list[str], capability: DynamicCapability) -> str:
    """Generate feedback for invalid field usage.

    Args:
        invalid_fields: List of invalid field names
        capability: DynamicCapability with available fields

    Returns:
        Feedback string explaining the error
    """
    valid_fields = sorted(capability.field_registry.get_available_fields())

    lines = [
        "## Field Validation Error",
        "",
        f"Your expression uses {len(invalid_fields)} invalid field(s):",
        "",
    ]

    for field in invalid_fields:
        lines.append(f"- `{field}` is not available")

    lines.extend([
        "",
        "## Available Fields",
        "",
        ", ".join(f"`{f}`" for f in valid_fields),
        "",
        "Please regenerate the expression using only the available fields.",
    ])

    return "\n".join(lines)


# ============================================================================
# B5: Technical Indicator Catalog (指标/算子字典)
# ============================================================================


@dataclass
class TechnicalIndicatorDefinition:
    """Definition of a technical indicator with Qlib expression formula."""

    name: str
    description: str
    category: str  # momentum, volatility, trend, volume, oscillator
    qlib_formula: str  # Qlib expression implementation
    parameters: dict[str, str]  # parameter name -> description
    default_params: dict[str, int]  # parameter name -> default value
    value_range: tuple[float | None, float | None]  # (min, max) or None if unbounded
    interpretation: str  # How to interpret the indicator value
    aliases: list[str] = field(default_factory=list)  # Alternative names


# Complete Technical Indicator Library
TECHNICAL_INDICATORS: dict[str, TechnicalIndicatorDefinition] = {
    # ============ Momentum Indicators ============
    "RSI": TechnicalIndicatorDefinition(
        name="RSI",
        description="Relative Strength Index - measures momentum and overbought/oversold",
        category="oscillator",
        qlib_formula="RSI($close, {period})",
        parameters={"period": "lookback period (typically 14)"},
        default_params={"period": 14},
        value_range=(0, 100),
        interpretation="RSI > 70 = overbought, RSI < 30 = oversold",
        aliases=["rsi", "relative strength index", "相对强弱"],
    ),
    "MACD": TechnicalIndicatorDefinition(
        name="MACD",
        description="Moving Average Convergence Divergence - trend following momentum",
        category="momentum",
        qlib_formula="EMA($close, {fast}) - EMA($close, {slow})",
        parameters={
            "fast": "fast EMA period (typically 12)",
            "slow": "slow EMA period (typically 26)",
        },
        default_params={"fast": 12, "slow": 26},
        value_range=(None, None),
        interpretation="MACD > 0 = bullish momentum, MACD < 0 = bearish momentum",
        aliases=["macd", "异同平均线", "指数平滑异同移动平均线"],
    ),
    "MOMENTUM": TechnicalIndicatorDefinition(
        name="MOMENTUM",
        description="Simple price momentum - price change over N periods",
        category="momentum",
        qlib_formula="$close - Ref($close, {period})",
        parameters={"period": "lookback period"},
        default_params={"period": 10},
        value_range=(None, None),
        interpretation="Positive = upward momentum, Negative = downward momentum",
        aliases=["mom", "动量指标"],
    ),
    "ROC": TechnicalIndicatorDefinition(
        name="ROC",
        description="Rate of Change - percentage price change",
        category="momentum",
        qlib_formula="($close / Ref($close, {period}) - 1) * 100",
        parameters={"period": "lookback period"},
        default_params={"period": 10},
        value_range=(None, None),
        interpretation="Positive % = price increase, Negative % = price decrease",
        aliases=["roc", "变动率", "rate of change"],
    ),

    # ============ Trend Indicators ============
    "SMA": TechnicalIndicatorDefinition(
        name="SMA",
        description="Simple Moving Average",
        category="trend",
        qlib_formula="Mean($close, {period})",
        parameters={"period": "averaging period"},
        default_params={"period": 20},
        value_range=(None, None),
        interpretation="Price above SMA = uptrend, below = downtrend",
        aliases=["sma", "简单移动平均", "ma"],
    ),
    "EMA": TechnicalIndicatorDefinition(
        name="EMA",
        description="Exponential Moving Average - gives more weight to recent prices",
        category="trend",
        qlib_formula="EMA($close, {period})",
        parameters={"period": "averaging period"},
        default_params={"period": 20},
        value_range=(None, None),
        interpretation="More responsive to recent price changes than SMA",
        aliases=["ema", "指数移动平均"],
    ),

    # ============ Volatility Indicators ============
    "BOLLINGER_UPPER": TechnicalIndicatorDefinition(
        name="BOLLINGER_UPPER",
        description="Bollinger Band Upper - Mean + N*Std",
        category="volatility",
        qlib_formula="Mean($close, {period}) + {mult} * Std($close, {period})",
        parameters={
            "period": "averaging period (typically 20)",
            "mult": "standard deviation multiplier (typically 2)",
        },
        default_params={"period": 20, "mult": 2},
        value_range=(None, None),
        interpretation="Price near upper band = potentially overbought",
        aliases=["bollinger upper", "布林上轨"],
    ),
    "BOLLINGER_LOWER": TechnicalIndicatorDefinition(
        name="BOLLINGER_LOWER",
        description="Bollinger Band Lower - Mean - N*Std",
        category="volatility",
        qlib_formula="Mean($close, {period}) - {mult} * Std($close, {period})",
        parameters={
            "period": "averaging period",
            "mult": "standard deviation multiplier",
        },
        default_params={"period": 20, "mult": 2},
        value_range=(None, None),
        interpretation="Price near lower band = potentially oversold",
        aliases=["bollinger lower", "布林下轨"],
    ),
    "ATR": TechnicalIndicatorDefinition(
        name="ATR",
        description="Average True Range - volatility indicator",
        category="volatility",
        qlib_formula="Mean(Max($high - $low, Max(Abs($high - Ref($close, 1)), Abs($low - Ref($close, 1)))), {period})",
        parameters={"period": "averaging period (typically 14)"},
        default_params={"period": 14},
        value_range=(0, None),
        interpretation="Higher ATR = higher volatility, lower = lower volatility",
        aliases=["atr", "平均真实波幅", "average true range"],
    ),

    # ============ Oscillator Indicators ============
    "STOCHASTIC_K": TechnicalIndicatorDefinition(
        name="STOCHASTIC_K",
        description="Stochastic %K - fast stochastic oscillator",
        category="oscillator",
        qlib_formula="($close - Min($low, {period})) / (Max($high, {period}) - Min($low, {period})) * 100",
        parameters={"period": "lookback period (typically 14)"},
        default_params={"period": 14},
        value_range=(0, 100),
        interpretation="%K > 80 = overbought, %K < 20 = oversold",
        aliases=["stochastic k", "stoch k", "随机指标k"],
    ),
    "WILLIAMS_R": TechnicalIndicatorDefinition(
        name="WILLIAMS_R",
        description="Williams %R - momentum oscillator",
        category="oscillator",
        qlib_formula="(Max($high, {period}) - $close) / (Max($high, {period}) - Min($low, {period})) * -100",
        parameters={"period": "lookback period (typically 14)"},
        default_params={"period": 14},
        value_range=(-100, 0),
        interpretation="%R > -20 = overbought, %R < -80 = oversold",
        aliases=["wr", "williams", "威廉指标", "williams %r"],
    ),
    "CCI": TechnicalIndicatorDefinition(
        name="CCI",
        description="Commodity Channel Index - identifies cyclical trends",
        category="oscillator",
        qlib_formula="(($high + $low + $close) / 3 - Mean(($high + $low + $close) / 3, {period})) / (0.015 * Mean(Abs(($high + $low + $close) / 3 - Mean(($high + $low + $close) / 3, {period})), {period}))",
        parameters={"period": "lookback period (typically 20)"},
        default_params={"period": 20},
        value_range=(None, None),
        interpretation="CCI > 100 = overbought, CCI < -100 = oversold",
        aliases=["cci", "商品通道指数", "commodity channel index"],
    ),

    # ============ Volume Indicators ============
    "VOLUME_RATIO": TechnicalIndicatorDefinition(
        name="VOLUME_RATIO",
        description="Volume Ratio - current volume vs average",
        category="volume",
        qlib_formula="$volume / Mean($volume, {period})",
        parameters={"period": "averaging period"},
        default_params={"period": 20},
        value_range=(0, None),
        interpretation="Ratio > 1 = above average volume, < 1 = below average",
        aliases=["volume ratio", "成交量比"],
    ),

    # ============ Channel Indicators ============
    "DONCHIAN_UPPER": TechnicalIndicatorDefinition(
        name="DONCHIAN_UPPER",
        description="Donchian Channel Upper - highest high over N periods",
        category="volatility",
        qlib_formula="Max($high, {period})",
        parameters={"period": "lookback period (typically 20)"},
        default_params={"period": 20},
        value_range=(None, None),
        interpretation="Breakout above = bullish signal",
        aliases=["donchian upper", "唐奇安上轨"],
    ),
    "DONCHIAN_LOWER": TechnicalIndicatorDefinition(
        name="DONCHIAN_LOWER",
        description="Donchian Channel Lower - lowest low over N periods",
        category="volatility",
        qlib_formula="Min($low, {period})",
        parameters={"period": "lookback period (typically 20)"},
        default_params={"period": 20},
        value_range=(None, None),
        interpretation="Breakout below = bearish signal",
        aliases=["donchian lower", "唐奇安下轨"],
    ),
}


class TechnicalIndicatorCatalog:
    """Catalog of technical indicators with formula lookup.

    B5 Task: Complete indicator/operator dictionary for LLM reference.
    """

    def __init__(self) -> None:
        """Initialize with all technical indicators."""
        self._indicators: dict[str, TechnicalIndicatorDefinition] = TECHNICAL_INDICATORS.copy()
        self._alias_map: dict[str, str] = self._build_alias_map()

    def _build_alias_map(self) -> dict[str, str]:
        """Build mapping from aliases to canonical names."""
        alias_map: dict[str, str] = {}
        for name, indicator in self._indicators.items():
            alias_map[name.lower()] = name
            for alias in indicator.aliases:
                alias_map[alias.lower()] = name
        return alias_map

    def get_indicator(self, name: str) -> TechnicalIndicatorDefinition | None:
        """Get indicator by name or alias."""
        canonical = self._alias_map.get(name.lower())
        if canonical:
            return self._indicators.get(canonical)
        return None

    def get_formula(self, name: str, **params: int) -> str | None:
        """Get Qlib formula for indicator with parameters filled in.

        Args:
            name: Indicator name or alias
            **params: Parameter values to substitute

        Returns:
            Formatted Qlib expression or None if not found
        """
        indicator = self.get_indicator(name)
        if not indicator:
            return None

        # Merge defaults with provided params
        all_params = {**indicator.default_params, **params}
        return indicator.qlib_formula.format(**all_params)

    def get_indicators_by_category(self, category: str) -> list[TechnicalIndicatorDefinition]:
        """Get all indicators in a category."""
        return [ind for ind in self._indicators.values() if ind.category == category]

    def get_all_indicators(self) -> list[TechnicalIndicatorDefinition]:
        """Get all indicators."""
        return list(self._indicators.values())

    def get_categories(self) -> list[str]:
        """Get list of indicator categories."""
        return list({ind.category for ind in self._indicators.values()})

    def search(self, query: str) -> list[TechnicalIndicatorDefinition]:
        """Search indicators by name, alias, or description."""
        query_lower = query.lower()
        results = []
        for indicator in self._indicators.values():
            if (query_lower in indicator.name.lower() or
                query_lower in indicator.description.lower() or
                any(query_lower in alias for alias in indicator.aliases)):
                results.append(indicator)
        return results

    def generate_reference_section(self, categories: list[str] | None = None) -> str:
        """Generate indicator reference section for prompts.

        Args:
            categories: Categories to include (None = all)

        Returns:
            Formatted markdown reference
        """
        lines = ["## Technical Indicator Reference (B5)", ""]

        target_categories = categories or self.get_categories()

        for category in sorted(target_categories):
            indicators = self.get_indicators_by_category(category)
            if not indicators:
                continue

            lines.append(f"### {category.title()} Indicators")
            for ind in indicators:
                lines.append(f"**{ind.name}**: {ind.description}")
                lines.append(f"  - Formula: `{ind.qlib_formula}`")
                lines.append(f"  - Interpretation: {ind.interpretation}")
                if ind.value_range != (None, None):
                    low = ind.value_range[0] if ind.value_range[0] is not None else "-inf"
                    high = ind.value_range[1] if ind.value_range[1] is not None else "+inf"
                    lines.append(f"  - Range: [{low}, {high}]")
                lines.append("")

        return "\n".join(lines)


# Global catalog instance for easy access
INDICATOR_CATALOG = TechnicalIndicatorCatalog()


# ============================================================================
# C9: FieldSchema - Declarative Field Availability
# ============================================================================


@dataclass
class FieldSchema:
    """Declarative schema for field availability (C9).

    This schema declares which fields are actually available in the data,
    allowing the LLM to generate only valid expressions.

    Attributes:
        symbol: Trading pair this schema applies to.
        available_fields: List of available field names.
        field_metadata: Detailed metadata for each field.
        data_sources: Which data sources are available.
        warnings: Any warnings about data gaps or issues.
    """

    symbol: str
    available_fields: list[str]
    field_metadata: dict[str, FieldDefinition] = field(default_factory=dict)
    data_sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    has_derivatives: bool = False
    has_crypto_perpetuals: bool = False

    def to_prompt_context(self) -> str:
        """Generate prompt context from this schema."""
        lines = [
            f"## Available Data Fields for {self.symbol}",
            "",
        ]

        # List data sources
        if self.data_sources:
            lines.append(f"**Data Sources**: {', '.join(self.data_sources)}")
            lines.append("")

        # List all fields
        lines.append(f"**Available Fields** ({len(self.available_fields)} total):")
        for field_name in sorted(self.available_fields):
            meta = self.field_metadata.get(field_name)
            if meta:
                lines.append(f"- `{field_name}`: {meta.description}")
            else:
                lines.append(f"- `{field_name}`")
        lines.append("")

        # Warnings
        if self.warnings:
            lines.append("**Warnings:**")
            for warning in self.warnings:
                lines.append(f"- ⚠️ {warning}")
            lines.append("")

        # Field constraint
        field_list = ", ".join(f"`{f}`" for f in sorted(self.available_fields))
        lines.append(f"**ONLY use these fields**: {field_list}")
        lines.append("")

        return "\n".join(lines)


def create_field_schema_from_availability(
    symbol: str,
    availability: dict[str, bool],
) -> FieldSchema:
    """Create FieldSchema from data availability check.

    Args:
        symbol: Trading pair.
        availability: Dictionary from check_data_availability().

    Returns:
        FieldSchema with available fields.
    """
    registry = FieldRegistry()
    data_sources = ["ohlcv"]

    # Enable sources based on availability
    if availability.get("funding_rate") or availability.get("open_interest"):
        registry.enable_source(DataSourceType.DERIVATIVES)
        data_sources.append("derivatives")

    available_fields = registry.get_available_fields()
    field_metadata = {name: registry.get_field(name) for name in available_fields}

    warnings = []
    if not availability.get("ohlcv"):
        warnings.append("No OHLCV data found - basic price fields may not work")
    if not availability.get("funding_rate"):
        warnings.append("No funding rate data - $funding_* fields unavailable")
    if not availability.get("open_interest"):
        warnings.append("No open interest data - $open_interest* fields unavailable")
    if not availability.get("liquidation"):
        warnings.append("No liquidation data - $liquidation_* fields unavailable")

    return FieldSchema(
        symbol=symbol,
        available_fields=available_fields,
        field_metadata=field_metadata,
        data_sources=data_sources,
        warnings=warnings,
        has_derivatives="derivatives" in data_sources,
        has_crypto_perpetuals=availability.get("funding_rate", False),
    )


def create_field_schema_for_crypto_perpetuals(symbol: str) -> FieldSchema:
    """Create FieldSchema optimized for crypto perpetuals trading.

    This is the default schema for crypto futures/perpetuals with
    all derivative fields enabled.

    Args:
        symbol: Trading pair (e.g., "BTC/USDT:USDT").

    Returns:
        FieldSchema with all crypto perpetual fields.
    """
    registry = FieldRegistry()
    registry.enable_source(DataSourceType.DERIVATIVES, include_crypto_perpetuals=True)

    available_fields = registry.get_available_fields()
    field_metadata = {name: registry.get_field(name) for name in available_fields}

    return FieldSchema(
        symbol=symbol,
        available_fields=available_fields,
        field_metadata=field_metadata,
        data_sources=["ohlcv", "derivatives", "crypto_perpetuals"],
        warnings=[],
        has_derivatives=True,
        has_crypto_perpetuals=True,
    )
