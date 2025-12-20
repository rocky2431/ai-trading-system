"""Intelligent Indicator Detection and Feedback System.

This module implements the smart feedback loop for factor generation:
1. Extract requested indicators from user hypothesis
2. Detect which indicators are present in generated expressions
3. Generate specific feedback for missing indicators
4. Enable LLM to learn through iteration, not hardcoded formulas

Design Philosophy:
- Prompt provides ONLY syntax, no hardcoded formulas
- LLM attempts to implement requested indicators
- Feedback loop tells LLM exactly what's missing
- LLM learns through 5 rounds of iteration
"""

import re
from dataclasses import dataclass
from typing import Optional


# Indicator pattern mapping: name variations -> canonical name
# This helps recognize different ways users might refer to the same indicator
# Note: Aliases are matched with word boundaries to avoid false positives
INDICATOR_ALIASES: dict[str, list[str]] = {
    # Technical Indicators - order matters for priority
    "MACD": ["macd", "指数平滑异同移动平均线", "异同平均线", "macd指标"],  # MACD before MA
    "RSI": ["rsi", "相对强弱", "relative strength", "超买超卖"],
    "WR": ["wr指标", "wr ", " wr", "williams", "威廉指标", "威廉%r", "williams %r", "william"],
    "SSL": ["ssl", "ssl通道", "ssl channel", "ssl indicator"],
    "ZIGZAG": ["zigzag", "之字形", "zig zag", "ziggy", "zigzag指标", "zig-zag"],
    "BOLLINGER": ["bollinger", "布林", "boll", "布林带", "bollinger bands"],
    "ATR": ["atr指标", "atr ", " atr", "平均真实波幅", "average true range", "真实波幅"],
    "SMA": ["sma指标", "sma ", " sma", "简单移动平均", "simple moving average"],  # Removed "ma" to avoid MACD conflict
    "EMA": ["ema指标", "ema ", " ema", "指数移动平均", "exponential moving average"],
    "WMA": ["wma指标", "wma ", " wma", "加权移动平均", "weighted moving average"],
    "STOCHASTIC": ["stochastic", "随机指标", "kdj", "stoch"],
    "CCI": ["cci指标", "cci ", " cci", "商品通道指数", "commodity channel index"],
    "ADX": ["adx指标", "adx ", " adx", "趋势指标", "average directional index", "平均趋向指数"],
    "OBV": ["obv指标", "obv ", " obv", "能量潮", "on balance volume"],
    "VWAP": ["vwap", "成交量加权平均价"],
    "MOMENTUM": ["momentum", "动量指标", "mom指标"],
    "ROC": ["roc指标", "roc ", " roc", "变动率", "rate of change"],
    "MFI": ["mfi指标", "mfi ", " mfi", "资金流量指标", "money flow index"],
    "DMI": ["dmi指标", "dmi ", " dmi", "directional movement index"],
    "ICHIMOKU": ["ichimoku", "一目均衡", "云图"],
    "PIVOT": ["pivot", "枢轴点", "pivot point", "支撑阻力"],
    "DONCHIAN": ["donchian", "唐奇安通道", "donchian channel"],
    "KELTNER": ["keltner", "肯特纳通道", "keltner channel"],
    "PARABOLIC": ["parabolic", "抛物线", "parabolic sar", "psar"],
    "AROON": ["aroon", "阿隆指标"],
    "TRIX": ["trix", "三重指数平滑"],
    "PPO": ["ppo", "价格震荡器", "percentage price oscillator"],
    "CMO": ["cmo", "钱德动量振荡器", "chande momentum oscillator"],
}


# Detection patterns for each indicator in Qlib expressions
# These are used to check if an indicator is implemented in the expression
INDICATOR_DETECTION_PATTERNS: dict[str, list[str]] = {
    "RSI": ["RSI(", "rsi("],
    "MACD": ["MACD(", "macd(", "EMA($close, 12)", "EMA($close, 26)"],
    "WR": [
        "($high - $close)", "($close - $low)",  # Williams %R components
        "Max($high,", "Min($low,",  # Highest high, lowest low
        "/ (Max($high", "/ (Min($low",  # Ratio patterns
    ],
    "SSL": [
        "EMA($high", "EMA($low",  # SSL uses EMA of high/low
        "If(", ">",  # Crossover logic
    ],
    "ZIGZAG": [
        "Max($high,", "Min($low,",  # Local extremes
        "If(", "Ref(",  # Trend change detection
        "$high - $low",  # Range detection
    ],
    "BOLLINGER": [
        "Mean($close", "Std($close",  # BB uses mean and std
        "Mean($close, 20)", "Std($close, 20)",  # Classic parameters
        "+ 2", "- 2", "* 2",  # Band multipliers
    ],
    "ATR": [
        "$high - $low",  # True range component
        "Max(", "Ref($close",  # True range calculation
        "Mean(", "EMA(",  # Averaging
    ],
    "SMA": ["Mean($close,"],
    "EMA": ["EMA($close,", "EMA("],
    "WMA": ["WMA($close,", "WMA("],
    "STOCHASTIC": [
        "($close - Min($low", "(Max($high",  # %K formula
        "Min($low,", "Max($high,",
    ],
    "CCI": [
        "$high + $low + $close", "/ 3",  # Typical price
        "Mean(", "Std(",  # Mean deviation
    ],
    "MOMENTUM": ["Ref($close,", "$close / Ref(", "$close - Ref("],
    "ROC": ["$close / Ref($close", "- 1"],
}


@dataclass
class IndicatorAnalysis:
    """Result of indicator analysis."""

    requested: set[str]  # Indicators user requested
    found: set[str]  # Indicators detected in expression
    missing: set[str]  # Indicators that are missing
    confidence: dict[str, float]  # Confidence score for each found indicator

    @property
    def is_complete(self) -> bool:
        """Check if all requested indicators are implemented."""
        return len(self.missing) == 0

    @property
    def completion_rate(self) -> float:
        """Calculate how many requested indicators are implemented."""
        if not self.requested:
            return 1.0
        return len(self.found & self.requested) / len(self.requested)


def extract_requested_indicators(hypothesis: str) -> set[str]:
    """Extract indicator names from user hypothesis.

    Handles:
    - Chinese indicator names: "威廉指标结合MACD"
    - English indicator names: "WR combined with MACD"
    - Common variations: "ziggy" -> ZIGZAG, "boll" -> BOLLINGER
    - Mixed language: "WR结合macd结合SSL"

    Args:
        hypothesis: User's factor hypothesis/description

    Returns:
        Set of canonical indicator names (uppercase)
    """
    if not hypothesis:
        return set()

    # Pad with spaces and add common separators for word boundary matching
    # This helps aliases like " wr" or "wr " match correctly
    hypothesis_normalized = f" {hypothesis.lower()} "
    # Replace common separators with spaces
    for sep in ["结合", "和", "与", "+", ",", "、", "加上", "配合"]:
        hypothesis_normalized = hypothesis_normalized.replace(sep, " ")

    found_indicators: set[str] = set()

    for canonical_name, aliases in INDICATOR_ALIASES.items():
        for alias in aliases:
            alias_lower = alias.lower()
            # Check if alias is in normalized hypothesis
            if alias_lower in hypothesis_normalized:
                found_indicators.add(canonical_name)
                break

    return found_indicators


def detect_indicators_in_expression(expression: str) -> dict[str, float]:
    """Detect which indicators are implemented in a Qlib expression.

    Returns a confidence score for each indicator based on how many
    characteristic patterns are found.

    Args:
        expression: Qlib expression string

    Returns:
        Dictionary mapping indicator name to confidence score (0.0-1.0)
    """
    if not expression:
        return {}

    expression_lower = expression.lower()
    indicator_scores: dict[str, float] = {}

    for indicator, patterns in INDICATOR_DETECTION_PATTERNS.items():
        matches = 0
        for pattern in patterns:
            if pattern.lower() in expression_lower:
                matches += 1

        if matches > 0:
            # Confidence = matches / patterns, capped at 1.0
            confidence = min(matches / len(patterns), 1.0)
            # Require at least 30% pattern match to count as implemented
            if confidence >= 0.3 or matches >= 2:
                indicator_scores[indicator] = confidence

    return indicator_scores


def analyze_indicator_coverage(
    hypothesis: str,
    expression: str,
) -> IndicatorAnalysis:
    """Analyze how well the expression covers the requested indicators.

    Args:
        hypothesis: User's factor hypothesis
        expression: Generated Qlib expression

    Returns:
        IndicatorAnalysis with requested, found, and missing indicators
    """
    requested = extract_requested_indicators(hypothesis)
    found_with_confidence = detect_indicators_in_expression(expression)
    found = set(found_with_confidence.keys())

    # Calculate missing indicators
    missing = requested - found

    return IndicatorAnalysis(
        requested=requested,
        found=found,
        missing=missing,
        confidence=found_with_confidence,
    )


def generate_missing_indicator_feedback(analysis: IndicatorAnalysis) -> Optional[str]:
    """Generate specific feedback for missing indicators.

    This is the core of the intelligent feedback loop - it tells the LLM
    exactly what indicators are missing so it can implement them.

    Args:
        analysis: IndicatorAnalysis from analyze_indicator_coverage

    Returns:
        Feedback string if indicators are missing, None if complete
    """
    if analysis.is_complete:
        return None

    missing_list = sorted(analysis.missing)

    # Build feedback message
    feedback_parts = [
        f"## Missing Indicators Detected",
        f"",
        f"Your factor is missing {len(missing_list)} indicator(s) that the user requested:",
        "",
    ]

    for indicator in missing_list:
        feedback_parts.append(f"- **{indicator}**: Not detected in your expression. Please implement it.")
        # Add hints for common indicators
        hints = _get_implementation_hints(indicator)
        if hints:
            feedback_parts.append(f"  Hint: {hints}")

    feedback_parts.extend([
        "",
        "## What You Implemented",
        "",
    ])

    if analysis.found:
        for ind, conf in sorted(analysis.confidence.items()):
            status = "✓" if ind in analysis.requested else "(extra)"
            feedback_parts.append(f"- {ind}: confidence {conf:.0%} {status}")
    else:
        feedback_parts.append("- No recognized indicators detected")

    feedback_parts.extend([
        "",
        "## Required Action",
        "",
        "Please generate a NEW Qlib expression that includes ALL requested indicators.",
        f"Missing: {', '.join(missing_list)}",
        "",
        "Research each indicator's formula and implement it using the available operators:",
        "Ref, Mean, Std, Sum, Max, Min, Delta, Rank, Abs, Log, Sign, Corr, EMA, WMA, RSI, MACD, If",
    ])

    return "\n".join(feedback_parts)


def _get_implementation_hints(indicator: str) -> str:
    """Get implementation hints for an indicator.

    These are NOT formulas - just guidance on what the indicator typically uses.
    The LLM should research and implement the actual formula.
    """
    hints = {
        "WR": "Williams %R uses highest high and lowest low over N periods, typically 14",
        "SSL": "SSL Channel uses EMA of highs and lows with crossover detection (If operator)",
        "ZIGZAG": "ZigZag identifies local extremes - needs threshold-based peak/trough detection",
        "BOLLINGER": "Bollinger uses Mean for middle band, +/- N*Std for upper/lower bands",
        "ATR": "ATR = Average of True Range where TR = Max(high-low, |high-prevClose|, |low-prevClose|)",
        "STOCHASTIC": "Stochastic %K = (close - lowest low) / (highest high - lowest low)",
        "CCI": "CCI uses (TypicalPrice - Mean(TP)) / (0.015 * MeanDeviation)",
        "ADX": "ADX uses +DI and -DI directional indicators with smoothing",
        "OBV": "OBV is cumulative: +volume if close > prev, -volume if close < prev",
        "MOMENTUM": "Simple momentum: close - close_N_periods_ago",
        "ROC": "Rate of Change: (close / close_N_periods_ago) - 1",
    }
    return hints.get(indicator, "")


def check_factor_completeness(
    hypothesis: str,
    expression: str,
) -> tuple[bool, Optional[str]]:
    """Check if a factor expression fully implements the user's hypothesis.

    This is the main entry point for the feedback loop integration.

    Args:
        hypothesis: User's factor hypothesis
        expression: Generated Qlib expression

    Returns:
        Tuple of (is_complete, feedback_message)
        - is_complete: True if all indicators are implemented
        - feedback_message: Specific feedback if incomplete, None otherwise
    """
    analysis = analyze_indicator_coverage(hypothesis, expression)

    if analysis.is_complete:
        return True, None

    feedback = generate_missing_indicator_feedback(analysis)
    return False, feedback


def get_indicator_summary(hypothesis: str) -> str:
    """Get a summary of indicators detected in a hypothesis.

    Useful for logging and debugging.
    """
    indicators = extract_requested_indicators(hypothesis)
    if not indicators:
        return "No specific indicators detected"
    return f"Detected indicators: {', '.join(sorted(indicators))}"
