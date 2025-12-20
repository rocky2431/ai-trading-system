"""Read-only tool helpers for agents.

This module intentionally provides ONLY read operations so it can be safely used
to augment prompts (fields, data summary, similar factors) without allowing an
LLM to execute arbitrary code or mutate state.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from iqfmp.llm.validation import FieldRegistry, FieldSet


def list_available_fields(
    *,
    field_set: FieldSet = FieldSet.CRYPTO,
    include_prefix: bool = True,
) -> list[str]:
    """List allowed expression fields for a field set.

    Args:
        field_set: Field set to list.
        include_prefix: Whether to prefix fields with '$'.

    Returns:
        Sorted list of field names.
    """
    fields = sorted(FieldRegistry.get_fields(field_set))
    if include_prefix:
        return [f"${f}" for f in fields]
    return fields


def search_similar_factors(*, hypothesis: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search similar factors by hypothesis in the vector store (best-effort).

    Args:
        hypothesis: Text hypothesis / user request.
        limit: Max results.

    Returns:
        List of dicts with basic factor metadata; empty on failure.
    """
    if limit <= 0 or not hypothesis.strip():
        return []

    try:
        from iqfmp.vector.search import SimilaritySearcher

        searcher = SimilaritySearcher()
        results = searcher.search_by_hypothesis(hypothesis=hypothesis, limit=limit)
        return [
            {
                "factor_id": r.factor_id,
                "score": float(r.score),
                "name": r.name,
                "family": r.family,
            }
            for r in results
        ]
    except Exception:
        return []


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize a DataFrame for prompt context (no heavy stats).

    Returns:
        A JSON-serializable summary dict.
    """
    if df.empty:
        return {"rows": 0, "cols": 0, "columns": []}

    summary: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns.astype(str)),
    }

    # Time range heuristics
    for time_col in ("date", "datetime", "timestamp"):
        if time_col in df.columns:
            try:
                series = pd.to_datetime(df[time_col], errors="coerce")
                series = series.dropna()
                if not series.empty:
                    summary["time_column"] = time_col
                    summary["time_start"] = series.min().to_pydatetime().isoformat()
                    summary["time_end"] = series.max().to_pydatetime().isoformat()
            except Exception:
                pass
            break

    if "symbol" in df.columns:
        try:
            summary["n_symbols"] = int(df["symbol"].nunique())
        except Exception:
            pass

    # Missingness (coarse)
    try:
        missing_ratio = float(df.isna().mean().mean())
        summary["missing_ratio"] = round(missing_ratio, 6)
    except Exception:
        pass

    summary["generated_at"] = datetime.utcnow().isoformat()
    return summary

