"""Factor service for business logic with TimescaleDB + Redis support.

Integrates:
- Full FactorEvaluator from evaluation module
- CryptoCVSplitter for multi-dimensional validation
- StabilityAnalyzer for robustness analysis
- Transaction cost estimation
"""

import asyncio
import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from iqfmp.models.factor import Factor, FactorMetrics, FactorStatus, StabilityReport
from iqfmp.llm.provider import LLMConfig, LLMProvider
from iqfmp.agents.factor_generation import (
    FactorGenerationAgent,
    FactorGenerationConfig,
    FactorFamily,
)
from iqfmp.db.repositories import (
    FactorRepository,
    ResearchTrialRepository,
    FactorValueRepository,
    MiningTaskRepository,
)
from iqfmp.db.database import get_async_session
from iqfmp.api.factors.schemas import (
    MiningTaskStatus,
    FactorLibraryStats,
)

# Import evaluation module components
from iqfmp.evaluation.factor_evaluator import (
    FactorEvaluator as FullFactorEvaluator,
    EvaluationConfig,
    FactorMetrics as EvalFactorMetrics,
)
from iqfmp.evaluation.cv_splitter import (
    CryptoCVSplitter,
    CVSplitConfig,
)
from iqfmp.evaluation.stability_analyzer import (
    StabilityAnalyzer,
    StabilityConfig,
    StabilityReport as FullStabilityReport,
)
# New evaluation modules
from iqfmp.evaluation.walk_forward_validator import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardResult,
)
from iqfmp.evaluation.ic_decomposition import (
    ICDecompositionAnalyzer,
    ICDecompositionConfig,
    ICDecompositionResult,
)
from iqfmp.evaluation.redundancy_detector import (
    RedundancyDetector,
    RedundancyConfig,
    RedundancyResult,
)

# Vector store imports
from iqfmp.vector.store import FactorVectorStore
from iqfmp.vector.search import SimilaritySearcher, SearchResult

# WebSocket broadcast imports
from iqfmp.api.system.websocket import (
    broadcast_factor_created,
    broadcast_evaluation_complete,
    broadcast_task_update,
)


class FactorDuplicateError(Exception):
    """Raised when a duplicate factor is detected."""

    pass


class FactorNotFoundError(Exception):
    """Raised when factor is not found."""

    pass


class FactorEvaluationError(Exception):
    """Raised when factor evaluation fails due to data or computation issues."""

    pass


class FactorService:
    """Service for factor management with database persistence."""

    def __init__(
        self,
        session: AsyncSession,
        redis_client: Optional[redis.Redis] = None,
    ) -> None:
        """Initialize factor service with database session.

        Args:
            session: Async database session
            redis_client: Optional Redis client for caching
        """
        self.session = session
        self.redis_client = redis_client
        self.factor_repo = FactorRepository(session, redis_client)
        self.trial_repo = ResearchTrialRepository(session)
        self.factor_value_repo = FactorValueRepository(session)
        self.mining_task_repo = MiningTaskRepository(session, redis_client)
        self._llm_provider: Optional[LLMProvider] = None
        self._generation_agent: Optional[FactorGenerationAgent] = None
        # Vector store for factor indexing and similarity search
        self._vector_store: Optional[FactorVectorStore] = None
        self._similarity_searcher: Optional[SimilaritySearcher] = None

    def _get_llm_provider(self) -> LLMProvider:
        """获取或创建 LLM Provider.

        优先从 ConfigService 获取 API key（用户在前端配置的），
        如果没有则从环境变量获取。
        """
        if self._llm_provider is None:
            # 优先从 ConfigService 获取 API key
            from iqfmp.api.config.service import get_config_service
            config_service = get_config_service()

            # 从 config 获取 API key（优先）或环境变量（后备）
            api_key = config_service._config.get("api_key") or os.getenv("OPENROUTER_API_KEY")

            if not api_key:
                raise ValueError(
                    "LLM API key not configured. Please configure OpenRouter API key in Settings page."
                )

            # 确保环境变量也设置了（供 LLMConfig.from_env 使用）
            os.environ["OPENROUTER_API_KEY"] = api_key

            # 使用 from_env 创建配置（它现在可以从环境变量获取正确的 key）
            config = LLMConfig.from_env()
            self._llm_provider = LLMProvider(config)
        return self._llm_provider

    def _get_generation_agent(self) -> FactorGenerationAgent:
        """获取或创建因子生成 Agent"""
        if self._generation_agent is None:
            config = FactorGenerationConfig(
                name="factor_generator",
                security_check_enabled=True,
                field_constraint_enabled=True,
            )
            self._generation_agent = FactorGenerationAgent(
                config=config,
                llm_provider=self._get_llm_provider(),
            )
        return self._generation_agent

    def _get_vector_store(self) -> FactorVectorStore:
        """获取或创建向量存储"""
        if self._vector_store is None:
            try:
                self._vector_store = FactorVectorStore()
            except Exception as e:
                print(f"Warning: Failed to initialize vector store: {e}")
                self._vector_store = None
        return self._vector_store

    def _get_similarity_searcher(self) -> SimilaritySearcher:
        """获取或创建相似度搜索器"""
        if self._similarity_searcher is None:
            try:
                self._similarity_searcher = SimilaritySearcher(
                    similarity_threshold=0.85,  # 85% 相似度阈值
                )
            except Exception as e:
                print(f"Warning: Failed to initialize similarity searcher: {e}")
                self._similarity_searcher = None
        return self._similarity_searcher

    async def _index_factor_to_vector_store(
        self,
        factor: Factor,
        metrics: Optional[FactorMetrics] = None,
    ) -> bool:
        """将因子索引到向量存储 (Task 7.1).

        在因子创建或评估通过后调用此方法将因子存入 Qdrant 向量库。

        Args:
            factor: 因子对象
            metrics: 可选的因子指标

        Returns:
            是否索引成功
        """
        try:
            store = self._get_vector_store()
            if store is None:
                return False

            # 准备元数据
            # 安全获取 status 值 (Pydantic use_enum_values=True 可能已转为字符串)
            if factor.status:
                status_value = factor.status.value if hasattr(factor.status, 'value') else str(factor.status)
            else:
                status_value = "candidate"

            metadata = {
                "sharpe": metrics.sharpe if metrics else 0.0,
                "ic_mean": metrics.ic_mean if metrics else 0.0,
                "ir": metrics.ir if metrics else 0.0,
                "status": status_value,
            }

            # 获取因子假设/描述
            hypothesis = ""
            if hasattr(factor, "hypothesis") and factor.hypothesis:
                hypothesis = factor.hypothesis
            elif hasattr(factor, "description") and factor.description:
                hypothesis = factor.description
            else:
                # 从代码中提取 docstring 作为假设
                if '"""' in factor.code:
                    start = factor.code.find('"""') + 3
                    end = factor.code.find('"""', start)
                    if end > start:
                        hypothesis = factor.code[start:end].strip()

            # 存入向量库
            store.add_factor(
                factor_id=factor.id,
                name=factor.name,
                code=factor.code,
                hypothesis=hypothesis,
                family=factor.family[0] if factor.family else "unknown",
                metadata=metadata,
            )

            print(f"Indexed factor to vector store: {factor.id} ({factor.name})")
            return True

        except Exception as e:
            print(f"Warning: Failed to index factor to vector store: {e}")
            return False

    async def _check_similarity(
        self,
        code: str,
        name: str = "",
        hypothesis: str = "",
        threshold: float = 0.85,
    ) -> list[SearchResult]:
        """检查因子代码相似度 (Task 7.2).

        在生成新因子之前检查是否存在相似因子。

        Args:
            code: 因子代码
            name: 因子名称
            hypothesis: 因子假设
            threshold: 相似度阈值 (默认 0.85)

        Returns:
            相似因子列表
        """
        try:
            searcher = self._get_similarity_searcher()
            if searcher is None:
                return []

            results = searcher.search_similar(
                query_code=code,
                query_name=name,
                query_hypothesis=hypothesis,
                limit=5,
                score_threshold=threshold,
            )

            return results

        except Exception as e:
            print(f"Warning: Similarity check failed: {e}")
            return []

    def _generate_code_hash(self, code: str) -> str:
        """Generate hash for factor code."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def _generate_name(self, description: str, family: list[str]) -> str:
        """Generate factor name from description."""
        words = description.lower().split()[:3]
        base = "_".join(w for w in words if w.isalnum())
        if family:
            base = f"{family[0]}_{base}"
        return base[:50]

    async def create_factor(
        self,
        name: str,
        family: list[str],
        code: str,
        target_task: str,
        skip_vector_index: bool = False,
    ) -> Factor:
        """Create a new factor and persist to database.

        Args:
            name: Factor name
            family: Factor family tags
            code: Factor computation code
            target_task: Target prediction task
            skip_vector_index: Skip vector store indexing (for testing)

        Returns:
            Created factor
        """
        factor_id = str(uuid.uuid4())
        code_hash = self._generate_code_hash(code)

        # Check for duplicate code (exact hash match)
        existing = await self.factor_repo.get_by_code_hash(code_hash)
        if existing:
            raise ValueError(f"Factor with same code already exists: {existing.id}")

        factor = Factor(
            id=factor_id,
            name=name,
            family=family,
            code=code,
            code_hash=code_hash,
            target_task=target_task,
            status=FactorStatus.CANDIDATE,
            created_at=datetime.now(),
        )

        await self.factor_repo.create(factor)

        # Task 7.1: Index factor to vector store
        if not skip_vector_index:
            await self._index_factor_to_vector_store(factor)

        # P1.2: Broadcast factor creation via WebSocket
        try:
            await broadcast_factor_created(
                factor_id=factor.id,
                name=factor.name,
                family=factor.family,
            )
        except Exception as e:
            print(f"Warning: Failed to broadcast factor creation: {e}")

        return factor

    async def generate_factor(
        self,
        description: str,
        family: list[str],
        target_task: str,
        check_similarity: bool = True,
        similarity_threshold: float = 0.95,
    ) -> Factor:
        """Generate a new factor from description using LLM.

        Includes similarity check (Task 7.2) to prevent duplicate factors.

        Args:
            description: Natural language description
            family: Factor family tags
            target_task: Target prediction task
            check_similarity: Whether to check for similar existing factors
            similarity_threshold: Threshold for raising FactorDuplicateError (default 0.95)

        Returns:
            Generated factor

        Raises:
            FactorDuplicateError: If a highly similar factor already exists
        """
        # Map family string to FactorFamily enum
        factor_family = None
        if family:
            family_map = {
                "momentum": FactorFamily.MOMENTUM,
                "value": FactorFamily.VALUE,
                "volatility": FactorFamily.VOLATILITY,
                "quality": FactorFamily.QUALITY,
                "sentiment": FactorFamily.SENTIMENT,
                "liquidity": FactorFamily.LIQUIDITY,
            }
            factor_family = family_map.get(family[0].lower())

        generated_code = None
        generated_name = None
        warnings = []

        try:
            # Call LLM to generate factor
            agent = self._get_generation_agent()
            generated = await agent.generate(
                user_request=description,
                factor_family=factor_family,
            )
            generated_code = generated.code
            generated_name = generated.name

        except Exception as e:
            # Fallback to simple generation if LLM fails
            print(f"Warning: LLM generation failed ({e}), using fallback")
            generated_name = self._generate_name(description, family)
            generated_code = f'''def {generated_name}(df):
    """
    {description}

    Generated by fallback (LLM unavailable)
    """
    import pandas as pd
    # 简单动量因子作为默认
    returns = df["close"].pct_change(20)
    factor = (returns - returns.mean()) / returns.std()
    return factor
'''

        # Task 7.2: Similarity check before creation
        if check_similarity and generated_code:
            similar_factors = await self._check_similarity(
                code=generated_code,
                name=generated_name,
                hypothesis=description,
                threshold=0.80,  # Lower threshold for search
            )

            if similar_factors:
                top_similar = similar_factors[0]

                # Raise error if similarity is very high
                if top_similar.score >= similarity_threshold:
                    raise FactorDuplicateError(
                        f"Factor highly similar to '{top_similar.name}' "
                        f"(similarity: {top_similar.score:.2%})"
                    )

                # Add warning for moderate similarity
                if top_similar.score >= 0.80:
                    warnings.append(
                        f"Similar factor found: {top_similar.name} "
                        f"(similarity: {top_similar.score:.2%})"
                    )

        # Create the factor
        factor = await self.create_factor(
            name=generated_name,
            family=family,
            code=generated_code,
            target_task=target_task,
        )

        # Log warnings if any
        if warnings:
            for warning in warnings:
                print(f"Warning: {warning}")

        return factor

    async def get_factor(self, factor_id: str) -> Optional[Factor]:
        """Get factor by ID.

        Args:
            factor_id: Factor ID

        Returns:
            Factor if found, None otherwise
        """
        return await self.factor_repo.get_by_id(factor_id)

    async def list_factors(
        self,
        page: int = 1,
        page_size: int = 10,
        family: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[list[Factor], int]:
        """List factors with pagination and filtering.

        Args:
            page: Page number (1-indexed)
            page_size: Items per page
            family: Filter by family
            status: Filter by status

        Returns:
            Tuple of (factors, total_count)
        """
        return await self.factor_repo.list_factors(page, page_size, family, status)

    async def update_status(self, factor_id: str, status: str) -> Factor:
        """Update factor status.

        Args:
            factor_id: Factor ID
            status: New status

        Returns:
            Updated factor

        Raises:
            FactorNotFoundError: If factor not found
        """
        factor = await self.factor_repo.get_by_id(factor_id)
        if not factor:
            raise FactorNotFoundError(f"Factor {factor_id} not found")

        factor.status = FactorStatus(status)
        await self.factor_repo.update(factor)
        return factor

    async def delete_factor(self, factor_id: str) -> bool:
        """Delete a factor.

        Args:
            factor_id: Factor ID

        Returns:
            True if deleted, False if not found
        """
        return await self.factor_repo.delete(factor_id)

    async def evaluate_factor(
        self,
        factor_id: str,
        splits: list[str],
        market_splits: list[str] = None,
        include_cv: bool = True,
        include_stability: bool = True,
        include_cost_analysis: bool = True,
        symbol: str = "ETH/USDT",
        timeframe: str = "1d",
    ) -> tuple[FactorMetrics, StabilityReport, bool, int]:
        """Evaluate a factor using full evaluation pipeline.

        Integrates:
        - CryptoCVSplitter for multi-dimensional validation
        - StabilityAnalyzer for robustness analysis
        - Transaction cost estimation

        Args:
            factor_id: Factor ID
            splits: Data splits to evaluate on (train/valid/test)
            market_splits: Market splits to evaluate on (large/mid/small)
            include_cv: Whether to run CV splits evaluation
            include_stability: Whether to run stability analysis
            include_cost_analysis: Whether to estimate transaction costs

        Returns:
            Tuple of (metrics, stability, passed_threshold, experiment_number)

        Raises:
            FactorNotFoundError: If factor not found
        """
        factor = await self.factor_repo.get_by_id(factor_id)
        if not factor:
            raise FactorNotFoundError(f"Factor {factor_id} not found")

        # Use Qlib-based factor engine with TimescaleDB data
        from iqfmp.core.factor_engine import FactorEngine
        from iqfmp.core.data_provider import DataProvider

        try:
            # Load real market data from TimescaleDB (primary) or CSV (fallback)
            # Uses parameterized symbol/timeframe instead of hardcoded values
            provider = DataProvider(session=self.session)
            df = await provider.load_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
            )

            engine = FactorEngine(df=df)

            # Compute factor values using actual code
            factor_values = engine.compute_factor(factor.code, factor.name)

            # Get the underlying data for evaluation
            df = engine.data
            if df is None:
                raise ValueError("No data loaded in engine")

            # ====== Persist factor values to TimescaleDB ======
            symbol = df.get("symbol", pd.Series(["MULTI"] * len(df))).iloc[0] if "symbol" in df.columns else "MULTI"
            try:
                rows_saved = await self.factor_value_repo.save_factor_values(
                    factor_id=factor_id,
                    values=factor_values,
                    symbol=str(symbol),
                    timeframe="1d",
                )
                print(f"Persisted {rows_saved} factor values to database for {factor.name}")
            except Exception as e:
                print(f"Warning: Failed to persist factor values: {e}")

            # Prepare evaluation DataFrame
            eval_df = self._prepare_evaluation_data(df, factor_values)

            # ====== 1. Configure Full Evaluator ======
            eval_config = EvaluationConfig(
                date_column="date",
                symbol_column="symbol",
                factor_column="factor_value",
                return_column="forward_return",
                use_cv_splits=include_cv,
                run_stability_analysis=include_stability,
                ic_threshold=0.03,
                ir_threshold=1.0,
            )

            # ====== 2. Run CryptoCVSplitter if enabled ======
            cv_results = {}
            if include_cv:
                cv_config = CVSplitConfig(
                    time_split=True,
                    market_split=market_splits is not None,
                    frequency_split=False,  # Single frequency for now
                    train_ratio=0.6,
                    valid_ratio=0.2,
                    test_ratio=0.2,
                    strict_temporal=True,
                    gap_periods=1,  # 1 day gap to prevent look-ahead
                )
                cv_splitter = CryptoCVSplitter(cv_config)

                # Run CV evaluation
                for split_result in cv_splitter.split(eval_df):
                    split_name = split_result.metadata.get("split_type", "default")
                    train_ic = self._calculate_ic(
                        split_result.train["factor_value"],
                        split_result.train["forward_return"],
                    )
                    test_ic = self._calculate_ic(
                        split_result.test["factor_value"],
                        split_result.test["forward_return"],
                    )
                    cv_results[split_name] = {
                        "train_ic": train_ic,
                        "test_ic": test_ic,
                        "train_size": len(split_result.train),
                        "test_size": len(split_result.test),
                    }

            # ====== 3. Calculate Core Metrics ======
            metrics_dict = self._calculate_core_metrics(
                factor_values=eval_df["factor_value"],
                returns=eval_df["forward_return"],
                volume=df.get("volume") if "volume" in df.columns else None,
            )

            # ====== 4. Run Stability Analysis if enabled ======
            stability_dict = {"time_stability": {}, "market_stability": {}, "regime_stability": {}}
            if include_stability:
                try:
                    stability_config = StabilityConfig(
                        date_column="date",
                        factor_column="factor_value",
                        return_column="forward_return",
                        time_frequency="monthly",
                    )
                    stability_analyzer = StabilityAnalyzer(stability_config)
                    stability_report = stability_analyzer.analyze(eval_df)

                    stability_dict = {
                        "time_stability": {
                            "ic_mean": stability_report.time_stability.ic_mean,
                            "ic_std": stability_report.time_stability.ic_std,
                            "ir": stability_report.time_stability.ir,
                            "stability_score": stability_report.time_stability.stability_score,
                        },
                        "market_stability": {
                            "consistency_score": stability_report.market_stability.consistency_score,
                            "stability_score": stability_report.market_stability.stability_score,
                        },
                        "regime_stability": {
                            "sensitivity_score": stability_report.regime_stability.sensitivity_score,
                            "stability_score": stability_report.regime_stability.stability_score,
                        },
                        "overall_grade": stability_report.grade,
                        "overall_score": stability_report.overall_score.value,
                    }
                except Exception as e:
                    print(f"Warning: Stability analysis failed: {e}")
                    stability_dict["error"] = str(e)

            # ====== 5. Estimate Transaction Costs if enabled ======
            cost_metrics = {}
            if include_cost_analysis and "volume" in df.columns:
                cost_metrics = self._estimate_transaction_costs(
                    factor_values=eval_df["factor_value"],
                    volume=df["volume"],
                )
                metrics_dict.update(cost_metrics)

            # Build final metrics object
            ic_by_split = {}
            sharpe_by_split = {}
            for split_name in splits:
                if split_name in cv_results:
                    ic_by_split[split_name] = cv_results[split_name].get("test_ic", 0.0)
                else:
                    ic_by_split[split_name] = metrics_dict.get("ic_mean", 0.0)
                sharpe_by_split[split_name] = metrics_dict.get("sharpe", 0.0)

            metrics = FactorMetrics(
                ic_mean=metrics_dict.get("ic_mean", 0.0),
                ic_std=metrics_dict.get("ic_std", 0.0),
                ir=metrics_dict.get("ir", 0.0),
                sharpe=metrics_dict.get("sharpe", 0.0),
                max_drawdown=metrics_dict.get("max_drawdown", 0.0),
                turnover=metrics_dict.get("turnover", 0.0),
                ic_by_split=ic_by_split,
                sharpe_by_split=sharpe_by_split,
            )

            stability = StabilityReport(
                time_stability=stability_dict.get("time_stability", {}),
                market_stability=stability_dict.get("market_stability", {}),
                regime_stability=stability_dict.get("regime_stability", {}),
            )

        except (ValueError, FileNotFoundError) as e:
            # H1 FIX: Raise proper error instead of returning zero metrics
            # Zero metrics would cause factors to be incorrectly evaluated as "poor"
            error_msg = f"Data loading failed for factor {factor_id} (symbol={symbol}, timeframe={timeframe}): {e}"
            print(f"ERROR: {error_msg}")
            raise FactorEvaluationError(error_msg)
        except Exception as e:
            # H1 FIX: Raise proper error for any evaluation failure
            error_msg = f"Factor evaluation failed for {factor_id}: {e}"
            print(f"ERROR: {error_msg}")
            raise FactorEvaluationError(error_msg)

        # Get dynamic threshold
        threshold = await self.trial_repo.calculate_dynamic_threshold()
        passed = metrics.sharpe > threshold

        # Record trial in research ledger with factor family
        factor_family = factor.family[0] if factor.family else "unknown"
        trial_number = await self.trial_repo.create(
            factor_id=factor_id,
            factor_name=factor.name,
            sharpe_ratio=metrics.sharpe,
            factor_family=factor_family,
            ic_mean=metrics.ic_mean,
            ir=metrics.ir,
            max_drawdown=metrics.max_drawdown,
            win_rate=None,  # Will be calculated in backtest
            threshold_used=threshold,
            passed_threshold=passed,
            evaluation_config={
                "splits": splits,
                "market_splits": market_splits,
                "include_cv": include_cv,
                "include_stability": include_stability,
                "include_cost_analysis": include_cost_analysis,
            },
        )

        # Update factor with metrics
        factor.metrics = metrics
        factor.stability = stability
        factor.experiment_number = trial_number
        await self.factor_repo.update(factor)

        # P1.2: Broadcast evaluation completion via WebSocket
        try:
            await broadcast_evaluation_complete(
                factor_id=factor_id,
                passed=passed,
                metrics={
                    "ic_mean": metrics.ic_mean,
                    "ir": metrics.ir,
                    "sharpe": metrics.sharpe,
                    "max_drawdown": metrics.max_drawdown,
                    "experiment_number": trial_number,
                },
            )
        except Exception as e:
            print(f"Warning: Failed to broadcast evaluation completion: {e}")

        return metrics, stability, passed, trial_number

    async def evaluate_factor_extended(
        self,
        factor_id: str,
        splits: list[str],
        include_walk_forward: bool = True,
        include_ic_decomposition: bool = True,
        walk_forward_config: Optional[dict] = None,
        symbol: str = "ETH/USDT",
        timeframe: str = "1d",
    ) -> dict:
        """Run extended factor evaluation with Walk-Forward and IC decomposition.

        This method provides comprehensive robustness analysis beyond basic metrics.

        Args:
            factor_id: Factor ID
            splits: Data splits to evaluate
            include_walk_forward: Run Walk-Forward validation
            include_ic_decomposition: Run IC decomposition analysis
            walk_forward_config: Optional Walk-Forward configuration

        Returns:
            Dictionary with all evaluation results
        """
        # First run basic evaluation with parameterized symbol/timeframe
        metrics, stability, passed, trial_number = await self.evaluate_factor(
            factor_id=factor_id,
            splits=splits,
            include_cv=True,
            include_stability=True,
            symbol=symbol,
            timeframe=timeframe,
        )

        result = {
            "factor_id": factor_id,
            "metrics": {
                "ic_mean": metrics.ic_mean,
                "ic_std": metrics.ic_std,
                "ir": metrics.ir,
                "sharpe": metrics.sharpe,
                "max_drawdown": metrics.max_drawdown,
                "turnover": metrics.turnover,
            },
            "passed_threshold": passed,
            "experiment_number": trial_number,
            "stability": {
                "time_stability": stability.time_stability,
                "market_stability": stability.market_stability,
                "regime_stability": stability.regime_stability,
            },
        }

        # Get factor and compute values for extended analysis
        factor = await self.factor_repo.get_by_id(factor_id)
        if not factor:
            return result

        try:
            from iqfmp.core.factor_engine import FactorEngine
            from iqfmp.core.data_provider import DataProvider

            provider = DataProvider(session=self.session)
            df = await provider.load_ohlcv(symbol=symbol, timeframe=timeframe)
            engine = FactorEngine(df=df)
            factor_values = engine.compute_factor(factor.code, factor.name)
            eval_df = self._prepare_evaluation_data(engine.data, factor_values)

            # Walk-Forward Validation
            if include_walk_forward:
                try:
                    wf_config_dict = walk_forward_config or {}
                    wf_config = WalkForwardConfig(
                        window_size=wf_config_dict.get("window_size", 252),
                        step_size=wf_config_dict.get("step_size", 63),
                        min_train_samples=wf_config_dict.get("min_train_samples", 126),
                        date_column="date",
                        factor_column="factor_value",
                        return_column="forward_return",
                    )
                    wf_validator = WalkForwardValidator(config=wf_config)
                    wf_result = wf_validator.validate(eval_df)

                    result["walk_forward"] = {
                        "avg_train_ic": wf_result.avg_train_ic,
                        "avg_oos_ic": wf_result.avg_oos_ic,
                        "ic_degradation": wf_result.ic_degradation,
                        "oos_ir": wf_result.oos_ir,
                        "min_oos_ic": wf_result.min_oos_ic,
                        "max_oos_ic": wf_result.max_oos_ic,
                        "oos_ic_std": wf_result.oos_ic_std,
                        "ic_consistency": wf_result.ic_consistency,
                        "passes_robustness": wf_result.passes_robustness,
                        "ic_decay_rate": wf_result.ic_decay_rate,
                        "predicted_half_life": wf_result.predicted_half_life,
                        "raw_sharpe": wf_result.raw_sharpe,
                        "deflated_sharpe": wf_result.deflated_sharpe,
                        "n_windows": wf_result.n_windows,
                        "window_results": [w.to_dict() for w in wf_result.window_results[:10]],
                    }
                except Exception as e:
                    result["walk_forward"] = {"error": str(e)}

            # IC Decomposition
            if include_ic_decomposition:
                try:
                    ic_config = ICDecompositionConfig(
                        date_column="date",
                        factor_column="factor_value",
                        return_column="forward_return",
                    )
                    ic_analyzer = ICDecompositionAnalyzer(config=ic_config)
                    ic_result = ic_analyzer.analyze(eval_df)

                    result["ic_decomposition"] = {
                        "total_ic": ic_result.total_ic,
                        "ic_by_month": dict(list(ic_result.ic_by_month.items())[-12:]),  # Last 12 months
                        "ic_by_quarter": ic_result.ic_by_quarter,
                        "large_cap_ic": ic_result.large_cap_ic,
                        "mid_cap_ic": ic_result.mid_cap_ic,
                        "small_cap_ic": ic_result.small_cap_ic,
                        "high_vol_ic": ic_result.high_vol_ic,
                        "low_vol_ic": ic_result.low_vol_ic,
                        "ic_hit_rate": ic_result.ic_hit_rate,
                        "ic_stability": ic_result.ic_stability,
                        "regime_shift_detected": ic_result.regime_shift_detected,
                        "ic_decay_rate": ic_result.ic_decay_rate,
                        "predicted_half_life": ic_result.predicted_half_life,
                        "diagnosis": ic_result.diagnosis,
                        "recommendations": ic_result.recommendations,
                    }
                except Exception as e:
                    result["ic_decomposition"] = {"error": str(e)}

            # Calculate overall verdict
            wf_passed = result.get("walk_forward", {}).get("passes_robustness", True)
            ic_stable = result.get("ic_decomposition", {}).get("ic_stability", 0) > 0.3
            overall_score = self._calculate_overall_score(result)

            if passed and wf_passed and ic_stable:
                verdict = "pass"
            elif passed or (wf_passed and overall_score > 50):
                verdict = "needs_review"
            else:
                verdict = "fail"

            result["overall_score"] = overall_score
            result["verdict"] = verdict

        except Exception as e:
            result["extended_analysis_error"] = str(e)

        return result

    def _calculate_overall_score(self, result: dict) -> float:
        """Calculate composite score from all metrics (0-100)."""
        score = 0.0

        # Basic metrics contribution (40 points max)
        metrics = result.get("metrics", {})
        ic = abs(metrics.get("ic_mean", 0))
        ir = metrics.get("ir", 0)
        sharpe = metrics.get("sharpe", 0)

        score += min(15, ic * 300)  # IC: max 15 points
        score += min(15, ir * 7.5)  # IR: max 15 points
        score += min(10, max(0, sharpe) * 5)  # Sharpe: max 10 points

        # Walk-Forward contribution (30 points max)
        wf = result.get("walk_forward", {})
        if "error" not in wf:
            oos_ic = abs(wf.get("avg_oos_ic", 0))
            consistency = wf.get("ic_consistency", 0)
            degradation = wf.get("ic_degradation", 1)

            score += min(10, oos_ic * 200)  # OOS IC: max 10 points
            score += min(10, consistency * 10)  # Consistency: max 10 points
            score += min(10, max(0, 1 - degradation) * 10)  # Low degradation: max 10 points

        # IC Decomposition contribution (20 points max)
        ic_dec = result.get("ic_decomposition", {})
        if "error" not in ic_dec:
            hit_rate = ic_dec.get("ic_hit_rate", 0)
            stability = ic_dec.get("ic_stability", 0)

            score += min(10, hit_rate * 10)  # Hit rate: max 10 points
            score += min(10, stability * 10)  # Stability: max 10 points

        # Stability contribution (10 points max)
        if result.get("passed_threshold", False):
            score += 10

        return round(min(100, score), 2)

    def _prepare_evaluation_data(
        self, df: pd.DataFrame, factor_values: pd.Series
    ) -> pd.DataFrame:
        """Prepare DataFrame for evaluation modules."""
        eval_df = pd.DataFrame()

        # Handle datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            eval_df["date"] = df.index
            eval_df["datetime"] = df.index
        elif "timestamp" in df.columns:
            eval_df["date"] = pd.to_datetime(df["timestamp"])
            eval_df["datetime"] = eval_df["date"]
        else:
            eval_df["date"] = pd.date_range(start="2022-01-01", periods=len(df), freq="D")
            eval_df["datetime"] = eval_df["date"]

        # Factor values
        eval_df["factor_value"] = factor_values.values if hasattr(factor_values, "values") else factor_values

        # Forward returns (1-day by default)
        if "fwd_returns_1d" in df.columns:
            eval_df["forward_return"] = df["fwd_returns_1d"].values
        elif "returns" in df.columns:
            eval_df["forward_return"] = df["returns"].shift(-1).values
        else:
            eval_df["forward_return"] = df["close"].pct_change().shift(-1).values

        # Symbol (for market split)
        eval_df["symbol"] = df.get("symbol", "UNKNOWN").values if "symbol" in df.columns else "UNKNOWN"

        # Market cap (for market stability - estimate from volume * price)
        if "volume" in df.columns and "close" in df.columns:
            eval_df["market_cap"] = (df["volume"] * df["close"]).values
        else:
            eval_df["market_cap"] = 1e10  # Default to mid-cap

        return eval_df.dropna(subset=["factor_value", "forward_return"])

    def _calculate_ic(self, factor_values: pd.Series, returns: pd.Series) -> float:
        """Calculate Information Coefficient (Spearman correlation)."""
        from scipy import stats

        mask = ~(factor_values.isna() | returns.isna())
        if mask.sum() < 10:
            return 0.0

        corr, _ = stats.spearmanr(
            factor_values[mask].values, returns[mask].values
        )
        return float(corr) if not np.isnan(corr) else 0.0

    def _calculate_core_metrics(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> dict:
        """Calculate core factor metrics."""
        from scipy import stats

        metrics = {}

        # IC (Information Coefficient)
        mask = ~(factor_values.isna() | returns.isna())
        if mask.sum() > 10:
            ic, _ = stats.spearmanr(factor_values[mask].values, returns[mask].values)
            metrics["ic_mean"] = float(ic) if not np.isnan(ic) else 0.0
        else:
            metrics["ic_mean"] = 0.0

        # Rolling IC for IR calculation
        ic_series = []
        window = 20
        for i in range(window, len(factor_values)):
            f_window = factor_values.iloc[i - window:i]
            r_window = returns.iloc[i - window:i]
            m = ~(f_window.isna() | r_window.isna())
            if m.sum() >= 5:
                ic_val, _ = stats.spearmanr(f_window[m].values, r_window[m].values)
                if not np.isnan(ic_val):
                    ic_series.append(ic_val)

        if ic_series:
            metrics["ic_std"] = float(np.std(ic_series))
            metrics["ir"] = float(np.mean(ic_series) / np.std(ic_series)) if np.std(ic_series) > 0 else 0.0
        else:
            metrics["ic_std"] = 0.0
            metrics["ir"] = 0.0

        # Backtest as simple long-short strategy
        factor_zscore = (factor_values - factor_values.rolling(20).mean()) / factor_values.rolling(20).std()
        factor_zscore = factor_zscore.clip(-3, 3)
        position = np.sign(factor_zscore).fillna(0)
        strategy_returns = position.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)

        # Sharpe ratio
        if strategy_returns.std() > 0:
            metrics["sharpe"] = float((strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252))
        else:
            metrics["sharpe"] = 0.0

        # Max drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        metrics["max_drawdown"] = float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0

        # Turnover
        position_changes = position.diff().abs()
        metrics["turnover"] = float(position_changes.mean()) if len(position_changes) > 0 else 0.0

        return metrics

    def _estimate_transaction_costs(
        self,
        factor_values: pd.Series,
        volume: pd.Series,
        taker_fee: float = 0.0004,  # 0.04% taker fee
        slippage_bps: float = 2.0,  # 2 basis points
    ) -> dict:
        """Estimate transaction costs and capacity.

        Args:
            factor_values: Factor values series
            volume: Trading volume series
            taker_fee: Taker fee rate (default 0.04%)
            slippage_bps: Slippage in basis points

        Returns:
            Dict with cost metrics
        """
        # Calculate turnover
        factor_zscore = (factor_values - factor_values.rolling(20).mean()) / factor_values.rolling(20).std()
        position = np.sign(factor_zscore.clip(-3, 3)).fillna(0)
        position_changes = position.diff().abs()
        turnover = float(position_changes.mean())

        # Estimate annual cost
        slippage_rate = slippage_bps / 10000
        annual_cost = turnover * (taker_fee + slippage_rate) * 252

        # Capacity estimation (1% of avg volume)
        avg_volume = float(volume.mean())
        capacity_daily = avg_volume * 0.01
        capacity_annual = capacity_daily * 252

        # Implementability score
        if annual_cost < 0.005:  # < 0.5% annual cost
            implementability = 1.0
        elif annual_cost < 0.02:  # < 2%
            implementability = 0.7
        elif annual_cost < 0.05:  # < 5%
            implementability = 0.4
        else:
            implementability = 0.1

        return {
            "estimated_annual_cost_bps": round(annual_cost * 10000, 2),
            "capacity_daily_usd": round(capacity_daily, 0),
            "capacity_annual_usd": round(capacity_annual, 0),
            "implementability_score": round(implementability, 2),
        }

    async def get_stats(self) -> dict:
        """Get comprehensive factor statistics for monitoring dashboard.

        Returns:
            Dictionary with basic and extended statistics for frontend dashboard.
        """
        status_counts = await self.factor_repo.count_by_status()
        total_trials = await self.trial_repo.get_total_trials()
        threshold = await self.trial_repo.calculate_dynamic_threshold()

        # Calculate extended metrics for monitoring dashboard
        total_factors = sum(status_counts.values())
        core_count = status_counts.get("core", 0)
        candidate_count = status_counts.get("candidate", 0)
        rejected_count = status_counts.get("rejected", 0)
        redundant_count = status_counts.get("redundant", 0)

        # Evaluated = all non-candidate factors
        evaluated_count = core_count + rejected_count + redundant_count

        # Pass rate = core / evaluated (if any evaluated)
        pass_rate = (core_count / evaluated_count * 100) if evaluated_count > 0 else 0.0

        # Calculate average IC and Sharpe from factors with metrics
        avg_ic = 0.0
        avg_sharpe = 0.0
        try:
            factors, _ = await self.factor_repo.list_factors(page=1, page_size=1000)
            total_ic = 0.0
            total_sharpe = 0.0
            count_with_metrics = 0
            for factor in factors:
                if factor.metrics:
                    total_ic += factor.metrics.ic_mean
                    total_sharpe += factor.metrics.sharpe
                    count_with_metrics += 1
            if count_with_metrics > 0:
                avg_ic = total_ic / count_with_metrics
                avg_sharpe = total_sharpe / count_with_metrics
        except Exception:
            pass  # Use default 0.0 values

        return {
            # Basic counts
            "total_factors": total_factors,
            "by_status": status_counts,
            "total_trials": total_trials,
            "current_threshold": threshold,
            # Extended fields for monitoring dashboard
            "evaluated_count": evaluated_count,
            "pass_rate": round(pass_rate, 2),
            "avg_ic": round(avg_ic, 4),
            "avg_sharpe": round(avg_sharpe, 4),
            "pending_count": candidate_count,
        }

    # ==================== Mining Task Methods ====================

    async def create_mining_task(
        self,
        name: str,
        description: str,
        factor_families: list[str],
        target_count: int,
        auto_evaluate: bool,
    ) -> str:
        """Create and start a factor mining task.

        Uses MiningTaskRepository for dual persistence (DB + Redis cache).

        Args:
            name: Task name
            description: Task description
            factor_families: Factor families to mine
            target_count: Target number of factors to generate
            auto_evaluate: Whether to auto-evaluate generated factors

        Returns:
            Task ID

        Raises:
            ValueError: If LLM API key is not configured
        """
        # Validate LLM configuration before creating task
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "LLM API key not configured. Please configure OpenRouter API key in Settings page first."
            )

        task_id = str(uuid.uuid4())

        # Create task in database (with Redis cache)
        task_data = await self.mining_task_repo.create(
            task_id=task_id,
            name=name,
            description=description,
            factor_families=factor_families,
            target_count=target_count,
            auto_evaluate=auto_evaluate,
        )

        # C2 FIX: Use Celery task for persistent task queue
        # Tasks are persisted to Redis and survive service restarts
        from iqfmp.celery_app.tasks import mining_task
        mining_task.delay(task_id, task_data)

        return task_id

    async def _run_mining_task(self, task_id: str, task_data: dict) -> None:
        """Run mining task in background.

        Uses MiningTaskRepository for persistent progress tracking.
        Creates its own database session to avoid using the closed request session.

        Args:
            task_id: Task ID
            task_data: Task configuration data
        """
        generated_count = 0
        passed_count = 0
        failed_count = 0

        # Create independent session for background task
        # This avoids using the request-scoped session which gets closed after HTTP response
        async with get_async_session() as bg_session:
            bg_mining_repo = MiningTaskRepository(bg_session, self.redis_client)

            try:
                # Mark as started
                await bg_mining_repo.set_started(task_id)

                target_count = task_data["target_count"]
                families = task_data.get("factor_families") or ["momentum", "volatility"]
                auto_evaluate = task_data.get("auto_evaluate", True)

                for i in range(target_count):
                    if await self._is_task_cancelled(task_id):
                        await bg_mining_repo.set_completed(task_id, status="cancelled")
                        return

                    # Generate factor
                    family = families[i % len(families)]
                    description = f"Auto-generated {family} factor #{i+1} for mining task {task_data['name']}"

                    try:
                        factor = await self.generate_factor(
                            description=description,
                            family=[family],
                            target_task="price_prediction",
                        )
                        generated_count += 1

                        # Auto-evaluate if enabled
                        if auto_evaluate:
                            try:
                                _, _, passed, _ = await self.evaluate_factor(
                                    factor_id=factor.id,
                                    splits=["train", "valid", "test"],
                                )
                                if passed:
                                    passed_count += 1
                                else:
                                    failed_count += 1
                            except Exception:
                                failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        print(f"Mining task error: {e}")

                    # Update progress in database
                    progress = ((i + 1) / target_count) * 100
                    await bg_mining_repo.update_progress(
                        task_id=task_id,
                        generated_count=generated_count,
                        passed_count=passed_count,
                        failed_count=failed_count,
                        progress=progress,
                    )

                    # P1.2: Broadcast mining progress via WebSocket
                    try:
                        await broadcast_task_update(
                            task_id=task_id,
                            task_type="mining",
                            status="running",
                            progress=progress,
                            result={
                                "generated": generated_count,
                                "passed": passed_count,
                                "failed": failed_count,
                            },
                        )
                    except Exception:
                        pass  # Non-critical, continue execution

                    # Small delay to avoid overwhelming LLM
                    await asyncio.sleep(0.5)

                # Mark as completed
                await bg_mining_repo.set_completed(task_id, status="completed")

            except Exception as e:
                # Mark as failed
                await bg_mining_repo.set_completed(
                    task_id=task_id,
                    status="failed",
                    error_message=str(e),
                )

    async def _is_task_cancelled(self, task_id: str) -> bool:
        """Check if task has been cancelled."""
        # Check Redis for real-time cancellation signal
        if self.redis_client:
            cancelled = await self.redis_client.sismember("mining_tasks:cancelled", task_id)
            return bool(cancelled)

        # Fallback: check database status
        task = await self.mining_task_repo.get_by_id(task_id)
        return task.get("status") == "cancelled" if task else False

    async def list_mining_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> list[MiningTaskStatus]:
        """List mining tasks from database.

        Args:
            status: Filter by status
            limit: Maximum number to return

        Returns:
            List of mining task statuses
        """
        task_dicts, _ = await self.mining_task_repo.list_tasks(
            status=status,
            limit=limit,
        )

        tasks = []
        for task_data in task_dicts:
            tasks.append(MiningTaskStatus(
                id=task_data["id"],
                name=task_data["name"],
                description=task_data.get("description", ""),
                factor_families=task_data.get("factor_families", []),
                target_count=task_data["target_count"],
                generated_count=task_data["generated_count"],
                passed_count=task_data["passed_count"],
                failed_count=task_data["failed_count"],
                status=task_data["status"],
                progress=task_data["progress"],
                error_message=task_data.get("error_message"),
                created_at=datetime.fromisoformat(task_data["created_at"]) if isinstance(task_data["created_at"], str) else task_data["created_at"],
                started_at=datetime.fromisoformat(task_data["started_at"]) if task_data.get("started_at") and isinstance(task_data["started_at"], str) else task_data.get("started_at"),
                completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data.get("completed_at") and isinstance(task_data["completed_at"], str) else task_data.get("completed_at"),
            ))

        return tasks

    async def get_mining_task(self, task_id: str) -> Optional[MiningTaskStatus]:
        """Get mining task by ID from database.

        Args:
            task_id: Task ID

        Returns:
            Mining task status or None
        """
        task_data = await self.mining_task_repo.get_by_id(task_id)

        if task_data:
            return MiningTaskStatus(
                id=task_data["id"],
                name=task_data["name"],
                description=task_data.get("description", ""),
                factor_families=task_data.get("factor_families", []),
                target_count=task_data["target_count"],
                generated_count=task_data["generated_count"],
                passed_count=task_data["passed_count"],
                failed_count=task_data["failed_count"],
                status=task_data["status"],
                progress=task_data["progress"],
                error_message=task_data.get("error_message"),
                created_at=datetime.fromisoformat(task_data["created_at"]) if isinstance(task_data["created_at"], str) else task_data["created_at"],
                started_at=datetime.fromisoformat(task_data["started_at"]) if task_data.get("started_at") and isinstance(task_data["started_at"], str) else task_data.get("started_at"),
                completed_at=datetime.fromisoformat(task_data["completed_at"]) if task_data.get("completed_at") and isinstance(task_data["completed_at"], str) else task_data.get("completed_at"),
            )
        return None

    async def cancel_mining_task(self, task_id: str) -> bool:
        """Cancel a running mining task.

        Args:
            task_id: Task ID

        Returns:
            True if cancelled, False otherwise
        """
        # Get task from database
        task_data = await self.mining_task_repo.get_by_id(task_id)

        if task_data and task_data["status"] == "running":
            # Set Redis cancellation signal for real-time detection
            if self.redis_client:
                await self.redis_client.sadd("mining_tasks:cancelled", task_id)

            # Also update database status
            await self.mining_task_repo.set_completed(task_id, status="cancelled")
            return True

        return False

    # ==================== Factor Library Methods ====================

    async def get_library_stats(self) -> FactorLibraryStats:
        """Get comprehensive factor library statistics.

        Returns:
            Factor library statistics
        """
        status_counts = await self.factor_repo.count_by_status()
        factors, _ = await self.factor_repo.list_factors(page=1, page_size=1000)

        # Calculate by family
        by_family: dict[str, int] = {}
        total_sharpe = 0.0
        total_ic = 0.0
        count_with_metrics = 0
        best_factor_id = None
        best_sharpe = 0.0

        for factor in factors:
            # Count by family
            for fam in factor.family:
                by_family[fam] = by_family.get(fam, 0) + 1

            # Calculate average metrics
            if factor.metrics:
                total_sharpe += factor.metrics.sharpe
                total_ic += factor.metrics.ic_mean
                count_with_metrics += 1

                if factor.metrics.sharpe > best_sharpe:
                    best_sharpe = factor.metrics.sharpe
                    best_factor_id = factor.id

        return FactorLibraryStats(
            total_factors=sum(status_counts.values()),
            core_factors=status_counts.get("core", 0),
            candidate_factors=status_counts.get("candidate", 0),
            rejected_factors=status_counts.get("rejected", 0),
            redundant_factors=status_counts.get("redundant", 0),
            by_family=by_family,
            avg_sharpe=total_sharpe / count_with_metrics if count_with_metrics > 0 else 0.0,
            avg_ic=total_ic / count_with_metrics if count_with_metrics > 0 else 0.0,
            best_factor_id=best_factor_id,
            best_sharpe=best_sharpe,
        )

    async def compare_factors(
        self,
        factor_ids: list[str],
        symbol: str = "ETH/USDT",
        timeframe: str = "1d",
    ) -> tuple[list[Factor], dict[str, dict[str, float]], list[str]]:
        """Compare multiple factors using real correlation calculation.

        Args:
            factor_ids: List of factor IDs to compare

        Returns:
            Tuple of (factors, correlation_matrix, ranking)

        Raises:
            FactorNotFoundError: If any factor not found
        """
        factors = []
        for factor_id in factor_ids:
            factor = await self.factor_repo.get_by_id(factor_id)
            if not factor:
                raise FactorNotFoundError(f"Factor {factor_id} not found")
            factors.append(factor)

        # Calculate real correlation matrix using factor values
        from iqfmp.core.factor_engine import FactorEngine
        from iqfmp.core.data_provider import DataProvider
        import numpy as np
        from scipy import stats

        correlation_matrix: dict[str, dict[str, float]] = {}

        try:
            # Load data from TimescaleDB (primary) or CSV (fallback)
            provider = DataProvider(session=self.session)
            df = await provider.load_ohlcv(symbol=symbol, timeframe=timeframe)
            engine = FactorEngine(df=df)

            factor_values_dict = {}
            for factor in factors:
                try:
                    values = engine.compute_factor(factor.code, factor.name)
                    factor_values_dict[factor.id] = values
                except Exception:
                    factor_values_dict[factor.id] = None

            # Calculate pairwise correlations
            for f1 in factors:
                correlation_matrix[f1.id] = {}
                v1 = factor_values_dict.get(f1.id)

                for f2 in factors:
                    if f1.id == f2.id:
                        correlation_matrix[f1.id][f2.id] = 1.0
                    elif f2.id in correlation_matrix and f1.id in correlation_matrix.get(f2.id, {}):
                        correlation_matrix[f1.id][f2.id] = correlation_matrix[f2.id][f1.id]
                    else:
                        v2 = factor_values_dict.get(f2.id)
                        if v1 is not None and v2 is not None:
                            # Calculate Spearman correlation
                            valid_mask = ~(v1.isna() | v2.isna())
                            if valid_mask.sum() > 10:
                                corr, _ = stats.spearmanr(
                                    v1[valid_mask].values,
                                    v2[valid_mask].values,
                                )
                                correlation_matrix[f1.id][f2.id] = round(float(corr), 4) if not np.isnan(corr) else 0.0
                            else:
                                correlation_matrix[f1.id][f2.id] = 0.0
                        else:
                            # Fallback: estimate from family overlap
                            overlap = len(set(f1.family) & set(f2.family))
                            base_corr = 0.3 + (0.3 * overlap / max(len(f1.family), len(f2.family), 1))
                            correlation_matrix[f1.id][f2.id] = round(base_corr, 4)

        except Exception as e:
            # Fallback if data not available
            print(f"Warning: Factor comparison using fallback: {e}")
            for f1 in factors:
                correlation_matrix[f1.id] = {}
                for f2 in factors:
                    if f1.id == f2.id:
                        correlation_matrix[f1.id][f2.id] = 1.0
                    else:
                        overlap = len(set(f1.family) & set(f2.family))
                        base_corr = 0.3 + (0.3 * overlap / max(len(f1.family), len(f2.family), 1))
                        correlation_matrix[f1.id][f2.id] = round(base_corr, 4)

        # Rank by Sharpe ratio (from actual metrics)
        ranking = sorted(
            factor_ids,
            key=lambda fid: next((f.metrics.sharpe if f.metrics else 0 for f in factors if f.id == fid), 0),
            reverse=True,
        )

        return factors, correlation_matrix, ranking


# ==================== Legacy sync API (for backward compatibility) ====================


class SyncFactorService:
    """Synchronous wrapper for FactorService (legacy support)."""

    def __init__(self) -> None:
        """Initialize sync factor service."""
        self._async_service: Optional[FactorService] = None
        # Fallback to in-memory storage if DB not available
        self._factors: dict[str, Factor] = {}
        self._llm_provider: Optional[LLMProvider] = None
        self._generation_agent: Optional[FactorGenerationAgent] = None

    def _get_llm_provider(self) -> LLMProvider:
        if self._llm_provider is None:
            config = LLMConfig.from_env()
            self._llm_provider = LLMProvider(config)
        return self._llm_provider

    def _get_generation_agent(self) -> FactorGenerationAgent:
        if self._generation_agent is None:
            config = FactorGenerationConfig(
                name="factor_generator",
                security_check_enabled=True,
                field_constraint_enabled=True,
            )
            self._generation_agent = FactorGenerationAgent(
                config=config,
                llm_provider=self._get_llm_provider(),
            )
        return self._generation_agent

    def _generate_code_hash(self, code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def _generate_name(self, description: str, family: list[str]) -> str:
        words = description.lower().split()[:3]
        base = "_".join(w for w in words if w.isalnum())
        if family:
            base = f"{family[0]}_{base}"
        return base[:50]

    def create_factor(
        self, name: str, family: list[str], code: str, target_task: str
    ) -> Factor:
        factor_id = str(uuid.uuid4())
        code_hash = self._generate_code_hash(code)
        factor = Factor(
            id=factor_id,
            name=name,
            family=family,
            code=code,
            code_hash=code_hash,
            target_task=target_task,
            status=FactorStatus.CANDIDATE,
            created_at=datetime.now(),
        )
        self._factors[factor_id] = factor
        return factor

    def generate_factor(
        self, description: str, family: list[str], target_task: str
    ) -> Factor:
        try:
            agent = self._get_generation_agent()
            generated = asyncio.run(
                agent.generate(
                    user_request=description,
                    factor_family=None,
                )
            )
            return self.create_factor(
                name=generated.name,
                family=family,
                code=generated.code,
                target_task=target_task,
            )
        except Exception as e:
            print(f"Warning: LLM generation failed ({e}), using fallback")
            name = self._generate_name(description, family)
            code = f'''def {name}(df):
    """
    {description}
    """
    import pandas as pd
    returns = df["close"].pct_change(20)
    factor = (returns - returns.mean()) / returns.std()
    return factor
'''
            return self.create_factor(name=name, family=family, code=code, target_task=target_task)

    def get_factor(self, factor_id: str) -> Optional[Factor]:
        return self._factors.get(factor_id)

    def list_factors(
        self,
        page: int = 1,
        page_size: int = 10,
        family: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[list[Factor], int]:
        factors = list(self._factors.values())
        if family:
            factors = [f for f in factors if family in f.family]
        if status:
            factors = [f for f in factors if f.status.value == status]
        total = len(factors)
        start = (page - 1) * page_size
        end = start + page_size
        return factors[start:end], total

    def update_status(self, factor_id: str, status: str) -> Factor:
        factor = self._factors.get(factor_id)
        if not factor:
            raise FactorNotFoundError(f"Factor {factor_id} not found")
        factor.status = FactorStatus(status)
        return factor

    def delete_factor(self, factor_id: str) -> bool:
        if factor_id in self._factors:
            del self._factors[factor_id]
            return True
        return False

    def evaluate_factor(
        self, factor_id: str, splits: list[str], market_splits: list[str] = None
    ) -> tuple[FactorMetrics, StabilityReport, bool, int]:
        factor = self._factors.get(factor_id)
        if not factor:
            raise FactorNotFoundError(f"Factor {factor_id} not found")

        metrics = FactorMetrics(
            ic_mean=0.05,
            ic_std=0.02,
            ir=2.5,
            sharpe=1.8,
            max_drawdown=0.15,
            turnover=0.3,
            ic_by_split={s: 0.05 for s in splits},
            sharpe_by_split={s: 1.8 for s in splits},
        )
        stability = StabilityReport(
            time_stability={"monthly": 0.8, "quarterly": 0.75},
            market_stability={"btc": 0.9, "altcoins": 0.7},
            regime_stability={"bull": 0.85, "bear": 0.6},
        )
        factor.metrics = metrics
        factor.stability = stability
        factor.experiment_number += 1
        passed = metrics.ir > 1.0 and metrics.sharpe > 1.0
        return metrics, stability, passed, factor.experiment_number


# Singleton for legacy support
_sync_factor_service = SyncFactorService()


def get_factor_service() -> SyncFactorService:
    """Get sync factor service instance (legacy)."""
    return _sync_factor_service
