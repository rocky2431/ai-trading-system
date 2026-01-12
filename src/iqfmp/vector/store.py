"""
因子向量存储服务
管理因子 embedding 的存储和检索
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from qdrant_client.http import models as qdrant_models

from .client import QdrantClient, QdrantConfig, get_qdrant_client
from .embedding import EmbeddingGenerator, get_embedding_generator

logger = logging.getLogger(__name__)

# 默认集合名称
DEFAULT_COLLECTION = "factors"


@dataclass
class FactorDocument:
    """因子文档"""
    id: str
    name: str
    code: str
    hypothesis: str
    family: str
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class FactorVectorStore:
    """
    因子向量存储服务
    提供因子的存储、更新和删除功能
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        qdrant_client: Optional[QdrantClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        qdrant_config: Optional[QdrantConfig] = None,
    ):
        """
        初始化因子向量存储

        Args:
            collection_name: 集合名称
            qdrant_client: Qdrant 客户端
            embedding_generator: Embedding 生成器
            qdrant_config: Qdrant 配置（用于控制严格模式）
        """
        self.collection_name = collection_name
        if qdrant_client:
            self.qdrant = qdrant_client
        elif qdrant_config:
            self.qdrant = QdrantClient(qdrant_config)
        else:
            self.qdrant = get_qdrant_client()
        self.embedding = embedding_generator or get_embedding_generator()

        # 确保集合存在
        self._ensure_collection()

    def _ensure_collection(self):
        """确保集合存在"""
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vector_size=self.embedding.dimension,
                distance="Cosine",
            )
            logger.info(f"Created collection: {self.collection_name}")

    def add_factor(
        self,
        factor_id: str,
        name: str,
        code: str,
        hypothesis: str,
        family: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        添加因子到向量存储

        Args:
            factor_id: 因子 ID
            name: 因子名称
            code: 因子代码
            hypothesis: 因子假设
            family: 因子家族
            metadata: 额外元数据

        Returns:
            存储的因子 ID
        """
        # 生成 embedding
        embedding = self.embedding.generate_factor_embedding(
            factor_code=code,
            factor_name=name,
            hypothesis=hypothesis,
        )

        # 构建 payload
        payload = {
            "id": factor_id,
            "name": name,
            "code": code,
            "hypothesis": hypothesis,
            "family": family,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }

        # 存储到 Qdrant - 严格模式，无 Mock 降级
        try:
            self.qdrant.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=factor_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            logger.info(f"Added factor to vector store: {factor_id}")
            return factor_id

        except Exception as e:
            logger.error(f"Failed to add factor {factor_id}: {e}")
            raise

    def add_factors_batch(
        self,
        factors: list[dict[str, Any]],
    ) -> list[str]:
        """
        批量添加因子

        Args:
            factors: 因子列表，每个因子包含 id, name, code, hypothesis, family

        Returns:
            成功添加的因子 ID 列表
        """
        if not factors:
            return []

        # 生成所有 embedding
        texts = []
        for f in factors:
            combined = f"Factor Name: {f['name']}\nHypothesis: {f['hypothesis']}\nCode:\n{f['code']}"
            texts.append(combined)

        embeddings = self.embedding.generate_batch(texts)

        # 构建 points
        points = []
        for i, f in enumerate(factors):
            factor_id = f.get("id") or str(uuid.uuid4())
            payload = {
                "id": factor_id,
                "name": f["name"],
                "code": f["code"],
                "hypothesis": f["hypothesis"],
                "family": f["family"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                **(f.get("metadata") or {}),
            }

            # 严格模式：必须使用 Qdrant models，无 Mock 降级
            points.append(qdrant_models.PointStruct(
                id=factor_id,
                vector=embeddings[i],
                payload=payload,
            ))

        # 批量存储
        try:
            self.qdrant.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            factor_ids = [f.get("id") or points[i]["id"] for i, f in enumerate(factors)]
            logger.info(f"Added {len(factor_ids)} factors to vector store")
            return factor_ids

        except Exception as e:
            logger.error(f"Failed to add factors batch: {e}")
            raise

    def update_factor(
        self,
        factor_id: str,
        name: Optional[str] = None,
        code: Optional[str] = None,
        hypothesis: Optional[str] = None,
        family: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        更新因子

        Args:
            factor_id: 因子 ID
            name: 新名称
            code: 新代码
            hypothesis: 新假设
            family: 新家族
            metadata: 新元数据

        Returns:
            是否更新成功
        """
        try:
            # 获取现有因子
            existing = self.get_factor(factor_id)
            if not existing:
                logger.warning(f"Factor not found: {factor_id}")
                return False

            # 合并更新
            updated = {
                "name": name or existing.get("name", ""),
                "code": code or existing.get("code", ""),
                "hypothesis": hypothesis or existing.get("hypothesis", ""),
                "family": family or existing.get("family", ""),
            }

            # 重新添加（会覆盖旧的）
            self.add_factor(
                factor_id=factor_id,
                name=updated["name"],
                code=updated["code"],
                hypothesis=updated["hypothesis"],
                family=updated["family"],
                metadata={**existing.get("metadata", {}), **(metadata or {})},
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update factor {factor_id}: {e}")
            return False

    def delete_factor(self, factor_id: str) -> bool:
        """
        删除因子

        Args:
            factor_id: 因子 ID

        Returns:
            是否删除成功
        """
        # 严格模式：必须使用真实 Qdrant，无 Mock 降级
        try:
            self.qdrant.client.delete(
                collection_name=self.collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[factor_id],
                ),
            )

            logger.info(f"Deleted factor: {factor_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete factor {factor_id}: {e}")
            return False

    def get_factor(self, factor_id: str) -> Optional[dict[str, Any]]:
        """
        获取因子信息

        Args:
            factor_id: 因子 ID

        Returns:
            因子信息字典
        """
        try:
            result = self.qdrant.client.retrieve(
                collection_name=self.collection_name,
                ids=[factor_id],
                with_payload=True,
            )

            if result:
                return result[0].payload

            return None

        except Exception as e:
            logger.error(f"Failed to get factor {factor_id}: {e}")
            return None

    def get_collection_stats(self) -> dict[str, Any]:
        """获取集合统计信息"""
        return self.qdrant.get_collection_info(self.collection_name)

    def health_check(self) -> tuple[bool, str]:
        """Check health of the vector store.

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # 检查 Qdrant 连接
            info = self.qdrant.get_collection_info(self.collection_name)
            if info:
                point_count = info.get("points_count", 0)
                return True, f"Qdrant healthy, collection '{self.collection_name}' has {point_count} points"
            else:
                return True, f"Qdrant healthy, collection '{self.collection_name}' is empty"
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False, f"Qdrant unhealthy: {e}"

    def is_available(self) -> bool:
        """检查向量存储是否可用"""
        healthy, _ = self.health_check()
        return healthy

    # =========================================================================
    # Pattern Memory Support (Closed-Loop Factor Mining)
    # =========================================================================

    def ensure_pattern_collection(self, collection_name: str = "patterns") -> None:
        """Ensure the patterns collection exists.

        Args:
            collection_name: Name of the patterns collection
        """
        if not self.qdrant.collection_exists(collection_name):
            self.qdrant.create_collection(
                collection_name=collection_name,
                vector_size=self.embedding.dimension,
                distance="Cosine",
            )
            logger.info(f"Created patterns collection: {collection_name}")

    def add_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        hypothesis: str,
        factor_code: str,
        family: str,
        metrics: dict[str, float],
        feedback: Optional[str] = None,
        failure_reasons: Optional[list[str]] = None,
        trial_id: Optional[str] = None,
        collection_name: str = "patterns",
    ) -> str:
        """Add a pattern to the vector store.

        Patterns are success/failure records from factor evaluation,
        used for similarity-based retrieval during factor generation.

        Args:
            pattern_id: Unique pattern identifier
            pattern_type: "success" or "failure"
            hypothesis: The research hypothesis
            factor_code: Generated factor code
            family: Factor family
            metrics: Evaluation metrics (ic, ir, sharpe, etc.)
            feedback: Structured feedback text (for failures)
            failure_reasons: List of FailureReason values
            trial_id: Reference to evaluation trial
            collection_name: Qdrant collection name

        Returns:
            Stored pattern ID
        """
        # Ensure collection exists
        self.ensure_pattern_collection(collection_name)

        # Generate embedding from hypothesis + code
        embedding = self.embedding.generate_factor_embedding(
            factor_code=factor_code,
            factor_name=f"{pattern_type}_pattern",
            hypothesis=hypothesis,
        )

        # Build payload
        payload = {
            "pattern_id": pattern_id,
            "pattern_type": pattern_type,
            "hypothesis": hypothesis,
            "factor_code": factor_code,
            "family": family,
            "metrics": metrics,
            "feedback": feedback,
            "failure_reasons": failure_reasons or [],
            "trial_id": trial_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Store in Qdrant
        try:
            self.qdrant.client.upsert(
                collection_name=collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=pattern_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            logger.info(f"Added {pattern_type} pattern to vector store: {pattern_id}")
            return pattern_id

        except Exception as e:
            logger.error(f"Failed to add pattern {pattern_id}: {e}")
            raise

    def search_patterns(
        self,
        hypothesis: str,
        pattern_type: Optional[str] = None,
        family: Optional[str] = None,
        limit: int = 5,
        collection_name: str = "patterns",
    ) -> list[dict[str, Any]]:
        """Search for similar patterns by hypothesis.

        Args:
            hypothesis: Query hypothesis for similarity search
            pattern_type: Filter by "success" or "failure" (optional)
            family: Filter by factor family (optional)
            limit: Maximum number of results
            collection_name: Qdrant collection name

        Returns:
            List of similar patterns with scores
        """
        # Ensure collection exists
        self.ensure_pattern_collection(collection_name)

        # Generate query embedding
        query_embedding = self.embedding.generate_factor_embedding(
            factor_code="",
            factor_name="query",
            hypothesis=hypothesis,
        )

        # Build filter conditions
        filter_conditions = []

        if pattern_type:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="pattern_type",
                    match=qdrant_models.MatchValue(value=pattern_type),
                )
            )

        if family:
            filter_conditions.append(
                qdrant_models.FieldCondition(
                    key="family",
                    match=qdrant_models.MatchValue(value=family),
                )
            )

        query_filter = None
        if filter_conditions:
            query_filter = qdrant_models.Filter(must=filter_conditions)

        try:
            results = self.qdrant.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

            patterns = []
            for result in results:
                pattern = result.payload.copy()
                pattern["score"] = result.score
                patterns.append(pattern)

            logger.debug(f"Found {len(patterns)} similar patterns for hypothesis")
            return patterns

        except Exception as e:
            # Log detailed error info for debugging
            logger.error(
                f"Failed to search patterns: {e}",
                extra={
                    "hypothesis_preview": hypothesis[:100] if hypothesis else None,
                    "pattern_type": pattern_type,
                    "family": family,
                    "collection": collection_name,
                },
            )
            # Return empty list to allow caller to continue gracefully
            return []

    def delete_pattern(
        self,
        pattern_id: str,
        collection_name: str = "patterns",
    ) -> bool:
        """Delete a pattern from the vector store.

        Args:
            pattern_id: Pattern ID to delete
            collection_name: Qdrant collection name

        Returns:
            True if deleted successfully
        """
        try:
            self.qdrant.client.delete(
                collection_name=collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[pattern_id],
                ),
            )

            logger.info(f"Deleted pattern: {pattern_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete pattern {pattern_id}: {e}")
            return False

    def get_pattern_stats(
        self,
        collection_name: str = "patterns",
    ) -> dict[str, Any]:
        """Get statistics for the patterns collection.

        Args:
            collection_name: Qdrant collection name

        Returns:
            Collection statistics
        """
        self.ensure_pattern_collection(collection_name)
        return self.qdrant.get_collection_info(collection_name)
