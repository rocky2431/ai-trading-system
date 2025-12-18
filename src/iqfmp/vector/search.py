"""
相似度检索模块
实现因子相似度搜索和重复检测
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .client import QdrantClient, get_qdrant_client
from .embedding import EmbeddingGenerator, get_embedding_generator
from .store import DEFAULT_COLLECTION

logger = logging.getLogger(__name__)

# 默认相似度阈值
DEFAULT_SIMILARITY_THRESHOLD = 0.85


@dataclass
class SearchResult:
    """搜索结果"""
    factor_id: str
    score: float
    name: str
    hypothesis: str
    family: str
    code: str
    metadata: dict[str, Any]

    @property
    def is_duplicate(self) -> bool:
        """是否为重复因子（相似度超过阈值）"""
        return self.score >= DEFAULT_SIMILARITY_THRESHOLD


class SimilaritySearcher:
    """
    相似度检索器
    用于搜索相似因子和检测重复
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        qdrant_client: Optional[QdrantClient] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        """
        初始化相似度检索器

        Args:
            collection_name: 集合名称
            qdrant_client: Qdrant 客户端
            embedding_generator: Embedding 生成器
            similarity_threshold: 相似度阈值
        """
        self.collection_name = collection_name
        self.qdrant = qdrant_client or get_qdrant_client()
        self.embedding = embedding_generator or get_embedding_generator()
        self.similarity_threshold = similarity_threshold

    def search_similar(
        self,
        query_code: str,
        query_name: str = "",
        query_hypothesis: str = "",
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_family: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        搜索相似因子

        Args:
            query_code: 查询因子代码
            query_name: 查询因子名称
            query_hypothesis: 查询因子假设
            limit: 返回结果数量
            score_threshold: 分数阈值
            filter_family: 过滤家族

        Returns:
            相似因子列表
        """
        # 生成查询向量
        query_vector = self.embedding.generate_factor_embedding(
            factor_code=query_code,
            factor_name=query_name,
            hypothesis=query_hypothesis,
        )

        # 构建过滤条件
        query_filter = None
        if filter_family:
            try:
                from qdrant_client.http import models
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="family",
                            match=models.MatchValue(value=filter_family),
                        )
                    ]
                )
            except ImportError:
                pass

        # 执行搜索
        try:
            # qdrant-client v1.7+ 使用 query_points 替代 search
            response = self.qdrant.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold or self.similarity_threshold,
                query_filter=query_filter,
                with_payload=True,
            )

            # query_points 返回 QueryResponse 对象，需要访问 .points 属性
            results = response.points if hasattr(response, 'points') else response

            return [
                SearchResult(
                    factor_id=str(r.id),
                    score=r.score,
                    name=r.payload.get("name", ""),
                    hypothesis=r.payload.get("hypothesis", ""),
                    family=r.payload.get("family", ""),
                    code=r.payload.get("code", ""),
                    metadata={
                        k: v for k, v in r.payload.items()
                        if k not in ["name", "hypothesis", "family", "code", "id"]
                    },
                )
                for r in results
            ]

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def check_duplicate(
        self,
        code: str,
        name: str = "",
        hypothesis: str = "",
        threshold: Optional[float] = None,
    ) -> tuple[bool, Optional[SearchResult]]:
        """
        检查是否存在重复因子

        Args:
            code: 因子代码
            name: 因子名称
            hypothesis: 因子假设
            threshold: 相似度阈值

        Returns:
            (是否重复, 最相似的因子)
        """
        threshold = threshold or self.similarity_threshold

        results = self.search_similar(
            query_code=code,
            query_name=name,
            query_hypothesis=hypothesis,
            limit=1,
            score_threshold=threshold,
        )

        if results and results[0].score >= threshold:
            return True, results[0]

        return False, None

    def find_duplicates(
        self,
        factors: list[dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> list[tuple[dict[str, Any], SearchResult]]:
        """
        批量检查重复因子

        Args:
            factors: 因子列表
            threshold: 相似度阈值

        Returns:
            重复因子对列表 [(新因子, 已存在的相似因子)]
        """
        threshold = threshold or self.similarity_threshold
        duplicates = []

        for factor in factors:
            is_dup, similar = self.check_duplicate(
                code=factor.get("code", ""),
                name=factor.get("name", ""),
                hypothesis=factor.get("hypothesis", ""),
                threshold=threshold,
            )

            if is_dup and similar:
                duplicates.append((factor, similar))

        return duplicates

    def search_by_hypothesis(
        self,
        hypothesis: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        根据假设搜索相似因子

        Args:
            hypothesis: 因子假设描述
            limit: 返回结果数量

        Returns:
            相似因子列表
        """
        # 只使用假设生成 embedding
        query_vector = self.embedding.generate(hypothesis)

        try:
            # qdrant-client v1.7+ 使用 query_points 替代 search
            response = self.qdrant.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )

            results = response.points if hasattr(response, 'points') else response

            return [
                SearchResult(
                    factor_id=str(r.id),
                    score=r.score,
                    name=r.payload.get("name", ""),
                    hypothesis=r.payload.get("hypothesis", ""),
                    family=r.payload.get("family", ""),
                    code=r.payload.get("code", ""),
                    metadata={},
                )
                for r in results
            ]

        except Exception as e:
            logger.error(f"Search by hypothesis failed: {e}")
            return []

    def search_by_family(
        self,
        family: str,
        limit: int = 100,
    ) -> list[SearchResult]:
        """
        获取指定家族的所有因子

        Args:
            family: 因子家族
            limit: 返回结果数量

        Returns:
            因子列表
        """
        try:
            from qdrant_client.http import models

            results = self.qdrant.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="family",
                            match=models.MatchValue(value=family),
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            return [
                SearchResult(
                    factor_id=str(r.id),
                    score=1.0,  # 精确匹配
                    name=r.payload.get("name", ""),
                    hypothesis=r.payload.get("hypothesis", ""),
                    family=r.payload.get("family", ""),
                    code=r.payload.get("code", ""),
                    metadata={},
                )
                for r in results[0]
            ]

        except Exception as e:
            logger.error(f"Search by family failed: {e}")
            return []

    def get_similarity_matrix(
        self,
        factor_ids: list[str],
    ) -> dict[str, dict[str, float]]:
        """
        计算因子相似度矩阵

        Args:
            factor_ids: 因子 ID 列表

        Returns:
            相似度矩阵 {factor_id: {other_id: score}}
        """
        # 获取所有因子的向量
        try:
            points = self.qdrant.client.retrieve(
                collection_name=self.collection_name,
                ids=factor_ids,
                with_vectors=True,
            )

            if not points:
                return {}

            # 计算两两相似度
            import numpy as np
            from numpy.linalg import norm

            vectors = {str(p.id): np.array(p.vector) for p in points}
            matrix = {}

            for id1, v1 in vectors.items():
                matrix[id1] = {}
                for id2, v2 in vectors.items():
                    if id1 == id2:
                        matrix[id1][id2] = 1.0
                    else:
                        # Cosine similarity
                        similarity = float(np.dot(v1, v2) / (norm(v1) * norm(v2)))
                        matrix[id1][id2] = similarity

            return matrix

        except Exception as e:
            logger.error(f"Failed to compute similarity matrix: {e}")
            return {}
