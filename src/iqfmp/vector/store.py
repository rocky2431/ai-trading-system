"""
因子向量存储服务
管理因子 embedding 的存储和检索
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .client import QdrantClient, get_qdrant_client
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
    ):
        """
        初始化因子向量存储

        Args:
            collection_name: 集合名称
            qdrant_client: Qdrant 客户端
            embedding_generator: Embedding 生成器
        """
        self.collection_name = collection_name
        self.qdrant = qdrant_client or get_qdrant_client()
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
            "created_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        # 存储到 Qdrant
        try:
            from qdrant_client.http import models

            self.qdrant.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=factor_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            logger.info(f"Added factor to vector store: {factor_id}")
            return factor_id

        except ImportError:
            # 使用 Mock 客户端
            self.qdrant.client.upsert(
                collection_name=self.collection_name,
                points=[{"id": factor_id, "vector": embedding, "payload": payload}],
            )
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
                "created_at": datetime.utcnow().isoformat(),
                **(f.get("metadata") or {}),
            }

            try:
                from qdrant_client.http import models
                points.append(models.PointStruct(
                    id=factor_id,
                    vector=embeddings[i],
                    payload=payload,
                ))
            except ImportError:
                points.append({
                    "id": factor_id,
                    "vector": embeddings[i],
                    "payload": payload,
                })

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
        try:
            from qdrant_client.http import models

            self.qdrant.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[factor_id],
                ),
            )

            logger.info(f"Deleted factor: {factor_id}")
            return True

        except ImportError:
            # Mock 实现
            logger.info(f"[Mock] Deleted factor: {factor_id}")
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
