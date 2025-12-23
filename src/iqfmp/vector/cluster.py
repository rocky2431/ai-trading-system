"""
因子聚类模块
基于向量相似度对因子进行聚类分析
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .client import QdrantClient, get_qdrant_client
from .store import DEFAULT_COLLECTION

logger = logging.getLogger(__name__)


class ClusteringDependencyError(Exception):
    """聚类依赖包未安装时抛出的异常。

    严格模式 - 禁止静默降级为 Mock 聚类。
    """
    pass


@dataclass
class ClusterInfo:
    """聚类信息"""
    cluster_id: int
    center: list[float]
    factor_ids: list[str] = field(default_factory=list)
    size: int = 0
    representative_factor: Optional[str] = None
    avg_similarity: float = 0.0


@dataclass
class ClusterResult:
    """聚类结果"""
    n_clusters: int
    clusters: list[ClusterInfo]
    factor_assignments: dict[str, int]  # factor_id -> cluster_id
    silhouette_score: float = 0.0


class FactorClusterer:
    """
    因子聚类器
    使用 K-Means 或层次聚类对因子进行分组
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        qdrant_client: Optional[QdrantClient] = None,
    ):
        """
        初始化因子聚类器

        Args:
            collection_name: 集合名称
            qdrant_client: Qdrant 客户端
        """
        self.collection_name = collection_name
        self.qdrant = qdrant_client or get_qdrant_client()

    def cluster_factors(
        self,
        n_clusters: int = 5,
        method: str = "kmeans",
        factor_ids: Optional[list[str]] = None,
    ) -> ClusterResult:
        """
        对因子进行聚类

        Args:
            n_clusters: 聚类数量
            method: 聚类方法 (kmeans, hierarchical)
            factor_ids: 要聚类的因子 ID 列表，为空则聚类所有因子

        Returns:
            聚类结果
        """
        # 获取因子向量
        vectors, ids, payloads = self._get_factor_vectors(factor_ids)

        if len(vectors) < n_clusters:
            logger.warning(f"Not enough factors ({len(vectors)}) for {n_clusters} clusters")
            n_clusters = max(1, len(vectors))

        if method == "kmeans":
            return self._kmeans_cluster(vectors, ids, payloads, n_clusters)
        elif method == "hierarchical":
            return self._hierarchical_cluster(vectors, ids, payloads, n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def _get_factor_vectors(
        self,
        factor_ids: Optional[list[str]] = None,
    ) -> tuple[list[list[float]], list[str], list[dict]]:
        """获取因子向量"""
        try:
            if factor_ids:
                # 获取指定因子
                points = self.qdrant.client.retrieve(
                    collection_name=self.collection_name,
                    ids=factor_ids,
                    with_vectors=True,
                    with_payload=True,
                )
            else:
                # 获取所有因子
                points, _ = self.qdrant.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,
                    with_vectors=True,
                    with_payload=True,
                )

            vectors = [p.vector for p in points]
            ids = [str(p.id) for p in points]
            payloads = [p.payload for p in points]

            return vectors, ids, payloads

        except Exception as e:
            logger.error(f"Failed to get factor vectors: {e}")
            return [], [], []

    def _kmeans_cluster(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict],
        n_clusters: int,
    ) -> ClusterResult:
        """使用 K-Means 聚类"""
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            X = np.array(vectors)

            # K-Means 聚类
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)

            # 计算轮廓系数
            sil_score = 0.0
            if len(set(labels)) > 1:
                sil_score = float(silhouette_score(X, labels))

            # 构建聚类结果
            clusters = []
            factor_assignments = {}

            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                cluster_ids = [ids[i] for i in range(len(ids)) if mask[i]]
                cluster_vectors = X[mask]

                if len(cluster_ids) == 0:
                    continue

                # 计算聚类中心
                center = kmeans.cluster_centers_[cluster_id].tolist()

                # 找到最接近中心的因子作为代表
                distances = np.linalg.norm(cluster_vectors - center, axis=1)
                representative_idx = np.argmin(distances)
                representative_id = cluster_ids[representative_idx]

                # 计算平均相似度
                avg_sim = 1.0 - np.mean(distances) / np.max(distances) if len(distances) > 0 else 1.0

                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    center=center,
                    factor_ids=cluster_ids,
                    size=len(cluster_ids),
                    representative_factor=representative_id,
                    avg_similarity=float(avg_sim),
                )
                clusters.append(cluster_info)

                for fid in cluster_ids:
                    factor_assignments[fid] = cluster_id

            return ClusterResult(
                n_clusters=len(clusters),
                clusters=clusters,
                factor_assignments=factor_assignments,
                silhouette_score=sil_score,
            )

        except ImportError as e:
            # 严格模式：sklearn 未安装时抛出异常，禁止 Mock 降级
            raise ClusteringDependencyError(
                "sklearn is required for K-Means clustering. "
                "Install with: pip install scikit-learn"
            ) from e

        except Exception as e:
            logger.error(f"K-Means clustering failed: {e}")
            raise

    def _hierarchical_cluster(
        self,
        vectors: list[list[float]],
        ids: list[str],
        payloads: list[dict],
        n_clusters: int,
    ) -> ClusterResult:
        """使用层次聚类"""
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score

            X = np.array(vectors)

            # 层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(X)

            # 计算轮廓系数
            sil_score = 0.0
            if len(set(labels)) > 1:
                sil_score = float(silhouette_score(X, labels, metric="cosine"))

            # 构建聚类结果
            clusters = []
            factor_assignments = {}

            for cluster_id in range(n_clusters):
                mask = labels == cluster_id
                cluster_ids = [ids[i] for i in range(len(ids)) if mask[i]]
                cluster_vectors = X[mask]

                if len(cluster_ids) == 0:
                    continue

                # 计算聚类中心
                center = np.mean(cluster_vectors, axis=0).tolist()

                # 找到最接近中心的因子
                distances = np.linalg.norm(cluster_vectors - center, axis=1)
                representative_idx = np.argmin(distances)
                representative_id = cluster_ids[representative_idx]

                cluster_info = ClusterInfo(
                    cluster_id=cluster_id,
                    center=center,
                    factor_ids=cluster_ids,
                    size=len(cluster_ids),
                    representative_factor=representative_id,
                    avg_similarity=float(1.0 - np.mean(distances)),
                )
                clusters.append(cluster_info)

                for fid in cluster_ids:
                    factor_assignments[fid] = cluster_id

            return ClusterResult(
                n_clusters=len(clusters),
                clusters=clusters,
                factor_assignments=factor_assignments,
                silhouette_score=sil_score,
            )

        except ImportError as e:
            # 严格模式：sklearn 未安装时抛出异常，禁止 Mock 降级
            raise ClusteringDependencyError(
                "sklearn is required for Hierarchical clustering. "
                "Install with: pip install scikit-learn"
            ) from e

        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            raise

    def _mock_cluster(self, ids: list[str], n_clusters: int) -> ClusterResult:
        """模拟聚类结果 - 仅用于测试环境，生产环境禁用。"""
        clusters = []
        factor_assignments = {}

        # 简单地将因子平均分配到各个聚类
        factors_per_cluster = max(1, len(ids) // n_clusters)

        for i in range(n_clusters):
            start = i * factors_per_cluster
            end = start + factors_per_cluster if i < n_clusters - 1 else len(ids)
            cluster_ids = ids[start:end]

            if not cluster_ids:
                continue

            cluster_info = ClusterInfo(
                cluster_id=i,
                center=[0.0] * 1536,  # 模拟中心
                factor_ids=cluster_ids,
                size=len(cluster_ids),
                representative_factor=cluster_ids[0] if cluster_ids else None,
                avg_similarity=0.8,
            )
            clusters.append(cluster_info)

            for fid in cluster_ids:
                factor_assignments[fid] = i

        return ClusterResult(
            n_clusters=len(clusters),
            clusters=clusters,
            factor_assignments=factor_assignments,
            silhouette_score=0.5,
        )

    def find_optimal_clusters(
        self,
        max_clusters: int = 10,
        factor_ids: Optional[list[str]] = None,
    ) -> int:
        """
        使用肘部法则找到最优聚类数

        Args:
            max_clusters: 最大聚类数
            factor_ids: 因子 ID 列表

        Returns:
            最优聚类数
        """
        try:
            import numpy as np
            from sklearn.cluster import KMeans

            vectors, ids, _ = self._get_factor_vectors(factor_ids)
            if len(vectors) < 3:
                return 1

            X = np.array(vectors)
            max_k = min(max_clusters, len(vectors) - 1)

            inertias = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertias.append(kmeans.inertia_)

            # 找到肘部点
            if len(inertias) < 3:
                return 1

            # 计算二阶差分
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)

            # 肘部点是二阶差分最大的点
            optimal_k = np.argmax(diffs2) + 2  # +2 因为两次差分

            return max(1, min(optimal_k, max_k))

        except Exception as e:
            logger.error(f"Failed to find optimal clusters: {e}")
            return 5  # 默认值

    def get_cluster_summary(self, result: ClusterResult) -> dict[str, Any]:
        """
        获取聚类摘要

        Args:
            result: 聚类结果

        Returns:
            摘要信息
        """
        return {
            "n_clusters": result.n_clusters,
            "silhouette_score": result.silhouette_score,
            "cluster_sizes": [c.size for c in result.clusters],
            "avg_cluster_size": sum(c.size for c in result.clusters) / len(result.clusters) if result.clusters else 0,
            "clusters": [
                {
                    "id": c.cluster_id,
                    "size": c.size,
                    "representative": c.representative_factor,
                    "avg_similarity": c.avg_similarity,
                }
                for c in result.clusters
            ],
        }
