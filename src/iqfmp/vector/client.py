"""
Qdrant 客户端配置
提供与 Qdrant 向量数据库的连接管理
"""

import os
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QdrantConfig:
    """Qdrant 配置"""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: Optional[str] = None
    https: bool = False
    timeout: float = 30.0
    prefer_grpc: bool = True


class QdrantClient:
    """
    Qdrant 客户端封装
    提供连接管理和基础操作
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        初始化 Qdrant 客户端

        Args:
            config: Qdrant 配置，如果为空则从环境变量读取
        """
        self.config = config or self._load_config_from_env()
        self._client = None

    def _load_config_from_env(self) -> QdrantConfig:
        """从环境变量加载配置"""
        return QdrantConfig(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            grpc_port=int(os.getenv("QDRANT_GRPC_PORT", "6334")),
            api_key=os.getenv("QDRANT_API_KEY"),
            https=os.getenv("QDRANT_HTTPS", "false").lower() == "true",
            timeout=float(os.getenv("QDRANT_TIMEOUT", "30.0")),
            prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true",
        )

    @property
    def client(self):
        """获取 Qdrant 客户端实例（懒加载）"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """创建 Qdrant 客户端"""
        try:
            from qdrant_client import QdrantClient as QdrantClientLib
            from qdrant_client.http import models

            client = QdrantClientLib(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port,
                api_key=self.config.api_key,
                https=self.config.https,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc,
            )

            logger.info(f"Connected to Qdrant at {self.config.host}:{self.config.port}")
            return client

        except ImportError:
            logger.warning("qdrant-client not installed, using mock client")
            return MockQdrantClient()

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance: str = "Cosine",
        on_disk_payload: bool = True,
    ) -> bool:
        """
        创建集合

        Args:
            collection_name: 集合名称
            vector_size: 向量维度
            distance: 距离度量 (Cosine, Euclid, Dot)
            on_disk_payload: 是否将 payload 存储在磁盘

        Returns:
            是否成功创建
        """
        try:
            from qdrant_client.http import models

            distance_map = {
                "Cosine": models.Distance.COSINE,
                "Euclid": models.Distance.EUCLID,
                "Dot": models.Distance.DOT,
            }

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, models.Distance.COSINE),
                    on_disk=True,
                ),
                on_disk_payload=on_disk_payload,
            )

            logger.info(f"Created collection: {collection_name}")
            return True

        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Collection {collection_name} already exists")
                return True
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> dict:
        """获取集合信息"""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            # 安全获取 status 值 (不同版本 qdrant-client 可能返回字符串或枚举)
            if info.status:
                status_str = info.status.value if hasattr(info.status, 'value') else str(info.status)
            else:
                status_str = "unknown"
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": status_str,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def close(self):
        """关闭连接"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None


class MockQdrantClient:
    """
    Mock Qdrant 客户端
    用于开发测试时无需真实 Qdrant 服务
    """

    def __init__(self):
        self._collections: dict[str, list] = {}
        logger.info("Using MockQdrantClient for development")

    def create_collection(self, **kwargs):
        collection_name = kwargs.get("collection_name")
        self._collections[collection_name] = []

    def get_collections(self):
        class Collections:
            def __init__(self, names):
                self.collections = [type("C", (), {"name": n})() for n in names]
        return Collections(list(self._collections.keys()))

    def delete_collection(self, collection_name: str):
        if collection_name in self._collections:
            del self._collections[collection_name]

    def get_collection(self, collection_name: str):
        points = self._collections.get(collection_name, [])
        class Info:
            vectors_count = len(points)
            points_count = len(points)
            status = type("S", (), {"value": "green"})()
        return Info()

    def upsert(self, collection_name: str, points: list):
        if collection_name not in self._collections:
            self._collections[collection_name] = []
        self._collections[collection_name].extend(points)

    def search(self, collection_name: str, query_vector: list, limit: int = 10, **kwargs):
        # 返回模拟结果 (deprecated, 保留向后兼容)
        return []

    def query_points(self, collection_name: str, query: list, limit: int = 10, **kwargs):
        """qdrant-client v1.7+ 新 API"""
        # 返回模拟的 QueryResponse 对象
        class QueryResponse:
            points = []
        return QueryResponse()

    def close(self):
        pass


# 单例实例
_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """获取 Qdrant 客户端单例"""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient()
    return _qdrant_client
