"""
向量存储模块
提供因子代码 Embedding 生成、向量存储和相似度检索功能

严格模式（默认）：
- 禁止静默降级为 Mock
- Qdrant 服务不可用时抛出 QdrantUnavailableError
- 确保数据持久性

宽松模式（allow_mock=True）：
- 仅用于单元测试
- 数据不会持久化
"""

from .client import (
    QdrantClient,
    QdrantConfig,
    QdrantUnavailableError,
    get_qdrant_client,
)
from .embedding import EmbeddingGenerator, get_embedding_generator
from .store import FactorVectorStore
from .search import SimilaritySearcher
from .cluster import FactorClusterer

__all__ = [
    "QdrantClient",
    "QdrantConfig",
    "QdrantUnavailableError",
    "get_qdrant_client",
    "EmbeddingGenerator",
    "get_embedding_generator",
    "FactorVectorStore",
    "SimilaritySearcher",
    "FactorClusterer",
]
