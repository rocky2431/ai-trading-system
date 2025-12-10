"""
Embedding 生成器
将因子代码转换为向量表示
"""

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    model_name: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class BaseEmbeddingGenerator(ABC):
    """Embedding 生成器基类"""

    @abstractmethod
    def generate(self, text: str) -> list[float]:
        """生成单个文本的 embedding"""
        pass

    @abstractmethod
    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回 embedding 维度"""
        pass


class OpenAIEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    OpenAI Embedding 生成器
    使用 OpenAI API 生成文本 embedding
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._client = None

    @property
    def client(self):
        """获取 OpenAI 客户端（懒加载）"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        """创建 OpenAI 客户端"""
        try:
            from openai import OpenAI

            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not configured")

            client = OpenAI(
                api_key=api_key,
                base_url=self.config.api_base or os.getenv("OPENAI_API_BASE"),
            )

            logger.info(f"OpenAI embedding client initialized with model: {self.config.model_name}")
            return client

        except ImportError:
            logger.warning("openai package not installed, using mock embeddings")
            return None

    def generate(self, text: str) -> list[float]:
        """生成单个文本的 embedding"""
        if self.client is None:
            return self._mock_embedding(text)

        try:
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=text,
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._mock_embedding(text)

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding"""
        if self.client is None:
            return [self._mock_embedding(t) for t in texts]

        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.config.model_name,
                    input=batch,
                )
                results.extend([d.embedding for d in response.data])

            except Exception as e:
                logger.error(f"Failed to generate batch embedding: {e}")
                results.extend([self._mock_embedding(t) for t in batch])

        return results

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def _mock_embedding(self, text: str) -> list[float]:
        """生成模拟 embedding（用于测试）"""
        import random
        # 使用文本哈希作为随机种子，确保相同文本生成相同 embedding
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        return [random.uniform(-1, 1) for _ in range(self.config.dimension)]


class LocalEmbeddingGenerator(BaseEmbeddingGenerator):
    """
    本地 Embedding 生成器
    使用 sentence-transformers 模型生成 embedding
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = None

    @property
    def model(self):
        """获取模型（懒加载）"""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded local embedding model: {self.model_name}")

        except ImportError:
            logger.warning("sentence-transformers not installed, using mock embeddings")
            self._model = None
            self._dimension = 384  # Default dimension for MiniLM

    def generate(self, text: str) -> list[float]:
        """生成单个文本的 embedding"""
        if self.model is None:
            return self._mock_embedding(text)

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate local embedding: {e}")
            return self._mock_embedding(text)

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding"""
        if self.model is None:
            return [self._mock_embedding(t) for t in texts]

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate batch local embedding: {e}")
            return [self._mock_embedding(t) for t in texts]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def _mock_embedding(self, text: str) -> list[float]:
        """生成模拟 embedding"""
        import random
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        return [random.uniform(-1, 1) for _ in range(self.dimension)]


class EmbeddingGenerator:
    """
    Embedding 生成器工厂类
    根据配置选择合适的 embedding 生成器
    """

    def __init__(
        self,
        provider: str = "openai",
        config: Optional[EmbeddingConfig] = None,
    ):
        """
        初始化 Embedding 生成器

        Args:
            provider: embedding 提供者 (openai, local)
            config: embedding 配置
        """
        self.provider = provider
        self.config = config

        if provider == "openai":
            self._generator = OpenAIEmbeddingGenerator(config)
        elif provider == "local":
            model_name = config.model_name if config else "all-MiniLM-L6-v2"
            self._generator = LocalEmbeddingGenerator(model_name)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def generate(self, text: str) -> list[float]:
        """生成 embedding"""
        return self._generator.generate(text)

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding"""
        return self._generator.generate_batch(texts)

    @property
    def dimension(self) -> int:
        """获取 embedding 维度"""
        return self._generator.dimension

    def generate_factor_embedding(
        self,
        factor_code: str,
        factor_name: str,
        hypothesis: str,
    ) -> list[float]:
        """
        生成因子 embedding

        将因子代码、名称和假设组合后生成 embedding

        Args:
            factor_code: 因子代码
            factor_name: 因子名称
            hypothesis: 因子假设

        Returns:
            因子的 embedding 向量
        """
        # 组合因子信息
        combined_text = f"""
Factor Name: {factor_name}
Hypothesis: {hypothesis}
Code:
{factor_code}
"""
        return self.generate(combined_text.strip())


# 单例实例
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator(
    provider: str = "openai",
    config: Optional[EmbeddingConfig] = None,
) -> EmbeddingGenerator:
    """获取 Embedding 生成器单例"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator(provider, config)
    return _embedding_generator
