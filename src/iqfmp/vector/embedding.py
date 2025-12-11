"""
Embedding 生成器
将因子代码转换为向量表示
统一使用 OpenRouter 作为 Embedding 供应商
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding 配置"""
    model_name: str = "openai/text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100
    api_key: Optional[str] = None


class EmbeddingGeneratorError(Exception):
    """Embedding 生成器异常"""
    pass


class EmbeddingGenerator:
    """
    Embedding 生成器
    统一使用 OpenRouter API 生成文本 embedding (OpenAI 兼容)
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._client = None
        self._initialized = False

    @property
    def client(self):
        """获取 OpenAI 客户端（懒加载）"""
        if not self._initialized:
            self._client = self._create_client()
            self._initialized = True
        return self._client

    def _create_client(self):
        """创建 OpenRouter 客户端 (OpenAI 兼容)"""
        try:
            from openai import OpenAI
        except ImportError:
            raise EmbeddingGeneratorError(
                "openai package not installed. Please run: pip install openai"
            )

        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EmbeddingGeneratorError(
                "OpenRouter API key not configured. "
                "Please set OPENROUTER_API_KEY environment variable or configure in Settings."
            )

        client = OpenAI(
            api_key=api_key,
            base_url=self.OPENROUTER_BASE_URL,
        )

        logger.info(f"OpenRouter embedding client initialized with model: {self.config.model_name}")
        return client

    def generate(self, text: str) -> list[float]:
        """生成单个文本的 embedding"""
        try:
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=text,
            )
            return response.data[0].embedding

        except EmbeddingGeneratorError:
            raise
        except Exception as e:
            raise EmbeddingGeneratorError(f"Failed to generate embedding: {e}")

    def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding"""
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.config.model_name,
                    input=batch,
                )
                results.extend([d.embedding for d in response.data])

            except EmbeddingGeneratorError:
                raise
            except Exception as e:
                raise EmbeddingGeneratorError(f"Failed to generate batch embedding: {e}")

        return results

    @property
    def dimension(self) -> int:
        """获取 embedding 维度"""
        return self.config.dimension

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
    config: Optional[EmbeddingConfig] = None,
) -> EmbeddingGenerator:
    """获取 Embedding 生成器单例"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator(config)
    return _embedding_generator


def reset_embedding_generator():
    """重置 Embedding 生成器单例（用于配置变更时）"""
    global _embedding_generator
    _embedding_generator = None
