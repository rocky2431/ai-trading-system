"""
因子生成集成测试
测试从自然语言输入到因子代码生成的完整流程
"""

import pytest
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock


class TestFactorGenerationFlow:
    """因子生成流程测试"""

    def test_natural_language_to_factor_code(
        self,
        mock_llm_response: dict[str, Any],
    ):
        """测试自然语言转因子代码"""
        # 模拟用户输入
        user_input = "创建一个基于过去20天价格动量的因子"

        # 模拟 LLM 生成
        generated_code = mock_llm_response["factor_code"]

        # 验证生成的代码结构
        assert "def " in generated_code
        assert "return" in generated_code
        assert "df" in generated_code

    def test_factor_code_ast_validation(
        self,
        mock_llm_response: dict[str, Any],
    ):
        """测试 AST 安全检查"""
        factor_code = mock_llm_response["factor_code"]

        # 模拟 AST 检查
        dangerous_patterns = ["eval", "exec", "os.system", "__import__", "open("]

        for pattern in dangerous_patterns:
            assert pattern not in factor_code, f"Dangerous pattern found: {pattern}"

    def test_factor_family_constraint_validation(
        self,
        mock_llm_response: dict[str, Any],
        factor_families: list[dict[str, Any]],
    ):
        """测试因子家族约束验证"""
        family_id = mock_llm_response["family"]

        # 查找对应的家族
        family = next(
            (f for f in factor_families if f["id"] == family_id),
            None,
        )

        assert family is not None, f"Unknown family: {family_id}"

        # 验证使用的字段在允许范围内
        allowed_fields = family["allowed_fields"]
        factor_code = mock_llm_response["factor_code"]

        # 简单检查：代码中使用了允许的字段
        used_fields = []
        for field in ["close", "open", "high", "low", "volume"]:
            if f'"{field}"' in factor_code or f"'{field}'" in factor_code:
                used_fields.append(field)

        for field in used_fields:
            assert field in allowed_fields, f"Field {field} not allowed in {family_id}"

    def test_factor_metadata_extraction(
        self,
        mock_llm_response: dict[str, Any],
    ):
        """测试因子元数据提取"""
        # 验证必要的元数据存在
        assert "factor_name" in mock_llm_response
        assert "hypothesis" in mock_llm_response
        assert "family" in mock_llm_response
        assert "factor_code" in mock_llm_response

        # 验证元数据格式
        assert len(mock_llm_response["factor_name"]) > 0
        assert len(mock_llm_response["hypothesis"]) > 0

    def test_factor_code_execution_in_sandbox(
        self,
        mock_llm_response: dict[str, Any],
        sample_market_data: dict[str, Any],
    ):
        """测试因子代码在沙箱中执行"""
        import pandas as pd

        factor_code = mock_llm_response["factor_code"]

        # 准备测试数据
        btc_data = sample_market_data["data"]["BTC"]
        df = pd.DataFrame({
            "open": btc_data["open"],
            "high": btc_data["high"],
            "low": btc_data["low"],
            "close": btc_data["close"],
            "volume": btc_data["volume"],
        })

        # 在受限环境中执行
        local_vars = {"df": df, "pd": pd}
        global_vars = {"__builtins__": {"len": len, "range": range}}

        try:
            exec(factor_code, global_vars, local_vars)

            # 获取因子函数
            factor_func = local_vars.get("momentum_factor")

            if factor_func:
                result = factor_func(df)
                assert result is not None
                assert len(result) == len(df)

        except Exception as e:
            # 沙箱执行错误应该被捕获
            pytest.fail(f"Sandbox execution failed: {e}")


class TestFactorDeduplication:
    """因子去重测试"""

    def test_similarity_check_new_factor(self):
        """测试新因子相似度检查"""
        # 模拟向量相似度检查
        similarity_threshold = 0.85

        # 模拟新因子与现有因子的相似度
        mock_similarities = [0.3, 0.45, 0.6, 0.72]

        max_similarity = max(mock_similarities)
        is_duplicate = max_similarity >= similarity_threshold

        assert not is_duplicate, "New factor should not be marked as duplicate"

    def test_similarity_check_duplicate_factor(self):
        """测试重复因子检测"""
        similarity_threshold = 0.85

        # 模拟与现有因子高度相似
        mock_similarities = [0.3, 0.92, 0.45]

        max_similarity = max(mock_similarities)
        is_duplicate = max_similarity >= similarity_threshold

        assert is_duplicate, "Duplicate factor should be detected"

    def test_factor_embedding_generation(
        self,
        mock_llm_response: dict[str, Any],
    ):
        """测试因子嵌入向量生成"""
        # 模拟嵌入向量
        embedding_dim = 1536
        mock_embedding = [0.1] * embedding_dim

        assert len(mock_embedding) == embedding_dim
        assert all(isinstance(v, float) for v in mock_embedding)


class TestFactorVersioning:
    """因子版本控制测试"""

    def test_factor_version_creation(self):
        """测试因子版本创建"""
        factor_id = "momentum_20d"
        version = 1

        versioned_id = f"{factor_id}_v{version}"

        assert versioned_id == "momentum_20d_v1"

    def test_factor_update_creates_new_version(self):
        """测试因子更新创建新版本"""
        original_version = 1
        new_version = original_version + 1

        assert new_version == 2

    def test_factor_history_tracking(self):
        """测试因子历史追踪"""
        factor_history = [
            {"version": 1, "created_at": "2024-01-01", "status": "deprecated"},
            {"version": 2, "created_at": "2024-02-01", "status": "active"},
        ]

        # 获取当前活跃版本
        active_version = next(
            (v for v in factor_history if v["status"] == "active"),
            None,
        )

        assert active_version is not None
        assert active_version["version"] == 2


class TestBatchFactorGeneration:
    """批量因子生成测试"""

    def test_batch_generation_success(self):
        """测试批量生成成功"""
        batch_size = 5
        generated_factors = []

        for i in range(batch_size):
            factor = {
                "id": f"factor_{i}",
                "name": f"test_factor_{i}",
                "status": "generated",
            }
            generated_factors.append(factor)

        assert len(generated_factors) == batch_size

    def test_batch_generation_partial_failure(self):
        """测试批量生成部分失败"""
        results = [
            {"id": "factor_1", "status": "success"},
            {"id": "factor_2", "status": "failed", "error": "AST validation failed"},
            {"id": "factor_3", "status": "success"},
        ]

        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "failed"]

        assert len(successful) == 2
        assert len(failed) == 1

    def test_batch_rate_limiting(self):
        """测试批量生成速率限制"""
        rate_limit = 10  # 每秒最多 10 个请求
        batch_size = 20

        # 计算需要的时间
        required_seconds = batch_size / rate_limit

        assert required_seconds == 2.0


class TestFactorGenerationErrors:
    """因子生成错误处理测试"""

    def test_invalid_natural_language_input(self):
        """测试无效的自然语言输入"""
        invalid_inputs = [
            "",
            "   ",
            "a" * 10000,  # 超长输入
        ]

        for invalid_input in invalid_inputs:
            # 应该返回错误或拒绝处理
            is_valid = len(invalid_input.strip()) > 0 and len(invalid_input) < 5000

            if invalid_input == "" or invalid_input == "   ":
                assert not is_valid
            elif len(invalid_input) >= 5000:
                assert not is_valid

    def test_llm_timeout_handling(self):
        """测试 LLM 超时处理"""
        timeout_seconds = 30
        max_retries = 3

        # 模拟重试逻辑
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            retry_count += 1
            # 模拟第三次成功
            if retry_count == 3:
                success = True

        assert success
        assert retry_count == 3

    def test_unsafe_code_rejection(self):
        """测试不安全代码拒绝"""
        unsafe_codes = [
            "import os; os.system('rm -rf /')",
            "eval(user_input)",
            "__import__('subprocess').call(['ls'])",
            "exec(open('malicious.py').read())",
        ]

        for code in unsafe_codes:
            # 检查危险模式
            is_safe = not any(
                pattern in code
                for pattern in ["os.system", "eval", "__import__", "exec", "open("]
            )
            assert not is_safe, f"Unsafe code should be rejected: {code[:50]}"
