"""Tests for Sandbox Executor.

Six-dimensional test coverage:
1. Functional: Core sandbox execution functionality
2. Boundary: Edge cases and limits
3. Exception: Error handling
4. Performance: Execution time limits
5. Security: Isolation and restriction
6. Compatibility: Different code patterns
"""

import pytest
import time

from iqfmp.core.sandbox import (
    SandboxExecutor,
    ExecutionResult,
    ExecutionStatus,
    SandboxConfig,
)


class TestExecutionResultModel:
    """Test ExecutionResult data model."""

    def test_result_creation_success(self) -> None:
        """Test creating a successful execution result."""
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output={"factor_value": 0.5},
            execution_time=0.1,
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output == {"factor_value": 0.5}
        assert result.execution_time == 0.1

    def test_result_creation_error(self) -> None:
        """Test creating an error execution result."""
        result = ExecutionResult(
            status=ExecutionStatus.ERROR,
            error_message="Division by zero",
        )
        assert result.status == ExecutionStatus.ERROR
        assert result.error_message == "Division by zero"

    def test_execution_status_values(self) -> None:
        """Test all execution status values exist."""
        assert ExecutionStatus.SUCCESS is not None
        assert ExecutionStatus.ERROR is not None
        assert ExecutionStatus.TIMEOUT is not None
        assert ExecutionStatus.SECURITY_VIOLATION is not None


class TestSandboxConfigModel:
    """Test SandboxConfig model."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = SandboxConfig()
        assert config.timeout_seconds > 0
        assert config.max_memory_mb > 0
        assert len(config.allowed_modules) > 0

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SandboxConfig(
            timeout_seconds=30,
            max_memory_mb=256,
            allowed_modules=["pandas", "numpy"],
        )
        assert config.timeout_seconds == 30
        assert config.max_memory_mb == 256
        assert "pandas" in config.allowed_modules


class TestSandboxExecutorFunctional:
    """Functional tests for SandboxExecutor."""

    @pytest.fixture
    def executor(self) -> SandboxExecutor:
        """Create a sandbox executor instance."""
        return SandboxExecutor()

    # === Basic Execution ===

    def test_execute_simple_code(self, executor: SandboxExecutor) -> None:
        """Test executing simple arithmetic code."""
        code = "result = 1 + 2 + 3"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 6

    def test_execute_with_variables(self, executor: SandboxExecutor) -> None:
        """Test executing code with variables."""
        code = """
x = 10
y = 20
result = x * y
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 200

    def test_execute_function_definition(self, executor: SandboxExecutor) -> None:
        """Test executing code with function definition."""
        code = """
def add(a, b):
    return a + b

result = add(5, 3)
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 8

    # === Allowed Modules ===

    def test_execute_with_math(self, executor: SandboxExecutor) -> None:
        """Test executing code with math module."""
        code = """
import math
result = math.sqrt(16)
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 4.0

    def test_execute_with_numpy(self, executor: SandboxExecutor) -> None:
        """Test executing code with numpy."""
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
result = np.mean(arr)
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 3.0

    def test_execute_with_pandas(self, executor: SandboxExecutor) -> None:
        """Test executing code with pandas."""
        code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
result = int(df['a'].sum())  # Convert to native int
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 6

    # === Factor-like Code ===

    def test_execute_factor_calculation(self, executor: SandboxExecutor) -> None:
        """Test executing realistic factor calculation code."""
        code = """
import pandas as pd
import numpy as np

# Simulate price data
prices = pd.Series([100, 102, 101, 103, 105, 104, 106])

# Calculate momentum factor
returns = prices.pct_change()
momentum = returns.rolling(3).mean()
result = momentum.iloc[-1]
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") is not None


class TestSandboxExecutorSecurity:
    """Security tests - verify dangerous operations are blocked."""

    @pytest.fixture
    def executor(self) -> SandboxExecutor:
        return SandboxExecutor()

    # === Dangerous Module Imports ===

    def test_block_os_import(self, executor: SandboxExecutor) -> None:
        """Test blocking os module import."""
        code = "import os"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_subprocess_import(self, executor: SandboxExecutor) -> None:
        """Test blocking subprocess module import."""
        code = "import subprocess"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_socket_import(self, executor: SandboxExecutor) -> None:
        """Test blocking socket module import."""
        code = "import socket"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    # === Dangerous Function Calls ===

    def test_block_eval(self, executor: SandboxExecutor) -> None:
        """Test blocking eval function."""
        code = "result = eval('1+1')"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_exec(self, executor: SandboxExecutor) -> None:
        """Test blocking exec function."""
        code = "exec('x=1')"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_open(self, executor: SandboxExecutor) -> None:
        """Test blocking open function."""
        code = "f = open('/etc/passwd', 'r')"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_dunder_import(self, executor: SandboxExecutor) -> None:
        """Test blocking __import__ function."""
        code = "__import__('os')"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    # === Dangerous Attribute Access ===

    def test_block_globals_access(self, executor: SandboxExecutor) -> None:
        """Test blocking __globals__ access."""
        code = """
def foo():
    pass
x = foo.__globals__
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_builtins_access(self, executor: SandboxExecutor) -> None:
        """Test blocking __builtins__ access via attribute."""
        code = "x = type.__builtins__"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION


class TestSandboxExecutorBoundary:
    """Boundary tests for edge cases."""

    @pytest.fixture
    def executor(self) -> SandboxExecutor:
        return SandboxExecutor()

    def test_empty_code(self, executor: SandboxExecutor) -> None:
        """Test executing empty code."""
        result = executor.execute("")
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output == {}

    def test_whitespace_only(self, executor: SandboxExecutor) -> None:
        """Test executing whitespace-only code."""
        result = executor.execute("   \n\t\n   ")
        assert result.status == ExecutionStatus.SUCCESS

    def test_comments_only(self, executor: SandboxExecutor) -> None:
        """Test executing comments-only code."""
        result = executor.execute("# This is a comment")
        assert result.status == ExecutionStatus.SUCCESS

    def test_large_output(self, executor: SandboxExecutor) -> None:
        """Test handling large output."""
        code = "result = list(range(1000))"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert len(result.output.get("result", [])) == 1000


class TestSandboxExecutorException:
    """Exception handling tests."""

    @pytest.fixture
    def executor(self) -> SandboxExecutor:
        return SandboxExecutor()

    def test_syntax_error(self, executor: SandboxExecutor) -> None:
        """Test handling syntax errors."""
        code = "def broken("
        result = executor.execute(code)
        # Syntax errors are caught during AST security check
        assert result.status == ExecutionStatus.SECURITY_VIOLATION
        assert "syntax" in result.error_message.lower()

    def test_runtime_error(self, executor: SandboxExecutor) -> None:
        """Test handling runtime errors."""
        code = "result = 1 / 0"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.ERROR
        assert "division" in result.error_message.lower() or "zero" in result.error_message.lower()

    def test_name_error(self, executor: SandboxExecutor) -> None:
        """Test handling name errors."""
        code = "result = undefined_variable"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.ERROR

    def test_type_error(self, executor: SandboxExecutor) -> None:
        """Test handling type errors."""
        code = "result = 'string' + 123"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.ERROR


class TestSandboxExecutorPerformance:
    """Performance tests."""

    @pytest.fixture
    def executor(self) -> SandboxExecutor:
        return SandboxExecutor(config=SandboxConfig(timeout_seconds=2))

    def test_timeout_enforcement(self, executor: SandboxExecutor) -> None:
        """Test that timeout is enforced via infinite loop."""
        code = """
# Infinite loop to trigger timeout
i = 0
while True:
    i += 1
result = i
"""
        start = time.time()
        result = executor.execute(code)
        elapsed = time.time() - start

        assert result.status == ExecutionStatus.TIMEOUT
        assert elapsed < 5  # Should timeout well before 10 seconds

    def test_fast_execution(self) -> None:
        """Test that fast code executes quickly.

        Note: Uses in-process mode since subprocess has ~4s startup overhead.
        """
        # Use in-process mode for performance testing
        executor = SandboxExecutor(config=SandboxConfig(
            timeout_seconds=2,
            use_subprocess=False,  # In-process mode for accurate timing
        ))
        code = "result = sum(range(1000))"
        start = time.time()
        result = executor.execute(code)
        elapsed = time.time() - start

        assert result.status == ExecutionStatus.SUCCESS
        assert elapsed < 1.0

    def test_execution_time_recorded(self, executor: SandboxExecutor) -> None:
        """Test that execution time is recorded."""
        code = "result = 1 + 1"
        result = executor.execute(code)
        assert result.execution_time is not None
        assert result.execution_time >= 0


class TestSandboxExecutorCompatibility:
    """Compatibility tests for different code patterns."""

    @pytest.fixture
    def executor(self) -> SandboxExecutor:
        return SandboxExecutor()

    def test_list_comprehension(self, executor: SandboxExecutor) -> None:
        """Test list comprehension."""
        code = "result = [x**2 for x in range(5)]"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == [0, 1, 4, 9, 16]

    def test_dict_comprehension(self, executor: SandboxExecutor) -> None:
        """Test dict comprehension."""
        code = "result = {x: x**2 for x in range(3)}"
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        # Dict keys are serialized to strings
        assert result.output.get("result") == {"0": 0, "1": 1, "2": 4}

    def test_lambda_function(self, executor: SandboxExecutor) -> None:
        """Test lambda function."""
        code = """
square = lambda x: x**2
result = square(5)
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 25

    def test_class_definition(self, executor: SandboxExecutor) -> None:
        """Test class definition."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

calc = Calculator()
result = calc.add(3, 4)
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 7

    def test_decorator(self, executor: SandboxExecutor) -> None:
        """Test decorator usage."""
        code = """
def double(func):
    def wrapper(*args):
        return func(*args) * 2
    return wrapper

@double
def add(a, b):
    return a + b

result = add(2, 3)
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output.get("result") == 10


class TestSandboxIntegration:
    """Integration tests with AST security checker."""

    @pytest.fixture
    def executor(self) -> SandboxExecutor:
        return SandboxExecutor()

    def test_ast_check_before_execution(self, executor: SandboxExecutor) -> None:
        """Test that AST check runs before execution."""
        # Code that would pass AST but fail sandbox
        code = "import os"
        result = executor.execute(code)
        # Should fail at AST check, not reach sandbox execution
        assert result.status == ExecutionStatus.SECURITY_VIOLATION

    def test_safe_code_passes_both_checks(self, executor: SandboxExecutor) -> None:
        """Test that safe code passes both AST and sandbox checks."""
        code = """
import pandas as pd
import numpy as np

data = pd.Series([1, 2, 3, 4, 5])
result = data.mean()
"""
        result = executor.execute(code)
        assert result.status == ExecutionStatus.SUCCESS
