"""Tests for AST Security Checker.

Six-dimensional test coverage:
1. Functional: Core security check functionality
2. Boundary: Edge cases and limits
3. Exception: Error handling
4. Performance: Check execution time
5. Security: Bypass attempt prevention
6. Compatibility: Different code patterns
"""

import pytest
import time

from iqfmp.core.security import (
    ASTSecurityChecker,
    SecurityViolation,
    ViolationType,
    SecurityCheckResult,
)


class TestSecurityViolationModel:
    """Test SecurityViolation data model."""

    def test_violation_creation(self) -> None:
        """Test creating a security violation."""
        violation = SecurityViolation(
            violation_type=ViolationType.DANGEROUS_FUNCTION,
            message="Dangerous function 'eval' detected",
            line_number=5,
            column=10,
            code_snippet="eval(user_input)",
        )
        assert violation.violation_type == ViolationType.DANGEROUS_FUNCTION
        assert violation.line_number == 5
        assert "eval" in violation.message

    def test_violation_types(self) -> None:
        """Test all violation types exist."""
        assert ViolationType.DANGEROUS_FUNCTION is not None
        assert ViolationType.DANGEROUS_IMPORT is not None
        assert ViolationType.DANGEROUS_ATTRIBUTE is not None
        assert ViolationType.SYNTAX_ERROR is not None


class TestASTSecurityCheckerFunctional:
    """Functional tests for ASTSecurityChecker."""

    @pytest.fixture
    def checker(self) -> ASTSecurityChecker:
        """Create a security checker instance."""
        return ASTSecurityChecker()

    # === Dangerous Function Detection ===

    def test_detect_eval(self, checker: ASTSecurityChecker) -> None:
        """Test detection of eval() function."""
        code = "result = eval('1 + 1')"
        result = checker.check(code)
        assert not result.is_safe
        assert any(v.violation_type == ViolationType.DANGEROUS_FUNCTION for v in result.violations)

    def test_detect_exec(self, checker: ASTSecurityChecker) -> None:
        """Test detection of exec() function."""
        code = "exec('print(1)')"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_compile(self, checker: ASTSecurityChecker) -> None:
        """Test detection of compile() function."""
        code = "code = compile('x=1', '<string>', 'exec')"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_os_system(self, checker: ASTSecurityChecker) -> None:
        """Test detection of os.system()."""
        code = "import os\nos.system('ls')"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_subprocess(self, checker: ASTSecurityChecker) -> None:
        """Test detection of subprocess calls."""
        code = "import subprocess\nsubprocess.run(['ls'])"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_open(self, checker: ASTSecurityChecker) -> None:
        """Test detection of open() function."""
        code = "f = open('/etc/passwd', 'r')"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_dunder_import(self, checker: ASTSecurityChecker) -> None:
        """Test detection of __import__()."""
        code = "__import__('os')"
        result = checker.check(code)
        assert not result.is_safe

    # === Dangerous Import Detection ===

    def test_detect_os_import(self, checker: ASTSecurityChecker) -> None:
        """Test detection of os module import."""
        code = "import os"
        result = checker.check(code)
        assert not result.is_safe
        assert any(v.violation_type == ViolationType.DANGEROUS_IMPORT for v in result.violations)

    def test_detect_sys_import(self, checker: ASTSecurityChecker) -> None:
        """Test detection of sys module import."""
        code = "import sys"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_socket_import(self, checker: ASTSecurityChecker) -> None:
        """Test detection of socket module import."""
        code = "import socket"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_pickle_import(self, checker: ASTSecurityChecker) -> None:
        """Test detection of pickle module import."""
        code = "import pickle"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_from_import(self, checker: ASTSecurityChecker) -> None:
        """Test detection of from ... import."""
        code = "from os import path"
        result = checker.check(code)
        assert not result.is_safe

    # === Dangerous Attribute Access ===

    def test_detect_globals_access(self, checker: ASTSecurityChecker) -> None:
        """Test detection of __globals__ access."""
        code = "func.__globals__"
        result = checker.check(code)
        assert not result.is_safe
        assert any(v.violation_type == ViolationType.DANGEROUS_ATTRIBUTE for v in result.violations)

    def test_detect_builtins_access(self, checker: ASTSecurityChecker) -> None:
        """Test detection of __builtins__ access."""
        code = "x.__builtins__"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_code_access(self, checker: ASTSecurityChecker) -> None:
        """Test detection of __code__ access."""
        code = "func.__code__"
        result = checker.check(code)
        assert not result.is_safe

    def test_detect_subclasses_access(self, checker: ASTSecurityChecker) -> None:
        """Test detection of __subclasses__ access."""
        code = "object.__subclasses__()"
        result = checker.check(code)
        assert not result.is_safe

    # === Safe Code ===

    def test_safe_math_code(self, checker: ASTSecurityChecker) -> None:
        """Test that safe math code passes."""
        code = """
def calculate_factor(df):
    return df['close'].pct_change(20)
"""
        result = checker.check(code)
        assert result.is_safe
        assert len(result.violations) == 0

    def test_safe_pandas_code(self, checker: ASTSecurityChecker) -> None:
        """Test that safe pandas code passes."""
        code = """
import pandas as pd
import numpy as np

def momentum_factor(df):
    returns = df['close'].pct_change()
    return returns.rolling(20).mean()
"""
        result = checker.check(code)
        assert result.is_safe

    def test_safe_numpy_code(self, checker: ASTSecurityChecker) -> None:
        """Test that safe numpy code passes."""
        code = """
import numpy as np

def volatility_factor(prices):
    log_returns = np.log(prices[1:] / prices[:-1])
    return np.std(log_returns) * np.sqrt(252)
"""
        result = checker.check(code)
        assert result.is_safe


class TestASTSecurityCheckerBoundary:
    """Boundary tests for edge cases."""

    @pytest.fixture
    def checker(self) -> ASTSecurityChecker:
        return ASTSecurityChecker()

    def test_empty_code(self, checker: ASTSecurityChecker) -> None:
        """Test handling of empty code."""
        result = checker.check("")
        assert result.is_safe

    def test_whitespace_only(self, checker: ASTSecurityChecker) -> None:
        """Test handling of whitespace-only code."""
        result = checker.check("   \n\t\n   ")
        assert result.is_safe

    def test_comments_only(self, checker: ASTSecurityChecker) -> None:
        """Test handling of comments-only code."""
        code = "# This is a comment\n# Another comment"
        result = checker.check(code)
        assert result.is_safe

    def test_very_long_code(self, checker: ASTSecurityChecker) -> None:
        """Test handling of very long safe code."""
        code = "x = 1\n" * 1000
        result = checker.check(code)
        assert result.is_safe

    def test_deeply_nested_code(self, checker: ASTSecurityChecker) -> None:
        """Test handling of deeply nested code."""
        code = """
def outer():
    def inner1():
        def inner2():
            def inner3():
                return 1
            return inner3()
        return inner2()
    return inner1()
"""
        result = checker.check(code)
        assert result.is_safe


class TestASTSecurityCheckerException:
    """Exception handling tests."""

    @pytest.fixture
    def checker(self) -> ASTSecurityChecker:
        return ASTSecurityChecker()

    def test_syntax_error_handling(self, checker: ASTSecurityChecker) -> None:
        """Test handling of syntax errors."""
        code = "def broken("
        result = checker.check(code)
        assert not result.is_safe
        assert any(v.violation_type == ViolationType.SYNTAX_ERROR for v in result.violations)

    def test_invalid_encoding(self, checker: ASTSecurityChecker) -> None:
        """Test handling of invalid encoding."""
        # This should not crash
        code = "x = '\\x80'"
        result = checker.check(code)
        # Should handle gracefully
        assert isinstance(result, SecurityCheckResult)

    def test_unicode_code(self, checker: ASTSecurityChecker) -> None:
        """Test handling of unicode in code."""
        code = "x = '你好世界'"
        result = checker.check(code)
        assert result.is_safe


class TestASTSecurityCheckerPerformance:
    """Performance tests."""

    @pytest.fixture
    def checker(self) -> ASTSecurityChecker:
        return ASTSecurityChecker()

    def test_check_completes_quickly(self, checker: ASTSecurityChecker) -> None:
        """Test that security check completes within reasonable time."""
        code = """
import pandas as pd
import numpy as np

def complex_factor(df):
    returns = df['close'].pct_change()
    vol = returns.rolling(20).std()
    mom = returns.rolling(20).mean()
    return mom / vol
"""
        start = time.time()
        for _ in range(100):
            checker.check(code)
        elapsed = time.time() - start
        # 100 checks should complete in under 1 second
        assert elapsed < 1.0, f"Performance issue: {elapsed:.2f}s for 100 checks"


class TestASTSecurityCheckerSecurity:
    """Security bypass attempt tests."""

    @pytest.fixture
    def checker(self) -> ASTSecurityChecker:
        return ASTSecurityChecker()

    def test_obfuscated_eval(self, checker: ASTSecurityChecker) -> None:
        """Test detection of obfuscated eval."""
        code = "getattr(__builtins__, 'eval')('1+1')"
        result = checker.check(code)
        assert not result.is_safe

    def test_string_concat_import(self, checker: ASTSecurityChecker) -> None:
        """Test detection of dynamic import attempts."""
        code = "__import__('o' + 's')"
        result = checker.check(code)
        assert not result.is_safe

    def test_class_escape(self, checker: ASTSecurityChecker) -> None:
        """Test detection of class-based escape attempts."""
        code = "().__class__.__bases__[0].__subclasses__()"
        result = checker.check(code)
        assert not result.is_safe

    def test_lambda_with_dangerous_call(self, checker: ASTSecurityChecker) -> None:
        """Test detection of dangerous calls in lambda."""
        code = "f = lambda: eval('1')"
        result = checker.check(code)
        assert not result.is_safe

    def test_list_comp_with_dangerous_call(self, checker: ASTSecurityChecker) -> None:
        """Test detection of dangerous calls in list comprehension."""
        code = "[eval(x) for x in ['1', '2']]"
        result = checker.check(code)
        assert not result.is_safe


class TestASTSecurityCheckerCompatibility:
    """Compatibility tests for different code patterns."""

    @pytest.fixture
    def checker(self) -> ASTSecurityChecker:
        return ASTSecurityChecker()

    def test_async_code(self, checker: ASTSecurityChecker) -> None:
        """Test handling of async code."""
        code = """
async def fetch_data():
    return await some_async_call()
"""
        result = checker.check(code)
        assert result.is_safe

    def test_type_hints(self, checker: ASTSecurityChecker) -> None:
        """Test handling of type hints."""
        code = """
from typing import List, Optional

def process(data: List[float]) -> Optional[float]:
    if not data:
        return None
    return sum(data) / len(data)
"""
        result = checker.check(code)
        assert result.is_safe

    def test_walrus_operator(self, checker: ASTSecurityChecker) -> None:
        """Test handling of walrus operator."""
        code = """
if (n := len(data)) > 0:
    average = sum(data) / n
"""
        result = checker.check(code)
        assert result.is_safe

    def test_match_statement(self, checker: ASTSecurityChecker) -> None:
        """Test handling of match statement (Python 3.10+)."""
        code = """
match status:
    case 200:
        return "OK"
    case 404:
        return "Not Found"
    case _:
        return "Unknown"
"""
        result = checker.check(code)
        assert result.is_safe

    def test_dataclass_code(self, checker: ASTSecurityChecker) -> None:
        """Test handling of dataclass code."""
        code = """
from dataclasses import dataclass

@dataclass
class Factor:
    name: str
    value: float
"""
        result = checker.check(code)
        assert result.is_safe


class TestSecurityCheckResult:
    """Test SecurityCheckResult functionality."""

    @pytest.fixture
    def checker(self) -> ASTSecurityChecker:
        return ASTSecurityChecker()

    def test_result_summary(self, checker: ASTSecurityChecker) -> None:
        """Test result summary generation."""
        code = "import os\neval('1')\nexec('2')"
        result = checker.check(code)
        summary = result.get_summary()
        assert "violations" in summary.lower()

    def test_result_to_dict(self, checker: ASTSecurityChecker) -> None:
        """Test result serialization."""
        code = "import os"
        result = checker.check(code)
        data = result.to_dict()
        assert "is_safe" in data
        assert "violations" in data
        assert data["is_safe"] is False

    def test_multiple_violations(self, checker: ASTSecurityChecker) -> None:
        """Test handling of multiple violations."""
        code = """
import os
import sys
eval('1')
exec('2')
"""
        result = checker.check(code)
        assert len(result.violations) >= 4
