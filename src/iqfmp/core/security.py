"""AST Security Checker for LLM-generated code.

This module provides static analysis of Python code to detect potentially
dangerous operations before execution. It serves as the first layer of
the three-layer security architecture.

Security Layers:
1. AST Security Checker (this module) - Static analysis
2. Sandbox Executor - Runtime isolation
3. Human Review Gate - Manual approval
"""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ViolationType(str, Enum):
    """Types of security violations."""

    DANGEROUS_FUNCTION = "dangerous_function"
    DANGEROUS_IMPORT = "dangerous_import"
    DANGEROUS_ATTRIBUTE = "dangerous_attribute"
    SYNTAX_ERROR = "syntax_error"


@dataclass
class SecurityViolation:
    """Represents a single security violation found in code."""

    violation_type: ViolationType
    message: str
    line_number: int = 0
    column: int = 0
    code_snippet: str = ""

    def to_dict(self) -> dict:
        """Convert violation to dictionary."""
        return {
            "type": self.violation_type.value,
            "message": self.message,
            "line": self.line_number,
            "column": self.column,
            "snippet": self.code_snippet,
        }


@dataclass
class SecurityCheckResult:
    """Result of a security check."""

    is_safe: bool
    violations: list[SecurityViolation] = field(default_factory=list)
    code_hash: str = ""

    def get_summary(self) -> str:
        """Generate a human-readable summary of the check result."""
        if self.is_safe:
            return "Code passed security check. No violations found."

        violation_counts = {}
        for v in self.violations:
            vtype = v.violation_type.value
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1

        summary_parts = [f"Security check failed with {len(self.violations)} violations:"]
        for vtype, count in violation_counts.items():
            summary_parts.append(f"  - {vtype}: {count}")

        return "\n".join(summary_parts)

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "is_safe": self.is_safe,
            "violations": [v.to_dict() for v in self.violations],
            "code_hash": self.code_hash,
            "violation_count": len(self.violations),
        }


class ASTSecurityChecker:
    """AST-based security checker for Python code.

    This class performs static analysis on Python code to detect
    potentially dangerous operations before execution.
    """

    # Dangerous built-in functions
    DANGEROUS_FUNCTIONS: set[str] = {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "getattr",  # Can be used to access dangerous attributes
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "input",  # Can hang execution
        "breakpoint",  # Debugging
    }

    # Dangerous modules
    DANGEROUS_MODULES: set[str] = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "pickle",
        "marshal",
        "shelve",
        "ctypes",
        "cffi",
        "multiprocessing",
        "threading",
        "asyncio",  # Can be used for network operations
        "importlib",
        "builtins",
        "code",
        "codeop",
        "pty",
        "fcntl",
        "signal",
        "resource",
        "sysconfig",
        "platform",
        "shutil",
        "tempfile",
        "pathlib",  # File system access
        "io",  # File operations
    }

    # Safe modules (whitelist)
    SAFE_MODULES: set[str] = {
        "pandas",
        "pd",
        "numpy",
        "np",
        "math",
        "statistics",
        "decimal",
        "fractions",
        "datetime",
        "time",  # Only for time calculations, not sleep
        "collections",
        "itertools",
        "functools",
        "operator",
        "typing",
        "dataclasses",
        "enum",
        "abc",
        "copy",
        "re",
        "json",
        "hashlib",
        "base64",
        "scipy",
        "sklearn",
        "talib",  # Technical analysis library
    }

    # Dangerous attribute names
    DANGEROUS_ATTRIBUTES: set[str] = {
        "__globals__",
        "__builtins__",
        "__code__",
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__dict__",
        "__module__",
        "__import__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__annotations__",
        "__qualname__",
        "__func__",
        "__self__",
        "__closure__",
        "__defaults__",
        "__kwdefaults__",
        "gi_frame",
        "gi_code",
        "f_globals",
        "f_locals",
        "f_code",
        "f_builtins",
    }

    # Dangerous method names on modules
    DANGEROUS_METHODS: set[str] = {
        "system",
        "popen",
        "spawn",
        "fork",
        "exec",
        "execv",
        "execve",
        "execl",
        "execlp",
        "execvp",
        "run",  # subprocess.run
        "call",  # subprocess.call
        "Popen",  # subprocess.Popen
        "check_output",
        "check_call",
    }

    def __init__(self) -> None:
        """Initialize the security checker."""
        self._violations: list[SecurityViolation] = []

    def check(self, code: str) -> SecurityCheckResult:
        """Check code for security violations.

        Args:
            code: Python source code to check.

        Returns:
            SecurityCheckResult with is_safe flag and list of violations.
        """
        self._violations = []

        # Handle empty or whitespace-only code
        if not code or not code.strip():
            return SecurityCheckResult(is_safe=True)

        # Try to parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violation = SecurityViolation(
                violation_type=ViolationType.SYNTAX_ERROR,
                message=f"Syntax error: {e.msg}",
                line_number=e.lineno or 0,
                column=e.offset or 0,
            )
            return SecurityCheckResult(is_safe=False, violations=[violation])

        # Walk the AST and check for violations
        self._check_node(tree, code)

        is_safe = len(self._violations) == 0
        return SecurityCheckResult(
            is_safe=is_safe,
            violations=self._violations.copy(),
            code_hash=self._compute_hash(code),
        )

    def _check_node(self, node: ast.AST, source: str) -> None:
        """Recursively check AST node for violations."""
        for child in ast.walk(node):
            # Check function calls
            if isinstance(child, ast.Call):
                self._check_call(child, source)

            # Check imports
            elif isinstance(child, ast.Import):
                self._check_import(child)

            elif isinstance(child, ast.ImportFrom):
                self._check_import_from(child)

            # Check attribute access
            elif isinstance(child, ast.Attribute):
                self._check_attribute(child, source)

    def _check_call(self, node: ast.Call, source: str) -> None:
        """Check function call for dangerous functions."""
        func = node.func

        # Direct function call: eval(), exec(), etc.
        if isinstance(func, ast.Name):
            if func.id in self.DANGEROUS_FUNCTIONS:
                self._add_violation(
                    ViolationType.DANGEROUS_FUNCTION,
                    f"Dangerous function '{func.id}' detected",
                    node,
                    source,
                )

        # Method call: os.system(), subprocess.run(), etc.
        elif isinstance(func, ast.Attribute):
            # Check for dangerous methods
            if func.attr in self.DANGEROUS_METHODS:
                self._add_violation(
                    ViolationType.DANGEROUS_FUNCTION,
                    f"Dangerous method '{func.attr}' detected",
                    node,
                    source,
                )

            # Check for dangerous module method calls
            if isinstance(func.value, ast.Name):
                module_name = func.value.id
                if module_name in self.DANGEROUS_MODULES:
                    self._add_violation(
                        ViolationType.DANGEROUS_FUNCTION,
                        f"Call to dangerous module '{module_name}.{func.attr}'",
                        node,
                        source,
                    )

    def _check_import(self, node: ast.Import) -> None:
        """Check import statement for dangerous modules."""
        for alias in node.names:
            module_name = alias.name.split(".")[0]  # Get top-level module
            if module_name in self.DANGEROUS_MODULES:
                self._add_violation(
                    ViolationType.DANGEROUS_IMPORT,
                    f"Import of dangerous module '{alias.name}'",
                    node,
                    "",
                )

    def _check_import_from(self, node: ast.ImportFrom) -> None:
        """Check from...import statement for dangerous modules."""
        if node.module:
            module_name = node.module.split(".")[0]
            if module_name in self.DANGEROUS_MODULES:
                self._add_violation(
                    ViolationType.DANGEROUS_IMPORT,
                    f"Import from dangerous module '{node.module}'",
                    node,
                    "",
                )

    def _check_attribute(self, node: ast.Attribute, source: str) -> None:
        """Check attribute access for dangerous attributes."""
        if node.attr in self.DANGEROUS_ATTRIBUTES:
            self._add_violation(
                ViolationType.DANGEROUS_ATTRIBUTE,
                f"Access to dangerous attribute '{node.attr}'",
                node,
                source,
            )

    def _add_violation(
        self,
        violation_type: ViolationType,
        message: str,
        node: ast.AST,
        source: str,
    ) -> None:
        """Add a violation to the list."""
        line_number = getattr(node, "lineno", 0)
        column = getattr(node, "col_offset", 0)

        # Extract code snippet
        snippet = ""
        if source and line_number > 0:
            lines = source.split("\n")
            if 0 < line_number <= len(lines):
                snippet = lines[line_number - 1].strip()

        violation = SecurityViolation(
            violation_type=violation_type,
            message=message,
            line_number=line_number,
            column=column,
            code_snippet=snippet,
        )
        self._violations.append(violation)

    def _compute_hash(self, code: str) -> str:
        """Compute hash of the code for caching."""
        import hashlib

        return hashlib.sha256(code.encode()).hexdigest()[:16]
