"""Sandbox Executor for secure code execution.

This module provides a sandboxed execution environment for LLM-generated
Python code. It serves as the second layer of the three-layer security
architecture.

Security Layers:
1. AST Security Checker - Static analysis (pre-execution)
2. Sandbox Executor (this module) - Runtime isolation
3. Human Review Gate - Manual approval
"""

import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from iqfmp.core.security import ASTSecurityChecker


class ExecutionStatus(str, Enum):
    """Status of code execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class ExecutionResult:
    """Result of sandboxed code execution."""

    status: ExecutionStatus
    output: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "status": self.status.value,
            "output": self.output,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
        }


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""

    timeout_seconds: int = 60
    max_memory_mb: int = 512
    allowed_modules: list[str] = field(default_factory=lambda: [
        "math",
        "statistics",
        "decimal",
        "fractions",
        "datetime",
        "time",
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
        "pandas",
        "numpy",
        "scipy",
    ])


class TimeoutException(Exception):
    """Exception raised when execution times out."""
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for timeout."""
    raise TimeoutException("Execution timed out")


class SandboxExecutor:
    """Sandboxed executor for Python code.

    This class provides a secure execution environment with:
    - Pre-execution AST security checking
    - Module whitelist enforcement
    - Execution timeout enforcement
    - Restricted builtins
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        """Initialize the sandbox executor.

        Args:
            config: Optional sandbox configuration.
        """
        self.config = config or SandboxConfig()
        self.security_checker = ASTSecurityChecker()
        self._allowed_builtins = self._create_allowed_builtins()

    def execute(self, code: str) -> ExecutionResult:
        """Execute code in the sandbox.

        Args:
            code: Python source code to execute.

        Returns:
            ExecutionResult with status and output.
        """
        start_time = time.time()

        # Handle empty code
        if not code or not code.strip():
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output={},
                execution_time=0.0,
            )

        # Step 1: AST security check
        security_result = self.security_checker.check(code)
        if not security_result.is_safe:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                error_message=security_result.get_summary(),
                execution_time=time.time() - start_time,
            )

        # Step 2: Execute in sandbox with timeout
        try:
            result = self._execute_with_timeout(code)
            result.execution_time = time.time() - start_time
            return result
        except TimeoutException:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Execution timed out after {self.config.timeout_seconds} seconds",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    def _execute_with_timeout(self, code: str) -> ExecutionResult:
        """Execute code with timeout enforcement.

        Args:
            code: Python source code to execute.

        Returns:
            ExecutionResult with status and output.
        """
        # Set up timeout (Unix only)
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.config.timeout_seconds)
        except (ValueError, AttributeError):
            # signal.alarm not available on Windows
            pass

        try:
            # Create execution environment
            exec_globals = self._create_execution_globals()
            exec_locals: dict[str, Any] = {}

            # Compile the code
            try:
                compiled = compile(code, "<sandbox>", "exec")
            except SyntaxError as e:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error_message=f"Syntax error: {e.msg} at line {e.lineno}",
                )

            # Execute the code
            exec(compiled, exec_globals, exec_locals)

            # Extract output (variables defined in the code)
            output = self._extract_output(exec_locals)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output,
            )

        except TimeoutException:
            raise
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"{type(e).__name__}: {str(e)}",
            )
        finally:
            # Reset alarm
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, AttributeError):
                pass

    def _create_execution_globals(self) -> dict[str, Any]:
        """Create the global namespace for execution.

        Returns:
            Dictionary with allowed modules and builtins.
        """
        exec_globals: dict[str, Any] = {
            "__builtins__": self._allowed_builtins,
            "__name__": "__sandbox__",
        }

        # Import allowed modules
        for module_name in self.config.allowed_modules:
            try:
                if module_name == "pandas":
                    import pandas as pd
                    exec_globals["pandas"] = pd
                    exec_globals["pd"] = pd
                elif module_name == "numpy":
                    import numpy as np
                    exec_globals["numpy"] = np
                    exec_globals["np"] = np
                elif module_name == "scipy":
                    import scipy
                    exec_globals["scipy"] = scipy
                elif module_name == "math":
                    import math
                    exec_globals["math"] = math
                elif module_name == "statistics":
                    import statistics
                    exec_globals["statistics"] = statistics
                elif module_name == "datetime":
                    import datetime
                    exec_globals["datetime"] = datetime
                elif module_name == "time":
                    import time as time_module
                    exec_globals["time"] = time_module
                elif module_name == "collections":
                    import collections
                    exec_globals["collections"] = collections
                elif module_name == "itertools":
                    import itertools
                    exec_globals["itertools"] = itertools
                elif module_name == "functools":
                    import functools
                    exec_globals["functools"] = functools
                elif module_name == "operator":
                    import operator
                    exec_globals["operator"] = operator
                elif module_name == "typing":
                    import typing
                    exec_globals["typing"] = typing
                elif module_name == "dataclasses":
                    import dataclasses
                    exec_globals["dataclasses"] = dataclasses
                elif module_name == "enum":
                    import enum
                    exec_globals["enum"] = enum
                elif module_name == "abc":
                    import abc
                    exec_globals["abc"] = abc
                elif module_name == "copy":
                    import copy
                    exec_globals["copy"] = copy
                elif module_name == "re":
                    import re
                    exec_globals["re"] = re
                elif module_name == "json":
                    import json
                    exec_globals["json"] = json
                elif module_name == "hashlib":
                    import hashlib
                    exec_globals["hashlib"] = hashlib
                elif module_name == "base64":
                    import base64
                    exec_globals["base64"] = base64
                elif module_name == "decimal":
                    import decimal
                    exec_globals["decimal"] = decimal
                elif module_name == "fractions":
                    import fractions
                    exec_globals["fractions"] = fractions
            except ImportError:
                # Module not available, skip
                pass

        return exec_globals

    def _create_allowed_builtins(self) -> dict[str, Any]:
        """Create restricted builtins dictionary.

        Returns:
            Dictionary with safe builtin functions only.
        """
        import builtins

        # Safe builtin functions
        safe_builtins = {
            # Type conversion
            "int": builtins.int,
            "float": builtins.float,
            "str": builtins.str,
            "bool": builtins.bool,
            "bytes": builtins.bytes,
            "bytearray": builtins.bytearray,
            "complex": builtins.complex,

            # Collections
            "list": builtins.list,
            "tuple": builtins.tuple,
            "dict": builtins.dict,
            "set": builtins.set,
            "frozenset": builtins.frozenset,

            # Iterators and generators
            "range": builtins.range,
            "iter": builtins.iter,
            "next": builtins.next,
            "enumerate": builtins.enumerate,
            "zip": builtins.zip,
            "map": builtins.map,
            "filter": builtins.filter,
            "reversed": builtins.reversed,
            "sorted": builtins.sorted,

            # Math and comparison
            "abs": builtins.abs,
            "min": builtins.min,
            "max": builtins.max,
            "sum": builtins.sum,
            "round": builtins.round,
            "pow": builtins.pow,
            "divmod": builtins.divmod,

            # String operations
            "len": builtins.len,
            "repr": builtins.repr,
            "format": builtins.format,
            "chr": builtins.chr,
            "ord": builtins.ord,
            "ascii": builtins.ascii,
            "bin": builtins.bin,
            "hex": builtins.hex,
            "oct": builtins.oct,

            # Boolean operations
            "all": builtins.all,
            "any": builtins.any,

            # Object introspection (safe subset)
            "type": builtins.type,
            "isinstance": builtins.isinstance,
            "issubclass": builtins.issubclass,
            "callable": builtins.callable,
            "hash": builtins.hash,
            "id": builtins.id,

            # Attribute access (restricted)
            "hasattr": builtins.hasattr,

            # Exceptions (for error handling in code)
            "Exception": builtins.Exception,
            "ValueError": builtins.ValueError,
            "TypeError": builtins.TypeError,
            "KeyError": builtins.KeyError,
            "IndexError": builtins.IndexError,
            "AttributeError": builtins.AttributeError,
            "ZeroDivisionError": builtins.ZeroDivisionError,
            "RuntimeError": builtins.RuntimeError,
            "StopIteration": builtins.StopIteration,

            # Constants
            "True": True,
            "False": False,
            "None": None,

            # Other safe functions
            "slice": builtins.slice,
            "object": builtins.object,
            "property": builtins.property,
            "staticmethod": builtins.staticmethod,
            "classmethod": builtins.classmethod,
            "super": builtins.super,
            "print": self._safe_print,  # Safe print that does nothing

            # Safe __import__ for allowed modules only
            "__import__": self._safe_import,

            # Required for class definitions
            "__build_class__": builtins.__build_class__,
        }

        return safe_builtins

    def _safe_import(
        self,
        name: str,
        globals: Optional[dict] = None,
        locals: Optional[dict] = None,
        fromlist: tuple = (),
        level: int = 0,
    ) -> Any:
        """Safe import function that only allows whitelisted modules.

        Args:
            name: Module name to import.
            globals: Global namespace (unused).
            locals: Local namespace (unused).
            fromlist: Names to import from module.
            level: Relative import level.

        Returns:
            The imported module.

        Raises:
            ImportError: If module is not in whitelist.
        """
        # Get the top-level module name
        top_level = name.split(".")[0]

        # Check if module is allowed
        if top_level not in self.config.allowed_modules:
            raise ImportError(f"Import of module '{name}' is not allowed")

        # Import the module
        import importlib
        return importlib.import_module(name)

    def _safe_print(self, *args: Any, **kwargs: Any) -> None:
        """Safe print function that does nothing.

        We don't want sandboxed code to produce output.
        """
        pass

    def _extract_output(self, exec_locals: dict[str, Any]) -> dict[str, Any]:
        """Extract output variables from execution locals.

        Args:
            exec_locals: Local namespace after execution.

        Returns:
            Dictionary with serializable output values.
        """
        output: dict[str, Any] = {}

        for name, value in exec_locals.items():
            # Skip private/magic names
            if name.startswith("_"):
                continue

            # Skip functions and classes (keep only data)
            if callable(value) and not isinstance(value, type):
                continue

            # Try to serialize the value
            try:
                # Handle common types
                if isinstance(value, (int, float, str, bool, type(None))):
                    output[name] = value
                elif isinstance(value, (list, tuple)):
                    output[name] = self._serialize_sequence(value)
                elif isinstance(value, dict):
                    output[name] = self._serialize_dict(value)
                elif hasattr(value, "tolist"):  # numpy arrays, pandas series
                    output[name] = value.tolist() if hasattr(value, "__len__") and len(value) < 10000 else str(value)
                elif hasattr(value, "to_dict"):  # pandas dataframes
                    output[name] = value.to_dict()
                elif hasattr(value, "item"):  # numpy scalars
                    output[name] = value.item()
                else:
                    # Fallback to string representation
                    output[name] = str(value)
            except Exception:
                # If serialization fails, use string representation
                output[name] = str(value)

        return output

    def _serialize_sequence(self, seq: Any) -> list:
        """Serialize a sequence to a list."""
        result = []
        for item in seq:
            if isinstance(item, (int, float, str, bool, type(None))):
                result.append(item)
            elif isinstance(item, (list, tuple)):
                result.append(self._serialize_sequence(item))
            elif isinstance(item, dict):
                result.append(self._serialize_dict(item))
            elif hasattr(item, "item"):
                result.append(item.item())
            else:
                result.append(str(item))
        return result

    def _serialize_dict(self, d: dict) -> dict:
        """Serialize a dictionary."""
        result = {}
        for key, value in d.items():
            # Convert key to string if needed
            str_key = str(key) if not isinstance(key, str) else key

            if isinstance(value, (int, float, str, bool, type(None))):
                result[str_key] = value
            elif isinstance(value, (list, tuple)):
                result[str_key] = self._serialize_sequence(value)
            elif isinstance(value, dict):
                result[str_key] = self._serialize_dict(value)
            elif hasattr(value, "item"):
                result[str_key] = value.item()
            else:
                result[str_key] = str(value)
        return result
