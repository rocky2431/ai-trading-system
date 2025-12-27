"""Sandbox Executor for secure code execution.

This module provides a sandboxed execution environment for LLM-generated
Python code. It serves as the second layer of the three-layer security
architecture.

Security Layers:
1. AST Security Checker - Static analysis (pre-execution)
2. Sandbox Executor (this module) - Runtime isolation with RestrictedPython
3. Human Review Gate - Manual approval

P0 Security Enhancement (2025-12-26):
- RestrictedPython integration for bytecode-level restrictions
- CPU/Memory resource limits via resource.setrlimit
- Subprocess isolation for additional safety
"""

import logging
import multiprocessing
import os
import platform
import signal
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# RestrictedPython imports - P0 Security Enhancement
from RestrictedPython import compile_restricted, safe_builtins, limited_builtins
from RestrictedPython.Guards import (
    safe_globals,
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
)
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem

from iqfmp.core.security import ASTSecurityChecker

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of code execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_EXCEEDED = "resource_exceeded"


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
    max_cpu_seconds: int = 30  # P0: CPU time limit
    use_subprocess: bool = True  # P0: Use subprocess for isolation
    # NOTE: scipy is intentionally EXCLUDED from allowed modules
    # All statistical operations must go through qlib_stats for Qlib architectural consistency
    # See: .ultra/docs/qlib-architecture-audit-report.md
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
        # "scipy" - REMOVED: Use iqfmp.evaluation.qlib_stats instead
        "iqfmp.evaluation.qlib_stats",  # Qlib-native statistical functions
    ])


class TimeoutException(Exception):
    """Exception raised when execution times out."""
    pass


class ResourceExceededException(Exception):
    """Exception raised when resource limits are exceeded."""
    pass


class RestrictedExecutionError(Exception):
    """Exception raised when RestrictedPython blocks an operation."""
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    """Signal handler for timeout."""
    raise TimeoutException("Execution timed out")


def _set_resource_limits(max_memory_mb: int, max_cpu_seconds: int) -> None:
    """Set resource limits for the current process.

    Only works on Unix-like systems (Linux, macOS).
    """
    if platform.system() == "Windows":
        logger.warning("Resource limits not supported on Windows")
        return

    try:
        import resource

        # Set CPU time limit (soft, hard)
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (max_cpu_seconds, max_cpu_seconds + 5)
        )

        # Set memory limit (soft, hard) in bytes
        max_memory_bytes = max_memory_mb * 1024 * 1024
        resource.setrlimit(
            resource.RLIMIT_AS,
            (max_memory_bytes, max_memory_bytes)
        )

        # Set maximum number of open files (prevent resource exhaustion)
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))

        logger.debug(
            f"Resource limits set: CPU={max_cpu_seconds}s, Memory={max_memory_mb}MB"
        )
    except (ImportError, ValueError, OSError) as e:
        logger.warning(f"Failed to set resource limits: {e}")


def _execute_in_subprocess(
    code: str,
    exec_globals: dict[str, Any],
    timeout_seconds: int,
    max_memory_mb: int,
    max_cpu_seconds: int,
) -> tuple[ExecutionStatus, dict[str, Any], str]:
    """Execute code in a subprocess with resource limits.

    Returns:
        Tuple of (status, output_dict, error_message)
    """
    def _worker(
        code: str,
        result_queue: multiprocessing.Queue,
        max_memory_mb: int,
        max_cpu_seconds: int,
    ) -> None:
        """Worker function that runs in subprocess."""
        try:
            # Set resource limits in subprocess
            _set_resource_limits(max_memory_mb, max_cpu_seconds)

            # Compile with RestrictedPython
            compiled = compile_restricted(code, "<sandbox>", "exec")

            if compiled.errors:
                result_queue.put((
                    ExecutionStatus.SECURITY_VIOLATION,
                    {},
                    f"RestrictedPython compilation errors: {compiled.errors}"
                ))
                return

            # Create restricted execution environment
            restricted_globals = _create_restricted_globals(exec_globals)
            exec_locals: dict[str, Any] = {}

            # Execute
            exec(compiled.code, restricted_globals, exec_locals)

            # Extract serializable output
            output = _extract_serializable_output(exec_locals)
            result_queue.put((ExecutionStatus.SUCCESS, output, ""))

        except MemoryError:
            result_queue.put((
                ExecutionStatus.RESOURCE_EXCEEDED,
                {},
                "Memory limit exceeded"
            ))
        except Exception as e:
            result_queue.put((
                ExecutionStatus.ERROR,
                {},
                f"{type(e).__name__}: {str(e)}"
            ))

    # Create queue for result communication
    result_queue = multiprocessing.Queue()

    # Start subprocess
    process = multiprocessing.Process(
        target=_worker,
        args=(code, result_queue, max_memory_mb, max_cpu_seconds),
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        # Timeout - kill the process
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
        return (ExecutionStatus.TIMEOUT, {}, f"Execution timed out after {timeout_seconds}s")

    # Get result from queue
    try:
        if not result_queue.empty():
            return result_queue.get_nowait()
        else:
            return (ExecutionStatus.ERROR, {}, "No result from subprocess")
    except Exception as e:
        return (ExecutionStatus.ERROR, {}, f"Failed to get result: {e}")


def _create_restricted_globals(allowed_modules_globals: dict[str, Any]) -> dict[str, Any]:
    """Create RestrictedPython-compatible global namespace.

    Combines safe_builtins from RestrictedPython with allowed modules.
    """
    # Start with RestrictedPython's safe globals
    restricted_globals = dict(safe_globals)

    # Use safe_builtins as the base
    restricted_builtins = dict(safe_builtins)

    # Add guarded operations required by RestrictedPython
    restricted_builtins["_getattr_"] = default_guarded_getattr
    restricted_builtins["_getitem_"] = default_guarded_getitem
    restricted_builtins["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
    restricted_builtins["_unpack_sequence_"] = guarded_unpack_sequence

    # Add safe exception types for error handling
    import builtins
    for exc_name in [
        "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "AttributeError", "ZeroDivisionError",
        "RuntimeError", "StopIteration",
    ]:
        restricted_builtins[exc_name] = getattr(builtins, exc_name)

    restricted_globals["__builtins__"] = restricted_builtins
    restricted_globals["__name__"] = "__sandbox__"

    # Add allowed modules (already imported)
    for key, value in allowed_modules_globals.items():
        if key not in ("__builtins__", "__name__"):
            restricted_globals[key] = value

    return restricted_globals


def _extract_serializable_output(exec_locals: dict[str, Any]) -> dict[str, Any]:
    """Extract serializable output from execution locals."""
    output: dict[str, Any] = {}

    for name, value in exec_locals.items():
        # Skip private/magic names
        if name.startswith("_"):
            continue

        # Skip functions and classes (keep only data)
        if callable(value) and not isinstance(value, type):
            continue

        try:
            # Handle common types
            if isinstance(value, (int, float, str, bool, type(None))):
                output[name] = value
            elif isinstance(value, (list, tuple)):
                output[name] = _serialize_sequence(value)
            elif isinstance(value, dict):
                output[name] = _serialize_dict(value)
            elif hasattr(value, "tolist"):  # numpy arrays, pandas series
                if hasattr(value, "__len__") and len(value) < 10000:
                    output[name] = value.tolist()
                else:
                    output[name] = str(value)
            elif hasattr(value, "to_dict"):  # pandas dataframes
                output[name] = value.to_dict()
            elif hasattr(value, "item"):  # numpy scalars
                output[name] = value.item()
            else:
                output[name] = str(value)
        except Exception:
            output[name] = str(value)

    return output


def _serialize_sequence(seq: Any) -> list:
    """Serialize a sequence to a list."""
    result = []
    for item in seq:
        if isinstance(item, (int, float, str, bool, type(None))):
            result.append(item)
        elif isinstance(item, (list, tuple)):
            result.append(_serialize_sequence(item))
        elif isinstance(item, dict):
            result.append(_serialize_dict(item))
        elif hasattr(item, "item"):
            result.append(item.item())
        else:
            result.append(str(item))
    return result


def _serialize_dict(d: dict) -> dict:
    """Serialize a dictionary."""
    result = {}
    for key, value in d.items():
        str_key = str(key) if not isinstance(key, str) else key

        if isinstance(value, (int, float, str, bool, type(None))):
            result[str_key] = value
        elif isinstance(value, (list, tuple)):
            result[str_key] = _serialize_sequence(value)
        elif isinstance(value, dict):
            result[str_key] = _serialize_dict(value)
        elif hasattr(value, "item"):
            result[str_key] = value.item()
        else:
            result[str_key] = str(value)
    return result


class SandboxExecutor:
    """Sandboxed executor for Python code using RestrictedPython.

    This class provides a secure execution environment with:
    - Pre-execution AST security checking
    - RestrictedPython bytecode-level restrictions (P0 Enhancement)
    - CPU/Memory resource limits (P0 Enhancement)
    - Module whitelist enforcement
    - Execution timeout enforcement
    - Optional subprocess isolation (P0 Enhancement)
    """

    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        """Initialize the sandbox executor.

        Args:
            config: Optional sandbox configuration.
        """
        self.config = config or SandboxConfig()
        self.security_checker = ASTSecurityChecker()
        self._allowed_modules_globals = self._import_allowed_modules()

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

        # Step 1: AST security check (first line of defense)
        security_result = self.security_checker.check(code)
        if not security_result.is_safe:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                error_message=security_result.get_summary(),
                execution_time=time.time() - start_time,
            )

        # Step 2: Execute with RestrictedPython
        if self.config.use_subprocess:
            # Execute in subprocess for maximum isolation
            status, output, error = _execute_in_subprocess(
                code=code,
                exec_globals=self._allowed_modules_globals,
                timeout_seconds=self.config.timeout_seconds,
                max_memory_mb=self.config.max_memory_mb,
                max_cpu_seconds=self.config.max_cpu_seconds,
            )
            return ExecutionResult(
                status=status,
                output=output,
                error_message=error,
                execution_time=time.time() - start_time,
            )
        else:
            # Execute in-process (faster but less isolated)
            return self._execute_in_process(code, start_time)

    def _execute_in_process(self, code: str, start_time: float) -> ExecutionResult:
        """Execute code in the current process with RestrictedPython."""
        # Set up timeout (Unix only)
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.config.timeout_seconds)
        except (ValueError, AttributeError):
            pass

        try:
            # Set resource limits
            _set_resource_limits(
                self.config.max_memory_mb,
                self.config.max_cpu_seconds,
            )

            # Compile with RestrictedPython (P0 Enhancement)
            compiled = compile_restricted(code, "<sandbox>", "exec")

            if compiled.errors:
                return ExecutionResult(
                    status=ExecutionStatus.SECURITY_VIOLATION,
                    error_message=f"RestrictedPython errors: {compiled.errors}",
                    execution_time=time.time() - start_time,
                )

            # Create restricted execution environment
            exec_globals = _create_restricted_globals(self._allowed_modules_globals)
            exec_locals: dict[str, Any] = {}

            # Execute the restricted bytecode
            exec(compiled.code, exec_globals, exec_locals)

            # Extract output
            output = _extract_serializable_output(exec_locals)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output,
                execution_time=time.time() - start_time,
            )

        except TimeoutException:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Execution timed out after {self.config.timeout_seconds} seconds",
                execution_time=time.time() - start_time,
            )
        except MemoryError:
            return ExecutionResult(
                status=ExecutionStatus.RESOURCE_EXCEEDED,
                error_message="Memory limit exceeded",
                execution_time=time.time() - start_time,
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error_message=f"{type(e).__name__}: {str(e)}",
                execution_time=time.time() - start_time,
            )
        finally:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (ValueError, AttributeError):
                pass

    def _import_allowed_modules(self) -> dict[str, Any]:
        """Import allowed modules and return as globals dict."""
        exec_globals: dict[str, Any] = {}

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
                elif module_name == "iqfmp.evaluation.qlib_stats":
                    from iqfmp.evaluation import qlib_stats
                    exec_globals["qlib_stats"] = qlib_stats
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
                logger.warning(f"Failed to import allowed module: {module_name}")

        return exec_globals
