"""IQFMP CLI - Command Line Interface for the trading system."""

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iqfmp.evaluation.lookahead_detector import DetectionResult

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="iqfmp",
    help="IQFMP - Intelligent Quantitative Factor Mining Platform CLI",
    add_completion=False,
)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    """Default command - start server if no subcommand provided."""
    if ctx.invoked_subcommand is None:
        # Backwards compatibility: `iqfmp` without args starts the server
        serve()


console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    reload: bool = typer.Option(True, help="Enable auto-reload"),
) -> None:
    """Start the IQFMP API server."""
    import uvicorn

    console.print(
        Panel.fit(
            f"[bold green]Starting IQFMP API Server[/bold green]\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"Reload: {reload}",
            title="IQFMP Server",
        )
    )

    uvicorn.run(
        "iqfmp.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def validate(
    factor: str | None = typer.Option(
        None, "--factor", "-f", help="Factor name or expression to validate"
    ),
    file: Path | None = typer.Option(
        None, "--file", help="Python file containing factor code"
    ),
    expression: str | None = typer.Option(
        None, "--expression", "-e", help="Qlib expression to check"
    ),
    check_lookahead: bool = typer.Option(
        True, "--check-lookahead/--no-check-lookahead", help="Check for lookahead bias"
    ),
    strict: bool = typer.Option(False, "--strict", help="Fail on any warning"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Validate factors for lookahead bias and other issues.

    Examples:

        # Check a Qlib expression
        iqfmp validate -e "Ref($close, -1) / $close"

        # Check a factor by name
        iqfmp validate -f momentum_20d

        # Check a Python file
        iqfmp validate --file factors/my_factor.py

        # Strict mode (exit with error on warnings)
        iqfmp validate -e "Ref($close, -1)" --strict
    """
    if not check_lookahead:
        console.print("[yellow]Lookahead bias checking disabled[/yellow]")
        return

    from iqfmp.evaluation.lookahead_detector import LookaheadBiasDetector

    detector = LookaheadBiasDetector()
    issues_found = False
    warnings_found = False

    console.print(
        Panel.fit(
            "[bold blue]Lookahead Bias Detection[/bold blue]\n"
            "Scanning for potential data leakage issues...",
            title="IQFMP Validate",
        )
    )

    # Check Qlib expression
    if expression:
        console.print(f"\n[bold]Checking expression:[/bold] {expression}")
        result = detector.check_qlib_expression(expression)
        has_critical, has_warn = _print_detection_result(result, verbose)
        if has_critical:
            issues_found = True
        if has_warn:
            warnings_found = True

    # Check Python file
    if file:
        if not file.exists():
            console.print(f"[red]Error: File not found: {file}[/red]")
            raise typer.Exit(1)

        console.print(f"\n[bold]Checking file:[/bold] {file}")
        try:
            code = file.read_text(encoding="utf-8")
        except OSError as e:
            console.print(f"[red]Error reading file: {e}[/red]")
            logger.error("Failed to read file %s: %s", file, e)
            raise typer.Exit(1)
        except UnicodeDecodeError as e:
            console.print(f"[red]Error: File is not valid UTF-8: {e}[/red]")
            logger.error("Unicode decode error for file %s: %s", file, e)
            raise typer.Exit(1)

        result = detector.check_python_code(code)
        has_critical, has_warn = _print_detection_result(result, verbose)
        if has_critical:
            issues_found = True
        if has_warn:
            warnings_found = True

    # Check factor by name
    if factor:
        console.print(f"\n[bold]Checking factor:[/bold] {factor}")
        # Factor registry is not yet implemented - treat as Qlib expression
        console.print(
            f"[yellow]Note: Factor registry not yet implemented. "
            f"Treating '{factor}' as a Qlib expression.[/yellow]"
        )
        result = detector.check_qlib_expression(factor)
        has_critical, has_warn = _print_detection_result(result, verbose)
        if has_critical:
            issues_found = True
        if has_warn:
            warnings_found = True

    # If no input provided, show help
    if not expression and not file and not factor:
        console.print(
            "[yellow]No input provided. Use --help for usage information.[/yellow]"
        )
        raise typer.Exit(0)

    # Summary
    console.print("\n" + "=" * 50)
    if issues_found:
        console.print("[bold red]FAILED: Lookahead bias detected![/bold red]")
        raise typer.Exit(1)
    elif warnings_found and strict:
        console.print("[bold yellow]FAILED: Warnings found (strict mode)[/bold yellow]")
        raise typer.Exit(1)
    elif warnings_found:
        console.print("[bold yellow]PASSED with warnings[/bold yellow]")
    else:
        console.print("[bold green]PASSED: No lookahead bias detected[/bold green]")


def _print_detection_result(
    result: "DetectionResult", verbose: bool
) -> tuple[bool, bool]:
    """Print detection result in a formatted way.

    Args:
        result: The detection result from LookaheadBiasDetector containing
            instances of detected bias issues with severity levels.
        verbose: If True, prints additional details including evidence and
            suggested fixes for each issue.

    Returns:
        Tuple of (has_critical_issues, has_warnings) where each boolean
        indicates if that severity level was found in the results.
    """
    from iqfmp.evaluation.lookahead_detector import SeverityLevel

    critical_issues = [
        i for i in result.instances if i.severity == SeverityLevel.CRITICAL
    ]
    warnings = [i for i in result.instances if i.severity == SeverityLevel.WARNING]
    infos = [i for i in result.instances if i.severity == SeverityLevel.INFO]

    if critical_issues:
        console.print("[red]  BIAS DETECTED[/red]")
        for issue in critical_issues:
            console.print(
                f"    [red]- [{issue.bias_type.value}] {issue.description}[/red]"
            )
            if verbose:
                console.print(f"      [dim]Evidence: {issue.evidence}[/dim]")
                console.print(f"      [dim]Fix: {issue.suggestion}[/dim]")
    elif warnings:
        console.print("[yellow]  WARNINGS[/yellow]")
        for warning in warnings:
            console.print(
                f"    [yellow]- [{warning.bias_type.value}] {warning.description}[/yellow]"
            )
            if verbose:
                console.print(f"      [dim]Evidence: {warning.evidence}[/dim]")
                console.print(f"      [dim]Fix: {warning.suggestion}[/dim]")
    else:
        console.print("[green]  OK[/green]")

    if verbose and infos:
        console.print("  [dim]Info:[/dim]")
        for info in infos:
            console.print(f"    [dim]- {info.description}[/dim]")

    return len(critical_issues) > 0, len(warnings) > 0


@app.command()
def info() -> None:
    """Show system information and configuration."""
    import platform

    from iqfmp import __version__

    table = Table(title="IQFMP System Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Version
    table.add_row("Version", __version__, "")

    # Python version
    table.add_row("Python", platform.python_version(), sys.executable)

    # Check database configuration
    pg_host = os.environ.get("PGHOST", "localhost")
    pg_port = os.environ.get("PGPORT", "5433")
    table.add_row("PostgreSQL", "Configured", f"{pg_host}:{pg_port}")

    # Check Redis configuration
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = os.environ.get("REDIS_PORT", "6379")
    table.add_row("Redis", "Configured", f"{redis_host}:{redis_port}")

    # Check Qlib
    try:
        from iqfmp.core.qlib_init import is_qlib_initialized

        status = "Initialized" if is_qlib_initialized() else "Not initialized"
        table.add_row("Qlib", status, "")
    except ImportError:
        table.add_row("Qlib", "Not installed", "")
        logger.debug("Qlib module not available")
    except RuntimeError as e:
        table.add_row("Qlib", "Error", str(e)[:50])
        logger.warning("Qlib initialization check failed: %s", e)

    # Check PyTorch
    try:
        import torch

        cuda = "CUDA" if torch.cuda.is_available() else "CPU"
        table.add_row("PyTorch", torch.__version__, cuda)
    except ImportError:
        table.add_row("PyTorch", "Not installed", "")

    console.print(table)


def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
