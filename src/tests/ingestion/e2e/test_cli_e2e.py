"""End-to-end tests for the ingestion CLI via ``typer.testing.CliRunner``.

These tests invoke the full CLI command with mocked internal services to
validate the Typer argument parsing, option wiring, and exit-code behaviour.

Currently a placeholder — CliRunner smoke tests will be added as the CLI
surface stabilises.
"""

from __future__ import annotations
