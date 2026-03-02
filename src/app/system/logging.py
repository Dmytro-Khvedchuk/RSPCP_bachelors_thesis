"""Loguru logging configuration.

Provides a single ``setup_logging`` entry-point that must be called once at
application startup.  All stdlib ``logging`` records (SQLAlchemy, Alembic, …)
are intercepted and routed through loguru so every log line shares the same
format and sinks.
"""

from __future__ import annotations

import logging
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Forward stdlib ``logging`` records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: PLR6301
        # Map stdlib level to loguru level name.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the originating frame so loguru reports the right caller.
        frame, depth = logging.currentframe(), 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    *,
    level: str = "DEBUG",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure loguru sinks and intercept stdlib logging.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, …).
        json_output: If *True*, emit JSON-serialised records to *stdout*
            (useful in production / structured-log pipelines).
        log_file: Optional path for a rotating file sink (10 MB, 7-day
            retention).
    """
    # Remove default loguru sink so we start from a clean slate.
    logger.remove()

    # Console sink — coloured for dev, JSON for production.
    if json_output:
        logger.add(sys.stdout, level=level, serialize=True)
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

    # Optional rotating file sink.
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="gz",
        )

    # Intercept stdlib logging → loguru.
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Ensure known noisy loggers are captured.
    for name in ("sqlalchemy.engine", "alembic"):
        stdlib_logger = logging.getLogger(name)
        stdlib_logger.handlers = [_InterceptHandler()]
        stdlib_logger.propagate = False

    logger.debug("Loguru logging initialised (level={})", level)
