"""
core/logging.py - Centralized Structured Logging Configuration

This module provides a unified logging setup using structlog for consistent,
production-grade logging across all services. It supports both human-readable
console output and JSON structured output for log aggregation systems.

Usage:
    from core.logging import get_logger
    log = get_logger(__name__)
    log.info("Trade executed", symbol="ETHBTC", side="BUY", qty=0.5)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from structlog.typing import Processor


def configure_logging(
    json_output: bool = False,
    log_level: str = "INFO",
    add_timestamp: bool = True,
) -> None:
    """
    Configure global structured logging.
    
    Args:
        json_output: If True, output JSON; otherwise human-readable.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        add_timestamp: Whether to add ISO timestamps to log entries.
    """
    # Get log level from environment or default
    level_str = os.getenv("LOGLEVEL", log_level).upper()
    level = getattr(logging, level_str, logging.INFO)
    
    # Determine output format from environment
    use_json = os.getenv("LOG_JSON", "false").lower() == "true" or json_output
    
    # Shared processors for all output types
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso") if add_timestamp else structlog.processors.TimeStamper(fmt=None),
        structlog.contextvars.merge_contextvars,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if use_json:
        # JSON output for production / log aggregation
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Human-readable console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            ),
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("binance").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__). If None, uses root logger.
    
    Returns:
        A bound structlog logger with consistent configuration.
    
    Example:
        log = get_logger(__name__)
        log.info("Order placed", order_id="12345", symbol="ETHBTC")
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables that will be included in all subsequent log entries.
    
    Useful for request-scoped or loop-scoped context like instance_name, symbol, etc.
    
    Example:
        bind_context(symbol="ETHBTC", mode="live")
        log.info("Starting loop")  # Will include symbol="ETHBTC", mode="live"
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()


# Auto-configure on import (can be overridden by calling configure_logging again)
configure_logging()
