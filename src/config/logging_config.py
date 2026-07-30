"""
Centralized logging configuration for ResearchRAG.

This module provides a single, reusable logging setup used consistently
across every component of the system (loaders, chunkers, embedders,
vector store, retriever, generation layer, API, and frontend). It reads
its behavior from the global application settings so that log level and
log location can be controlled through environment configuration rather
than being hardcoded in individual modules.

Usage:
    from src.config.logging_config import get_logger

    logger = get_logger(__name__)
    logger.info("Loading documents")
    logger.warning("Unsupported file skipped")
    logger.error("Embedding generation failed")
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.settings import settings

# Log format: timestamp | level | module | function:line | message
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Rotating file handler configuration. These bounds keep a single log
# file from growing without limit while preserving recent history,
# without requiring any change to calling code if adjusted later.
_MAX_LOG_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_LOG_FILE_COUNT = 5

_LOG_FILE_NAME = "researchrag.log"


def _resolve_log_level() -> int:
    """
    Determine the effective log level from application settings.

    Returns:
        The numeric logging level, defaulting to DEBUG when the
        application is running in debug mode and INFO otherwise.
    """
    return logging.DEBUG if settings.app.debug else logging.INFO


def _ensure_log_directory(log_dir: Path) -> None:
    """
    Ensure the configured log directory exists on disk.

    Args:
        log_dir: Directory where log files should be written.

    Raises:
        OSError: If the directory cannot be created due to a filesystem
            or permissions issue.
    """
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise OSError(f"Unable to create log directory '{log_dir}': {error}") from error


def _build_console_handler(formatter: logging.Formatter) -> logging.Handler:
    """
    Create a console (stream) handler for logging to stdout.

    Args:
        formatter: The formatter applied to emitted log records.

    Returns:
        A configured StreamHandler instance.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    return handler


def _build_file_handler(formatter: logging.Formatter, log_dir: Path) -> logging.Handler:
    """
    Create a rotating file handler for persisting logs to disk.

    Args:
        formatter: The formatter applied to emitted log records.
        log_dir: Directory in which the log file will be created.

    Returns:
        A configured RotatingFileHandler instance.
    """
    log_file_path = log_dir / _LOG_FILE_NAME
    handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=_MAX_LOG_FILE_BYTES,
        backupCount=_BACKUP_LOG_FILE_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a configured logger for the given module name.

    The returned logger writes formatted records to both the console
    and a rotating log file, with its level determined by the global
    application settings. Calling this function multiple times with the
    same name is safe: handlers are attached only once per logger.

    Args:
        name: The name of the requesting module, typically passed as
            `__name__` from the calling file.

    Returns:
        A `logging.Logger` instance ready for use.
    """
    logger = logging.getLogger(name)

    # Avoid attaching duplicate handlers if this logger has already
    # been configured, which can otherwise happen when a module is
    # imported multiple times (e.g. in tests or reload scenarios).
    if logger.handlers:
        return logger

    log_level = _resolve_log_level()
    logger.setLevel(log_level)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    log_dir = settings.paths.logs_dir
    _ensure_log_directory(log_dir)

    logger.addHandler(_build_console_handler(formatter))
    logger.addHandler(_build_file_handler(formatter, log_dir))

    # Prevent records from propagating to the root logger, which would
    # otherwise result in duplicate output if the root logger also has
    # handlers configured.
    logger.propagate = False

    return logger