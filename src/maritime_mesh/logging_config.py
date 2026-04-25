"""Shared logging configuration for maritime_mesh."""

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(processName)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO, force: bool = False) -> None:
    """Configure root logger with project defaults.

    If handlers already exist, keep them but still enforce the effective level.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers and not force:
        root_logger.setLevel(level)
        return
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT, force=force)
