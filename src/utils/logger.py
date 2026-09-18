"""Project-wide logging setup.

Every module obtains a named logger via get_logger(__name__). Output
goes to both stdout and a rotating log file (logs/pipeline.log) so
batch runs can be audited after the fact.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import LOG_DIR, LOG_LEVEL

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_DIR / "pipeline.log", maxBytes=2_000_000, backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger (pass __name__)."""
    _configure_root_logger()
    return logging.getLogger(name)
