"""Package-wide logging configuration.

Usage in package modules::

    from eeg_win_stack.tools.logging import get_logger
    logger = get_logger(__name__)

Usage in top-level scripts::

    from eeg_win_stack.tools.logging import configure_logging, get_logger
    configure_logging()
    logger = get_logger("eeg_win_stack.my_script")
"""

from __future__ import annotations

import logging

_PACKAGE = "eeg_win_stack"

# Prevents "No handlers could be found" when the package is used as a library
# without the caller having configured logging.
logging.getLogger(_PACKAGE).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped under the eeg_win_stack hierarchy."""
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    log_file: str | None = None,
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure console (and optionally file) output for the package logger.

    Call once at the top of a script before running experiments. Subsequent
    calls replace any previously added handlers.
    """
    root = logging.getLogger(_PACKAGE)
    root.setLevel(level)
    root.handlers.clear()

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
