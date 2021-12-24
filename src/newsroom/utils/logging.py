"""Logging configuration shared by the CLI, the API and library code."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

_LOGGER_NAME = "newsroom"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _ManagedHandler(logging.StreamHandler):
    """Marker base class so :func:`setup_logging` can replace only its own handlers."""


class _LazyStderrHandler(_ManagedHandler):
    """StreamHandler that always writes to the *current* ``sys.stderr``.

    Test runners (pytest capture, Typer's ``CliRunner``) swap ``sys.stderr`` and close the
    replacement afterwards; resolving the stream lazily avoids writing to a closed file.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> TextIO:  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value: TextIO) -> None:  # noqa: D401 - intentionally ignored
        return None


def setup_logging(level: str | int = "INFO", *, stream: TextIO | None = None) -> logging.Logger:
    """Configure the ``newsroom`` logger hierarchy.

    Calling this repeatedly is safe: existing handlers attached by this function are
    replaced rather than duplicated.

    Args:
        level: Logging level name or numeric level.
        stream: Optional explicit stream; by default the current ``sys.stderr`` is used.

    Returns:
        The root ``newsroom`` logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    numeric = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logger.setLevel(numeric)

    for existing in list(logger.handlers):
        if isinstance(existing, _ManagedHandler):
            logger.removeHandler(existing)

    handler: _ManagedHandler = (
        _ManagedHandler(stream) if stream is not None else _LazyStderrHandler()
    )
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``newsroom`` namespace."""
    if not name or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    if name.startswith(_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
