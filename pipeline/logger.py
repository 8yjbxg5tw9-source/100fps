"""Centralised, timestamped terminal logging for the pipeline.

Format example::

    [2026-09-09 14:55:02] [INFO] CUDA detected: NVIDIA RTX 3080 (10.0 GB VRAM)
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: set[str] = set()


def get_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    """Return a logger that prints timestamped messages to stdout.

    Handlers are attached only once per logger name, so calling this from
    every pipeline step is safe (no duplicated log lines).
    """
    logger = logging.getLogger(name)
    if name not in _configured_loggers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        # Avoid double logging via the root logger.
        logger.propagate = False
        _configured_loggers.add(name)
    return logger
