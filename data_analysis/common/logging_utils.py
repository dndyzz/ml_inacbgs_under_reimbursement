"""Logging seragam untuk semua pipeline (rapi di terminal maupun notebook)."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def banner(logger: logging.Logger, text: str) -> None:
    """Garis pemisah antartahap supaya log panjang mudah dibaca."""
    line = "=" * 68
    logger.info(line)
    logger.info(text.upper())
    logger.info(line)
