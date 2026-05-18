"""
logger.py — Central logging configuration for the Flight Fare pipeline.

Call setup_logging() once at application entry (main.py, DAG task, notebook)
before importing any pipeline module.  Every module then gets its own named
logger via logging.getLogger(__name__) — no further configuration needed.

Log format:  2025-01-15 02:00:00 | INFO     | src.data_loader        | message
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: Path | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Configure the root logger with a consistent format.
    If log_dir is provided, a 'pipeline.log' file handler is added and the
    directory is created if it does not exist.
    """
    fmt     = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Force UTF-8 on the console stream — Windows defaults to cp1252 which
    # cannot encode the box-drawing and tick characters used in log messages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "pipeline.log", mode="a", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,   # override any handlers already attached (e.g. Jupyter's)
    )

    # Suppress verbose third-party loggers
    for noisy in ("lightgbm", "xgboost", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
