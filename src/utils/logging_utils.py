"""
logging_utils.py
Simple logger setup used across modules.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name="alphon", logfile="reports/pipeline.log", level=logging.INFO):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = RotatingFileHandler(logfile, maxBytes=5_000_000, backupCount=3)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger
