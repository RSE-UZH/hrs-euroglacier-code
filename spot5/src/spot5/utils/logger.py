# -------------------------------------------------------------------------------
#
#  Python dual-logging setup (console and log file),supporting different log
#  levels and colorized output
#
#  Inpired from:
#  Fonic <https://github.com/fonic>
#  Date: 04/05/20
#
#  Based on:                                                                   -
#  https://stackoverflow.com/a/13733863/1976617
#  https://uran198.github.io/en/python/2016/07/12/colorful-python-logging.html
#  https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
#
# -------------------------------------------------------------------------------

import logging
import sys
from datetime import datetime
from pathlib import Path

# Constants
DEFAULT_FORMAT = "%(asctime)s | [%(levelname)-8s] %(message)s"
DEBUG_FORMAT = "%(asctime)s | [%(filename)s -> %(funcName)s], line %(lineno)d - [%(levelname)-8s] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ANSI color codes
COLORS = {
    "DEBUG": "\033[0;36m",  # Cyan
    "INFO": "\033[0;32m",  # Green
    "WARNING": "\033[0;33m",  # Yellow
    "ERROR": "\033[0;31m",  # Red
    "CRITICAL": "\033[0;37m\033[41m",  # White on Red
    "RESET": "\033[0m",  # Reset
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to levelname field."""

    def format(self, record):
        # Add color to levelname if running in terminal
        if sys.stderr.isatty():
            color = COLORS.get(record.levelname, COLORS["RESET"])
            record.levelname = f"{color}{record.levelname}{COLORS['RESET']}"
        return super().format(record)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get existing logger or create new one with default settings."""
    return (
        logging.getLogger(name)
        if logging.getLogger(name).handlers
        else setup_logger(name)
    )


def set_log_level(logger_name: str, level: str) -> logging.Logger:
    """Change logger level."""
    logger = logging.getLogger(logger_name)

    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)  # type: ignore

    logger.setLevel(level)

    return logger


def setup_logger(
    level: str | int | None = logging.INFO,
    name: str | None = None,
    log_to_file: bool = False,
    log_folder: Path | str | None = None,
) -> logging.Logger:
    """Setup and configure logger with color support.

    Args:
        name: Logger name (root logger if None)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_file: Whether to log to file
        log_folder: Path to logs directory (default: ./logs)

    Returns:
        Configured logger instance
    """

    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), logging.INFO)

    # Check if debug level is set
    debug = level == logging.DEBUG

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level=level)  # type: ignore

    # Clear existing handlers
    logger.handlers.clear()

    # Create console handler with color support
    console = logging.StreamHandler()
    console.setFormatter(
        ColoredFormatter(DEBUG_FORMAT if debug else DEFAULT_FORMAT, datefmt=DATE_FMT)
    )
    logger.addHandler(console)

    # Optional file logging
    if log_to_file:
        log_folder = Path(log_folder or "./logs")
        log_folder.mkdir(exist_ok=True, parents=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_name = f"log_{timestamp}.log"
        log_file = log_folder / log_name

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                DEBUG_FORMAT if debug else DEFAULT_FORMAT, datefmt=DATE_FMT
            )
        )
        logger.addHandler(file_handler)

    return logger
