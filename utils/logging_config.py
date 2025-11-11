"""Centralized logging configuration for FreshStart.

Provides file-based logging with rotation and structured formatting.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "freshstart",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """Configure file-based logging with rotation.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level (default INFO)
        max_bytes: Max file size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)

    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path / f"{name}.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Detailed format with timestamp, logger name, level, and message
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with standard configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Check if root freshstart logger is configured
    root_logger = logging.getLogger("freshstart")
    if not root_logger.handlers:
        setup_logging()

    # Return child logger
    return logging.getLogger(f"freshstart.{name}")
