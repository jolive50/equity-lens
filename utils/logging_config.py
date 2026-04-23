"""Centralized logging configuration for Equity Lens.

Provides file-based logging with rotation, structured formatting, and LangSmith tracing.
"""
import json
import logging
import logging.handlers
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Optional

# Context variables for tracking execution flow
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
workflow_stage_var: ContextVar[Optional[str]] = ContextVar('workflow_stage', default=None)
component_var: ContextVar[Optional[str]] = ContextVar('component', default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with context."""
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context variables
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        workflow_stage = workflow_stage_var.get()
        if workflow_stage:
            log_data["workflow_stage"] = workflow_stage

        component = component_var.get()
        if component:
            log_data["component"] = component

        # Add extra fields from record
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ContextualFormatter(logging.Formatter):
    """Enhanced text formatter with context tracking."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with context information."""
        # Build context prefix
        context_parts = []

        request_id = request_id_var.get()
        if request_id:
            context_parts.append(f"req={request_id[:8]}")

        workflow_stage = workflow_stage_var.get()
        if workflow_stage:
            context_parts.append(f"stage={workflow_stage}")

        component = component_var.get()
        if component:
            context_parts.append(f"comp={component}")

        context_str = f"[{' | '.join(context_parts)}] " if context_parts else ""

        # Format the base message
        base_msg = super().format(record)

        # Add context prefix if present
        if context_str:
            parts = base_msg.split(" - ", 3)
            if len(parts) >= 4:
                return f"{parts[0]} - {parts[1]} - {parts[2]} - {context_str}{parts[3]}"

        return base_msg


def setup_logging(
    name: str = "equity_lens",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    use_json: bool = False,
    enable_langchain_tracing: bool = True
) -> logging.Logger:
    """Configure file-based logging with rotation and optional structured formatting.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level (default INFO)
        max_bytes: Max file size before rotation (default 10MB)
        backup_count: Number of backup files to keep (default 5)
        use_json: Enable JSON structured logging (default False)
        enable_langchain_tracing: Enable LangSmith tracing (default True)

    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Allow environment override for verbosity (e.g., EQUITY_LENS_LOG_LEVEL=DEBUG)
    env_level = os.getenv("EQUITY_LENS_LOG_LEVEL")
    if env_level:
        log_level = getattr(logging, env_level.upper(), log_level)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # File handler with rotation (always detailed)
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

    # Choose formatter based on mode
    if use_json:
        file_formatter = StructuredFormatter(datefmt='%Y-%m-%d %H:%M:%S')
        console_formatter = StructuredFormatter(datefmt='%Y-%m-%d %H:%M:%S')

        # Also create JSON log file
        json_handler = logging.handlers.RotatingFileHandler(
            filename=log_path / f"{name}.json",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        json_handler.setLevel(log_level)
        json_handler.setFormatter(file_formatter)
        logger.addHandler(json_handler)
    else:
        file_formatter = ContextualFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = ContextualFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Configure LangChain tracing if enabled
    if enable_langchain_tracing:
        _configure_langchain_tracing(logger)

    return logger


def _configure_langchain_tracing(logger: logging.Logger) -> None:
    """Configure LangSmith tracing for LangChain/LangGraph operations."""
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    if langchain_tracing and langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "equity-lens")

        logger.info("✅ LangSmith tracing enabled")
        logger.info(f"   Project: {os.getenv('LANGCHAIN_PROJECT')}")
    elif langchain_tracing and not langchain_api_key:
        logger.warning("⚠️  LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY not set")
        logger.warning("   LangSmith tracing will be disabled")
    else:
        logger.info("ℹ️  LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with standard configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    # Check if root equity_lens logger is configured
    root_logger = logging.getLogger("equity_lens")
    if not root_logger.handlers:
        setup_logging()

    # Return child logger
    return logging.getLogger(f"equity_lens.{name}")


def set_request_context(request_id: str, stage: Optional[str] = None, component: Optional[str] = None) -> None:
    """Set context variables for request tracking.

    Args:
        request_id: Unique request identifier
        stage: Current workflow stage (e.g., 'validate', 'fetch_data', 'predict')
        component: Component name (e.g., 'PredictionAgent', 'ChromaDB')
    """
    request_id_var.set(request_id)
    if stage:
        workflow_stage_var.set(stage)
    if component:
        component_var.set(component)


def clear_request_context() -> None:
    """Clear all context variables."""
    request_id_var.set(None)
    workflow_stage_var.set(None)
    component_var.set(None)


def log_with_timing(logger: logging.Logger, level: int, message: str, start_time: float, **extra: Any) -> None:
    """Log message with elapsed time.

    Args:
        logger: Logger instance
        level: Log level (e.g., logging.INFO)
        message: Log message
        start_time: Start time from time.time()
        **extra: Additional fields to log
    """
    elapsed = time.time() - start_time
    extra_data = {"elapsed_seconds": round(elapsed, 3), **extra}

    # Create log record with extra data
    record = logger.makeRecord(
        logger.name,
        level,
        "(unknown file)",
        0,
        f"{message} ({elapsed:.3f}s)",
        (),
        None
    )
    record.extra_data = extra_data
    logger.handle(record)


def log_data_flow(
    logger: logging.Logger,
    source: str,
    destination: str,
    data_type: str,
    data_size: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log data flow between components.

    Args:
        logger: Logger instance
        source: Source component name
        destination: Destination component name
        data_type: Type of data being transferred
        data_size: Size of data (rows, articles, bytes, etc.)
        metadata: Additional metadata about the data
    """
    msg_parts = [f"📊 DATA FLOW: {source} → {destination}"]
    msg_parts.append(f"Type: {data_type}")

    if data_size is not None:
        msg_parts.append(f"Size: {data_size}")

    if metadata:
        msg_parts.append(f"Metadata: {metadata}")

    logger.info(" | ".join(msg_parts))
