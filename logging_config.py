"""
Logging configuration for the hikrobot-opencv-plc-pipeline project.

Sets up structured logging with rotating file handlers for different log levels,
enabling easy debugging and issue diagnosis in production.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    name: str = "hikrobot_pipeline"
) -> logging.Logger:
    """Configure application logging with rotating file handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # --- DEBUG log file (all messages) ---
    debug_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "debug.log",
        maxBytes=10_000_000,  # 10 MB
        backupCount=5
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    debug_handler.setFormatter(debug_formatter)
    
    # --- ERROR log file (errors and critical only) ---
    error_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "errors.log",
        maxBytes=5_000_000,  # 5 MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s\n%(exc_info)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    error_handler.setFormatter(error_formatter)
    
    # --- INFO log file (info and above, but not debug) ---
    info_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "info.log",
        maxBytes=10_000_000,  # 10 MB
        backupCount=5
    )
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    info_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    info_handler.setFormatter(info_formatter)
    
    # --- Console handler (INFO and above) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)-8s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add all handlers
    logger.addHandler(debug_handler)
    logger.addHandler(error_handler)
    logger.addHandler(info_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = "hikrobot_pipeline") -> logging.Logger:
    """Get the configured logger instance.
    
    Args:
        name: Logger name (should match setup_logging name parameter)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
