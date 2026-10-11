"""
Logging utilities for PAPPO training scripts.

Provides configurable logging setup with both file and console handlers.
Supports capturing stdout, stderr, and exceptions to the log file.
"""

import os
import sys
import re
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


# Shared re-entrancy guard for StreamToLogger.write (see below). Module-level and
# thread-local: once a write() triggers logger.log(), any nested write caused by a
# handler routing that record back into a redirected stream is passed to the
# console only and NOT re-logged -- otherwise it recurses until RecursionError.
_stream_local = threading.local()


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    
    This allows capturing stdout/stderr output (including print statements) 
    and writing it to both the console and log file.
    
    Filters out tqdm progress bars and ANSI escape sequences.
    """
    
    def __init__(self, logger: logging.Logger, level: int = logging.INFO, original_stream=None):
        self.logger = logger
        self.level = level
        self.linebuf = ''
        self.original_stream = original_stream
    
    def _should_skip_line(self, line: str) -> bool:
        """
        Check if a line should be skipped (progress bar or empty).
        
        Args:
            line: The line to check
            
        Returns:
            True if the line should be skipped
        """
        line = line.strip()
        
        # Skip empty lines
        if not line:
            return True
        
        # Skip tqdm progress bar lines (contain %| or [00:00<?, ?it/s] patterns)
        if '%|' in line or '?it/s' in line or 'examples/s' in line:
            return True
        
        # Skip ANSI cursor control sequences (like [A, [B, [K)
        if re.match(r'^\[\d*[A-Z]$', line):
            return True
        
        # Skip lines that are just progress bar remnants
        if re.match(r'^[\s\|\d#%<>\/,.?:\-\[\]A-Za-z]*$', line) and ('%' in line or '|' in line):
            return True
        
        return False
    
    def _clean_ansi(self, text: str) -> str:
        """
        Remove ANSI escape sequences from text.
        
        Args:
            text: Text that may contain ANSI codes
            
        Returns:
            Clean text without ANSI codes
        """
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def write(self, buf: str):
        # Write to original stream (console) if available
        if self.original_stream is not None:
            self.original_stream.write(buf)
            self.original_stream.flush()
        
        # Re-entrancy guard: a log record emitted by logger.log() below can be
        # routed by a handler back into a redirected stream (this class). Logging
        # it again would recurse forever (RecursionError). While a write-triggered
        # log call is in progress, hand nested writes to the console only.
        if getattr(_stream_local, "in_log", False):
            return
        
        # Process each line
        for line in buf.rstrip().splitlines():
            # Clean ANSI sequences
            clean_line = self._clean_ansi(line)
            
            # Skip progress bars and other unwanted lines
            if self._should_skip_line(clean_line):
                continue
            
            # Log the clean line (guarded so handler write-backs don't recurse)
            _stream_local.in_log = True
            try:
                self.logger.log(self.level, clean_line.rstrip())
            finally:
                _stream_local.in_log = False
    
    def flush(self):
        if self.original_stream is not None:
            self.original_stream.flush()


# Module-level storage of the true original stdio streams, set once on first setup
_true_original_stdout = None
_true_original_stderr = None


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    Configure logging to both file and console.

    Captures stdout, stderr, and all print statements to the log file.
    Filters out tqdm progress bars and ANSI escape sequences.

    Idempotent: if called multiple times, only adds a new FileHandler
    to the existing logger without re-redirecting streams.

    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment for log file naming

    Returns:
        Configured logger instance
    """
    global _true_original_stdout, _true_original_stderr

    os.makedirs(log_dir, exist_ok=True)

    # Create log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("pappo_anchor_train")
    logger.setLevel(logging.INFO)

    # Check if we've already set up log forwarding — don't re-redirect streams
    already_set_up = (sys.stdout is not None and isinstance(sys.stdout, StreamToLogger))

    # Save the true original streams the first time
    if _true_original_stdout is None:
        if already_set_up:
            # First StreamToLogger holds the real original via its original_stream
            _true_original_stdout = sys.stdout.original_stream
            _true_original_stderr = sys.stderr.original_stream
        else:
            _true_original_stdout = sys.stdout
            _true_original_stderr = sys.stderr

    original_stdout = _true_original_stdout
    original_stderr = _true_original_stderr

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # File handler with detailed format
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler: always write to the TRUE original stderr
    # This prevents infinite recursion when stdout/stderr are already StreamToLogger
    console_handler = logging.StreamHandler(original_stderr)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Only redirect streams the first time
    if not already_set_up:
        stdout_logger = StreamToLogger(logger, logging.INFO, original_stdout)
        stderr_logger = StreamToLogger(logger, logging.ERROR, original_stderr)

        sys.stdout = stdout_logger
        sys.stderr = stderr_logger

    # Store references to restore later if needed
    logger._original_stdout = original_stdout
    logger._original_stderr = original_stderr

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def restore_stdio(logger: Optional[logging.Logger] = None):
    """
    Restore original stdout and stderr.
    
    Args:
        logger: The logger that was used to redirect output
    """
    if logger is not None:
        if hasattr(logger, '_original_stdout'):
            sys.stdout = logger._original_stdout
        if hasattr(logger, '_original_stderr'):
            sys.stderr = logger._original_stderr
    else:
        # Try to restore from any logger
        root_logger = logging.getLogger("pappo_anchor_train")
        if hasattr(root_logger, '_original_stdout'):
            sys.stdout = root_logger._original_stdout
        if hasattr(root_logger, '_original_stderr'):
            sys.stderr = root_logger._original_stderr


def get_logger(name: str = "pappo") -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
