"""
Utility functions for jpgovsummary
"""

import os
from urllib.parse import urlparse


def is_local_file(path: str) -> bool:
    """
    Check if the given path is a local file path.

    Args:
        path (str): Path to check

    Returns:
        bool: True if it's a local file path, False otherwise
    """
    # Check for file:// protocol
    if path.startswith("file://"):
        return True

    # Check for absolute path (starts with /)
    if path.startswith("/"):
        return True

    # Check for relative path (doesn't start with http:// or https://)
    if not path.startswith(("http://", "https://")):
        return True

    return False


def get_local_file_path(path: str) -> str:
    """
    Convert a local file path (including file:// URLs) to a regular file path.

    Args:
        path (str): Local file path or file:// URL

    Returns:
        str: Regular file path
    """
    if path.startswith("file://"):
        parsed = urlparse(path)
        return parsed.path
    else:
        return path


def validate_local_file(file_path: str) -> None:
    """
    Validate that a local file exists and is readable.

    Args:
        file_path (str): Path to the file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is not a file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
