"""Input sanitization utilities for security and stability."""

import logging
import re
from typing import Optional, Pattern

logger = logging.getLogger(__name__)

# Maximum input lengths
MAX_USER_MESSAGE_LENGTH = 10000  # 10K characters
MAX_QUESTION_LENGTH = 5000  # 5K characters for questions

# Pre-compiled patterns to detect and remove (compiled once at module load)
PROMPT_INJECTION_PATTERNS: list[Pattern] = [
    re.compile(r"```.*?```", re.DOTALL | re.IGNORECASE),  # Code blocks
    re.compile(r"<\|.*?\|>", re.DOTALL | re.IGNORECASE),  # Special tokens
    re.compile(r"###\s*System:", re.IGNORECASE),  # System prompts
    re.compile(r"###\s*Assistant:", re.IGNORECASE),  # Assistant prompts
]

# Pre-compiled PII patterns for redaction in logs
PII_PATTERNS: dict[str, tuple[Pattern, str]] = {
    "ssn": (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    "email": (re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"), "[EMAIL]"),
    "phone": (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE]"),
    "credit_card": (
        re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        "[CARD]",
    ),
}

# Pre-compiled pattern for whitespace normalization
WHITESPACE_PATTERN = re.compile(r"\s+")


def sanitize_user_input(text: str, max_length: int = MAX_USER_MESSAGE_LENGTH) -> str:
    """
    Sanitize user input to prevent prompt injection and limit length.

    This removes potentially dangerous patterns that could be used for
    prompt injection attacks while preserving legitimate user content.

    Args:
        text: Raw user input
        max_length: Maximum allowed length

    Returns:
        Sanitized text

    Raises:
        ValueError: If input is invalid or too long

    Example:
        >>> sanitize_user_input("Hello, how are you?")
        'Hello, how are you?'
        >>> sanitize_user_input("```python\\nprint('hack')```")
        ''  # Code blocks removed
    """
    if not isinstance(text, str):
        raise ValueError(f"Input must be string, got {type(text)}")

    if not text.strip():
        raise ValueError("Input cannot be empty")

    # Trim whitespace
    text = text.strip()

    # Check length before processing
    if len(text) > max_length:
        logger.warning(f"Input truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]

    # Remove potential prompt injection patterns (using pre-compiled patterns)
    for pattern in PROMPT_INJECTION_PATTERNS:
        text = pattern.sub("", text)

    # Remove excessive whitespace (using pre-compiled pattern)
    text = WHITESPACE_PATTERN.sub(" ", text)
    text = text.strip()

    if not text:
        raise ValueError("Input became empty after sanitization")

    return text


def sanitize_question(question: str) -> str:
    """
    Sanitize a question for council/LLM processing.

    Questions have stricter limits than general messages.

    Args:
        question: Raw question text

    Returns:
        Sanitized question

    Raises:
        ValueError: If question is invalid
    """
    return sanitize_user_input(question, max_length=MAX_QUESTION_LENGTH)


def redact_pii_for_logging(text: str, max_len: int = 200) -> str:
    """
    Redact PII from text for safe logging.

    This should be used whenever logging user input to prevent
    accidentally logging sensitive information.

    Args:
        text: Text that may contain PII
        max_len: Maximum length to log

    Returns:
        Redacted and truncated text safe for logging

    Example:
        >>> redact_pii_for_logging("My email is john@example.com")
        'My email is [EMAIL]'
        >>> redact_pii_for_logging("Call me at 555-123-4567")
        'Call me at [PHONE]'
    """
    if not isinstance(text, str):
        return str(text)[:max_len]

    # Redact PII (using pre-compiled patterns)
    for pii_type, (pattern, replacement) in PII_PATTERNS.items():
        text = pattern.sub(replacement, text)

    # Truncate
    if len(text) > max_len:
        text = text[:max_len] + "..."

    return text


def validate_user_id(user_id: str) -> str:
    """
    Validate and sanitize user ID.

    Args:
        user_id: User identifier

    Returns:
        Validated user ID

    Raises:
        ValueError: If user ID is invalid
    """
    if not isinstance(user_id, str):
        raise ValueError(f"User ID must be string, got {type(user_id)}")

    user_id = user_id.strip()

    if not user_id:
        raise ValueError("User ID cannot be empty")

    # Only allow alphanumeric, hyphens, and underscores
    if not re.match(r"^[\w\-]+$", user_id):
        raise ValueError(f"Invalid user ID format: {user_id}")

    if len(user_id) > 255:
        raise ValueError(f"User ID too long: {len(user_id)} characters")

    return user_id
