"""Input validation for Legion persona system."""

import logging

logger = logging.getLogger(__name__)


class PersonaInputValidator:
    """Validates inputs for the persona generation system."""

    def validate_role(self, role: str) -> str:
        """Validate and sanitize role input."""
        if not isinstance(role, str):
            raise ValueError(f"Role must be a string, got {type(role).__name__}")

        # Sanitize input
        sanitized = role.strip()

        # Check length constraints
        if len(sanitized) == 0:
            raise ValueError("Role cannot be empty")
        if len(sanitized) > 100:
            raise ValueError("Role too long (max 100 characters)")

        # Check for potentially malicious patterns
        if any(char in sanitized for char in ["<", ">", "&", '"', "'"]):
            raise ValueError("Role contains invalid characters")

        return sanitized

    def validate_task_description(self, task_description: str) -> str:
        """Validate and sanitize task description input."""
        if not isinstance(task_description, str):
            # Convert to string if possible
            task_description = str(task_description)

        # Sanitize input
        sanitized = task_description.strip()

        # Truncate if too long (to prevent excessive processing)
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length] + "..."
            logger.warning(f"Task description truncated to {max_length} characters")

        return sanitized
