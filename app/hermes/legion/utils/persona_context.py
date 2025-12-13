"""Context manager for Legion persona usage."""

import logging
from contextvars import ContextVar
from typing import Optional

logger = logging.getLogger(__name__)

# Context variable to track current Legion persona
_current_legion_persona: ContextVar[Optional[str]] = ContextVar(
    "current_legion_persona", default=None
)


class LegionPersonaContext:
    """Context manager for Legion persona usage."""

    def __init__(self, persona: str):
        self.persona = persona
        self.token = None

    def __enter__(self):
        self.token = _current_legion_persona.set(self.persona)
        logger.debug(f"Set Legion persona context to: {self.persona}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_legion_persona.reset(self.token)
        logger.debug(f"Reset Legion persona context")


def get_current_legion_persona() -> str:
    """Get the current Legion persona, defaulting to 'legion'."""
    persona = _current_legion_persona.get()
    if persona is None:
        logger.debug("No Legion persona context set, using default 'legion'")
        return "legion"
    return persona


def use_legion_persona(persona: str):
    """Decorator to set Legion persona for a function."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            with LegionPersonaContext(persona):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_legion_persona(persona: str):
    """Context manager wrapper for Legion persona usage."""
    return LegionPersonaContext(persona)
