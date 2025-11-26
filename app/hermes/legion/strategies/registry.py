"""Registry for Legion strategies."""

import logging
from typing import Dict, List, Optional, Type

from .base import LegionStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Registry for managing available Legion strategies."""

    _instance = None
    _strategies: Dict[str, LegionStrategy] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StrategyRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, name: str, strategy: LegionStrategy) -> None:
        """
        Register a new strategy.

        Args:
            name: Unique name for the strategy
            strategy: Strategy implementation
        """
        self._strategies[name] = strategy
        logger.info(f"Registered Legion strategy: {name}")

    def get(self, name: str) -> Optional[LegionStrategy]:
        """
        Get a strategy by name.

        Args:
            name: Name of the strategy

        Returns:
            The strategy instance or None if not found
        """
        return self._strategies.get(name)

    def list_available(self) -> List[str]:
        """List names of all registered strategies."""
        return list(self._strategies.keys())


# Global registry accessor
def get_strategy_registry() -> StrategyRegistry:
    """Get the singleton strategy registry."""
    return StrategyRegistry()
