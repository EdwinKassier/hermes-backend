"""Registry for Legion strategies."""

import logging
from typing import Callable, Dict, List, Optional, Type, Union

from .base import LegionStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Registry for managing available Legion strategies with lazy initialization support."""

    _instance = None
    _strategies: Dict[str, LegionStrategy] = {}
    _factories: Dict[str, Callable[[], LegionStrategy]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StrategyRegistry, cls).__new__(cls)
        return cls._instance

    def register(
        self, name: str, strategy: Union[LegionStrategy, Callable[[], LegionStrategy]]
    ) -> None:
        """
        Register a new strategy or strategy factory.

        Args:
            name: Unique name for the strategy
            strategy: Strategy implementation or factory function that creates one
        """
        if callable(strategy) and not isinstance(strategy, LegionStrategy):
            # It's a factory function - store for lazy initialization
            self._factories[name] = strategy
            logger.info(f"Registered Legion strategy factory: {name}")
        else:
            # It's an instance - store directly
            self._strategies[name] = strategy
            logger.info(f"Registered Legion strategy: {name}")

    def get(self, name: str) -> Optional[LegionStrategy]:
        """
        Get a strategy by name, lazily initializing if needed.

        Args:
            name: Name of the strategy

        Returns:
            The strategy instance or None if not found
        """
        # Check if already instantiated
        if name in self._strategies:
            return self._strategies[name]

        # Check if we have a factory for it
        if name in self._factories:
            logger.debug(f"Lazy-initializing strategy: {name}")
            strategy = self._factories[name]()
            self._strategies[name] = strategy
            return strategy

        return None

    def list_available(self) -> List[str]:
        """List names of all registered strategies."""
        return list(self._strategies.keys())


# Global registry accessor
def get_strategy_registry() -> StrategyRegistry:
    """Get the singleton strategy registry."""
    return StrategyRegistry()
