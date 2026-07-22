from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import QuerySpec, TraversalResult


class BackendExecutor(ABC):
    system_name: str

    @abstractmethod
    def execute(self, query: QuerySpec) -> TraversalResult:
        raise NotImplementedError

    @abstractmethod
    def apply_write(self, query: QuerySpec) -> None:
        raise NotImplementedError

    def clear_application_cache(self) -> None:
        """Optional: used only when an experiment declares application-cold state."""

    def close(self) -> None:
        """Optional resource cleanup."""
