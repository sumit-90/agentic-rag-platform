from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> Any:
        """Execute the agent for the given query and return a structured response."""
