from abc import ABC, abstractmethod


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name, must be globally unique."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM to understand purpose."""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema parameter definition."""

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute tool and return text result."""
