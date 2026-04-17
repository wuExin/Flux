from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ToolCall:
    """LLM-returned tool call request."""
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """Tool execution result to send back to LLM."""
    tool_call_id: str
    content: str


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
