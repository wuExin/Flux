import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from .message import ToolCall

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM call response."""
    content: str
    finish_reason: str  # "stop" | "tool_calls"
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: object = None


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.model = model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://open.bigmodel.cn/api/coding/paas/v4",
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """Call GLM Chat Completions API."""
        api_messages = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def _parse_response(self, response) -> LLMResponse:
        """Parse API response into LLMResponse."""
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason or "stop"

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        return LLMResponse(
            content=message.content or "",
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw=response,
        )
