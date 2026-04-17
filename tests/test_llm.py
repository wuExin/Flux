import json
from unittest.mock import MagicMock, patch

import pytest

from flux.llm import LLMClient, LLMResponse


class TestLLMClient:
    def _make_client(self):
        with patch("flux.llm.OpenAI"):
            return LLMClient(api_key="test-key", model="glm-4")

    def _mock_response(self, content="Hello", finish_reason="stop", tool_calls=None):
        """Create a mock API response."""
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls

        choice = MagicMock()
        choice.message = message
        choice.finish_reason = finish_reason

        response = MagicMock()
        response.choices = [choice]
        return response

    def test_simple_chat(self):
        client = self._make_client()
        mock_resp = self._mock_response(content="Hi there")
        client._client.chat.completions.create.return_value = mock_resp

        result = client.chat(messages=[{"role": "user", "content": "hello"}])

        assert isinstance(result, LLMResponse)
        assert result.content == "Hi there"
        assert result.finish_reason == "stop"
        assert result.tool_calls == []

    def test_tool_calls_parsed(self):
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = "bash"
        tc.function.arguments = json.dumps({"command": "ls"})

        client = self._make_client()
        mock_resp = self._mock_response(
            content="", finish_reason="tool_calls", tool_calls=[tc]
        )
        client._client.chat.completions.create.return_value = mock_resp

        result = client.chat(
            messages=[{"role": "user", "content": "list files"}],
            tools=[{"type": "function", "function": {"name": "bash"}}],
        )

        assert result.finish_reason == "tool_calls"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].arguments == {"command": "ls"}

    def test_system_message_prepended(self):
        client = self._make_client()
        mock_resp = self._mock_response()
        client._client.chat.completions.create.return_value = mock_resp

        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            system="You are helpful.",
        )

        call_kwargs = client._client.chat.completions.create.call_args
        msgs = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."

    def test_malformed_tool_arguments(self):
        tc = MagicMock()
        tc.id = "call_bad"
        tc.function.name = "bash"
        tc.function.arguments = "not valid json{{"

        client = self._make_client()
        mock_resp = self._mock_response(
            content="", finish_reason="tool_calls", tool_calls=[tc]
        )
        client._client.chat.completions.create.return_value = mock_resp

        result = client.chat(messages=[{"role": "user", "content": "test"}])
        assert result.tool_calls[0].arguments == {}
