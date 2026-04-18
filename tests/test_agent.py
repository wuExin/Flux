from unittest.mock import MagicMock

import pytest

from flux.agent import Agent
from flux.llm import LLMResponse
from flux.message import ToolCall
from flux.tools.bash import BashTool
from flux.tools.registry import ToolRegistry
from flux.tools.todo import TodoState, TodoTool


def make_agent(llm_responses: list[LLMResponse], max_iterations=50) -> Agent:
    """Helper: create Agent with mocked LLM returning predetermined responses."""
    llm = MagicMock()
    llm.chat.side_effect = llm_responses

    registry = ToolRegistry()
    registry.register(BashTool())

    return Agent(llm=llm, tools=registry, system_prompt="test", max_iterations=max_iterations)


class TestAgent:
    def test_simple_text_response(self):
        agent = make_agent([
            LLMResponse(content="Hello!", finish_reason="stop"),
        ])
        result = agent.run("hi")
        assert result == "Hello!"
        assert len(agent.messages) == 2  # user + assistant

    def test_tool_call_then_response(self):
        agent = make_agent([
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "echo hi"})],
            ),
            LLMResponse(content="Done!", finish_reason="stop"),
        ])
        result = agent.run("run echo")
        assert result == "Done!"
        # user + assistant(tool_call) + tool_result + assistant(final)
        assert len(agent.messages) == 4

    def test_unknown_tool(self):
        agent = make_agent([
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c1", name="unknown_tool", arguments={})],
            ),
            LLMResponse(content="Sorry", finish_reason="stop"),
        ])
        result = agent.run("test")
        # The tool result should contain error about unknown tool
        tool_msg = agent.messages[2]
        assert tool_msg["role"] == "tool"
        assert "unknown tool" in tool_msg["content"].lower()

    def test_max_iterations(self):
        # Agent that always calls tools — should hit iteration limit
        responses = [
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id=f"c{i}", name="bash", arguments={"command": "echo loop"})],
            )
            for i in range(5)
        ]
        agent = make_agent(responses, max_iterations=3)
        result = agent.run("loop forever")
        assert "maximum iteration limit" in result.lower()

    def test_callbacks_called_on_tool_use(self):
        on_tool_call = MagicMock()
        on_tool_result = MagicMock()

        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "echo cb"})],
            ),
            LLMResponse(content="ok", finish_reason="stop"),
        ]
        registry = ToolRegistry()
        registry.register(BashTool())
        agent = Agent(
            llm=llm, tools=registry, system_prompt="test",
            on_tool_call=on_tool_call, on_tool_result=on_tool_result,
        )

        agent.run("test callbacks")
        on_tool_call.assert_called_once_with("bash", {"command": "echo cb"})
        on_tool_result.assert_called_once()
        # result should be the string output of `echo cb`
        result_arg = on_tool_result.call_args[0][0]
        assert isinstance(result_arg, str)

    def test_callbacks_not_called_without_tools(self):
        on_tool_call = MagicMock()
        on_tool_result = MagicMock()

        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResponse(content="Hi", finish_reason="stop"),
        ]
        registry = ToolRegistry()
        registry.register(BashTool())
        agent = Agent(
            llm=llm, tools=registry, system_prompt="test",
            on_tool_call=on_tool_call, on_tool_result=on_tool_result,
        )

        agent.run("hello")
        on_tool_call.assert_not_called()
        on_tool_result.assert_not_called()

    def test_callbacks_count_multiple_tools(self):
        on_tool_call = MagicMock()
        on_tool_result = MagicMock()

        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(id="c1", name="bash", arguments={"command": "echo a"}),
                    ToolCall(id="c2", name="bash", arguments={"command": "echo b"}),
                ],
            ),
            LLMResponse(content="done", finish_reason="stop"),
        ]
        registry = ToolRegistry()
        registry.register(BashTool())
        agent = Agent(
            llm=llm, tools=registry, system_prompt="test",
            on_tool_call=on_tool_call, on_tool_result=on_tool_result,
        )

        agent.run("two commands")
        assert on_tool_call.call_count == 2
        assert on_tool_result.call_count == 2

    def test_multiple_tool_calls_in_one_response(self):
        agent = make_agent([
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(id="c1", name="bash", arguments={"command": "echo a"}),
                    ToolCall(id="c2", name="bash", arguments={"command": "echo b"}),
                ],
            ),
            LLMResponse(content="Both done", finish_reason="stop"),
        ])
        result = agent.run("run two commands")
        assert result == "Both done"
        # user + assistant(2 tool_calls) + 2 tool_results + assistant(final)
        assert len(agent.messages) == 5


class TestAgentTodoIntegration:
    def test_nag_reminder_injected(self):
        """Verify nag reminder is injected after N iterations without todo calls."""
        state = TodoState()
        state.add("Test task", "")
        registry = ToolRegistry()
        registry.register(TodoTool(state))

        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResponse(content="response", finish_reason="stop"),
        ]

        agent = Agent(
            llm=llm,
            tools=registry,
            system_prompt="test",
            todo_state=state,
            nag_threshold=3,
        )

        # Advance past nag threshold
        for _ in range(4):
            state.advance_iteration()

        assert state.should_nag(threshold=3)

    def test_todo_call_resets_nag(self):
        """Verify todo call resets nag counter."""
        state = TodoState()
        state.add("Test task", "")
        registry = ToolRegistry()
        registry.register(TodoTool(state))

        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResponse(content="response", finish_reason="stop"),
        ]

        agent = Agent(
            llm=llm,
            tools=registry,
            system_prompt="test",
            todo_state=state,
            nag_threshold=3,
        )

        state.advance_iteration()
        state.advance_iteration()
        state.record_todo_call()
        state.advance_iteration()

        assert not state.should_nag(threshold=3)

    def test_nag_message_injected_during_run(self):
        """Verify nag reminder message is added to messages during run."""
        state = TodoState()
        state.add("Test task", "")
        registry = ToolRegistry()
        registry.register(BashTool())
        registry.register(TodoTool(state))

        llm = MagicMock()
        # Create enough responses to trigger nag (threshold=3)
        llm.chat.side_effect = [
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "echo 1"})],
            ),
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c2", name="bash", arguments={"command": "echo 2"})],
            ),
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c3", name="bash", arguments={"command": "echo 3"})],
            ),
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[ToolCall(id="c4", name="bash", arguments={"command": "echo 4"})],
            ),
            LLMResponse(content="done", finish_reason="stop"),
        ]

        agent = Agent(
            llm=llm,
            tools=registry,
            system_prompt="test",
            todo_state=state,
            nag_threshold=3,
        )

        result = agent.run("work on task")
        assert result == "done"

        # Check that a system reminder was injected
        system_msgs = [m for m in agent.messages if m.get("role") == "system"]
        assert len(system_msgs) > 0
        assert any("Reminder" in m.get("content", "") for m in system_msgs)
