import logging

from .llm import LLMClient, LLMResponse
from .message import ToolCall
from .tools.registry import ToolRegistry

try:
    from .tools.todo import TodoState
except ImportError:
    TodoState = None  # type: ignore


logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 50,
        on_tool_call=None,
        on_tool_result=None,
        todo_state: TodoState | None = None,
        nag_threshold: int = 3,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.messages: list[dict] = []
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.todo_state = todo_state
        self.nag_threshold = nag_threshold

    def run(self, user_query: str) -> str:
        """Execute the full agent loop, return final text reply."""
        self.messages.append({"role": "user", "content": user_query})

        for i in range(self.max_iterations):
            if self.todo_state:
                self.todo_state.advance_iteration()
                if self.todo_state.should_nag(self.nag_threshold):
                    self.messages.append({
                        "role": "system",
                        "content": (
                            "[Reminder] You have active todos. Use `todo` with action='list' to check progress. "
                            "Mark tasks complete with action='complete' when done."
                        )
                    })

            response = self.llm.chat(
                messages=self.messages,
                tools=self.tools.to_api_format(),
                system=self.system_prompt,
            )

            self.messages.append(self._build_assistant_message(response))

            if response.finish_reason != "tool_calls":
                return response.content or ""

            for tool_call in response.tool_calls:
                if self.on_tool_call:
                    self.on_tool_call(tool_call.name, tool_call.arguments)

                result = self._execute_tool(tool_call)

                if self.on_tool_result:
                    self.on_tool_result(result)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
                logger.info("tool=%s result_len=%d", tool_call.name, len(result))

        return "[Agent reached maximum iteration limit]"

    def _execute_tool(self, tool_call: ToolCall) -> str:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return f"[Error: unknown tool '{tool_call.name}']"
        try:
            return tool.execute(**tool_call.arguments)
        except Exception as e:
            return f"[Tool execution error: {e}]"

    def _build_assistant_message(self, response: LLMResponse) -> dict:
        """Convert LLMResponse to messages-format assistant message."""
        msg: dict = {"role": "assistant", "content": response.content or ""}
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": __import__("json").dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
        return msg
