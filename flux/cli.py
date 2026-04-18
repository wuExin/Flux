import os
import platform
import sys

from .agent import Agent
from .config import load_config
from .display import show_response, show_tool_call, show_tool_result
from .llm import LLMClient
from .tools.bash import BashTool
from .tools.edit import EditTool
from .tools.read import ReadTool
from .tools.registry import ToolRegistry
from .tools.todo import TodoState, TodoTool
from .tools.write import WriteTool

SYSTEM_PROMPT = """\
You are Flux, an AI coding assistant running in the user's terminal.
You help with software engineering tasks: writing code, debugging, explaining, and running commands.

## Environment
- Platform: {platform}
- Shell: {shell}
- Python: {python}
- Working directory: {cwd}

## Tools
You have access to the following tools:

- `bash` — Execute shell commands on the user's machine.
- `read` — Read file contents with line numbers. Use offset/limit for large files.
- `write` — Create new files or completely overwrite existing files.
- `edit` — Perform exact string replacement in a file. The old_string must match exactly once.
- `todo` — Track task progress for multi-step tasks.

## Guidelines
- Be concise and direct. Lead with the answer or action.
- **Read files before modifying them.** Use `read` to view a file before using `edit` or `write`.
- Prefer `read` over `bash` for reading files (e.g., use `read` instead of `cat`).
- Prefer `edit` for modifying existing files. Only use `write` for new files or full rewrites.
- Prefer `write` over `bash` for creating files (e.g., use `write` instead of `echo >>`).
- When running commands, prefer safe, non-destructive operations.
- If a command might be destructive (rm -rf, git reset --hard), warn the user first.
- Do not fabricate file contents or command outputs. Always use tools to verify.
- When you're done, stop. Do not call tools unnecessarily.
- **Use commands compatible with the current platform.** On Windows, avoid bash-specific syntax like heredoc (<<), use `python` instead of `python3`, and prefer `type` over `cat`.

## Task Tracking with Todo
For multi-step tasks, use the `todo` tool to track progress:
- Create tasks: `todo action='create' subject='...' description='...'`
- List tasks: `todo action='list'`
- Start a task: `todo action='start' task_id='...'`
- Complete a task: `todo action='complete' task_id='...'`
- Delete a task: `todo action='delete' task_id='...'`
- Update a task: `todo action='update' task_id='...' subject='...' description='...'`

Only one task can be in_progress at a time. Use `list` to check progress regularly.
"""


def _detect_shell() -> str:
    """Detect the current shell environment."""
    if os.name == "nt":
        # Check if running under Git Bash, MSYS2, or similar
        if os.environ.get("MSYSTEM") or os.environ.get("BASH"):
            return "git-bash"
        return "cmd.exe"
    return os.environ.get("SHELL", "/bin/sh")


def _build_system_prompt(cwd: str) -> str:
    """Build system prompt with environment information."""
    return SYSTEM_PROMPT.format(
        platform=platform.platform(),
        shell=_detect_shell(),
        python=f"{sys.executable} ({platform.python_version()})",
        cwd=cwd,
    )


def main():
    config = load_config()
    llm = LLMClient(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
    )
    registry = ToolRegistry()
    cwd = os.getcwd()
    registry.register(BashTool(timeout=config.bash_timeout, cwd=cwd))
    registry.register(ReadTool(cwd=cwd))
    registry.register(WriteTool(cwd=cwd))
    registry.register(EditTool(cwd=cwd))

    # Todo state management
    todo_state = TodoState()
    registry.register(TodoTool(state=todo_state))

    agent = Agent(
        llm=llm,
        tools=registry,
        system_prompt=_build_system_prompt(cwd),
        on_tool_call=show_tool_call,
        on_tool_result=show_tool_result,
        todo_state=todo_state,
        nag_threshold=getattr(config, "nag_threshold", 3),
    )

    print("Flux — AI Coding Assistant (type 'exit' to quit)")
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            response = agent.run(user_input)
            show_response(response)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except EOFError:
            break
