import os

from .agent import Agent
from .config import load_config
from .display import show_response, show_tool_call, show_tool_result
from .llm import LLMClient
from .tools.bash import BashTool
from .tools.registry import ToolRegistry

SYSTEM_PROMPT = """\
You are Flux, an AI coding assistant running in the user's terminal.
You help with software engineering tasks: writing code, debugging, explaining, and running commands.

## Tools
You have access to a `bash` tool that executes shell commands on the user's machine.
Use it to read files, run tests, check git status, install packages, etc.

## Guidelines
- Be concise and direct. Lead with the answer or action.
- Read files before modifying them. Understand context before suggesting changes.
- When running commands, prefer safe, non-destructive operations.
- If a command might be destructive (rm -rf, git reset --hard), warn the user first.
- Do not fabricate file contents or command outputs. Always use tools to verify.
- When you're done, stop. Do not call tools unnecessarily.
"""


def main():
    config = load_config()
    llm = LLMClient(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
    )
    registry = ToolRegistry()
    registry.register(BashTool(timeout=config.bash_timeout, cwd=os.getcwd()))
    agent = Agent(
        llm=llm,
        tools=registry,
        system_prompt=SYSTEM_PROMPT,
        on_tool_call=show_tool_call,
        on_tool_result=show_tool_result,
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
