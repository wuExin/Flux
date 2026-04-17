"""Terminal display for agent flow visualization."""

import shutil
import sys

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
GRAY = "\033[90m"

# Symbols
SYM_TOOL = ">"
SYM_RESULT = "|"
SYM_REPLY = "<"

MAX_RESULT_CHARS = 200


def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


def _truncate(text: str, max_chars: int = MAX_RESULT_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f" ... [{len(text) - max_chars} chars truncated] ... " + text[-half:]


def show_tool_call(name: str, arguments: dict) -> None:
    """Display a tool invocation."""
    if name == "bash":
        cmd = arguments.get("command", "")
        label = cmd if len(cmd) <= 80 else cmd[:77] + "..."
    else:
        label = str(arguments)[:80]

    line = f"  {CYAN}{BOLD}{SYM_TOOL} {name}{RESET}{GRAY}  {label}{RESET}"
    print(line, file=sys.stderr)


def show_tool_result(result: str) -> None:
    """Display truncated tool result."""
    truncated = _truncate(result)
    lines = truncated.splitlines()
    for line in lines:
        print(f"  {DIM}{SYM_RESULT} {line}{RESET}", file=sys.stderr)


def show_response(text: str) -> None:
    """Display the AI response."""
    if not text:
        return
    print(f"\n  {GREEN}{SYM_REPLY}{RESET} {text}")


def show_thinking() -> None:
    """Display a thinking indicator."""
    print(f"  {MAGENTA}...{RESET}", file=sys.stderr, end="\r")
