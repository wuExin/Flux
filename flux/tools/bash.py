import locale
import subprocess

from .base import Tool

MAX_OUTPUT_CHARS = 50000


class BashTool(Tool):
    """Execute shell commands and return output."""

    name = "bash"
    description = "Execute a command in the system shell and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    }

    def __init__(self, timeout: int = 120, cwd: str | None = None):
        self.timeout = timeout
        self.cwd = cwd

    def execute(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
                encoding=locale.getpreferredencoding(False),
                errors="replace",
            )
            output = result.stdout + result.stderr
            output = self._truncate(output)
            return f"{output}\n[exit code: {result.returncode}]"
        except subprocess.TimeoutExpired:
            return f"[Command timed out after {self.timeout}s]"

    def _truncate(self, text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        half = MAX_OUTPUT_CHARS // 2
        return (
            text[:half]
            + f"\n\n... [{len(text) - MAX_OUTPUT_CHARS} characters truncated] ...\n\n"
            + text[-half:]
        )
