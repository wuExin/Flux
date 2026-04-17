import os
from .base import Tool

MAX_READ_CHARS = 50000


class ReadTool(Tool):
    """Read file contents with line numbers."""

    name = "read"
    description = (
        "Read the contents of a file. Returns the file content with line numbers. "
        "Use offset and limit to read a specific range of lines for large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute or relative path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (0-based). Defaults to 0.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. Defaults to all lines.",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def _resolve(self, file_path: str) -> str:
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self.cwd or os.getcwd(), file_path)

    def execute(self, file_path: str, offset: int = 0, limit: int | None = None) -> str:
        path = self._resolve(file_path)

        if not os.path.exists(path):
            return f"[Error: file not found: {file_path}]"
        if os.path.isdir(path):
            return f"[Error: path is a directory, not a file: {file_path}]"

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except PermissionError:
            return f"[Error: permission denied: {file_path}]"
        except Exception as e:
            return f"[Error reading file: {e}]"

        total = len(lines)
        start = max(0, offset)
        end = min(total, start + limit) if limit is not None else total
        selected = lines[start:end]

        if not selected:
            if total == 0:
                return f"[File is empty: {file_path}]"
            return f"[No lines in range {start}-{end}, file has {total} lines]"

        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i}\t{line.rstrip()}")
        output = "\n".join(numbered)

        if len(output) > MAX_READ_CHARS:
            half = MAX_READ_CHARS // 2
            output = (
                output[:half]
                + f"\n\n... [{len(output) - MAX_READ_CHARS} characters truncated] ...\n\n"
                + output[-half:]
            )

        info = f"[{file_path}: {total} lines total"
        if start > 0 or end < total:
            info += f", showing lines {start + 1}-{end}"
        info += "]"
        return info + "\n" + output
