import os
from .base import Tool


class WriteTool(Tool):
    """Write content to a file, creating or overwriting."""

    name = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "or completely overwrites it if it does. "
        "Use this for creating new files or full rewrites. "
        "For partial modifications, prefer the 'edit' tool instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute or relative path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The complete content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def _resolve(self, file_path: str) -> str:
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self.cwd or os.getcwd(), file_path)

    def execute(self, file_path: str, content: str) -> str:
        path = self._resolve(file_path)

        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except OSError as e:
                return f"[Error creating directory: {e}]"

        if os.path.isdir(path):
            return f"[Error: path is a directory: {file_path}]"

        is_new = not os.path.exists(path)
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
        except PermissionError:
            return f"[Error: permission denied: {file_path}]"
        except Exception as e:
            return f"[Error writing file: {e}]"

        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        action = "Created" if is_new else "Wrote"
        return f"[{action} {file_path}: {lines} lines]"
