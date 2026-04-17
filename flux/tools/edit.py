import os
from .base import Tool


class EditTool(Tool):
    """Perform exact string replacement in a file."""

    name = "edit"
    description = (
        "Perform exact string replacement in a file. "
        "Replaces the first occurrence of old_string with new_string. "
        "The old_string must match exactly (including indentation and whitespace). "
        "For creating new files or full rewrites, use the 'write' tool instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute or relative path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace (must exist in the file)",
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace old_string with",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def _resolve(self, file_path: str) -> str:
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self.cwd or os.getcwd(), file_path)

    def execute(self, file_path: str, old_string: str, new_string: str) -> str:
        path = self._resolve(file_path)

        if not os.path.exists(path):
            return f"[Error: file not found: {file_path}]"
        if os.path.isdir(path):
            return f"[Error: path is a directory: {file_path}]"

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except PermissionError:
            return f"[Error: permission denied: {file_path}]"
        except Exception as e:
            return f"[Error reading file: {e}]"

        if old_string == new_string:
            return "[Error: old_string and new_string are identical]"

        count = content.count(old_string)
        if count == 0:
            return f"[Error: old_string not found in {file_path}]"
        if count > 1:
            return (
                f"[Error: old_string matches {count} locations in {file_path}. "
                f"Provide more surrounding context to make the match unique.]"
            )

        new_content = content.replace(old_string, new_string, 1)

        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_content)
        except Exception as e:
            return f"[Error writing file: {e}]"

        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        return f"[Edited {file_path}: replaced {old_lines} lines with {new_lines} lines]"
