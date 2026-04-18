import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from .base import Tool


@dataclass
class TodoItem:
    id: str
    subject: str
    description: str
    status: Literal["pending", "in_progress", "completed", "deleted"]
    created_at: float
    updated_at: float


class TodoState:
    """Task state manager."""

    def __init__(self) -> None:
        self.items: list[TodoItem] = []
        self._last_todo_call = 0
        self._current_iteration = 0

    def advance_iteration(self) -> None:
        """Advance iteration counter."""
        self._current_iteration += 1

    def record_todo_call(self) -> None:
        """Record a todo tool call."""
        self._last_todo_call = self._current_iteration

    def should_nag(self, threshold: int = 3) -> bool:
        """Check if nag reminder is needed."""
        return self._current_iteration - self._last_todo_call >= threshold

    def add(self, subject: str, description: str) -> TodoItem:
        item = TodoItem(
            id=str(uuid.uuid4())[:8],
            subject=subject,
            description=description,
            status="pending",
            created_at=time.time(),
            updated_at=time.time(),
        )
        self.items.append(item)
        return item

    def get(self, task_id: str) -> TodoItem | None:
        return next((i for i in self.items if i.id == task_id and i.status != "deleted"), None)

    def list_active(self) -> list[TodoItem]:
        return [i for i in self.items if i.status != "deleted"]

    def set_in_progress(self, task_id: str) -> TodoItem | None:
        for i in self.items:
            if i.status == "in_progress":
                i.status = "pending"
                i.updated_at = time.time()
        item = self.get(task_id)
        if item:
            item.status = "in_progress"
            item.updated_at = time.time()
        return item

    def update(self, task_id: str, **kwargs) -> TodoItem | None:
        item = self.get(task_id)
        if item:
            for k, v in kwargs.items():
                if hasattr(item, k) and v is not None:
                    setattr(item, k, v)
            item.updated_at = time.time()
        return item

    def set_status(self, task_id: str, status: str) -> TodoItem | None:
        item = self.get(task_id)
        if item:
            item.status = status
            item.updated_at = time.time()
        return item

    def format_list(self) -> str:
        active = self.list_active()
        if not active:
            return "[Todo List: empty]"

        lines = ["[Todo List]"]
        for i, item in enumerate(active, 1):
            status_icon = {"pending": " ", "in_progress": "*", "completed": "x"}
            icon = status_icon.get(item.status, "?")
            lines.append(f"({i}) [{icon}] {item.subject}")
            if item.description:
                for line in item.description.split("\n"):
                    lines.append(f"    {line}")
        return "\n".join(lines)


class TodoTool(Tool):
    """Task progress tracking tool."""

    @property
    def name(self) -> str:
        return "todo"

    @property
    def description(self) -> str:
        return (
            "Manage task progress. Use to create, list, start, complete, or delete tasks. "
            "Only one task can be in_progress at a time. "
            "Actions: create (requires subject, optional description), "
            "list (no params), start (requires task_id), "
            "complete (requires task_id), delete (requires task_id), "
            "update (requires task_id, optional subject/description)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "start", "update", "complete", "delete", "list"],
                },
                "task_id": {"type": "string"},
                "subject": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["action"],
        }

    def __init__(self, state: TodoState):
        self.state = state

    def execute(self, **kwargs) -> str:
        action = kwargs.get("action")
        task_id = kwargs.get("task_id")
        subject = kwargs.get("subject")
        description = kwargs.get("description")

        self.state.record_todo_call()

        if action == "create":
            if not subject:
                return "[Error: 'create' action requires 'subject']"
            item = self.state.add(subject, description or "")
            return f"[Todo created: {item.id}] {item.subject}"

        elif action == "list":
            return self.state.format_list()

        elif action == "start":
            if not task_id:
                return "[Error: 'start' action requires 'task_id']"
            item = self.state.set_in_progress(task_id)
            if not item:
                return f"[Error: task not found: {task_id}]"
            return f"[Todo started: {item.id}] {item.subject}"

        elif action == "complete":
            if not task_id:
                return "[Error: 'complete' action requires 'task_id']"
            item = self.state.set_status(task_id, "completed")
            if not item:
                return f"[Error: task not found: {task_id}]"
            return f"[Todo completed: {item.id}] {item.subject}"

        elif action == "delete":
            if not task_id:
                return "[Error: 'delete' action requires 'task_id']"
            item = self.state.set_status(task_id, "deleted")
            if not item:
                return f"[Error: task not found: {task_id}]"
            return f"[Todo deleted: {item.id}] {item.subject}"

        elif action == "update":
            if not task_id:
                return "[Error: 'update' action requires 'task_id']"
            updates = {}
            if subject is not None:
                updates["subject"] = subject
            if description is not None:
                updates["description"] = description
            if not updates:
                return "[Error: 'update' requires at least one of 'subject' or 'description']"
            item = self.state.update(task_id, **updates)
            if not item:
                return f"[Error: task not found: {task_id}]"
            return f"[Todo updated: {item.id}] {item.subject}"

        return f"[Error: unknown action '{action}']"
