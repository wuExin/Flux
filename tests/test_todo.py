import pytest
from flux.tools.todo import TodoItem, TodoState, TodoTool


class TestTodoState:
    def test_add_and_get(self):
        state = TodoState()
        item = state.add("Test task", "Description")
        assert state.get(item.id) == item
        assert item.status == "pending"

    def test_set_in_progress(self):
        state = TodoState()
        a = state.add("Task A", "")
        b = state.add("Task B", "")
        state.set_in_progress(a.id)
        assert state.get(a.id).status == "in_progress"
        state.set_in_progress(b.id)
        assert state.get(a.id).status == "pending"
        assert state.get(b.id).status == "in_progress"

    def test_list_active_excludes_deleted(self):
        state = TodoState()
        a = state.add("Task A", "")
        b = state.add("Task B", "")
        state.set_status(a.id, "deleted")
        assert state.list_active() == [state.get(b.id)]

    def test_should_nag(self):
        state = TodoState()
        assert not state.should_nag(threshold=3)
        state.advance_iteration()
        state.advance_iteration()
        state.advance_iteration()
        assert state.should_nag(threshold=3)
        state.record_todo_call()
        assert not state.should_nag(threshold=3)

    def test_update(self):
        state = TodoState()
        item = state.add("Task A", "Old desc")
        state.update(item.id, subject="Task B", description="New desc")
        assert item.subject == "Task B"
        assert item.description == "New desc"

    def test_set_status(self):
        state = TodoState()
        item = state.add("Task A", "")
        state.set_status(item.id, "completed")
        assert item.status == "completed"

    def test_format_list_empty(self):
        state = TodoState()
        assert state.format_list() == "[Todo List: empty]"

    def test_format_list_with_items(self):
        state = TodoState()
        a = state.add("Task A", "Desc A")
        state.set_in_progress(a.id)
        b = state.add("Task B", "Desc B")
        result = state.format_list()
        assert "[*] Task A" in result
        assert "[ ] Task B" in result
        assert "Desc A" in result
        assert "Desc B" in result


class TestTodoTool:
    def test_create(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="create", subject="Test")
        assert "[Todo created:" in result
        assert "Test" in result
        assert len(state.items) == 1

    def test_create_requires_subject(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="create")
        assert "requires 'subject'" in result

    def test_list_empty(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="list")
        assert "empty" in result

    def test_list_with_items(self):
        state = TodoState()
        tool = TodoTool(state)
        state.add("Task A", "Desc A")
        state.set_in_progress(state.items[0].id)
        state.add("Task B", "Desc B")
        result = tool.execute(action="list")
        assert "[*] Task A" in result
        assert "[ ] Task B" in result
        assert "Desc A" in result

    def test_start(self):
        state = TodoState()
        tool = TodoTool(state)
        state.add("Task A", "")
        task_id = state.items[0].id
        result = tool.execute(action="start", task_id=task_id)
        assert "[Todo started:" in result
        assert state.get(task_id).status == "in_progress"

    def test_complete(self):
        state = TodoState()
        tool = TodoTool(state)
        state.add("Task A", "")
        task_id = state.items[0].id
        result = tool.execute(action="complete", task_id=task_id)
        assert "[Todo completed:" in result
        assert state.get(task_id).status == "completed"

    def test_delete(self):
        state = TodoState()
        tool = TodoTool(state)
        state.add("Task A", "")
        task_id = state.items[0].id
        result = tool.execute(action="delete", task_id=task_id)
        assert "[Todo deleted:" in result
        assert state.get(task_id) is None

    def test_update(self):
        state = TodoState()
        tool = TodoTool(state)
        state.add("Task A", "Old desc")
        task_id = state.items[0].id
        result = tool.execute(
            action="update",
            task_id=task_id,
            subject="Task B",
            description="New desc"
        )
        assert "[Todo updated:" in result
        item = state.get(task_id)
        assert item.subject == "Task B"
        assert item.description == "New desc"

    def test_unknown_action(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="invalid")
        assert "unknown action" in result

    def test_start_requires_task_id(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="start")
        assert "requires 'task_id'" in result

    def test_complete_requires_task_id(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="complete")
        assert "requires 'task_id'" in result

    def test_delete_requires_task_id(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="delete")
        assert "requires 'task_id'" in result

    def test_update_requires_task_id(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="update")
        assert "requires 'task_id'" in result

    def test_task_not_found(self):
        state = TodoState()
        tool = TodoTool(state)
        result = tool.execute(action="start", task_id="nonexistent")
        assert "task not found" in result

    def test_record_todo_call(self):
        state = TodoState()
        tool = TodoTool(state)
        state.advance_iteration()
        state.advance_iteration()
        state.advance_iteration()
        assert state.should_nag()
        tool.execute(action="list")
        assert not state.should_nag()
