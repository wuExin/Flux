import os
import pytest
from flux.tools.read import ReadTool
from flux.tools.write import WriteTool
from flux.tools.edit import EditTool
from flux.tools.registry import ToolRegistry
from flux.tools.bash import BashTool


# ── ReadTool ──

class TestReadTool:
    def test_read_file(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("line1\nline2\nline3\n", encoding="utf-8")
        tool = ReadTool(cwd=str(tmp_path))
        result = tool.execute(file_path="hello.txt")
        assert "3 lines total" in result
        assert "1\tline1" in result
        assert "2\tline2" in result
        assert "3\tline3" in result

    def test_read_with_offset_limit(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        tool = ReadTool(cwd=str(tmp_path))
        result = tool.execute(file_path="data.txt", offset=1, limit=2)
        assert "showing lines 2-3" in result
        assert "2\tb" in result
        assert "3\tc" in result
        assert "1\ta" not in result

    def test_read_file_not_found(self, tmp_path):
        tool = ReadTool(cwd=str(tmp_path))
        result = tool.execute(file_path="nope.txt")
        assert "[Error: file not found:" in result

    def test_read_directory(self, tmp_path):
        tool = ReadTool(cwd=str(tmp_path))
        result = tool.execute(file_path=str(tmp_path))
        assert "[Error: path is a directory" in result

    def test_read_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        tool = ReadTool(cwd=str(tmp_path))
        result = tool.execute(file_path="empty.txt")
        assert "[File is empty:" in result

    def test_read_truncation(self, tmp_path):
        f = tmp_path / "big.txt"
        content = ("x" * 100 + "\n") * 1000
        f.write_text(content, encoding="utf-8")
        tool = ReadTool(cwd=str(tmp_path))
        result = tool.execute(file_path="big.txt")
        assert "truncated" in result

    def test_read_absolute_path(self, tmp_path):
        f = tmp_path / "abs.txt"
        f.write_text("hello\n", encoding="utf-8")
        tool = ReadTool(cwd=str(tmp_path))
        result = tool.execute(file_path=str(f))
        assert "1\thello" in result


# ── WriteTool ──

class TestWriteTool:
    def test_create_new_file(self, tmp_path):
        tool = WriteTool(cwd=str(tmp_path))
        result = tool.execute(file_path="new.txt", content="hello world\n")
        assert "[Created" in result
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello world\n"

    def test_overwrite_existing(self, tmp_path):
        f = tmp_path / "exist.txt"
        f.write_text("old content", encoding="utf-8")
        tool = WriteTool(cwd=str(tmp_path))
        result = tool.execute(file_path="exist.txt", content="new content\n")
        assert "[Wrote" in result
        assert f.read_text(encoding="utf-8") == "new content\n"

    def test_create_parent_dirs(self, tmp_path):
        tool = WriteTool(cwd=str(tmp_path))
        result = tool.execute(file_path="a/b/c.txt", content="deep\n")
        assert "[Created" in result
        assert (tmp_path / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "deep\n"

    def test_write_to_directory(self, tmp_path):
        d = tmp_path / "adir"
        d.mkdir()
        tool = WriteTool(cwd=str(tmp_path))
        result = tool.execute(file_path="adir", content="nope")
        assert "[Error: path is a directory" in result


# ── EditTool ──

class TestEditTool:
    def test_edit_unique_match(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n", encoding="utf-8")
        tool = EditTool(cwd=str(tmp_path))
        result = tool.execute(
            file_path="code.py",
            old_string="return 1",
            new_string="return 2",
        )
        assert "[Edited" in result
        assert "return 2" in f.read_text(encoding="utf-8")
        assert "return 1" not in f.read_text(encoding="utf-8")

    def test_edit_not_found(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello\n", encoding="utf-8")
        tool = EditTool(cwd=str(tmp_path))
        result = tool.execute(
            file_path="code.py",
            old_string="goodbye",
            new_string="hi",
        )
        assert "[Error: old_string not found" in result

    def test_edit_multiple_matches(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("aaa\nbbb\naaa\n", encoding="utf-8")
        tool = EditTool(cwd=str(tmp_path))
        result = tool.execute(
            file_path="code.py",
            old_string="aaa",
            new_string="ccc",
        )
        assert "matches 2 locations" in result
        # File should be unchanged
        assert f.read_text(encoding="utf-8") == "aaa\nbbb\naaa\n"

    def test_edit_identical_strings(self, tmp_path):
        f = tmp_path / "code.py"
        f.write_text("hello\n", encoding="utf-8")
        tool = EditTool(cwd=str(tmp_path))
        result = tool.execute(
            file_path="code.py",
            old_string="hello",
            new_string="hello",
        )
        assert "identical" in result

    def test_edit_file_not_found(self, tmp_path):
        tool = EditTool(cwd=str(tmp_path))
        result = tool.execute(
            file_path="nope.py",
            old_string="a",
            new_string="b",
        )
        assert "[Error: file not found" in result


# ── Integration ──

class TestIntegration:
    def test_registry_has_four_tools(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        registry.register(ReadTool())
        registry.register(WriteTool())
        registry.register(EditTool())
        tools = registry.to_api_format()
        names = {t["function"]["name"] for t in tools}
        assert names == {"bash", "read", "write", "edit"}

    def test_read_write_edit_roundtrip(self, tmp_path):
        """Write a file, read it, edit it, read again."""
        cwd = str(tmp_path)
        write = WriteTool(cwd=cwd)
        read = ReadTool(cwd=cwd)
        edit = EditTool(cwd=cwd)

        # Write
        result = write.execute(file_path="test.py", content="def hello():\n    return 'world'\n")
        assert "[Created" in result

        # Read
        result = read.execute(file_path="test.py")
        assert "return 'world'" in result

        # Edit
        result = edit.execute(
            file_path="test.py",
            old_string="return 'world'",
            new_string="return 'universe'",
        )
        assert "[Edited" in result

        # Verify
        result = read.execute(file_path="test.py")
        assert "return 'universe'" in result
