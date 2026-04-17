import sys

from flux.display import (
    MAX_RESULT_CHARS,
    _truncate,
    show_response,
    show_tool_call,
    show_tool_result,
)


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello") == "hello"

    def test_strips_whitespace(self):
        assert _truncate("  hello  ") == "hello"

    def test_exact_limit_unchanged(self):
        text = "x" * MAX_RESULT_CHARS
        assert _truncate(text) == text

    def test_long_text_truncated(self):
        text = "a" * (MAX_RESULT_CHARS + 100)
        result = _truncate(text)
        assert len(result) < len(text)
        assert "chars truncated" in result
        assert result.startswith("a" * (MAX_RESULT_CHARS // 2))
        assert result.endswith("a" * (MAX_RESULT_CHARS // 2))

    def test_custom_limit(self):
        text = "abcdefghij"  # 10 chars
        result = _truncate(text, max_chars=6)
        assert "chars truncated" in result


class TestShowToolCall:
    def test_bash_shows_command(self, capsys):
        show_tool_call("bash", {"command": "echo hi"})
        captured = capsys.readouterr()
        assert "bash" in captured.err
        assert "echo hi" in captured.err
        assert captured.out == ""

    def test_bash_truncates_long_command(self, capsys):
        long_cmd = "x" * 100
        show_tool_call("bash", {"command": long_cmd})
        captured = capsys.readouterr()
        assert "..." in captured.err

    def test_non_bash_tool(self, capsys):
        show_tool_call("read_file", {"path": "/tmp/foo.txt"})
        captured = capsys.readouterr()
        assert "read_file" in captured.err
        assert captured.out == ""


class TestShowToolResult:
    def test_short_result(self, capsys):
        show_tool_result("ok")
        captured = capsys.readouterr()
        assert "ok" in captured.err
        assert captured.out == ""

    def test_multiline_result(self, capsys):
        show_tool_result("line1\nline2\nline3")
        captured = capsys.readouterr()
        assert captured.err.count("\n") >= 3

    def test_long_result_truncated(self, capsys):
        show_tool_result("x" * (MAX_RESULT_CHARS + 100))
        captured = capsys.readouterr()
        assert "truncated" in captured.err


class TestShowResponse:
    def test_displays_to_stdout(self, capsys):
        show_response("Hello user")
        captured = capsys.readouterr()
        assert "Hello user" in captured.out
        assert captured.err == ""

    def test_empty_response_silent(self, capsys):
        show_response("")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_none_like_empty(self, capsys):
        show_response("")
        captured = capsys.readouterr()
        assert captured.out == ""
