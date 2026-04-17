import pytest
from flux.tools.bash import BashTool
from flux.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = BashTool()
        registry.register(tool)
        assert registry.get("bash") is tool

    def test_get_unknown(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_to_api_format(self):
        registry = ToolRegistry()
        registry.register(BashTool())
        result = registry.to_api_format()
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "bash"
        assert "parameters" in result[0]["function"]


class TestBashTool:
    def test_echo(self):
        tool = BashTool()
        result = tool.execute(command="echo hello")
        assert "hello" in result
        assert "[exit code: 0]" in result

    def test_nonzero_exit(self):
        tool = BashTool()
        result = tool.execute(command="exit 42")
        assert "[exit code: 42]" in result

    def test_timeout(self):
        tool = BashTool(timeout=1)
        result = tool.execute(command="sleep 10")
        assert "timed out" in result.lower()

    def test_truncation(self):
        tool = BashTool()
        # Generate output longer than 50000 chars
        result = tool.execute(command="python -c \"print('x' * 60000)\"")
        assert "truncated" in result.lower()
