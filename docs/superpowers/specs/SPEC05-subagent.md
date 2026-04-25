# SPEC05 — SubAgent 工具 (嵌套对话/任务委派)

> 对应需求文档: PRD04.md — SubAgents功能
> 创建日期: 2026-04-23

---

## 1. 目标

为 Flux 新增 `subagent` 工具，实现任务委派功能。主 Agent 可以将复杂任务委派给 SubAgent 执行，SubAgent 独立运行完成后返回结果。

---

## 2. 问题分析

| 场景 | 痛点 |
|------|------|
| 复杂多步任务 | 主 Agent 上下文过长，注意力分散 |
| 独立子任务 | 子任务完成后不需要保留在主对话历史中 |
| 临时探索性任务 | 需要独立执行但结果要返回主流程 |

---

## 3. 设计原则

| 原则 | 说明 |
|------|------|
| 同步阻塞 | SubAgent 在同一线程中运行，完成后返回结果 |
| 禁止嵌套 | SubAgent 不能调用 subagent 工具 |
| 纯文本返回 | SubAgent 返回纯文本结果，UI 包裹由 display 层处理 |
| 精简提示 | SubAgent 使用专门的精简系统提示词 |
| 工具限制 | SubAgent 可访问 bash/read/write/edit，不包含 todo/subagent |

---

## 4. 架构设计

```
+----------+     +------------+     +------------+
|  主Agent | --> | SubAgent   | --> |   Tools    |
|          |     |   Tool     |     | (bash/read |
|          |     |            |     | /write/..) |
+----------+     +------------+     +------------+
      |                                    ^
      |         (纯文本结果)                |
      +------------------------------------+
```

**工作流程：**

1. 主 Agent 调用 `subagent` 工具，传入 `task`（任务描述）
2. 创建新的 SubAgent 实例，使用精简系统提示词
3. SubAgent 运行独立的 agent loop
4. SubAgent 完成后，将最终回复作为纯文本返回给主 Agent
5. 主 Agent 继续自己的对话流程

---

## 5. 工具 API 设计

### SubAgentTool 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task` | string | 是 | 给 SubAgent 的任务描述 |
| `max_iterations` | int | 否 | SubAgent 最大迭代次数，默认 20 |

### JSON Schema

```python
parameters = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The task description to delegate to the sub-agent"
        },
        "max_iterations": {
            "type": "integer",
            "description": "Maximum iterations for sub-agent (default 20)"
        }
    },
    "required": ["task"]
}
```

### 调用示例

**输入：**
```json
{
  "task": "在 flux/ 目录下找出所有包含 'TODO' 注释的 Python 文件"
}
```

**输出：**
```
在 flux/ 目录下找到 3 个 包含 TODO 注释的文件：
- flux/agent.py: 第 45 行
- flux/tools/read.py: 第 12 行
- flux/cli.py: 第 78 行
```

---

## 6. SubAgent 类实现

### flux/subagent.py

```python
class SubAgent:
    """轻量子代理，用于执行委派任务。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str = SUBAGENT_SYSTEM_PROMPT,
        max_iterations: int = 20,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.messages: list[dict] = []

    def run(self, task: str) -> str:
        """执行任务，返回最终结果。"""
        self.messages.append({"role": "user", "content": task})

        for i in range(self.max_iterations):
            response = self.llm.chat(
                messages=self.messages,
                tools=self.tools.to_api_format(),
                system=self.system_prompt,
            )
            self.messages.append(self._build_assistant_message(response))

            if response.finish_reason != "tool_calls":
                return response.content or ""

            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "[SubAgent reached maximum iterations]"
```

### 精简系统提示词

```python
SUBAGENT_SYSTEM_PROMPT = """\
You are a task executor. Use the available tools to complete the assigned task.
Focus on efficiency and accuracy. Return concise results.

Available tools:
- bash: Execute shell commands
- read: Read file contents
- write: Create or overwrite files
- edit: Make exact string replacements in files

Complete the task and return your findings.
"""
```

---

## 7. SubAgentTool 实现

### flux/tools/subagenttool.py

```python
from ..subagent import SubAgent, SUBAGENT_SYSTEM_PROMPT
from ..tools.registry import ToolRegistry

class SubAgentTool(Tool):
    """将任务委派给子代理执行。"""

    name = "subagent"
    description = (
        "Delegate a task to a sub-agent for independent execution. "
        "The sub-agent will use available tools (bash, read, write, edit) "
        "to complete the task and return the result. "
        "Useful for multi-step tasks that should be isolated from the main conversation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The task description to delegate to the sub-agent"
            },
            "max_iterations": {
                "type": "integer",
                "description": "Maximum iterations for sub-agent (default 20)"
            }
        },
        "required": ["task"]
    }

    def __init__(self, llm, tools: ToolRegistry):
        self.llm = llm
        self.tools = tools

    def execute(self, task: str, max_iterations: int = 20) -> str:
        # 创建只包含 bash/read/write/edit 的工具注册表
        allowed_tools = ["bash", "read", "write", "edit"]
        sub_tools = ToolRegistry()
        for tool_name in allowed_tools:
            tool = self.tools.get(tool_name)
            if tool:
                sub_tools.register(tool)

        # 创建并运行 SubAgent
        subagent = SubAgent(
            llm=self.llm,
            tools=sub_tools,
            max_iterations=max_iterations,
        )
        return subagent.run(task)
```

---

## 8. 文件变更清单

```
flux/
├── subagent.py              # [新增] SubAgent 类
├── tools/
│   ├── subagenttool.py      # [新增] SubAgentTool
│   ├── __init__.py          # [修改] 导出 SubAgentTool
├── cli.py                   # [修改] 注册 SubAgentTool
└── display.py               # [修改] 添加 SubAgent UI 展示

tests/
├── test_subagent.py         # [新增] SubAgent 单元测试
├── test_subagenttool.py     # [新增] SubAgentTool 测试
└── test_integration.py      # [新增] 集成测试
```

### cli.py 修改

```python
from .tools.subagenttool import SubAgentTool

# 在 main() 中注册
registry.register(SubAgentTool(llm=llm, tools=registry))
```

### display.py 修改

```python
def show_subagent_start(task: str):
    """显示 SubAgent 开始执行。"""
    print(f"[SubAgent started] {task}")

def show_subagent_end():
    """显示 SubAgent 完成。"""
    print("[SubAgent completed]")
```

---

## 9. 错误处理

| 场景 | 处理方式 |
|------|----------|
| SubAgent 达到最大迭代次数 | 返回 `[SubAgent reached maximum iterations]` |
| SubAgent 执行工具失败 | 错误信息通过 tool 消息返回给 SubAgent |
| SubAgent 返回空结果 | 返回空字符串 `""` |
| LLM 调用失败 | 抛出异常，由主 Agent 的 `_execute_tool` 捕获 |

### 与主 Agent 的区别

| 特性 | 主 Agent | SubAgent |
|------|----------|----------|
| max_iterations | 默认 50 | 默认 20 |
| todo 支持 | 有 | 无 |
| nag reminder | 有 | 无 |

---

## 10. 测试计划

### 单元测试

```python
# tests/test_subagent.py

class TestSubAgent:
    def test_run_simple_task(self):
        """测试简单任务执行。"""
        subagent = SubAgent(llm, tools)
        result = subagent.run("列出当前目录的文件")
        assert "flux/" in result

    def test_max_iterations_limit(self):
        """测试达到最大迭代次数。"""
        subagent = SubAgent(llm, tools, max_iterations=2)
        result = subagent.run("复杂任务")
        assert "maximum iterations" in result

    def test_no_todo_tool(self):
        """验证 SubAgent 无法访问 todo 工具。"""
        subagent = SubAgent(llm, tools)
        assert not any(t.name == "todo" for t in subagent.tools.all())

# tests/test_subagenttool.py

class TestSubAgentTool:
    def test_execute_returns_result(self):
        """测试工具执行返回结果。"""
        tool = SubAgentTool(llm, registry)
        result = tool.execute(task="读取 flux/agent.py")
        assert "class Agent" in result

    def test_max_iterations_param(self):
        """测试 max_iterations 参数传递。"""
        tool = SubAgentTool(llm, registry)
        result = tool.execute(task="简单任务", max_iterations=5)
        assert result
```

### 集成测试

```python
# tests/test_integration.py

class TestSubAgentIntegration:
    def test_main_agent_delegates_to_subagent(self):
        """测试主 Agent 委派任务给 SubAgent。"""
        agent = Agent(llm, tools, system_prompt)
        response = agent.run("使用 subagent 工具统计 flux/ 目录的代码行数")
        assert "行数" in response or "lines" in response.lower()
```

---

## 11. 实现顺序

1. **第一步**：实现 `SubAgent` 类（flux/subagent.py）
2. **第二步**：实现 `SubAgentTool`（flux/tools/subagenttool.py）
3. **第三步**：修改 `cli.py` 注册工具
4. **第四步**：修改 `display.py` 添加 UI 展示
5. **第五步**：编写单元测试和集成测试
6. **第六步**：端到端测试验证完整流程
