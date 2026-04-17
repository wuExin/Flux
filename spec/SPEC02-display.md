# SPEC03 — AI 流程可视化

> 对应需求文档: PRD02.md  
> 创建日期: 2026-04-17

---

## 1. 目标

在终端中实时可视化 Agent 运行流程：用户能看到 AI 调用了哪些工具、传了什么参数、执行结果是什么，以及 AI 的最终回复。工具输出截断显示，保持终端整洁。

---

## 2. 现状分析

当前 Agent Loop 的输出方式：

| 环节 | 现状 |
|------|------|
| 工具调用 | 仅写入 logger.info，用户不可见 |
| 工具结果 | 仅回传给 LLM，用户不可见 |
| AI 回复 | cli.py 中 `print(f"\n{response}")` 直接输出原始文本 |

**问题**：用户无法感知 AI 正在做什么，只能等待最终结果。

---

## 3. 设计方案

### 3.1 架构：回调函数

在 `Agent` 中新增两个可选回调参数，agent 运行时在关键节点调用：

```python
class Agent:
    def __init__(self, ..., on_tool_call=None, on_tool_result=None):
```

**优点**：Agent 核心逻辑不耦合显示逻辑，回调可选，测试时不传即可。

### 3.2 显示时机与内容

| 事件 | 输出目标 | 显示内容 | 截断规则 |
|------|---------|---------|---------|
| 工具调用 | stderr | `> bash  <命令摘要>` | 命令超 80 字符截断 |
| 工具结果 | stderr | `\| <输出内容>` | 超 200 字符截断，中间省略 |
| AI 回复 | stdout | `< <完整回复>` | 不截断 |

- 工具事件输出到 **stderr**：不污染 stdout，可被管道过滤
- AI 回复输出到 **stdout**：与现有行为一致

### 3.3 视觉格式

```
> 用户输入

  > bash  ls -la src/
  | total 32
  | drwxr-xr-x  4 user user 4096 ...
  | -rw-r--r--  1 user user  234 ... [136 chars truncated]

  < 这个目录下有以下文件...
```

- `>` 青色加粗 — 工具调用
- `|` 灰色暗淡 — 工具输出
- `<` 绿色 — AI 回复
- 使用 ANSI 转义码着色，无需额外依赖

---

## 4. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `flux/display.py` | **新建** | 显示模块：格式化、截断、ANSI 着色 |
| `flux/agent.py` | 修改 | 添加 `on_tool_call` / `on_tool_result` 回调 |
| `flux/cli.py` | 修改 | 引入 display，传入回调函数，替换 print |
| `tests/test_display.py` | **新建** | display 模块单元测试 |
| `tests/test_agent.py` | 修改 | 验证回调被正确调用 |

---

## 5. 模块设计

### 5.1 `flux/display.py`

```python
"""终端显示模块 — AI 流程可视化"""

import shutil
import sys

# ANSI 颜色
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"

MAX_RESULT_CHARS = 200

def _truncate(text: str, max_chars: int = MAX_RESULT_CHARS) -> str:
    """超长文本中间截断"""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f" ... [{len(text) - max_chars} chars truncated] ... " + text[-half:]

def show_tool_call(name: str, arguments: dict) -> None:
    """显示工具调用: > bash  ls -la"""
    if name == "bash":
        label = arguments.get("command", "")
        if len(label) > 80:
            label = label[:77] + "..."
    else:
        label = str(arguments)[:80]
    print(f"  {CYAN}{BOLD}> {name}{RESET}  {DIM}{label}{RESET}", file=sys.stderr)

def show_tool_result(result: str) -> None:
    """显示工具结果: | <truncated output>"""
    truncated = _truncate(result)
    for line in truncated.splitlines():
        print(f"  {DIM}| {line}{RESET}", file=sys.stderr)

def show_response(text: str) -> None:
    """显示 AI 回复: < response"""
    if not text:
        return
    print(f"\n  {GREEN}<{RESET} {text}")
```

### 5.2 `flux/agent.py` 变更

```python
# __init__ 新增参数
def __init__(self, ..., on_tool_call=None, on_tool_result=None):
    self.on_tool_call = on_tool_call      # (name, arguments) -> None
    self.on_tool_result = on_tool_result   # (result) -> None

# run() 循环中，工具执行前后调用
for tool_call in response.tool_calls:
    if self.on_tool_call:
        self.on_tool_call(tool_call.name, tool_call.arguments)
    result = self._execute_tool(tool_call)
    if self.on_tool_result:
        self.on_tool_result(result)
    # ... append to messages
```

### 5.3 `flux/cli.py` 变更

```python
from .display import show_response, show_tool_call, show_tool_result

# 创建 Agent 时传入回调
agent = Agent(
    ...,
    on_tool_call=show_tool_call,
    on_tool_result=show_tool_result,
)

# 替换原始 print
response = agent.run(user_input)
show_response(response)  # 替代 print(f"\n{response}")
```

---

## 6. 验证方案

1. **单元测试**：`tests/test_display.py`
   - `_truncate` 截断逻辑（短文本不截断、长文本中间省略）
   - `show_tool_call` / `show_tool_result` 输出到 stderr
   - `show_response` 输出到 stdout
2. **Agent 回调测试**：`tests/test_agent.py`
   - mock 回调函数，验证 tool_call 和 tool_result 事件触发次数与参数
3. **手动验证**：运行 `python -m flux`，执行一次带工具调用的对话，确认终端显示效果
4. **全量测试**：`python -m pytest tests/ -v` 确保无回归

---

## 7. 注意事项

- 不引入新的第三方依赖，仅使用 ANSI 转义码
- Windows Terminal / PowerShell 7+ 支持 ANSI；旧版 cmd.exe 可能需要 `os.system("")` 激活
