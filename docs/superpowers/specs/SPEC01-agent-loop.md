# SPEC01 — The Agent Loop 实现方案

> 对应需求文档: PRD01.md — 第一阶段  
> 创建日期: 2026-04-17

---

## 1. 目标

实现一个 AI 编程助手的核心 Agent Loop：用户输入 prompt → LLM 推理 → 工具执行 → 结果回传 → 循环直到 LLM 不再调用工具。本阶段聚焦最小可用闭环，不涉及 UI、权限系统、多会话管理等后续功能。

---

## 2. 技术选型

| 维度 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | 生态成熟，LLM SDK 支持好 |
| LLM | GLM-4 (智谱 AI) | PRD 指定 GLM4.7 |
| HTTP 客户端 | `zhipuai` 官方 SDK / `httpx` | 官方 SDK 优先，fallback 用 httpx |
| 交互方式 | CLI (stdin/stdout) | 第一阶段只做终端交互 |
| 包管理 | `uv` / `pip` | 轻量快速 |

---

## 3. 项目结构

```
flux/
├── __init__.py
├── __main__.py          # 入口: python -m flux
├── cli.py               # CLI 交互层 (REPL)
├── agent.py             # Agent Loop 核心逻辑
├── llm.py               # LLM 客户端封装
├── tools/
│   ├── __init__.py      # 工具注册表
│   ├── registry.py      # 工具发现 & 定义生成
│   ├── bash.py          # Bash 命令执行
│   └── base.py          # 工具基类
├── message.py           # 消息构造 & 管理
└── config.py            # 配置 (API key, model, system prompt)
tests/
├── test_agent.py
├── test_tools.py
└── test_llm.py
pyproject.toml
```

---

## 4. 核心模块设计

### 4.1 消息模型 (`message.py`)

统一消息格式，对齐 GLM API 的 messages 协议。

```python
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ToolCall:
    """LLM 返回的工具调用请求"""
    id: str
    name: str
    arguments: dict

@dataclass
class ToolResult:
    """工具执行结果，回传给 LLM"""
    tool_call_id: str
    content: str

@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
```

### 4.2 LLM 客户端 (`llm.py`)

封装与 GLM API 的交互。职责单一：发送消息 + 工具定义，返回结构化响应。

```python
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """LLM 调用的标准化返回"""
    content: list[dict]       # 原始 content blocks
    finish_reason: str          # "stop" | "tool_calls" 等
    tool_calls: list[ToolCall]
    raw: object               # 原始 API 响应, 用于调试

class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        ...

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int = 8000,
    ) -> LLMResponse:
        """
        调用 GLM Chat API。
        - 将内部消息格式转为 GLM API 格式
        - 解析响应为 LLMResponse
        - 处理网络异常 & API 错误 (重试 / 抛出)
        """
        ...
```

**GLM API 适配要点：**

- tool_calls 在 `response.choices[0].message.tool_calls` 中
- `finish_reason` 在 `response.choices[0].finish_reason`，值为 `"stop"` 或 `"tool_calls"`
- 工具调用结果以 `role: "tool"` 消息回传，需要 `tool_call_id` 字段
- 工具定义采用 OpenAI 兼容的 function calling 格式

### 4.3 工具系统 (`tools/`)

#### 4.3.1 工具基类 (`tools/base.py`)

```python
from abc import ABC, abstractmethod

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，需全局唯一"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，供 LLM 理解用途"""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema 格式的参数定义"""

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具，返回文本结果"""
```

#### 4.3.2 Bash 工具 (`tools/bash.py`)

第一阶段唯一的内置工具。

```python
import subprocess

class BashTool(Tool):
    name = "bash"
    description = "在系统 shell 中执行命令并返回输出。"
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 bash 命令"
            }
        },
        "required": ["command"]
    }

    def __init__(self, timeout: int = 120, cwd: str | None = None):
        self.timeout = timeout
        self.cwd = cwd

    def execute(self, command: str) -> str:
        """
        执行命令，捕获 stdout+stderr。
        - 超时控制 (默认 120s)
        - 输出截断 (防止 token 爆炸, 默认 max 50000 字符)
        - 返回格式: stdout + stderr + exit_code
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
            )
            output = result.stdout + result.stderr
            return self._truncate(output) + f"\n[exit code: {result.returncode}]"
        except subprocess.TimeoutExpired:
            return f"[Command timed out after {self.timeout}s]"
```

#### 4.3.3 工具注册表 (`tools/registry.py`)

```python
class ToolRegistry:
    """管理所有可用工具，生成 API 所需的 tools 定义"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_api_format(self) -> list[dict]:
        """生成 GLM function calling 格式的工具定义列表"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            }
            for t in self._tools.values()
        ]
```

### 4.4 Agent Loop (`agent.py`)

核心循环逻辑。

```python
class Agent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 50,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.messages: list[dict] = []

    def run(self, user_query: str) -> str:
        """
        执行完整的 agent loop, 返回最终文本回复。

        流程:
        1. 追加 user message
        2. 调用 LLM
        3. 追加 assistant message
        4. 若 finish_reason 非工具调用 → 提取文本返回
        5. 执行所有工具调用, 收集结果
        6. 追加 tool result messages
        7. 回到步骤 2
        """
        self.messages.append({"role": "user", "content": user_query})

        for _ in range(self.max_iterations):
            # 调用 LLM
            response = self.llm.chat(
                messages=self.messages,
                tools=self.tools.to_api_format(),
                system=self.system_prompt,
            )

            # 追加 assistant 响应
            self.messages.append(self._build_assistant_message(response))

            # 检查是否结束
            if response.finish_reason != "tool_calls":
                return self._extract_text(response)

            # 执行工具调用
            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "[Agent reached maximum iteration limit]"

    def _execute_tool(self, tool_call: ToolCall) -> str:
        tool = self.tools.get(tool_call.name)
        if tool is None:
            return f"[Error: unknown tool '{tool_call.name}']"
        try:
            return tool.execute(**tool_call.arguments)
        except Exception as e:
            return f"[Tool execution error: {e}]"

    def _extract_text(self, response: LLMResponse) -> str:
        """从 response content 中提取纯文本部分"""
        ...

    def _build_assistant_message(self, response: LLMResponse) -> dict:
        """将 LLMResponse 转为 messages 格式的 assistant 消息"""
        ...
```

### 4.5 CLI 交互层 (`cli.py`)

简单 REPL，第一阶段不做复杂 UI。

```python
def main():
    config = load_config()
    llm = LLMClient(api_key=config.api_key, model=config.model)
    registry = ToolRegistry()
    registry.register(BashTool(cwd=os.getcwd()))
    agent = Agent(llm=llm, tools=registry, system_prompt=SYSTEM_PROMPT)

    print("Flux — AI Coding Assistant (type 'exit' to quit)")
    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue

            response = agent.run(user_input)
            print(f"\n{response}")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
```

### 4.6 配置 (`config.py`)

```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    api_key: str
    model: str = "glm-4"          # 默认模型
    base_url: str | None = None   # 自定义 API 端点
    max_tokens: int = 8000
    bash_timeout: int = 120

def load_config() -> Config:
    """
    配置优先级: 环境变量 > 配置文件 > 默认值
    - FLUX_API_KEY / ZHIPUAI_API_KEY
    - FLUX_MODEL
    - FLUX_BASE_URL
    """
    api_key = os.environ.get("FLUX_API_KEY") or os.environ.get("ZHIPUAI_API_KEY", "")
    if not api_key:
        raise SystemExit("Error: set FLUX_API_KEY or ZHIPUAI_API_KEY environment variable")
    return Config(
        api_key=api_key,
        model=os.environ.get("FLUX_MODEL", "glm-4"),
        base_url=os.environ.get("FLUX_BASE_URL"),
    )
```

---

## 5. System Prompt

第一阶段的系统提示词，定义 AI 助手的角色和行为边界。

```text
You are Flux, an AI coding assistant running in the user's terminal.
You help with software engineering tasks: writing code, debugging, explaining, and running commands.

## Tools
You have access to a `bash` tool that executes shell commands on the user's machine.
Use it to read files, run tests, check git status, install packages, etc.

## Guidelines
- Be concise and direct. Lead with the answer or action.
- Read files before modifying them. Understand context before suggesting changes.
- When running commands, prefer safe, non-destructive operations.
- If a command might be destructive (rm -rf, git reset --hard), warn the user first.
- Do not fabricate file contents or command outputs. Always use tools to verify.
- When you're done, stop. Do not call tools unnecessarily.
```

---

## 6. GLM API 消息格式说明

GLM 使用 OpenAI 兼容的 function calling 格式：

| 概念 | GLM API |
|------|---------|
| 停止原因判断 | `finish_reason == "tool_calls"` |
| 工具调用列表 | `message.tool_calls[i]` |
| 工具调用 ID | `tool_call.id` |
| 工具调用参数 | `json.loads(tool_call.function.arguments)` |
| 工具结果回传 | `role: "tool"` + `tool_call_id` 独立消息 |
| 工具定义格式 | OpenAI function calling 格式 |

---

## 7. 实现步骤 (里程碑)

### M1 — 基础骨架 (Day 1)

- [ ] 初始化项目结构 + `pyproject.toml`
- [ ] 实现 `config.py` — 环境变量读取
- [ ] 实现 `message.py` — 数据模型
- [ ] 实现 `tools/base.py` + `tools/registry.py`

### M2 — LLM 集成 (Day 1-2)

- [ ] 实现 `llm.py` — GLM API 调用 + 响应解析
- [ ] 单元测试：mock API 响应，验证解析逻辑
- [ ] 验证 function calling 格式正确

### M3 — Bash 工具 (Day 2)

- [ ] 实现 `tools/bash.py`
- [ ] 测试：正常执行、超时、输出截断、非零退出码

### M4 — Agent Loop (Day 2-3)

- [ ] 实现 `agent.py` — 核心循环
- [ ] 测试：单轮对话、多轮工具调用、未知工具处理、迭代上限
- [ ] 端到端测试：真实 API 调用 + bash 工具

### M5 — CLI + 集成 (Day 3)

- [ ] 实现 `cli.py` — REPL 交互
- [ ] 实现 `__main__.py` 入口
- [ ] 手动验收：能完成 "读取文件 → 修改 → 验证" 的编程任务

---

## 8. 第一阶段不做的事

以下功能明确排除在本阶段之外，留待后续迭代：

- 流式输出 (streaming)
- 多工具 (文件读写、搜索、编辑等) — 只做 bash
- 会话持久化 / 历史记录
- 权限控制 / 用户确认
- Token 计数 & 上下文窗口管理
- 多模型切换
- 配置文件 (只用环境变量)
- 插件系统
- Web UI / IDE 集成
- 错误处理与安全退出 (见 [SPEC02](SPEC02-error-handling.md))

---

## 9. 验收标准

1. **基本对话**：用户输入自然语言问题，LLM 返回文本回复
2. **工具调用闭环**：LLM 自主调用 bash 工具，获取结果后继续推理
3. **多轮工具调用**：LLM 能连续调用多次工具完成复杂任务（如：先 `ls` 查看文件，再 `cat` 读取内容，再给出分析）
