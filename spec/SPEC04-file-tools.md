# SPEC04 — 文件操作工具集 (Read / Write / Edit)

> 对应需求文档: PRD03.md — 新增工具  
> 创建日期: 2026-04-17

---

## 1. 目标

为 Flux 新增三个文件操作工具：`read`、`write`、`edit`，使 LLM 能够直接读写和编辑文件，而不必通过 `bash` 工具间接调用 `cat`、`echo`、`sed` 等命令。这三个工具提供更精确的文件操作语义，降低出错概率，提升 Agent 的编程能力。

---

## 2. 设计原则

| 原则 | 说明 |
|------|------|
| 与 BashTool 对齐 | 继承 `Tool` 基类，实现 `name` / `description` / `parameters` / `execute` |
| 输出截断 | 大文件读取使用截断机制，防止 token 溢出 |
| 编码安全 | 统一使用 UTF-8 读写，`errors="replace"` 兜底 |
| 最小参数集 | 每个工具只暴露必要参数，降低 LLM 调用出错率 |
| 幂等与安全 | `read` 只读无副作用；`write` 整体覆写；`edit` 精确替换 |

---

## 3. 文件变更清单

```
flux/tools/
├── read.py       # [新增] ReadTool
├── write.py      # [新增] WriteTool
├── edit.py       # [新增] EditTool
├── base.py       # [不变]
├── bash.py       # [不变]
├── registry.py   # [不变]
└── __init__.py   # [不变]
flux/
├── cli.py        # [修改] 注册新工具 + 更新 SYSTEM_PROMPT
tests/
├── test_file_tools.py  # [新增] 三个工具的单元测试
```

---

## 4. 核心模块设计

### 4.1 ReadTool (`tools/read.py`)

读取文件内容，支持行号显示和范围读取。

```python
import os
from .base import Tool

MAX_READ_CHARS = 50000

class ReadTool(Tool):
    name = "read"
    description = (
        "Read the contents of a file. Returns the file content with line numbers. "
        "Use offset and limit to read a specific range of lines for large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute or relative path to the file to read",
            },
            "offset": {
                "type": "integer",
                "description": "Line number to start reading from (0-based). Defaults to 0.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read. Defaults to all lines.",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, cwd: str | None = None):
        self.cwd = cwd

    def _resolve(self, file_path: str) -> str:
        """将相对路径解析为绝对路径。"""
        if os.path.isabs(file_path):
            return file_path
        return os.path.join(self.cwd or os.getcwd(), file_path)

    def execute(self, file_path: str, offset: int = 0, limit: int | None = None) -> str:
        path = self._resolve(file_path)

        if not os.path.exists(path):
            return f"[Error: file not found: {file_path}]"
        if os.path.isdir(path):
            return f"[Error: path is a directory, not a file: {file_path}]"

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except PermissionError:
            return f"[Error: permission denied: {file_path}]"
        except Exception as e:
            return f"[Error reading file: {e}]"

        total = len(lines)
        start = max(0, offset)
        end = min(total, start + limit) if limit is not None else total
        selected = lines[start:end]

        if not selected:
            if total == 0:
                return f"[File is empty: {file_path}]"
            return f"[No lines in range {start}-{end}, file has {total} lines]"

        # 添加行号 (1-based)
        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            numbered.append(f"{i}\t{line.rstrip()}")
        output = "\n".join(numbered)

        # 截断保护
        if len(output) > MAX_READ_CHARS:
            half = MAX_READ_CHARS // 2
            output = (
                output[:half]
                + f"\n\n... [{len(output) - MAX_READ_CHARS} characters truncated] ...\n\n"
                + output[-half:]
            )

        info = f"[{file_path}: {total} lines total"
        if start > 0 or end < total:
            info += f", showing lines {start + 1}-{end}"
        info += "]"
        return info + "\n" + output
```

**设计要点：**

- 行号从 1 开始，对齐常见编辑器约定 (`cat -n` 格式)
- `offset` 为 0-based 行偏移，`limit` 控制读取行数 — 便于 LLM 分段阅读大文件
- 截断阈值 `MAX_READ_CHARS = 50000`，与 BashTool 一致
- 通过 `cwd` 支持相对路径解析

---

### 4.2 WriteTool (`tools/write.py`)

将内容完整写入文件，覆盖已有内容。适用于创建新文件或完全重写。

```python
import os
from .base import Tool

class WriteTool(Tool):
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

        # 自动创建父目录
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
```

**设计要点：**

- 自动创建不存在的父目录 (`os.makedirs`)
- 区分 "Created"（新文件）与 "Wrote"（覆写已有）— 帮助 LLM 感知操作效果
- 强制 `newline="\n"` 避免 Windows 下产生 `\r\n`，保持跨平台一致性
- 写入使用 UTF-8 编码

---

### 4.3 EditTool (`tools/edit.py`)

精确字符串替换，适用于对已有文件的局部修改。

```python
import os
from .base import Tool

class EditTool(Tool):
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

        # 检查匹配数量
        count = content.count(old_string)
        if count == 0:
            return f"[Error: old_string not found in {file_path}]"
        if count > 1:
            return (
                f"[Error: old_string matches {count} locations in {file_path}. "
                f"Provide more surrounding context to make the match unique.]"
            )

        # 执行替换
        new_content = content.replace(old_string, new_string, 1)

        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(new_content)
        except Exception as e:
            return f"[Error writing file: {e}]"

        # 计算变更统计
        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        return f"[Edited {file_path}: replaced {old_lines} lines with {new_lines} lines]"
```

**设计要点：**

- **唯一性校验**：`old_string` 必须在文件中恰好匹配 1 次，否则报错并提示 LLM 提供更多上下文 — 这是防止误编辑的核心安全机制
- **精确匹配**：包括缩进和空白，不做模糊匹配
- **原子操作**：先读后写，中间只做字符串替换
- 与 `write` 明确分工：`edit` 用于局部修改，`write` 用于整体创建/覆写

---

## 5. CLI 集成

### 5.1 工具注册 (`cli.py`)

在 `main()` 中注册三个新工具：

```python
from .tools.read import ReadTool
from .tools.write import WriteTool
from .tools.edit import EditTool

def main():
    config = load_config()
    # ...
    registry = ToolRegistry()
    cwd = os.getcwd()
    registry.register(BashTool(timeout=config.bash_timeout, cwd=cwd))
    registry.register(ReadTool(cwd=cwd))
    registry.register(WriteTool(cwd=cwd))
    registry.register(EditTool(cwd=cwd))
    # ...
```

### 5.2 System Prompt 更新

更新 `SYSTEM_PROMPT`，让 LLM 了解新工具并遵循使用规范：

```text
You are Flux, an AI coding assistant running in the user's terminal.
You help with software engineering tasks: writing code, debugging, explaining, and running commands.

## Tools
You have access to the following tools:

- `bash` — Execute shell commands on the user's machine.
- `read` — Read file contents with line numbers. Use offset/limit for large files.
- `write` — Create new files or completely overwrite existing files.
- `edit` — Perform exact string replacement in a file. The old_string must match exactly once.

## Guidelines
- Be concise and direct. Lead with the answer or action.
- **Read files before modifying them.** Use `read` to view a file before using `edit` or `write`.
- Prefer `read` over `bash` for reading files (e.g., use `read` instead of `cat`).
- Prefer `edit` for modifying existing files. Only use `write` for new files or full rewrites.
- Prefer `write` over `bash` for creating files (e.g., use `write` instead of `echo >>`).
- When running commands, prefer safe, non-destructive operations.
- If a command might be destructive (rm -rf, git reset --hard), warn the user first.
- Do not fabricate file contents or command outputs. Always use tools to verify.
- When you're done, stop. Do not call tools unnecessarily.
```

### 5.3 Display 适配

现有 `display.py` 的 `show_tool_call` 对非 bash 工具 fallback 为 `str(arguments)[:80]`，已可展示 read/write/edit 的参数摘要。可选优化在后续迭代中添加。

---

## 6. 测试计划

### 6.1 ReadTool 测试

| 用例 | 预期 |
|------|------|
| 读取已有文件 | 返回带行号的内容 |
| 使用 offset + limit | 只返回指定范围 |
| 文件不存在 | `[Error: file not found: ...]` |
| 路径是目录 | `[Error: path is a directory: ...]` |
| 空文件 | `[File is empty: ...]` |
| 大文件截断 | 内容被截断，出现 truncated 提示 |

### 6.2 WriteTool 测试

| 用例 | 预期 |
|------|------|
| 创建新文件 | `[Created ...]`，文件内容正确 |
| 覆写已有文件 | `[Wrote ...]`，旧内容被替换 |
| 自动创建父目录 | 目录链自动建立 |
| 路径是已有目录 | `[Error: path is a directory: ...]` |

### 6.3 EditTool 测试

| 用例 | 预期 |
|------|------|
| 正常替换 (唯一匹配) | `[Edited ...]`，内容正确替换 |
| old_string 不存在 | `[Error: old_string not found ...]` |
| old_string 多次匹配 | `[Error: old_string matches N locations ...]` |
| old_string == new_string | `[Error: ... identical]` |
| 文件不存在 | `[Error: file not found: ...]` |

### 6.4 集成测试

- 注册三个工具后 `to_api_format()` 返回 4 个工具定义 (bash + read + write + edit)
- Agent 能正确路由到新工具并返回结果

---

## 7. 实现步骤

### M1 — ReadTool

- [ ] 创建 `flux/tools/read.py`
- [ ] 编写 `tests/test_file_tools.py` ReadTool 测试
- [ ] 验证通过

### M2 — WriteTool

- [ ] 创建 `flux/tools/write.py`
- [ ] 编写 WriteTool 测试
- [ ] 验证通过

### M3 — EditTool

- [ ] 创建 `flux/tools/edit.py`
- [ ] 编写 EditTool 测试
- [ ] 验证通过

### M4 — CLI 集成

- [ ] 修改 `cli.py` — 导入 & 注册三个工具
- [ ] 更新 `SYSTEM_PROMPT`
- [ ] 编写集成测试 (registry 包含 4 个工具)
- [ ] 全量测试通过 (`pytest`)

---

## 8. 本阶段不做的事

- 文件权限确认 / 用户审批机制
- 二进制文件处理 (图片、PDF 等)
- 文件 diff 预览
- 目录树浏览工具 (ls/glob/find)
- 正则替换 (edit 只做精确字符串替换)
- 文件锁 / 并发写入保护
- undo / 撤销编辑

---

## 9. 验收标准

1. **Read**：`ReadTool().execute(file_path="existing_file.py")` 返回带行号的文件内容
2. **Write**：`WriteTool().execute(file_path="new.txt", content="hello")` 创建文件并返回确认
3. **Edit**：`EditTool().execute(file_path="f.py", old_string="foo", new_string="bar")` 精确替换并返回确认
4. **错误处理**：所有工具对非法输入返回 `[Error: ...]` 格式信息，不抛异常
5. **集成**：`flux` CLI 启动后 LLM 能感知并调用 read/write/edit 工具
6. **全量测试**：`pytest` 全部通过，无回归
