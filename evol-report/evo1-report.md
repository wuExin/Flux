# Flux Meta-Harness Evo1 运行报告

**日期**: 2026-04-21
**运行环境**: Windows 11, GLM-4.7 (ZhipuAI, temp=0.0), Claude Opus (proposer)
**评估任务**: 14 个 search split 编程任务（5 add + 5 fix + 4 refactor）

---

## 一、系统是什么

Flux 是一个**自研的编程 agent**——用 LLM (GLM-4.7) 驱动，通过循环调用工具（读文件、写文件、跑命令等）来完成编程任务。Meta-Harness 是一套**自动化进化框架**，用来寻找"最好的方式"来指挥这个 agent 工作。

### 1.1 核心概念

```
┌─────────────────────────────────────────────────────┐
│  编程任务 (task.md)                                   │
│  例: "repo/app.py 有 import 错误，请修复"              │
└──────────────┬──────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────┐
│  Harness (指挥策略)                                   │
│  · system prompt: 告诉 LLM 它是谁、有什么工具、怎么做  │
│  · prepare_messages(): 每一步送什么上下文给 LLM        │
│  · post_step(): 每一步执行完做什么后处理               │
│  · select_tools(): 暴露哪些工具                       │
│  · get_config(): 循环参数（最大步数、提醒频率等）       │
└──────────────┬──────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────┐
│  Agent 循环 (flux/agent.py)                          │
│  for step in range(max_iterations):                  │
│    1. harness.prepare_messages() → 处理历史消息        │
│    2. LLM.chat(messages, tools, system_prompt)       │
│    3. 如果 LLM 说"完成" → 退出循环                    │
│    4. 如果 LLM 调工具 → 执行工具，记录结果             │
│    5. harness.post_step() → 后处理                    │
└──────────────┬──────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────┐
│  验证 (verify.sh)                                    │
│  运行 pytest 检查 agent 的改动是否正确                 │
│  returncode == 0 → PASS, 否则 FAIL                   │
└─────────────────────────────────────────────────────┘
```

### 1.2 一个任务的完整执行过程

以 `fix_import_error` 为例，task.md 内容：

> `repo/app.py` 在运行时会报 `ModuleNotFoundError`。项目结构中有 `utils/` 包提供工具函数。请找出导入错误并修复。

实际 bug：`app.py` 第 1 行写的是 `from utils.helper import ...`，正确的是 `from utils.helpers import ...`（少了个 s）。

执行链条：

```
1. runner.py 把 benchmark/tasks/search/fix_import_error/repo/ 拷贝到临时目录
2. 读取 task.md 作为用户输入
3. harness.build_system_prompt() → 生成 system prompt
4. 构造 Agent：注入 LLM client、工具集、system prompt、harness 的回调函数
5. Agent.run(task_text) 开始循环：

   Step 0: LLM 看到 system prompt + "请修复 import 错误"
         → LLM 决定调 read(file="app.py")
         → 工具执行，返回文件内容
         → harness.post_step() 被调用

   Step 1: LLM 看到文件内容，发现 `from utils.helper import ...`
         → LLM 决定调 bash(command="ls utils/")
         → 发现目录里是 helpers.py 不是 helper.py
         → harness.post_step() 被调用

   Step 2: LLM 决定调 edit(file="app.py", old="utils.helper", new="utils.helpers")
         → 工具执行，修改成功
         → harness.post_step() 被调用

   Step 3: LLM 决定调 bash(command="python -m pytest test_app.py -v")
         → 测试通过
         → harness.post_step() 被调用

   Step 4: LLM 输出 "修复完成" → finish_reason="stop" → 退出循环

6. runner.py 运行 verify.sh（即 pytest）→ returncode=0 → PASS
7. 保存 result.json（pass/steps/tokens/wall_time）和 trace.jsonl（完整对话记录）
```

### 1.3 不同 Harness 如何影响这个过程

Harness 框架代码（`base.py`, `runner.py`, `utils.py`）**始终不变**。进化改变的是 `harness/agents/` 下的配置文件——每个 Python 文件定义一种"指挥策略"。

**同一个任务，不同 harness 下 agent 的行为可能完全不同**：

| 维度 | 控制什么 | 对 agent 行为的影响 |
|---|---|---|
| **`build_system_prompt()`** | 告诉 LLM 它是谁、有什么规则 | prompt 越长越详细 → LLM 可能更规范但更啰嗦；prompt 精简 → LLM 更直接 |
| **`prepare_messages()`** | 每步送给 LLM 的历史消息 | 全量发送 → token 消耗大但信息全；压缩/摘要 → 省 token 但可能丢细节 |
| **`post_step()`** | 每步执行后的状态处理 | 可以追踪文件变更、检测错误模式、注入纠正消息 |
| **`select_tools()`** | 暴露哪些工具 | 有 todo 工具 → 会花步数做任务管理；没有 → 直接干活 |
| **`get_config()`** | 循环参数 | nag_threshold 低 → 频繁提醒检查 todo 进度，消耗额外步数 |

---

## 二、三个 Baseline 具体做了什么

### baseline_default（复刻当前 Flux agent）

```python
# system prompt: 49 行，详细列出环境信息、5 个工具说明、使用指南、todo 语法示例
# prepare_messages(): 原样返回，不做任何处理
# select_tools(): bash, read, write, edit, todo（5 个工具）
# post_step(): 无操作
# get_config(): max_iterations=50, nag_threshold=3（每 3 步没完成 todo 就提醒一次）
```

**行为特征**：agent 每 3 步收到一条 todo 提醒消息，system prompt 较长，LLM 会倾向于先规划再执行。平均 7.1 步完成任务，307K tokens。

### baseline_minimal（极简指令）

```python
# system prompt: 仅 3 行 —— "You are a coding assistant. Working directory: {cwd}\n
#                            Tools: bash, read, write, edit. Read files before modifying. When done, stop."
# prepare_messages(): 原样返回，不做任何处理
# select_tools(): bash, read, write, edit（4 个工具，没有 todo）
# post_step(): 无操作
# get_config(): max_iterations=50, nag_threshold=999（实际永远不提醒）, use_todo=False
```

**行为特征**：没有 todo 工具，不会被提醒检查进度，prompt 极短。LLM 直奔主题，不做多余的任务管理。平均 6.8 步，167K tokens——比 default 省一半 token。

### baseline_structured（强制结构化流程）

```python
# system prompt: 中等长度，强制 Plan→Read→Edit→Verify→Track 五步流程
#                "1. Plan first: 用 todo 创建任务列表"
#                "2. Read before edit: 必须先读再改"
#                "3. Verify: 改完跑测试"
#                "4. Track progress: 完成后标记 todo"
# prepare_messages(): step > 5 后启用 sliding_window（裁剪到 30K token 预算）
# select_tools(): bash, read, write, edit, todo（5 个工具）
# post_step(): 无操作
# get_config(): max_iterations=50, nag_threshold=2（每 2 步就提醒一次）
```

**行为特征**：LLM 被强制要求先建 todo 列表再干活，每 2 步被提醒一次进度，5 步后开始裁剪老消息。结果是平均 9.4 步——强制流程增加了不必要的"管理开销"。

### Baseline 结论

| Baseline | 通过率 | 平均步数 | Token | 耗时 |
|---|---|---|---|---|
| baseline_default | 100% (14/14) | 7.1 | 307K | 9m34s |
| baseline_minimal | 100% (14/14) | 6.8 | 167K | 7m14s |
| baseline_structured | 100% (14/14) | 9.4 | — | 9m26s |

三个都 100% 通过，说明 GLM-4.7 在这 14 个任务上能力足够。区别只在效率：**指令越精简、管理开销越少，agent 越高效**。

---

## 三、进化循环做了什么

### 3.1 进化循环的整体流程

```
meta_harness.py 启动
│
├── Phase 0: 基线评估
│   对 3 个 baseline 各跑 14 个任务，建立 frontier（当前最优: baseline_default @ 100%）
│
├── Iteration 1:
│   ├── 1. Propose（Claude Opus 作为 proposer）
│   │   └── 分析 baseline trace → 输出 2 个新 harness .py 文件 + pending_eval.json
│   ├── 2. Validate
│   │   └── import 检查 + 在 fix_import_error 上 smoke test
│   ├── 3. Benchmark
│   │   └── 2 个候选各跑 14 个任务，记录 result.json + trace.jsonl
│   └── 4. Update frontier
│       └── 如果新候选 pass_rate > 当前 frontier → 替换
│
├── Iteration 2: 同上，proposer 额外读取 iteration 1 的 trace 数据
├── Iteration 3: 同上，proposer 额外读取 iteration 1+2 的 trace 数据
│
└── 输出: evolution_summary.jsonl（每个候选的指标），frontier_val.json（最优 harness）
```

### 3.2 Proposer 具体做了什么

Proposer 是一个 **Claude Opus agent**，通过 `claude_wrapper.run()` 调用，带有一份 9,410 字符的 skill 指令（`.claude/skills/meta-harness/SKILL.md`），指导它：

1. **分析阶段**（用 Explore subagent）：
   - 读 `evolution_summary.jsonl` 了解历史尝试
   - 读 `frontier_val.json` 看当前最优
   - 读最优 harness 的源码
   - 读 14 个任务的 `trace.jsonl`（完整对话记录），找效率瓶颈

2. **实现阶段**（用 Agent subagent）：
   - 基于分析提出 2 个假设
   - 写 2 个新的 `harness/agents/xxx.py` 文件
   - 每个文件继承 `AgentHarness`，覆盖 5 个方法
   - 运行 import 验证

3. **输出**：`pending_eval.json`，包含候选名称、假设、变异轴、变更描述

### 3.3 每轮迭代的具体候选

**Iteration 1**

| 候选 | 变异轴 | 做了什么 |
|---|---|---|
| **systematic_workflow** | A (system prompt) | 重写 system prompt，强制 Explore→Understand→Plan→Implement→Verify 流程；step 8 后启用 25K token 的 sliding window；max_iterations=40, nag_threshold=5 |
| **active_context** | B+D (上下文+钩子) | `post_step()` 解析每步的 tool_call，追踪 files_read/files_modified/errors；`prepare_messages()` 在 step 3 后注入上下文摘要，step 6 后将旧 tool output 截断到 8K 字符，step 10 后启用 20K token 的 sliding window |

**Iteration 2**（proposer 读了 iter1 的 trace，发现 active_context 步数更少）

| 候选 | 变异轴 | 做了什么 |
|---|---|---|
| **direct_action** | A+B (精简指令+压缩) | 500 字符超短 system prompt；去掉 todo 工具；`prepare_messages()` 将最近 2 轮之前的**所有**旧 tool output 替换为一行语义摘要（如 `"Read src/main.py → 45 lines"`），不是截断而是完全替换；max_iterations=35 |
| **guided_recovery** | D+B (错误恢复+压缩) | `post_step()` 检测 3 种错误模式：路径混淆、edit 内容不匹配、重复失败；检测到后注入针对性纠正消息；`prepare_messages()` 保留最近 3 轮，其余做语义摘要；去掉 todo 工具 |

**Iteration 3**（proposer 读了 iter1+2 的 trace，发现 direct_action token 最少）

| 候选 | 变异轴 | 做了什么 |
|---|---|---|
| **hybrid_compress** | B+D (压缩+纠错) | 合并 direct_action 的 2 轮语义压缩 + guided_recovery 的 3 模式错误检测；纠正消息作为紧凑 system message 注入；去掉 todo；超短 prompt |
| **working_memory** | B+D (结构化记忆) | `post_step()` 构建结构化状态（files_read 含行数、files_modified 含操作数、bash 历史、错误日志）；`prepare_messages()` 注入 working memory 摘要 + 将除最近一次 read 外的所有旧 tool output 压缩为一行；prompt 告诉 agent "信任 working memory，避免重复读" |

---

## 四、过程中遇到的问题

运行过程中遇到 3 个 Windows 平台兼容性 Bug，均在 `claude_wrapper.py` 中。

### Bug 1: CLI 路径解析失败 (exit=127)

**现象**: 进化迭代全部以 exit=127 失败，报错 `[WinError 2] 系统找不到指定的文件`。

**根因**: `build_command()` 硬编码 `"claude"` 字符串。Windows 上 npm 安装的 CLI 工具以 `.cmd` 批处理文件存在（`claude.cmd`），而 `subprocess.Popen(shell=False)` 无法解析 `.cmd` 扩展名。

**修复**: 新增 `_find_claude_cli()` 函数，通过 `shutil.which("claude")` 解析完整路径。

### Bug 2: cmd.exe 命令行长度超限 (exit=1)

**现象**: 修复 Bug 1 后，proposer 改为 exit=1，报错为乱码（GBK 编码的中文"参数太长"）。

**根因**: `--append-system-prompt` 参数携带 9,410 字符的 skill 内容，总命令行长度 10,366 字符，超过 cmd.exe 的 8,191 字符硬限制。通过 `.cmd` 文件调用时必须经过 cmd.exe。

**修复**: 解析 `claude.CMD` 内容，提取 Node.js 入口点 `cli2.js` 的路径，改为返回 `[node.exe, cli2.js]` 列表，直接调用 `node.exe`，走 Windows `CreateProcess` API（32,767 字符限制），绕过 cmd.exe。

### Bug 3: GBK 编码崩溃

**现象**: 两类编码错误交替出现：
- `UnicodeDecodeError`: subprocess 读取 Claude UTF-8 输出时以 GBK 解码
- `UnicodeEncodeError`: `write_text()` 写入含 Unicode 字符（如 ✓）的内容时 GBK 编码失败

**根因**: 中文 Windows 系统默认编码为 GBK (CP936)，Python 的 `subprocess.Popen(text=True)` 和 `pathlib.Path.write_text()` / `read_text()` 不指定编码时均使用系统默认编码。

**修复**:
- subprocess: 加 `encoding='utf-8', errors='replace'`
- 5 处 `write_text()`: 加 `encoding="utf-8"`
- 5 处 `read_text()`: 加 `encoding="utf-8"`
- 共 10 处文件 I/O + 1 处 subprocess 全部显式指定 UTF-8

---

## 五、代码修改清单

所有修改均在 `E:\Project\github\private\Flux\claude_wrapper.py`：

| 修改 | 位置 | 内容 |
|---|---|---|
| 新增 `import sys` | 顶层 imports | 支持 `sys.platform` 检测 |
| 新增 `_find_claude_cli()` | `build_command()` 之前 | Windows CLI 路径解析，返回 `[node, cli2.js]` |
| 修改 `build_command()` | cmd 列表构造 | `_find_claude_cli()` → `*_find_claude_cli()` splat 解包 |
| 修改 `subprocess.Popen()` | `run()` 函数 | 加 `encoding='utf-8', errors='replace'` |
| 修改 5 处 `write_text()` | `log_session()` | 全部加 `encoding="utf-8"` |
| 修改 5 处 `read_text()` | `load_skill()` / `load_skills()` | 全部加 `encoding="utf-8"` |

---

## 六、最终结果

### 全部评估数据

| 阶段 | 候选 | 变异轴 | 通过率 | 平均步数 | 总 Token | 核心策略 |
|---|---|---|---|---|---|---|
| Phase 0 | baseline_default | — | 100% | 7.1 | 307K | 完整指令 + todo |
| Phase 0 | baseline_minimal | — | 100% | 6.8 | 167K | 极简指令，无 todo |
| Phase 0 | baseline_structured | — | 100% | 9.4 | — | 强制流程 + sliding window |
| Iter 1 | systematic_workflow | A | 100% | 8.9 | **390K** | 强制 5 步流程（退步） |
| Iter 1 | active_context | B+D | 100% | 5.4 | 215K | 文件追踪 + 渐进截断 |
| Iter 2 | direct_action | A+B | 100% | 5.8 | 150K | 超短 prompt + 语义摘要 |
| Iter 2 | guided_recovery | D+B | 100% | 5.6 | 174K | 错误检测 + 纠正注入 |
| Iter 3 | hybrid_compress | B+D | 100% | 5.4 | **147K** | 语义压缩 + 错误纠正 |
| Iter 3 | working_memory | B+D | 100% | 11.9 | 357K | 结构化记忆（退步） |

### Token 效率提升趋势

```
baseline_default (起点):    307K tokens, 7.1 步
         ↓ iter 1: active_context      215K tokens (-30%)
         ↓ iter 2: direct_action       150K tokens (-51%)
         ↓ iter 3: hybrid_compress      147K tokens (-52%)  ← 最优
```

最优候选 **hybrid_compress** 相比 baseline_default：
- Token 消耗：307K → 147K，**节省 52%**
- 平均步数：7.1 → 5.4，**减少 24%**
- 通过率保持 100%

### 关键发现

1. **精简优于繁复**: 500 字符 prompt 达到了和 49 行 prompt 相同的通过率，但省一半 token
2. **语义摘要 > 截断**: 把旧 tool output 替换为 `"Read src/main.py → 45 lines"` 比截断到 8K 字符更有效
3. **去掉 todo 工具有益**: todo 工具的提醒机制（nag）会注入额外 system message，消耗步数和 token
4. **过度结构化有害**: systematic_workflow 的强制流程增加 27% token，working_memory 的状态维护使步数翻倍
5. **错误检测有价值但不够**: guided_recovery 的纠错能减少错误恢复步数，但在这些简单任务上效果有限

### 局限性

- **测试集太简单**: 所有 harness 均达 100% 通过率，correctness 无法产生进化压力
- **Frontier 指标单一**: 仅看 pass_rate，无法感知效率差异，frontier 始终未被替换（始终是 baseline_default）
- **建议**: 下一轮需用更难的 test split，或将 token efficiency 纳入 fitness 函数

---

## 七、下一步计划

1. **更新 fitness 函数**: 在 pass_rate 之外加入 token_efficiency 权重，形成 `score = pass_rate * 0.7 + token_efficiency * 0.3`
2. **使用更难的测试集**: 切换到 harder split 或增加更复杂的任务，使通过率产生分化
3. **运行 evo2**: 在新条件下重新运行进化循环，验证 hybrid_compress 策略的泛化能力
