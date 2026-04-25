实现一个AI编程助手，类似 open code, codex。
# 第一阶段 the-agent-loop

## 解决方案

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> |  Tool   |
| prompt |      |       |      | execute |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                    (loop until finish_reason != "tool_calls")
```

## 工作原理

1. 用户 prompt 作为第一条消息。

```python
messages.append({"role": "user", "content": query})
```

2. 将消息和工具定义一起发给 LLM。

```python
response = client.chat.completions.create(
    model=MODEL, messages=[{"role": "system", "content": SYSTEM}] + messages,
    tools=TOOLS, max_tokens=8000,
)
```

3. 追加助手响应。检查 `finish_reason` -- 如果模型没有调用工具, 结束。

```python
message = response.choices[0].message
messages.append(message)
if response.choices[0].finish_reason != "tool_calls":
    return
```

4. 执行每个工具调用, 收集结果, 作为 tool 消息追加。回到第 2 步。

```python
for tool_call in message.tool_calls:
    output = run_bash(json.loads(tool_call.function.arguments)["command"])
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": output,
    })
```

## 模型选择
1. GLM4.7