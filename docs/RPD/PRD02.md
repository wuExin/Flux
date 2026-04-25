# AI流程可视化
1. 需要把AI的回复，工具调用的结果可视化
2. 不要超过200个字

## 设计文档

详见 [SPEC03-display.md](../spec/SPEC03-display.md)

## 实现状态: 已完成

- `flux/display.py` — ANSI终端可视化模块（show_tool_call, show_tool_result, show_response）
- `flux/agent.py` — on_tool_call / on_tool_result 回调参数
- `flux/cli.py` — 回调绑定，AI回复使用 show_response 输出
- `tests/test_display.py` — 11项显示模块测试
- `tests/test_agent.py` — 3项回调测试