# PRD04 — SubAgents 功能

## 背景

Flux 当前是单 Agent 架构，所有任务由一个 Agent 处理。对于复杂的多步任务，主 Agent 的上下文会变得很长，影响效率和准确性。

## 目标

实现 SubAgent（子代理）功能，允许主 Agent 将任务委派给独立的 SubAgent 执行。

## 核心需求

1. **任务委派** - 主 Agent 可以将任务委派给 SubAgent
2. **独立执行** - SubAgent 拥有独立的对话历史和执行循环
3. **结果返回** - SubAgent 完成后将结果返回给主 Agent
4. **工具限制** - SubAgent 可访问 bash/read/write/edit，不包含 todo/subagent
5. **禁止嵌套** - SubAgent 不能调用 subagent 工具

## 设计要点

- SubAgent 作为工具（`subagent`）实现，主 Agent 像调用其他工具一样调用它
- 同步阻塞模式：SubAgent 在同一线程中运行，完成后返回结果
- 纯文本返回：SubAgent 返回纯文本结果，UI 包裹由 display 层处理
- 精简提示词：SubAgent 使用专门的精简系统提示词

## 参考文档

详细设计见 [SPEC05-subagent.md](../spec/SPEC05-subagent.md)
