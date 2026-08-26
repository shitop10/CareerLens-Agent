# Agent 输出「思考与答案分离」设计文档

> 日期：2026-08-26 ｜ 作者：WorkBuddy（遵循 superpowers 工作流）
> 状态：**已批准**（用户确认方案 A + 直接实施）

## 1. 背景与目标

当前 `/chat` 返回 `text/plain` 流，Agent 的思考过程（状态/计划/工具调用/反思/记忆更新）与 LLM 最终答案 token 混在一个通道输出，前端整体渲染进一个气泡，无法像 WorkBuddy 一样"思考过程折叠展示、答案正文展示"。

**目标**：将思考过程与最终答案在协议层分离，前端分开展示。
**非目标**：不改 Agent 业务逻辑、不动数据库结构、不做 Markdown 渲染、不做引用高亮（预留协议，后续可扩展）。

## 2. 方案：SSE 结构化事件（已选）

### 2.1 协议设计

响应升级为标准 `text/event-stream`，三类事件：

| 事件名 | data 结构 | 说明 |
|--------|-----------|------|
| `reasoning` | `{"type": "status\|plan\|tool\|reflection\|memory", "content": "...", "title": "..."}` | 思考过程（title 仅 tool 事件使用） |
| `answer` | `{"type": "token", "content": "..."}` | LLM 答案 token |
| `done` | `{}` | 流结束标记 |

事件帧格式（标准 SSE）：
```
event: reasoning
data: {"type": "status", "content": "Agent 正在识别任务意图..."}

event: answer
data: {"type": "token", "content": "基于"}

event: done
data: {}

```

### 2.2 数据流

```
prepare() → reasoning 事件（status×3 + plan×6 + tool×N + reflection）
    ↓
RAG 链流式生成（redirect_stdout 捕获的检索状态 → reasoning/status 事件）
    ↓
qwen-max token → answer 事件（逐 token）
    ↓
finalize() → reasoning/memory 事件 → done 事件
```

### 2.3 前端展示

每条 AI 消息结构：

```
┌─ Agent 工作流（可折叠，默认展开，首个 answer 到达后自动收起）──┐
│  状态行（灰字小字）                                            │
│  计划（带序号的条目）                                          │
│  工具调用（卡片：标题 + 内容摘要）                             │
│  反思（高亮提示框）                                            │
│  记忆更新（提示行）                                            │
└───────────────────────────────────────────────────────────────┘
┌─ 答案气泡（正文）──────────────────────────────────────────────┐
│  （流式追加 answer token）                                     │
└───────────────────────────────────────────────────────────────┘
```

## 3. 组件改造

### 3.1 `app/core/agent.py`
- `AgentRun` 新增 `reasoning_events() -> list[dict]`：把状态行转成结构化事件（status/plan/tool/reflection），`status_lines()` 保留兼容。
- memory 事件在 `finalize()` 后单独产生（`finalize` 返回 updates，调用方包装成 memory 事件）。

### 3.2 `app/api/api_service.py`
- 新增 `sse_event(event, data) -> str` 格式化函数（JSON ensure_ascii=False）。
- `event_generator()` 改造：
  - prepare 后逐条发 `reasoning` 事件（含 plan/tool/reflection）
  - 检索状态 print（stdout_capture）→ `reasoning/status` 事件
  - 答案 chunk → `answer/token` 事件
  - finalize 后发 `reasoning/memory` 事件 → `done` 事件
- `StreamingResponse(media_type="text/event-stream")`。

### 3.3 `html/index.html`
- `sendMessage` 改为 SSE 帧解析（按 `\n\n` 分帧、`event:`/`data:` 拆行）。
- 新增渲染：`reasoning` 事件 → 折叠面板（toggle 按钮 + 分区内容）；`answer` 事件 → 答案气泡 textContent 追加；`done` → 结束。
- 首个 answer 到达时自动收起思考面板（WorkBuddy 交互）。
- 事件内容一律 `escapeHtml`。

## 4. 错误处理
- SSE 生成异常：由 FastAPI 流断开兜底；前端 `finally` 恢复发送按钮。
- LLM 调用失败（无 key/限流）：answer 事件为空，前端保持已有行为（显示已收到的 reasoning）。

## 5. 测试策略
- 单元：`unittest`（stdlib）验证 `sse_event` 帧格式与 `reasoning_events` 结构（不依赖 LLM）。
- 冒烟：启动服务 + 用配置好的 API key 走一次真实 `/chat` 流，验证事件序列。

## 6. 变更清单
| 文件 | 变更 |
|------|------|
| `app/core/agent.py` | +`AgentRun.reasoning_events()` |
| `app/api/api_service.py` | +`sse_event()`，改造 `event_generator` |
| `html/index.html` | SSE 解析 + 折叠思考面板 + 答案分离渲染 |
| `docs/plans/2026-08-26-agent-output-separation-design.md` | 本文档 |
