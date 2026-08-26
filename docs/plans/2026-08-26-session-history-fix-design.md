# 问题1：历史会话消息回显 — 设计文档

> 日期：2026-08-26 ｜ 流程：superpowers（design → plan → build → verify）
> 状态：待实施

## 1. 问题描述

**现象**：每次对话后会留下历史会话记录，但离开当前会话再点击回到之前的会话，界面空白——对话记录全部消失。

**根因（已定位，非猜测）**：
- 后端 `GET /chat/{session_uuid}` 端点**已存在且可用**，返回该会话全部消息（`user_input` + `raw_output`）。
- 前端点击历史会话时（`html/index.html` 的 `data-sid` 事件处理），仅设置 `state.currentSessionUuid` 并调用 `render()`；`renderChat()` 只是把 stage 清空并显示"开始新的分析"空状态，**从未调用 `GET /chat/{uuid}` 拉取并渲染历史消息**。因此切换会话后界面空白。

## 2. 方案

纯前端改造，后端零改动：

1. 新增 `loadHistory(uuid)`：
   - `fetch GET /chat/{uuid}`（带 cookie，复用现有 `api()` 封装）
   - 遍历返回的 `data` 数组，逐条渲染：
     - 用户消息 → 右侧用户气泡（复用 `appendMessage("user", ...)`）
     - AI 消息 → 左侧答案气泡（`raw_output` 作为正文，复用 `appendMessage("ai", ...)`）
   - 历史消息只有答案（数据库未存思考过程，不回显 reasoning 面板）
2. 触发点：
   - 点击会话列表项：设 uuid → `render()`（清空 stage）→ `loadHistory(uuid)`（追加历史）
   - 点击"新建分析"：清空 uuid → `render()`（空状态，保持现状）
3. 异常处理：
   - 加载失败（404/网络）：`console.error` 提示，stage 显示"历史加载失败"占位，不崩溃
   - 空会话（无消息）：保持空状态提示

## 3. 组件变更

| 文件 | 变更 |
|------|------|
| `html/index.html` | 新增 `loadHistory()`；`data-sid` 点击处理中调用 |

后端、数据库、Agent 逻辑均不变。

## 4. 验证策略

- 浏览器端到端：新建会话 → 发 2 条消息 → 新建另一会话 → 发 1 条消息 → 点击切回第一个会话 → 确认 2 轮对话完整回显 → 再切回第二个 → 确认 1 轮回显
- 空会话显示"暂无消息"占位
- 回归：新建会话、发送消息、SSE 分离展示不受影响
