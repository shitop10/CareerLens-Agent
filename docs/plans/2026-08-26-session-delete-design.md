# 问题2：会话删除功能 — 设计文档

> 日期：2026-08-26 ｜ 流程：superpowers（design → plan → build → verify）
> 状态：待实施

## 1. 问题描述

历史会话没有删除功能，前端 UI 也没有删除交互键。

**现状**：
- 后端已有 `DELETE /delete/{uuid}` 端点，但**无任何鉴权/归属校验**：未登录也能删、任何登录用户能删任意人的会话（技术债 §9.3 已记录）。
- 前端会话列表（`renderSessions()`）无删除按钮，无交互入口。

## 2. 方案

### 2.1 后端加固 `DELETE /delete/{uuid}`
| 场景 | 行为 |
|------|------|
| 无 Cookie / Cookie 无效 | 401（未登录） |
| 会话不存在 | 404 |
| 会话属于其他用户 | 403（无权删除） |
| 删除自己的会话 | 200 + 级联删除消息 |

实现：增加 `session_id: str = Cookie(None)` 参数；按 `last_cookie` 解析用户；校验 `target.user_id == user.id`。

### 2.2 前端交互
- 会话列表项右侧新增删除按钮（`×`，hover 才显示，避免误触）
- 点击 → `confirm("确定删除该会话？")` → `DELETE /delete/{uuid}` → 成功后：
  - 刷新会话列表
  - 若删除的是当前会话：重置 `currentSessionUuid`、清空聊天区（显示空状态）
- 删除按钮点击用 `stopPropagation` 语义隔离（事件委托中显式判断），不触发会话选中
- 删除失败（401/403/网络）→ alert 提示

## 3. 组件变更

| 文件 | 变更 |
|------|------|
| `app/api/api_service.py` | `delete_s` 增加鉴权与归属校验 |
| `html/index.html` | 会话项删除按钮（CSS + renderSessions + 事件委托 + deleteSession） |
| `tests/test_delete_session.py` | 新增：401/404/403/200 四类场景 |

## 4. 验证策略
- 后端 unittest：未登录 401；删不存在会话 404；A 删 B 的会话 403；删自己会话 200（TestClient，临时用户数据测试后清理）
- 前端浏览器验证：删除当前会话/非当前会话、取消确认、hover 显示按钮
