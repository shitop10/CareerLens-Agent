# Agent 输出「思考与答案分离」实现计划

> **For implementer:** Use TDD throughout. Write failing test first. Watch it fail. Then implement.

**Goal:** 将 `/chat` 的 SSE 流升级为结构化事件（reasoning/answer 分离），前端分开展示思考过程与最终答案。

**Architecture:** 后端把 `text/plain` 流改为标准 `text/event-stream`（`sse_event()` 格式化函数 + `event_generator` 分类发射）；`agent.py` 增加 `AgentRun.reasoning_events()` 结构化状态输出；前端按 SSE 帧解析渲染到可折叠思考面板 + 答案气泡。

**Tech Stack:** Python 3.13 / FastAPI / 原生 JS（无新依赖）

**关联文档:** `docs/plans/2026-08-26-agent-output-separation-design.md`

---

### Task 1: agent.py 增加 reasoning_events 结构化输出

**Files:**
- Modify: `app/core/agent.py`（`AgentRun` 类）
- Test: `tests/test_agent_reasoning.py`（unittest）

**Step 1: 写失败测试**
```python
# tests/test_agent_reasoning.py
import unittest
from app.core.agent import AgentRun, AgentToolResult

class TestAgentReasoningEvents(unittest.TestCase):
    def test_reasoning_events_structure(self):
        run = AgentRun(
            mode="岗位匹配",
            plan=["步骤A", "步骤B"],
            tool_results=[AgentToolResult("hybrid_retrieval", "混合检索", "证据", {"doc_count": 1})],
            reflection="已召回证据",
            memory_updates=[],
        )
        events = run.reasoning_events()
        types = [e["type"] for e in events]
        self.assertIn("status", types)
        self.assertIn("plan", types)
        self.assertIn("tool", types)
        self.assertIn("reflection", types)
        tool_evt = next(e for e in events if e["type"] == "tool")
        self.assertEqual(tool_evt["title"], "混合检索")
        self.assertEqual(tool_evt["content"], "证据")

if __name__ == "__main__":
    unittest.main()
```

**Step 2: 运行测试确认失败**
`python -m unittest tests.test_agent_reasoning -v`
Expected: FAIL — `AttributeError: 'AgentRun' object has no attribute 'reasoning_events'`

**Step 3: 最小实现**（在 `AgentRun` 中添加方法）
```python
def reasoning_events(self) -> list[dict]:
    events = [
        {"type": "status", "content": "Agent 正在识别任务意图..."},
        {"type": "status", "content": f"Agent 已选择工作模式：{self.mode}"},
        {"type": "status", "content": "Agent 已生成自动任务计划..."},
        *[{"type": "plan", "content": f"- {step}"} for step in self.plan],
        *[{"type": "tool", "title": item.title, "content": item.content} for item in self.tool_results],
        {"type": "status", "content": "Agent 正在进行反思与自我修正..."},
        {"type": "reflection", "content": self.reflection},
    ]
    return events
```

**Step 4: 运行测试确认通过**
`python -m unittest tests.test_agent_reasoning -v`
Expected: PASS

**Step 5: Commit**
`git add app/core/agent.py tests/test_agent_reasoning.py && git commit -m "feat(agent): add structured reasoning_events output"`

---

### Task 2: api_service.py 升级为结构化 SSE

**Files:**
- Modify: `app/api/api_service.py`
- Test: `tests/test_sse_format.py`（unittest）

**Step 1: 写失败测试**
```python
# tests/test_sse_format.py
import unittest, json
from app.api.api_service import sse_event

class TestSseEventFormat(unittest.TestCase):
    def test_frame_format(self):
        frame = sse_event("answer", {"type": "token", "content": "你好"})
        self.assertTrue(frame.startswith("event: answer\n"))
        self.assertTrue(frame.endswith("\n\n"))
        self.assertIn('"content": "你好"', frame)

    def test_json_data(self):
        frame = sse_event("reasoning", {"type": "status", "content": "x"})
        data_line = [l for l in frame.splitlines() if l.startswith("data: ")][0]
        payload = json.loads(data_line[6:])
        self.assertEqual(payload["type"], "status")

if __name__ == "__main__":
    unittest.main()
```

**Step 2: 运行测试确认失败**
Expected: FAIL — `ImportError: cannot import name 'sse_event'`

**Step 3: 最小实现**
```python
def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
```
并改造 `event_generator`：
- 顶部加 `from app.api.api_service import ...` 无需（同文件）；`agent_run` 阶段：
```python
agent_run = await asyncio.to_thread(current_agent.prepare, input_text)
for evt in agent_run.reasoning_events():
    yield sse_event("reasoning", evt)
    await asyncio.sleep(0.08)
```
- 检索状态 stdout 捕获：`yield captured_output` → `yield sse_event("reasoning", {"type": "status", "content": captured_output})`
- 答案：`yield content` → `yield sse_event("answer", {"type": "token", "content": content})`（跳过 `[状态]` 前缀的旧逻辑——该逻辑原用于过滤 LLM 偶发输出，改为按原样发 answer）
- finalize 后：
```python
for item in memory_updates:
    yield sse_event("reasoning", {"type": "memory", "content": item})
yield sse_event("done", {})
```
- `StreamingResponse(event_generator(), media_type="text/event-stream")`

**Step 4: 运行测试确认通过** + `python -m py_compile app/api/api_service.py`

**Step 5: Commit**
`git add app/api/api_service.py tests/test_sse_format.py && git commit -m "feat(api): structured SSE with reasoning/answer events"`

---

### Task 3: 前端 SSE 解析与分离渲染

**Files:**
- Modify: `html/index.html`

**Step 1: 说明**（无自动化测试框架，以浏览器冒烟验证为准）
**Step 2: 实现**
- `sendMessage` 中改为 SSE 帧解析：
```js
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
const thinking = appendThinkingPanel();   // 新增：思考折叠面板
const bubble = appendMessage("ai", "");   // 答案气泡
let answerStarted = false;

function parseFrame(raw) {
  let event = "message", data = "";
  for (const line of raw.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7).trim();
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!data) return;
  let payload; try { payload = JSON.parse(data); } catch { return; }
  if (event === "reasoning") renderReasoning(thinking, payload);
  else if (event === "answer") {
    if (!answerStarted) { answerStarted = true; collapseThinking(thinking); }
    bubble.textContent += payload.content;
  }
}
```
- 新增渲染函数：`appendThinkingPanel()`、`renderReasoning(panel, evt)`（status/plan/tool/reflection/memory 分区样式）、`collapseThinking(panel)`。
- 思考面板默认展开；首个 answer 到达自动收起；toggle 可再展开/收起。
- 所有内容 `escapeHtml` 后注入。

**Step 3: 浏览器冒烟**（启动服务，登录，发消息，肉眼验证：思考折叠区 + 答案分离）
**Step 4: Commit**
`git add html/index.html && git commit -m "feat(ui): split reasoning panel from answer display"`

---

### Task 4: 端到端冒烟 + 收尾

**Files:** 无新增
**Step 1:** `python -m app.api.api_service` 启动，curl 登录 → 建会话 → POST /chat 验证事件序列（reasoning... → answer... → done）
**Step 2:** 回归确认 `/health`、上传等不受影响
**Step 3:** 更新 `架构摘要与开发指南.md`（API 表 + 技术债中"SSE 状态转发 hack"条目）
**Step 4:** Commit
