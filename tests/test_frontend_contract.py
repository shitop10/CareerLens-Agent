"""前端契约测试。

不需要浏览器：直接解析 html/index.html，断言前端引用的 DOM 元素、
SSE 事件分支和后端实际产出的事件结构一致。

这能挡住最致命的一类回归——`$("xxx")` 引用了不存在的元素，
`render()` 抛 TypeError 导致整个应用在登录后白屏。
"""

import json
import os
import re
import unittest

from app.core.agent_types import (
    PLANNER_LLM,
    PLANNER_RULE,
    AgentRun,
    AgentToolResult,
    build_metrics_event,
    build_planner_event,
)

_HTML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "html", "index.html"
)


def _load_html() -> str:
    with open(_HTML_PATH, encoding="utf-8") as f:
        return f.read()


class TestDomContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = _load_html()
        cls.ids_in_html = set(re.findall(r'id="([^"]+)"', cls.html))

    def test_every_dollar_lookup_resolves(self):
        referenced = set(re.findall(r'\$\("([^"]+)"\)', self.html))
        missing = sorted(referenced - self.ids_in_html)
        self.assertEqual(missing, [], f"前端引用了不存在的元素 id: {missing}")

    def test_planner_controls_present(self):
        for element_id in ("sidePlanners", "plannerHint", "plannerBadge"):
            self.assertIn(element_id, self.ids_in_html)

    def test_planner_buttons_are_rendered_from_state(self):
        self.assertIn("data-planner=", self.html)
        self.assertIn("renderPlanners()", self.html)

    def test_chat_request_sends_planner_mode(self):
        self.assertIn("planner_mode: state.planner", self.html)

    def test_renderer_handles_planner_and_metrics(self):
        self.assertIn('evt.type === "planner"', self.html)
        self.assertIn('evt.type === "metrics"', self.html)

    def test_planner_banner_uses_backend_label(self):
        """横幅文案来自后端下发的 label，前端不硬编码，避免前后端说法不一致。"""
        self.assertIn('evt.label || evt.planner || "未知"', self.html)

    def test_side_hints_distinguish_both_routes(self):
        for phrase in ("规则路由", "LLM 规划"):
            self.assertTrue(phrase in self.html, f"前端缺少「{phrase}」说明")


class TestHonestLabelling(unittest.TestCase):
    """规则路线不得被表述成模型推理 —— 这是本项目对外表述的红线。"""

    @classmethod
    def setUpClass(cls):
        cls.html = _load_html()

    def test_rule_label_states_no_model_inference(self):
        from app.core.agent_types import PLANNER_LABELS

        label = PLANNER_LABELS[PLANNER_RULE]
        self.assertIn("规则", label)
        self.assertIn("无模型推理", label)
        self.assertNotIn("LLM", label)

    def test_llm_label_states_function_calling(self):
        from app.core.agent_types import PLANNER_LABELS

        self.assertIn("LLM", PLANNER_LABELS[PLANNER_LLM])
        self.assertIn("Function Calling", PLANNER_LABELS[PLANNER_LLM])

    def test_panel_title_carries_planner_label(self):
        self.assertIn('thinkingCtx.setTitle("Agent 工作流 · "', self.html)

    def test_javascript_parses(self):
        """用 node --check 做语法校验（node 不可用时跳过）。"""
        import shutil
        import subprocess
        import tempfile

        node = shutil.which("node")
        if not node:
            self.skipTest("未找到 node，跳过 JS 语法检查")

        blocks = re.findall(r"<script>(.*?)</script>", self.html, re.DOTALL)
        self.assertTrue(blocks, "未找到 script 块")

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write("\n".join(blocks))
            path = f.name
        try:
            result = subprocess.run([node, "--check", path], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            os.unlink(path)


class TestSsePayloadMatchesFrontend(unittest.TestCase):
    """后端事件字段与前端读取的字段必须一致。"""

    def _run(self, planner):
        return AgentRun(
            mode="岗位匹配",
            plan=["步骤A"],
            tool_results=[AgentToolResult("hybrid_retrieval", "混合检索", "证据", {"doc_count": 1})],
            reflection="反思",
            memory_updates=[],
            planner=planner,
            metrics={"plan_latency_ms": 12.5, "total_tokens": 340, "llm_rounds": 2},
        )

    def test_planner_event_fields_read_by_frontend(self):
        event = build_planner_event(self._run(PLANNER_LLM))
        for key in ("label", "planner", "note"):
            self.assertIn(key, event)
        json.dumps(event, ensure_ascii=False)

    def test_metrics_event_fields_read_by_frontend(self):
        event = build_metrics_event(self._run(PLANNER_LLM))
        for key in ("plan_latency_ms", "total_tokens", "llm_rounds", "tool_count"):
            self.assertIn(key, event)
        json.dumps(event, ensure_ascii=False)

    def test_rule_metrics_are_all_zero(self):
        run = self._run(PLANNER_RULE)
        run.metrics = {}
        event = build_metrics_event(run)
        self.assertEqual(event["total_tokens"], 0)
        self.assertEqual(event["llm_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
