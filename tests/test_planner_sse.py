import unittest

from app.api import api_service
from app.core.agent_types import (
    PLANNER_LLM,
    PLANNER_RULE,
    AgentRun,
    AgentToolResult,
    build_metrics_event,
    build_planner_event,
)


class _FakeVectorService:
    def hybrid_search_workflow(self, query):
        return ["[状态] ok\n"], []


class _FakeRagService:
    def __init__(self):
        self.vector_service = _FakeVectorService()


def _run(planner=PLANNER_RULE, reason="", metrics=None):
    return AgentRun(
        mode="岗位匹配",
        plan=["步骤A"],
        tool_results=[AgentToolResult("hybrid_retrieval", "混合检索", "证据", {"doc_count": 1})],
        reflection="反思",
        memory_updates=[],
        planner=planner,
        fallback_reason=reason,
        metrics=metrics or {},
    )


class TestPlannerEvent(unittest.TestCase):
    def test_rule_event_is_honest(self):
        event = build_planner_event(_run(PLANNER_RULE))
        self.assertEqual(event["type"], "planner")
        self.assertEqual(event["planner"], PLANNER_RULE)
        self.assertIn("规则路由", event["label"])
        self.assertNotIn("LLM", event["label"])

    def test_llm_event_label(self):
        event = build_planner_event(_run(PLANNER_LLM))
        self.assertIn("LLM", event["label"])

    def test_fallback_reason_is_surfaced(self):
        event = build_planner_event(_run(PLANNER_RULE, reason="LLM 规划器执行失败：boom"))
        self.assertIn("boom", event["note"])

    def test_event_keys_complete(self):
        self.assertEqual(
            set(build_planner_event(_run()).keys()),
            {"type", "planner", "label", "note"},
        )


class TestMetricsEvent(unittest.TestCase):
    def test_full_metrics(self):
        metrics = {
            "plan_latency_ms": 1234.5,
            "planner_llm_calls": 2,
            "llm_rounds": 2,
            "invalid_tool_calls": 1,
            "input_tokens": 900,
            "output_tokens": 100,
            "total_tokens": 1000,
        }
        event = build_metrics_event(_run(PLANNER_LLM, metrics=metrics))
        self.assertEqual(event["type"], "metrics")
        self.assertEqual(event["total_tokens"], 1000)
        self.assertEqual(event["planner_llm_calls"], 2)
        self.assertEqual(event["plan_latency_ms"], 1234.5)
        self.assertEqual(event["tool_count"], 1)

    def test_missing_metrics_default_to_zero(self):
        event = build_metrics_event(_run())
        for key in ("plan_latency_ms", "total_tokens", "planner_llm_calls", "llm_rounds"):
            self.assertEqual(event[key], 0, key)

    def test_frame_is_serializable(self):
        frame = api_service.sse_event("reasoning", build_metrics_event(_run(metrics={"total_tokens": 5})))
        self.assertTrue(frame.startswith("event: reasoning\n"))
        self.assertIn('"total_tokens": 5', frame)


class TestAgentServiceCache(unittest.TestCase):
    def setUp(self):
        api_service._agent_cache.clear()
        self._original = api_service.get_rag_service
        api_service.get_rag_service = lambda: _FakeRagService()

    def tearDown(self):
        api_service.get_rag_service = self._original
        api_service._agent_cache.clear()

    def test_rule_and_llm_are_distinct(self):
        rule_agent = api_service.get_agent_service(PLANNER_RULE)
        llm_agent = api_service.get_agent_service(PLANNER_LLM)
        self.assertNotIsInstance(llm_agent, type(rule_agent))

    def test_same_mode_returns_cached_instance(self):
        first = api_service.get_agent_service(PLANNER_RULE)
        second = api_service.get_agent_service(PLANNER_RULE)
        self.assertIs(first, second)

    def test_invalid_mode_falls_back_to_rule(self):
        agent = api_service.get_agent_service("nonsense")
        self.assertEqual(agent.planner, PLANNER_RULE)


class TestHealthEndpoint(unittest.TestCase):
    def test_health_exposes_planner_capabilities(self):
        from fastapi.testclient import TestClient

        with TestClient(api_service.app) as client:
            payload = client.get("/health").json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("planner", payload)
        self.assertIn(PLANNER_LLM, payload["planner_modes"])


if __name__ == "__main__":
    unittest.main()
