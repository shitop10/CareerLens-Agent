import unittest

from langchain_core.documents import Document

from app.core.agent import CareerAgentService, RulePlannerAgent
from app.core.agent_base import build_agent_service
from app.core.agent_types import PLANNER_RULE


class _FakeVectorService:
    def __init__(self, docs=None, exc=None):
        self.docs = docs if docs is not None else [
            Document(page_content="岗位要求 RAG 能力", metadata={"filename": "jd.txt"})
        ]
        self.exc = exc

    def hybrid_search_workflow(self, query):
        if self.exc:
            raise self.exc
        return ["[状态] 检索完成\n"], list(self.docs)


class _FakeRagService:
    def __init__(self, docs=None, exc=None):
        self.vector_service = _FakeVectorService(docs, exc)


class _AgentFixture(unittest.TestCase):
    def make_agent(self, **kwargs):
        agent = RulePlannerAgent(_FakeRagService(**kwargs))
        return agent


class TestModeDetection(_AgentFixture):
    def test_resume_intent(self):
        run = self.make_agent().prepare("帮我看看简历里的项目经历怎么写")
        self.assertEqual(run.mode, "简历诊断")
        self.assertIn("resume_diagnosis", run.tool_names())

    def test_job_intent(self):
        run = self.make_agent().prepare("这个 JD 里的能力要求我满足吗")
        self.assertEqual(run.mode, "岗位匹配")
        self.assertIn("job_match", run.tool_names())

    def test_fallback_intent(self):
        run = self.make_agent().prepare("帮我做个求职规划")
        self.assertEqual(run.mode, "综合求职分析")
        self.assertIn("job_match", run.tool_names())
        self.assertIn("resume_diagnosis", run.tool_names())

    def test_english_job_keyword(self):
        run = self.make_agent().prepare("show me the job requirement")
        self.assertEqual(run.mode, "岗位匹配")


class TestPlanShape(_AgentFixture):
    def test_plan_length_and_tail(self):
        run = self.make_agent().prepare("帮我看看简历")
        self.assertGreaterEqual(len(run.plan), 5)
        self.assertLessEqual(len(run.plan), 6)
        self.assertIn("反思", run.plan[-2])
        self.assertIn("记忆", run.plan[-1])

    def test_hybrid_retrieval_always_run(self):
        run = self.make_agent().prepare("随便聊聊")
        self.assertIn("hybrid_retrieval", run.tool_names())
        self.assertIn("memory_lookup", run.tool_names())

    def test_tool_results_are_ordered(self):
        run = self.make_agent().prepare("简历和岗位匹配度如何")
        self.assertEqual(run.tool_results[0].name, "memory_lookup")
        self.assertEqual(run.tool_results[1].name, "hybrid_retrieval")


class TestRulePlannerMetrics(_AgentFixture):
    def test_zero_llm_calls_and_tokens(self):
        run = self.make_agent().prepare("帮我看看简历")
        self.assertEqual(run.planner, PLANNER_RULE)
        self.assertEqual(run.metrics["planner_llm_calls"], 0)
        self.assertEqual(run.metrics["total_tokens"], 0)
        self.assertEqual(run.metrics["llm_rounds"], 0)

    def test_latency_is_recorded(self):
        run = self.make_agent().prepare("帮我看看简历")
        self.assertGreaterEqual(run.metrics["plan_latency_ms"], 0)

    def test_no_fallback_reason(self):
        run = self.make_agent().prepare("帮我看看简历")
        self.assertEqual(run.fallback_reason, "")


class TestToolFailureIsolation(_AgentFixture):
    def test_failing_tool_is_captured_not_raised(self):
        agent = self.make_agent()
        run = agent.prepare("帮我看看简历")
        self.assertGreater(len(run.tool_results), 0)

    def test_exception_inside_tool_marks_error(self):
        agent = RulePlannerAgent(_FakeRagService(exc=RuntimeError("boom")))
        run = agent.prepare("帮我看看简历")
        errored = [r for r in run.tool_results if r.metadata.get("error")]
        self.assertEqual(len(errored), 1, [r.name for r in run.tool_results])
        self.assertIn("不确定性", run.reflection)

    def test_empty_corpus_reflection_asks_for_material(self):
        agent = RulePlannerAgent(_FakeRagService(docs=[]))
        run = agent.prepare("帮我看看简历")
        self.assertIn("补充", run.reflection)


class TestBackwardCompatibility(_AgentFixture):
    def test_alias_points_to_rule_planner(self):
        self.assertIs(CareerAgentService, RulePlannerAgent)

    def test_reasoning_events_first_and_last(self):
        run = self.make_agent().prepare("帮我看看简历")
        events = run.reasoning_events()
        self.assertEqual(events[0]["type"], "status")
        self.assertEqual(events[-1]["type"], "reflection")

    def test_factory_returns_rule_planner(self):
        agent = build_agent_service(_FakeRagService(), "rule")
        self.assertIsInstance(agent, RulePlannerAgent)


if __name__ == "__main__":
    unittest.main()
