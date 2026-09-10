import os
import shutil
import tempfile
import unittest

from langchain_core.documents import Document

from app.core.agent_base import BaseCareerAgent, LongTermMemory, build_agent_service
from app.core.agent_types import PLANNER_RULE, AgentRun, AgentToolResult


class _FakeVectorService:
    def hybrid_search_workflow(self, query):
        return ["[状态] ok\n"], [Document(page_content="证据", metadata={"filename": "f.txt"})]


class _FakeRagService:
    def __init__(self):
        self.vector_service = _FakeVectorService()


def _run(**overrides):
    payload = {
        "mode": "岗位匹配",
        "plan": ["解析目标", "读取记忆", "混合检索"],
        "tool_results": [AgentToolResult("hybrid_retrieval", "混合检索工具", "资料[1]: 证据")],
        "reflection": "证据充分",
        "memory_updates": [],
    }
    payload.update(overrides)
    return AgentRun(**payload)


class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cl_mem_")
        self.path = os.path.join(self.dir, "memory.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_load_recent_empty_when_missing(self):
        self.assertEqual(LongTermMemory(self.path).load_recent(), [])

    def test_update_then_load(self):
        memory = LongTermMemory(self.path)
        updates = memory.update("简历怎么写", "建议量化表达", "简历诊断")
        self.assertTrue(updates)
        contents = [item["content"] for item in memory.load_recent()]
        self.assertIn("简历优化偏好：优先给出可直接替换的项目表达", contents)

    def test_update_is_idempotent(self):
        memory = LongTermMemory(self.path)
        first = memory.update("RAG 怎么讲", "用 RRF 融合", "综合求职分析")
        self.assertTrue(first)
        second = memory.update("RAG 怎么讲", "用 RRF 融合", "综合求职分析")
        self.assertEqual(second, ["本轮未发现新的稳定偏好或长期待办，保持现有记忆"])

    def test_load_recent_respects_limit(self):
        memory = LongTermMemory(self.path)
        memory.update("补齐短板", "需要补强评估", "岗位匹配")
        memory.update("简历量化", "量化表达", "简历诊断")
        self.assertLessEqual(len(memory.load_recent(limit=1)), 1)

    def test_corrupt_line_is_skipped(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not json}\n")
            f.write('{"content": "合法记录", "mode": "x", "time": "t"}\n')
        items = LongTermMemory(self.path).load_recent()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["content"], "合法记录")


class TestComposeAgentContext(unittest.TestCase):
    def setUp(self):
        self.agent = build_agent_service(_FakeRagService(), PLANNER_RULE)

    def test_contains_plan_tools_and_reflection(self):
        context = self.agent.compose_agent_context(_run())
        self.assertIn("解析目标", context)
        self.assertIn("混合检索工具", context)
        self.assertIn("证据充分", context)

    def test_declares_required_output_shape(self):
        context = self.agent.compose_agent_context(_run())
        for section in ("任务规划", "结论", "建议", "下一步"):
            self.assertIn(section, context)


class TestFinalize(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cl_fin_")
        self.agent = build_agent_service(_FakeRagService(), PLANNER_RULE)
        self.agent.memory = LongTermMemory(os.path.join(self.dir, "m.jsonl"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_finalize_writes_memory_updates_back_to_run(self):
        run = _run()
        updates = self.agent.finalize("RAG 项目怎么讲", "回答内容", run)
        self.assertEqual(updates, run.memory_updates)
        self.assertTrue(updates)


class TestBuildAgentService(unittest.TestCase):
    def test_rule_mode_returns_rule_agent(self):
        agent = build_agent_service(_FakeRagService(), PLANNER_RULE)
        self.assertIsInstance(agent, BaseCareerAgent)
        self.assertEqual(agent.planner, PLANNER_RULE)

    def test_unknown_mode_falls_back_to_rule(self):
        agent = build_agent_service(_FakeRagService(), "nonsense")
        self.assertEqual(agent.planner, PLANNER_RULE)

    def test_shares_tool_registry(self):
        agent = build_agent_service(_FakeRagService(), PLANNER_RULE)
        self.assertEqual(
            agent.tools.names(),
            ["memory_lookup", "hybrid_retrieval", "resume_diagnosis", "job_match"],
        )

    def test_base_prepare_is_abstract(self):
        with self.assertRaises(NotImplementedError):
            BaseCareerAgent(_FakeRagService()).prepare("q")

    def test_llm_mode_returns_llm_planner(self):
        from app.core.agent_types import PLANNER_LLM
        from app.core.llm_agent import LlmPlannerAgent

        agent = build_agent_service(_FakeRagService(), PLANNER_LLM)
        self.assertIsInstance(agent, LlmPlannerAgent)
        self.assertEqual(agent.planner, PLANNER_LLM)


if __name__ == "__main__":
    unittest.main()
