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

    def test_reasoning_events_memory_empty(self):
        run = AgentRun(
            mode="综合求职分析",
            plan=["步骤A"],
            tool_results=[],
            reflection="无证据",
            memory_updates=[],
        )
        events = run.reasoning_events()
        self.assertEqual(events[0]["type"], "status")
        self.assertEqual(events[-1]["type"], "reflection")


if __name__ == "__main__":
    unittest.main()
