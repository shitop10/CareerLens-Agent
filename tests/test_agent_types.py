import unittest

from app.core.agent_types import (
    PLANNER_LLM,
    PLANNER_LABELS,
    PLANNER_RULE,
    AgentRun,
    AgentToolResult,
    extract_token_usage,
)


class _Msg:
    """可变响应的假消息对象。"""

    def __init__(self, usage_metadata=None, response_metadata=None):
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata
        if response_metadata is not None:
            self.response_metadata = response_metadata


class TestAgentRunBackwardCompat(unittest.TestCase):
    """既有 tests/test_agent_reasoning.py 的构造方式是外部契约，必须保持。"""

    def test_legacy_keyword_construction(self):
        run = AgentRun(
            mode="岗位匹配",
            plan=["步骤A"],
            tool_results=[],
            reflection="无证据",
            memory_updates=[],
        )
        self.assertEqual(run.planner, PLANNER_RULE)
        self.assertEqual(run.fallback_reason, "")
        self.assertEqual(run.metrics, {})

    def test_reasoning_events_shape_unchanged(self):
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

    def test_tool_event_keys_preserved(self):
        run = AgentRun(
            mode="岗位匹配",
            plan=["步骤A"],
            tool_results=[AgentToolResult("hybrid_retrieval", "混合检索", "证据", {"doc_count": 1})],
            reflection="已召回证据",
            memory_updates=[],
        )
        tool_evt = next(e for e in run.reasoning_events() if e["type"] == "tool")
        self.assertEqual(tool_evt["title"], "混合检索")
        self.assertEqual(tool_evt["content"], "证据")

    def test_events_carry_planner_marker(self):
        run = AgentRun(
            mode="岗位匹配",
            plan=["步骤A"],
            tool_results=[],
            reflection="r",
            memory_updates=[],
            planner=PLANNER_LLM,
        )
        self.assertTrue(all(e.get("planner") == PLANNER_LLM for e in run.reasoning_events()))


class TestPlannerLabel(unittest.TestCase):
    def test_known_labels(self):
        run = AgentRun("m", [], [], "r", [], planner=PLANNER_LLM)
        self.assertEqual(run.planner_label, PLANNER_LABELS[PLANNER_LLM])

    def test_unknown_label_falls_back_to_raw_value(self):
        run = AgentRun("m", [], [], "r", [], planner="weird")
        self.assertEqual(run.planner_label, "weird")


class TestExtractTokenUsage(unittest.TestCase):
    def test_tongyi_response_metadata_format(self):
        msg = _Msg(response_metadata={
            "token_usage": {"input_tokens": 18, "output_tokens": 1, "total_tokens": 19}
        })
        self.assertEqual(
            extract_token_usage(msg),
            {"input_tokens": 18, "output_tokens": 1, "total_tokens": 19},
        )

    def test_langchain_usage_metadata_format(self):
        msg = _Msg(usage_metadata={
            "input_tokens": 5, "output_tokens": 7, "total_tokens": 12
        })
        self.assertEqual(
            extract_token_usage(msg),
            {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
        )

    def test_missing_usage_returns_zeros(self):
        self.assertEqual(
            extract_token_usage(_Msg()),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    def test_none_message_is_safe(self):
        self.assertEqual(
            extract_token_usage(None),
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )

    def test_total_synthesized_when_absent(self):
        msg = _Msg(response_metadata={"token_usage": {"input_tokens": 3, "output_tokens": 4}})
        self.assertEqual(extract_token_usage(msg)["total_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
