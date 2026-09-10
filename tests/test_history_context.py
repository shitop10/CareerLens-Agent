import unittest

from langchain_core.messages import AIMessage, HumanMessage

from app.core.agent import RulePlannerAgent
from app.core.agent_base import (
    build_history_digest,
    default_history_loader,
    format_turn_digest,
    parse_history_messages,
)


def _msgs():
    return [
        HumanMessage(content="帮我看看简历里专业技能怎么改"),
        AIMessage(content="建议把「熟悉 Python」改成「用 Python 重构了检索模块，召回率提升 12%」。" * 3),
        HumanMessage(content="大模型算法岗要求 LoRA 微调吗"),
        AIMessage(content="JD 里确有 LoRA 要求，建议补一个微调实验。" * 3),
        HumanMessage(content="那这个呢"),
    ]


class TestParseHistoryMessages(unittest.TestCase):
    def test_pairs_user_and_ai(self):
        turns = parse_history_messages(_msgs())
        self.assertEqual(len(turns), 3)
        self.assertIn("专业技能", turns[0]["user"])
        self.assertIn("召回率", turns[0]["ai"])

    def test_trailing_user_without_answer(self):
        turns = parse_history_messages([HumanMessage(content="第一个问题")])
        self.assertEqual(turns, [{"user": "第一个问题", "ai": ""}])

    def test_empty_input(self):
        self.assertEqual(parse_history_messages([]), [])

    def test_tolerates_plain_strings(self):
        turns = parse_history_messages(["纯字符串", None])
        self.assertEqual(turns, [{"user": "纯字符串", "ai": ""}])

    def test_ai_without_preceding_user(self):
        turns = parse_history_messages([AIMessage(content="孤立回答")])
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["ai"], "孤立回答")


class TestBuildHistoryDigest(unittest.TestCase):
    def test_keeps_only_recent_turns(self):
        turns = parse_history_messages(_msgs())
        digest = build_history_digest(turns, max_turns=2, max_chars=80)
        self.assertNotIn("专业技能", digest)
        self.assertIn("LoRA", digest)
        self.assertIn("那这个呢", digest)

    def test_truncates_each_side(self):
        turns = [{"user": "长" * 300, "ai": "答" * 300}]
        digest = build_history_digest(turns, max_turns=1, max_chars=20)
        self.assertIn("…", digest)
        self.assertLess(len(digest), 120)

    def test_empty_turns_returns_empty_string(self):
        self.assertEqual(build_history_digest([], max_turns=3, max_chars=80), "")

    def test_blank_turns_are_skipped(self):
        digest = build_history_digest([{"user": "", "ai": ""}], max_turns=3, max_chars=80)
        self.assertEqual(digest, "")

    def test_digest_is_numbered(self):
        turns = parse_history_messages(_msgs())
        digest = build_history_digest(turns, max_turns=2, max_chars=80)
        self.assertIn("[1]", digest)
        self.assertIn("[2]", digest)


class TestFormatTurnDigest(unittest.TestCase):
    def test_no_truncation_when_short(self):
        self.assertEqual(format_turn_digest("问题", "回答", 80), "问题 ｜ 回答")

    def test_truncation_adds_ellipsis(self):
        text = format_turn_digest("问" * 50, "答" * 50, 10)
        self.assertIn("…", text)
        self.assertLessEqual(len(text), 10 * 2 + 10)

    def test_short_side_not_truncated(self):
        text = format_turn_digest("问", "答" * 50, 10)
        self.assertEqual(text.count("…"), 1)

    def test_empty_ai_side_renders_placeholder(self):
        self.assertIn("(无回答)", format_turn_digest("问题", "", 80))


class TestDefaultHistoryLoaderShape(unittest.TestCase):
    def test_returns_empty_without_session(self):
        self.assertEqual(default_history_loader(None), [])
        self.assertEqual(default_history_loader(""), [])

    def test_unknown_session_returns_empty(self):
        self.assertEqual(default_history_loader("no-such-uuid-xyz"), [])


class _FakeVectorService:
    def hybrid_search_workflow(self, query):
        return ["[状态] ok\n"], []


class _FakeRagService:
    def __init__(self):
        self.vector_service = _FakeVectorService()


class TestRulePlannerUsesHistoryForIntent(unittest.TestCase):
    """规则路线：当前问题无意图关键词时，回落到历史对话推断意图。"""

    def _agent(self, turns):
        agent = RulePlannerAgent(_FakeRagService())
        agent._history_loader = lambda session_uuid: turns
        return agent

    def test_pronoun_query_inherits_resume_intent(self):
        agent = self._agent([{"user": "帮我看看简历里的项目经历", "ai": "好的"}])
        run = agent.prepare("那这个呢", session_uuid="s1")
        self.assertEqual(run.mode, "简历诊断")
        self.assertIn("resume_diagnosis", run.tool_names())

    def test_pronoun_query_inherits_job_intent(self):
        agent = self._agent([{"user": "这个 JD 的能力要求是什么", "ai": "好的"}])
        run = agent.prepare("继续", session_uuid="s1")
        self.assertEqual(run.mode, "岗位匹配")

    def test_explicit_query_wins_over_history(self):
        agent = self._agent([{"user": "这个 JD 怎么看", "ai": "好的"}])
        run = agent.prepare("帮我改简历表达", session_uuid="s1")
        self.assertEqual(run.mode, "简历诊断")

    def test_no_history_keeps_old_behaviour(self):
        agent = self._agent([])
        run = agent.prepare("那这个呢", session_uuid="s1")
        self.assertEqual(run.mode, "综合求职分析")

    def test_no_session_uuid_keeps_old_behaviour(self):
        agent = self._agent([{"user": "这个 JD 怎么看", "ai": "好的"}])
        run = agent.prepare("那这个呢")
        self.assertEqual(run.mode, "综合求职分析")

    def test_metrics_record_history_turns(self):
        agent = self._agent([{"user": "简历怎么写", "ai": "这样写"}])
        run = agent.prepare("那这个呢", session_uuid="s1")
        self.assertGreaterEqual(run.metrics["history_turns"], 1)

    def test_history_loader_failure_is_not_fatal(self):
        agent = RulePlannerAgent(_FakeRagService())

        def boom(session_uuid):
            raise RuntimeError("db locked")

        agent._history_loader = boom
        run = agent.prepare("那这个呢", session_uuid="s1")
        self.assertEqual(run.mode, "综合求职分析")
        self.assertEqual(run.metrics["history_turns"], 0)


if __name__ == "__main__":
    unittest.main()
