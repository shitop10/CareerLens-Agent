import unittest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.core.agent_types import PLANNER_LLM, PLANNER_RULE
from app.core.llm_agent import LlmPlannerAgent


class _FakeVectorService:
    def hybrid_search_workflow(self, query):
        return ["[状态] 检索完成\n"], [
            Document(page_content="岗位要求 RAG 能力", metadata={"filename": "jd.txt"})
        ]


class _FakeRagService:
    def __init__(self):
        self.vector_service = _FakeVectorService()


def _ai(tool_calls=None, content="", usage=None):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    if usage is not None:
        msg.response_metadata = {"token_usage": usage}
    return msg


def _call(name, query, call_id="c1"):
    return {"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}


class _FakePlannerLLM:
    """按脚本顺序返回响应；error 非空时每次 invoke 都抛错。"""

    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.invokes = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.invokes.append(messages)
        if self.error:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return _ai()


class _FakeReflector:
    def __init__(self, text="证据部分充分，建议补充项目量化数据。", error=None):
        self.text = text
        self.error = error
        self.invokes = []

    def invoke(self, messages):
        self.invokes.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.text)


def _agent(planner_llm=None, reflector=None, history_loader=None):
    return LlmPlannerAgent(
        _FakeRagService(),
        planner_llm=planner_llm if planner_llm is not None else _FakePlannerLLM(),
        reflector=reflector if reflector is not None else _FakeReflector(),
        history_loader=history_loader,
    )


def _history(turns):
    return lambda session_uuid: list(turns)


def _prompt_text(llm) -> str:
    """把规划器第一次收到的 messages 拼成纯文本，便于断言。"""
    first = llm.invokes[0]
    return "\n".join(str(getattr(m, "content", "")) for m in first)


def _human_text(llm) -> str:
    """只取 human 消息 —— 系统提示词里也含「历史对话摘要」字样，需区分开。"""
    return str(getattr(llm.invokes[0][-1], "content", ""))


class TestSuccessfulPlanning(unittest.TestCase):
    def test_two_valid_tool_calls_are_executed(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "RAG 能力"), _call("job_match", "岗位要求", "c2")]),
            _ai(),
        ])
        run = _agent(llm).prepare("大模型算法岗需要哪些能力")
        self.assertEqual(run.planner, PLANNER_LLM)
        self.assertEqual(run.tool_names(), ["hybrid_retrieval", "job_match"])
        self.assertEqual(run.fallback_reason, "")

    def test_plan_records_tool_and_query(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "RAG 能力")]),
            _ai(),
        ])
        run = _agent(llm).prepare("q")
        joined = "\n".join(run.plan)
        self.assertIn("hybrid_retrieval", joined)
        self.assertIn("RAG 能力", joined)

    def test_binds_all_four_tools(self):
        llm = _FakePlannerLLM([_ai([_call("job_match", "q")]), _ai()])
        _agent(llm).prepare("q")
        self.assertEqual(
            [t.name for t in llm.bound_tools],
            ["memory_lookup", "hybrid_retrieval", "resume_diagnosis", "job_match"],
        )

    def test_second_round_receives_tool_output(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "RAG")]),
            _ai(),
        ])
        _agent(llm).prepare("q")
        self.assertEqual(len(llm.invokes), 2)
        second_round_text = " ".join(
            str(getattr(m, "content", "")) for m in llm.invokes[1]
        )
        self.assertIn("jd.txt", second_round_text)

    def test_reflection_comes_from_reflector(self):
        reflector = _FakeReflector("这是一段自定义反思")
        llm = _FakePlannerLLM([_ai([_call("job_match", "q")]), _ai()])
        run = _agent(llm, reflector).prepare("q")
        self.assertEqual(run.reflection, "这是一段自定义反思")


class TestModeDerivation(unittest.TestCase):
    def test_job_only(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "q"), _call("job_match", "q", "c2")]),
            _ai(),
        ])
        self.assertEqual(_agent(llm).prepare("q").mode, "岗位匹配")

    def test_resume_only(self):
        llm = _FakePlannerLLM([
            _ai([_call("resume_diagnosis", "q")]),
            _ai(),
        ])
        self.assertEqual(_agent(llm).prepare("q").mode, "简历诊断")

    def test_both_intents(self):
        llm = _FakePlannerLLM([
            _ai([_call("job_match", "q"), _call("resume_diagnosis", "q", "c2")]),
            _ai(),
        ])
        self.assertEqual(_agent(llm).prepare("q").mode, "综合求职分析")

    def test_retrieval_only(self):
        llm = _FakePlannerLLM([_ai([_call("hybrid_retrieval", "q")]), _ai()])
        self.assertEqual(_agent(llm).prepare("q").mode, "综合求职分析")


class TestHallucinatedTools(unittest.TestCase):
    def test_unknown_tool_is_discarded_and_counted(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "q"), _call("nonexistent_tool", "q", "c2")]),
            _ai(),
        ])
        run = _agent(llm).prepare("q")
        self.assertEqual(run.tool_names(), ["hybrid_retrieval"])
        self.assertEqual(run.metrics["invalid_tool_calls"], 1)

    def test_all_hallucinated_falls_back(self):
        llm = _FakePlannerLLM([_ai([_call("ghost_tool", "q")]), _ai()])
        run = _agent(llm).prepare("帮我看看简历")
        self.assertEqual(run.planner, PLANNER_RULE)
        self.assertIn("不存在的工具名", run.fallback_reason)
        self.assertEqual(run.metrics["invalid_tool_calls"], 1)


class TestNoToolDecision(unittest.TestCase):
    """模型主动判断「无需工具」是合法决策，不应降级为规则路线。"""

    def test_zero_tool_calls_is_not_a_fallback(self):
        run = _agent(_FakePlannerLLM([_ai(content="这是一个寒暄，无需检索。"), _ai()])).prepare("你好")
        self.assertEqual(run.planner, PLANNER_LLM)
        self.assertEqual(run.fallback_reason, "")
        self.assertEqual(run.tool_names(), [])
        self.assertEqual(run.metrics["no_tool_decision"], 1)

    def test_zero_tool_run_is_still_usable(self):
        run = _agent(_FakePlannerLLM([_ai(content="无需检索")])).prepare("你好")
        self.assertTrue(run.reflection)
        self.assertEqual(run.reasoning_events()[0]["type"], "status")
        self.assertEqual(run.reasoning_events()[-1]["type"], "reflection")

    def test_reflector_still_runs_on_empty_evidence(self):
        reflector = _FakeReflector("本轮无需检索，属于寒暄。")
        run = _agent(_FakePlannerLLM([_ai()]), reflector).prepare("你好")
        self.assertEqual(run.reflection, "本轮无需检索，属于寒暄。")


class TestDuplicateToolCalls(unittest.TestCase):
    def test_same_tool_same_query_in_one_round_executed_once(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "同一个检索词", "c1"),
                 _call("hybrid_retrieval", "同一个检索词", "c2")]),
            _ai(),
        ])
        run = _agent(llm).prepare("q")
        self.assertEqual(run.tool_names(), ["hybrid_retrieval"])
        self.assertEqual(run.metrics["duplicate_tool_calls"], 1)

    def test_same_tool_different_query_is_allowed(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "检索词A", "c1"),
                 _call("hybrid_retrieval", "检索词B", "c2")]),
            _ai(),
        ])
        run = _agent(llm).prepare("q")
        self.assertEqual(run.tool_names(), ["hybrid_retrieval", "hybrid_retrieval"])
        self.assertEqual(run.metrics["duplicate_tool_calls"], 0)


class TestToolRoundSemantics(unittest.TestCase):
    """max_tool_rounds 控制的是**工具执行轮次**，规划器调用次数上限为 max+1。"""

    def test_two_execution_rounds_are_both_honoured(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "第一轮")]),
            _ai([_call("job_match", "第二轮")]),
            _ai(),
        ])
        run = _agent(llm).prepare("q")
        self.assertEqual(run.tool_names(), ["hybrid_retrieval", "job_match"])
        self.assertEqual(run.metrics["llm_rounds"], 3)
        self.assertEqual(run.metrics["tool_rounds"], 2)

    def test_third_round_calls_are_not_executed(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "1")]),
            _ai([_call("job_match", "2")]),
            _ai([_call("resume_diagnosis", "3")]),
        ])
        run = _agent(llm).prepare("q")
        self.assertNotIn("resume_diagnosis", run.tool_names())


class TestFallback(unittest.TestCase):
    def test_llm_exception_falls_back_without_raising(self):
        run = _agent(_FakePlannerLLM(error=RuntimeError("network down"))).prepare("帮我看看简历")
        self.assertEqual(run.planner, PLANNER_RULE)
        self.assertIn("network down", run.fallback_reason)

    def test_reflector_failure_uses_rule_reflection(self):
        llm = _FakePlannerLLM([_ai([_call("hybrid_retrieval", "q")]), _ai()])
        run = _agent(llm, _FakeReflector(error=RuntimeError("boom"))).prepare("q")
        self.assertEqual(run.planner, PLANNER_LLM)
        self.assertIn("知识库证据", run.reflection)

    def test_fallback_preserves_attempt_metrics(self):
        llm = _FakePlannerLLM([
            _ai([_call("ghost", "q")], usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}),
            _ai(),
        ])
        run = _agent(llm).prepare("帮我看看简历")
        self.assertEqual(run.planner, PLANNER_RULE)
        self.assertEqual(run.metrics["total_tokens"], 120)
        self.assertGreaterEqual(run.metrics["planner_llm_calls"], 1)
        self.assertEqual(run.metrics["fallback"], 1)

    def test_fallback_still_produces_usable_run(self):
        run = _agent(_FakePlannerLLM(error=RuntimeError("x"))).prepare("帮我做个求职规划")
        self.assertTrue(run.plan)
        self.assertTrue(run.reflection)
        self.assertEqual(run.reasoning_events()[0]["type"], "status")


class TestTokenAccounting(unittest.TestCase):
    def test_tokens_accumulated_across_rounds(self):
        llm = _FakePlannerLLM([
            _ai([_call("hybrid_retrieval", "q")],
                usage={"input_tokens": 300, "output_tokens": 30, "total_tokens": 330}),
            _ai(usage={"input_tokens": 200, "output_tokens": 10, "total_tokens": 210}),
        ])
        run = _agent(llm).prepare("q")
        self.assertEqual(run.metrics["total_tokens"], 540)
        self.assertEqual(run.metrics["input_tokens"], 500)
        self.assertEqual(run.metrics["planner_llm_calls"], 2)


class TestConversationHistory(unittest.TestCase):
    """省略式追问（「那这个呢」）必须靠历史才能解析，规划器需拿到压缩摘要。"""

    _TURNS = [
        {"user": "帮我看看简历里的项目经历", "ai": "建议量化指标" * 20},
        {"user": "那这个呢", "ai": ""},
    ]

    def test_digest_is_injected_into_planner_prompt(self):
        llm = _FakePlannerLLM([_ai([_call("hybrid_retrieval", "简历 项目经历")]), _ai()])
        _agent(llm, history_loader=_history(self._TURNS)).prepare("那这个呢", session_uuid="s1")
        human = _human_text(llm)
        self.assertIn("历史对话摘要", human)
        self.assertIn("项目经历", human)
        self.assertIn("当前问题：那这个呢", human)

    def test_digest_is_compressed_not_full(self):
        llm = _FakePlannerLLM([_ai([_call("hybrid_retrieval", "q")]), _ai()])
        _agent(llm, history_loader=_history(self._TURNS)).prepare("那这个呢", session_uuid="s1")
        human = _human_text(llm)
        # 历史里 ai 侧有 100+ 字，注入后应被截断
        self.assertIn("…", human)

    def test_no_session_uuid_means_no_history_block(self):
        llm = _FakePlannerLLM([_ai([_call("hybrid_retrieval", "q")]), _ai()])
        _agent(llm, history_loader=_history(self._TURNS)).prepare("那这个呢")
        self.assertNotIn("历史对话摘要", _human_text(llm))
        self.assertEqual(_human_text(llm), "那这个呢")

    def test_loader_failure_is_not_fatal(self):
        def boom(session_uuid):
            raise RuntimeError("db locked")

        llm = _FakePlannerLLM([_ai([_call("hybrid_retrieval", "q")]), _ai()])
        run = _agent(llm, history_loader=boom).prepare("那这个呢", session_uuid="s1")
        self.assertEqual(run.metrics["history_turns"], 0)
        self.assertEqual(run.tool_names(), ["hybrid_retrieval"])

    def test_metrics_record_history_turns(self):
        llm = _FakePlannerLLM([_ai([_call("hybrid_retrieval", "q")]), _ai()])
        run = _agent(llm, history_loader=_history(self._TURNS)).prepare("那这个呢", session_uuid="s1")
        self.assertEqual(run.metrics["history_turns"], 2)

    def test_rule_planner_also_receives_session_uuid(self):
        """两条路线接口签名必须一致，否则 api_service 无法统一调用。"""
        import inspect

        from app.core.agent import RulePlannerAgent
        from app.core.agent_base import BaseCareerAgent

        for cls in (BaseCareerAgent, RulePlannerAgent, LlmPlannerAgent):
            params = list(inspect.signature(cls.prepare).parameters)
            self.assertEqual(params[:3], ["self", "query", "session_uuid"], cls.__name__)


class TestAggregateForHistoryNotBreakingMetrics(unittest.TestCase):
    def test_rule_metrics_keep_new_keys(self):
        from app.core.agent import RulePlannerAgent

        agent = RulePlannerAgent(_FakeRagService())
        run = agent.prepare("帮我看看简历")
        self.assertIn("history_turns", run.metrics)
        self.assertIn("tool_rounds", run.metrics)


if __name__ == "__main__":
    unittest.main()
