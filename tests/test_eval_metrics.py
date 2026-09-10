import unittest

from app.evaluation.dataset import PLANNER_CASES, validate_dataset
from app.evaluation.evaluator import (
    aggregate,
    aggregate_by_intent,
    exact_match,
    extra_calls,
    percentile,
    required_recall,
)


class TestDatasetIntegrity(unittest.TestCase):
    def test_dataset_passes_validation(self):
        self.assertEqual(validate_dataset(), [])

    def test_expected_size(self):
        self.assertEqual(len(PLANNER_CASES), 20)

    def test_ids_unique(self):
        ids = [c["id"] for c in PLANNER_CASES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_covers_all_intents(self):
        intents = {c["intent"] for c in PLANNER_CASES}
        self.assertEqual(intents, {"岗位匹配", "简历诊断", "复合意图", "历史语境", "边界"})

    def test_detects_duplicate_id(self):
        cases = [
            {"id": "x", "query": "q", "intent": "i", "expected_tools": ["hybrid_retrieval"]},
            {"id": "x", "query": "q", "intent": "i", "expected_tools": ["hybrid_retrieval"]},
        ]
        self.assertTrue(any("重复" in p for p in validate_dataset(cases)))

    def test_detects_unknown_tool(self):
        cases = [{"id": "x", "query": "q", "intent": "i", "expected_tools": ["ghost"]}]
        self.assertTrue(any("未注册工具" in p for p in validate_dataset(cases)))

    def test_detects_non_list_expected(self):
        cases = [{"id": "x", "query": "q", "intent": "i", "expected_tools": "hybrid_retrieval"}]
        self.assertTrue(any("必须是列表" in p for p in validate_dataset(cases)))

    def test_empty_expected_is_legal(self):
        cases = [{"id": "x", "query": "你好", "intent": "边界", "expected_tools": []}]
        self.assertEqual(validate_dataset(cases), [])

    def test_edge_greeting_expects_no_tools(self):
        greeting = next(c for c in PLANNER_CASES if c["id"] == "edge_01")
        self.assertEqual(greeting["expected_tools"], [])


class TestRequiredRecall(unittest.TestCase):
    def test_full_recall(self):
        self.assertEqual(required_recall(["a", "b"], ["a", "b", "c"]), 1.0)

    def test_half_recall(self):
        self.assertEqual(required_recall(["a", "b"], ["a"]), 0.5)

    def test_zero_recall(self):
        self.assertEqual(required_recall(["a"], ["z"]), 0.0)

    def test_empty_expected_is_perfect(self):
        self.assertEqual(required_recall([], ["a"]), 1.0)


class TestExtraCalls(unittest.TestCase):
    def test_no_extra(self):
        self.assertEqual(extra_calls(["a"], ["a"]), 0)

    def test_counts_only_unexpected(self):
        self.assertEqual(extra_calls(["a"], ["a", "b", "c"]), 2)

    def test_missing_expected_is_not_extra(self):
        self.assertEqual(extra_calls(["a", "b"], ["a"]), 0)


class TestExactMatch(unittest.TestCase):
    def test_identical_sets_ignoring_order(self):
        self.assertTrue(exact_match(["a", "b"], ["b", "a"]))

    def test_superset_is_not_exact(self):
        self.assertFalse(exact_match(["a"], ["a", "b"]))

    def test_subset_is_not_exact(self):
        self.assertFalse(exact_match(["a", "b"], ["a"]))


class TestPercentile(unittest.TestCase):
    def test_empty_returns_zero(self):
        self.assertEqual(percentile([], 50), 0.0)

    def test_single_value(self):
        self.assertEqual(percentile([7], 95), 7.0)

    def test_median(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)

    def test_p95_of_ten_values(self):
        self.assertAlmostEqual(percentile(list(range(1, 11)), 95), 9.55, places=2)

    def test_p100_is_max(self):
        self.assertEqual(percentile([3, 1, 9], 100), 9.0)


def _record(planner, recall=1.0, extra=0, exact=True, latency=10.0,
            tokens=0, llm_calls=0, fallback=False, intent="岗位匹配"):
    return {
        "planner": planner, "id": "x", "intent": intent,
        "expected": ["a"], "produced": ["a"],
        "recall": recall, "extra": extra, "exact": exact,
        "latency_ms": latency, "total_tokens": tokens,
        "llm_calls": llm_calls, "fallback": fallback,
        "fallback_reason": "", "mode": "", "reflection": "", "error": "",
    }


class TestAggregate(unittest.TestCase):
    def test_empty_groups(self):
        summary = aggregate([])
        self.assertEqual(summary["rule"]["n"], 0)
        self.assertEqual(summary["llm"]["n"], 0)

    def test_groups_split_by_planner(self):
        records = [
            _record("rule", recall=0.5, latency=1.0),
            _record("rule", recall=1.0, latency=3.0),
            _record("llm", recall=1.0, latency=100.0, tokens=500, llm_calls=2),
        ]
        summary = aggregate(records)
        self.assertEqual(summary["rule"]["n"], 2)
        self.assertAlmostEqual(summary["rule"]["recall_mean"], 0.75)
        self.assertEqual(summary["llm"]["n"], 1)
        self.assertEqual(summary["llm"]["tokens_mean"], 500.0)

    def test_latency_percentiles(self):
        records = [_record("llm", latency=v) for v in (10, 20, 30, 40)]
        summary = aggregate(records)
        self.assertAlmostEqual(summary["llm"]["latency_p50"], 25.0)

    def test_fallback_rate(self):
        records = [
            _record("llm", fallback=True),
            _record("llm", fallback=False),
        ]
        self.assertAlmostEqual(aggregate(records)["llm"]["fallback_rate"], 0.5)

    def test_exact_rate(self):
        records = [
            _record("rule", exact=True),
            _record("rule", exact=False),
        ]
        self.assertAlmostEqual(aggregate(records)["rule"]["exact_rate"], 0.5)


class TestAggregateByIntent(unittest.TestCase):
    def test_splits_by_intent_and_planner(self):
        records = [
            _record("rule", recall=0.0, intent="复合意图"),
            _record("llm", recall=1.0, intent="复合意图"),
            _record("rule", recall=1.0, intent="岗位匹配"),
        ]
        result = aggregate_by_intent(records)
        self.assertEqual(result["复合意图"]["rule"], 0.0)
        self.assertEqual(result["复合意图"]["llm"], 1.0)
        self.assertEqual(result["岗位匹配"]["rule"], 1.0)
        self.assertNotIn("llm", result["岗位匹配"])


# ---------------------------------------------------------------------------
# 评测器：用假 Agent 注入，全程零网络
# ---------------------------------------------------------------------------

from app.core.agent_types import AgentRun, AgentToolResult  # noqa: E402
from app.evaluation.evaluator import PlannerABEvaluator  # noqa: E402


class _FakeAgent:
    def __init__(self, planner, tool_names, metrics=None, error=None):
        self.planner = planner
        self.tool_names = tool_names
        self.metrics = metrics or {}
        self.error = error

    def prepare(self, query):
        if self.error:
            raise self.error
        return AgentRun(
            mode="岗位匹配",
            plan=[f"步骤-{n}" for n in self.tool_names],
            tool_results=[AgentToolResult(n, n, "内容", {}) for n in self.tool_names],
            reflection="反思",
            memory_updates=[],
            planner=self.planner,
            metrics=self.metrics,
        )


_EVAL_CASES = [
    {"id": "c1", "intent": "岗位匹配", "query": "JD 要求什么",
     "expected_tools": ["hybrid_retrieval", "job_match"]},
    {"id": "c2", "intent": "复合意图", "query": "简历投这个岗位行不行",
     "expected_tools": ["hybrid_retrieval", "job_match", "resume_diagnosis"]},
]


def _evaluator(rule_agent, llm_agent):
    evaluator = PlannerABEvaluator(rag_service=object(), cases=_EVAL_CASES)
    evaluator._agents = {"rule": rule_agent, "llm": llm_agent}
    return evaluator


class TestEvaluatorRunCase(unittest.TestCase):
    def test_produces_one_record_per_planner(self):
        evaluator = _evaluator(
            _FakeAgent("rule", ["hybrid_retrieval", "job_match"]),
            _FakeAgent("llm", ["hybrid_retrieval", "job_match"]),
        )
        records = evaluator.run_case(_EVAL_CASES[0])
        self.assertEqual([r["planner"] for r in records], ["rule", "llm"])
        self.assertTrue(all(r["exact"] for r in records))

    def test_rule_planner_extra_calls_detected(self):
        evaluator = _evaluator(
            _FakeAgent("rule", ["hybrid_retrieval", "memory_lookup", "job_match"]),
            _FakeAgent("llm", ["hybrid_retrieval", "job_match"]),
        )
        records = evaluator.run_case(_EVAL_CASES[0])
        rule_rec = next(r for r in records if r["planner"] == "rule")
        self.assertEqual(rule_rec["extra"], 1)
        self.assertFalse(rule_rec["exact"])
        self.assertEqual(rule_rec["recall"], 1.0)

    def test_composite_intent_recall_gap(self):
        # 规则路线只能命中一类分支，这是本评测想暴露的核心差异
        evaluator = _evaluator(
            _FakeAgent("rule", ["hybrid_retrieval", "memory_lookup", "resume_diagnosis"]),
            _FakeAgent("llm", ["hybrid_retrieval", "job_match", "resume_diagnosis"]),
        )
        records = evaluator.run_case(_EVAL_CASES[1])
        rule_rec = next(r for r in records if r["planner"] == "rule")
        llm_rec = next(r for r in records if r["planner"] == "llm")
        self.assertAlmostEqual(rule_rec["recall"], 2 / 3)
        self.assertEqual(llm_rec["recall"], 1.0)

    def test_agent_exception_becomes_error_record(self):
        evaluator = _evaluator(
            _FakeAgent("rule", [], error=RuntimeError("db down")),
            _FakeAgent("llm", ["hybrid_retrieval"]),
        )
        records = evaluator.run_case(_EVAL_CASES[0])
        rule_rec = next(r for r in records if r["planner"] == "rule")
        self.assertIn("db down", rule_rec["error"])
        self.assertEqual(rule_rec["recall"], 0.0)

    def test_fallback_metrics_recorded(self):
        evaluator = _evaluator(
            _FakeAgent("rule", ["hybrid_retrieval"]),
            _FakeAgent("llm", ["hybrid_retrieval"],
                       metrics={"fallback": 1, "planner_llm_calls": 1, "total_tokens": 300}),
        )
        records = evaluator.run_case(_EVAL_CASES[0])
        llm_rec = next(r for r in records if r["planner"] == "llm")
        self.assertTrue(llm_rec["fallback"])
        self.assertEqual(llm_rec["total_tokens"], 300)


class TestEvaluatorReport(unittest.TestCase):
    def _run(self):
        evaluator = _evaluator(
            _FakeAgent("rule", ["hybrid_retrieval", "memory_lookup", "resume_diagnosis"],
                       metrics={"plan_latency_ms": 1.2}),
            _FakeAgent("llm", ["hybrid_retrieval", "job_match", "resume_diagnosis"],
                       metrics={"plan_latency_ms": 1800.0, "total_tokens": 1200,
                                "planner_llm_calls": 2}),
        )
        return evaluator, evaluator.evaluate()

    def test_report_structure(self):
        _, results = self._run()
        report = _evaluator(_FakeAgent("rule", []), _FakeAgent("llm", [])).report(results)
        for section in ("# Planner A/B 评测报告", "## 汇总对比",
                        "## 按意图分组的必需工具召回率", "## 逐条明细", "## 回退记录"):
            self.assertIn(section, report)

    def test_report_reveals_recall_gap(self):
        evaluator, results = self._run()
        summary = results["summary"]
        self.assertLess(summary["rule"]["recall_mean"], summary["llm"]["recall_mean"])
        report = evaluator.report(results)
        self.assertIn("复合意图", report)

    def test_report_states_scope_limitation(self):
        evaluator, results = self._run()
        report = evaluator.report(results)
        self.assertIn("仅反映**规划阶段**", report)
        self.assertIn("不具备统计显著性", report)

    def test_evaluate_limit(self):
        evaluator, _ = self._run()
        results = evaluator.evaluate(limit=1)
        self.assertEqual(results["n_cases"], 1)

    def test_evaluate_single_planner(self):
        evaluator, _ = self._run()
        results = evaluator.evaluate(planners=("rule",))
        planners = {r["planner"] for r in results["records"]}
        self.assertEqual(planners, {"rule"})
        self.assertEqual(len(results["records"]), len(_EVAL_CASES))


if __name__ == "__main__":
    unittest.main()
