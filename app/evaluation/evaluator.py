"""Planner A/B 评测器。

对比两条规划路线在**规划阶段**（不含最终回答生成）的表现：

| 指标 | 含义 | 方向 |
|---|---|---|
| 必需工具召回率 | 期望工具中被实际调用的比例 | 越高越好 |
| 多余调用数 | 实际调用中超出期望的工具数 | 越低越好 |
| 精确匹配率 | 实际工具集合与期望完全一致的比例 | 越高越好 |
| 规划延迟 P50/P95 | ``prepare()`` 耗时 | 越低越好 |
| 规划 token | 规划阶段消耗的 token 均值 | 越低越好 |
| 回退率 | LLM 路线回退到规则路线的比例 | 越低越好 |

设计要点：默认只跑规划阶段（快、省 token）；``--with-answer`` 才生成最终回答，
``--with-judge`` 才调用裁判模型，避免评测本身烧掉大量额度。
"""

from __future__ import annotations

import math
import re
from typing import Any

from app.core import config_data as config
from app.core.agent_types import PLANNER_LLM, PLANNER_RULE, extract_token_usage
from app.core.logger import logger
from app.core.prompts import judge_prompt
from app.evaluation.dataset import PLANNER_CASES, validate_dataset

_TOTAL_PATTERN = re.compile(r"TOTAL\s*=\s*(\d+)")


# ---------------------------------------------------------------------------
# 纯函数指标（无副作用，可离线单测）
# ---------------------------------------------------------------------------


def required_recall(expected, produced) -> float:
    """期望工具中被实际调用的比例。期望为空时视为 1.0。"""
    expected_set = set(expected or [])
    if not expected_set:
        return 1.0
    return len(expected_set & set(produced or [])) / len(expected_set)


def extra_calls(expected, produced) -> int:
    """实际调用中超出期望的工具数量。"""
    return len(set(produced or []) - set(expected or []))


def exact_match(expected, produced) -> bool:
    """实际工具集合与期望是否完全一致。"""
    return set(expected or []) == set(produced or [])


def percentile(values, p: float) -> float:
    """线性插值分位数。空列表返回 0.0。"""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(v) for v in values) / len(values)


def aggregate(records: list[dict]) -> dict:
    """按 planner 分组汇总。records 为 ``evaluate()`` 产出的单条记录列表。"""
    summary: dict[str, dict] = {}
    for planner in (PLANNER_RULE, PLANNER_LLM):
        group = [r for r in records if r.get("planner") == planner]
        if not group:
            summary[planner] = {"n": 0}
            continue

        latencies = [r.get("latency_ms", 0) for r in group]
        summary[planner] = {
            "n": len(group),
            "recall_mean": round(_mean(r.get("recall", 0) for r in group), 4),
            "extra_mean": round(_mean(r.get("extra", 0) for r in group), 3),
            "exact_rate": round(_mean(1.0 if r.get("exact") else 0.0 for r in group), 4),
            "latency_p50": round(percentile(latencies, 50), 2),
            "latency_p95": round(percentile(latencies, 95), 2),
            "tokens_mean": round(_mean(r.get("total_tokens", 0) for r in group), 1),
            "llm_calls_mean": round(_mean(r.get("llm_calls", 0) for r in group), 2),
            "fallback_rate": round(_mean(1.0 if r.get("fallback") else 0.0 for r in group), 4),
        }
    return summary


def aggregate_by_intent(records: list[dict]) -> dict:
    """按意图分组汇总召回率，用于定位规则路线的具体短板。"""
    buckets: dict[str, dict[str, list]] = {}
    for record in records:
        key = record.get("intent", "未分类")
        buckets.setdefault(key, {}).setdefault(record.get("planner"), []).append(record)

    result: dict[str, dict] = {}
    for intent, by_planner in sorted(buckets.items()):
        result[intent] = {
            planner: round(_mean(r.get("recall", 0) for r in group), 4)
            for planner, group in by_planner.items()
        }
    return result


# ---------------------------------------------------------------------------
# 评测器
# ---------------------------------------------------------------------------


class PlannerABEvaluator:
    """对数据集逐条跑两条路线，产出可比指标。"""

    def __init__(self, rag_service=None, cases: list[dict] | None = None) -> None:
        problems = validate_dataset(cases)
        if problems:
            raise ValueError("评测数据集校验失败:\n  - " + "\n  - ".join(problems))

        self.cases = list(cases if cases is not None else PLANNER_CASES)
        self._rag_service = rag_service
        self._agents: dict[str, Any] = {}

    # -- 依赖装配（惰性，避免无谓加载向量库） -----------------------------

    @property
    def rag_service(self):
        if self._rag_service is None:
            from app.core.rag import RagService

            self._rag_service = RagService()
        return self._rag_service

    def _agent(self, planner: str):
        if planner not in self._agents:
            from app.core.agent_base import build_agent_service

            self._agents[planner] = build_agent_service(self.rag_service, planner)
        return self._agents[planner]

    # -- 执行 -------------------------------------------------------------

    def run_case(self, case: dict, planners: tuple[str, ...] | None = None) -> list[dict]:
        records = []
        for planner in (planners or (PLANNER_RULE, PLANNER_LLM)):
            try:
                run = self._agent(planner).prepare(case["query"])
                produced = run.tool_names()
                metrics = run.metrics or {}
                records.append({
                    "planner": planner,
                    "id": case["id"],
                    "intent": case["intent"],
                    "expected": list(case["expected_tools"]),
                    "produced": produced,
                    "recall": required_recall(case["expected_tools"], produced),
                    "extra": extra_calls(case["expected_tools"], produced),
                    "exact": exact_match(case["expected_tools"], produced),
                    "latency_ms": metrics.get("plan_latency_ms", 0),
                    "total_tokens": metrics.get("total_tokens", 0),
                    "llm_calls": metrics.get("planner_llm_calls", 0),
                    "fallback": bool(metrics.get("fallback", 0)),
                    "fallback_reason": run.fallback_reason,
                    "mode": run.mode,
                    "reflection": run.reflection,
                    "error": "",
                })
            except Exception as exc:  # 单条失败不应中断整轮评测
                logger.error(f"[Eval] 用例 {case['id']} 在 {planner} 路线失败: {exc}")
                records.append({
                    "planner": planner,
                    "id": case["id"],
                    "intent": case["intent"],
                    "expected": list(case["expected_tools"]),
                    "produced": [],
                    "recall": 0.0,
                    "extra": 0,
                    "exact": False,
                    "latency_ms": 0,
                    "total_tokens": 0,
                    "llm_calls": 0,
                    "fallback": False,
                    "fallback_reason": "",
                    "mode": "",
                    "reflection": "",
                    "error": str(exc),
                })
        return records

    def evaluate(
        self,
        limit: int | None = None,
        with_judge: bool = False,
        planners: tuple[str, ...] | None = None,
    ) -> dict:
        """跑完整数据集。``with_judge`` 会额外调用裁判模型（消耗额度）。"""
        cases = self.cases[:limit] if limit else list(self.cases)
        records: list[dict] = []

        for index, case in enumerate(cases, 1):
            logger.info(f"[Eval] ({index}/{len(cases)}) {case['id']} · {case['query'][:32]}")
            records.extend(self.run_case(case, planners))

        if with_judge:
            self._attach_judgements(cases, records)

        return {
            "n_cases": len(cases),
            "summary": aggregate(records),
            "by_intent": aggregate_by_intent(records),
            "records": records,
        }

    # -- 可选：LLM-as-judge ----------------------------------------------

    def _attach_judgements(self, cases: list[dict], records: list[dict]) -> None:
        questions = {c["id"]: c["query"] for c in cases}
        for record in records:
            expected = set(record["expected"])
            evidence = "（本项评测仅跑规划阶段，未生成最终回答）"
            answer = (
                f"规划结果：调用了 {record['produced'] or '（无）'}"
                f"；期望工具：{sorted(expected)}"
            )
            try:
                record["judge_total"] = self._judge(
                    questions[record["id"]], evidence, answer
                )
            except Exception as exc:
                logger.error(f"[Eval] 裁判调用失败 {record['id']}: {exc}")
                record["judge_total"] = None

    def _judge(self, query: str, evidence: str, answer: str) -> int | None:
        from langchain_community.chat_models.tongyi import ChatTongyi
        from langchain_core.messages import HumanMessage

        judge = ChatTongyi(model=config.JUDGE_MODEL, temperature=0)
        response = judge.invoke([HumanMessage(content=judge_prompt.format(
            query=query, evidence=evidence, answer=answer,
        ))])
        usage = extract_token_usage(response)
        if usage["total_tokens"]:
            logger.debug(f"[Eval] 裁判消耗 token: {usage['total_tokens']}")
        match = _TOTAL_PATTERN.search(response.content or "")
        return int(match.group(1)) if match else None

    # -- 报告 -------------------------------------------------------------

    def report(self, results: dict) -> str:
        summary = results["summary"]
        rule = summary.get(PLANNER_RULE, {})
        llm = summary.get(PLANNER_LLM, {})

        def render(block: dict, key: str, fmt: str) -> str:
            value = block.get(key)
            return "-" if value is None else fmt.format(value)

        rows = [
            ("用例数", "{:.0f}", "n"),
            ("必需工具召回率", "{:.4f}", "recall_mean"),
            ("多余调用数(均值)", "{:.2f}", "extra_mean"),
            ("精确匹配率", "{:.4f}", "exact_rate"),
            ("规划延迟 P50 (ms)", "{:.1f}", "latency_p50"),
            ("规划延迟 P95 (ms)", "{:.1f}", "latency_p95"),
            ("规划 token/轮", "{:.1f}", "tokens_mean"),
            ("规划 LLM 调用次数", "{:.2f}", "llm_calls_mean"),
            ("回退率", "{:.4f}", "fallback_rate"),
        ]

        lines = [
            "# Planner A/B 评测报告",
            "",
            f"- 用例数：{results['n_cases']}",
            "- 评测范围：`prepare()` 全流程（含两条路线共享的检索耗时，不含最终回答生成）",
            f"- 规则路线：`{PLANNER_RULE}` · 零 token 确定性路由",
            f"- LLM 路线：`{PLANNER_LLM}` · `{config.LLM_PLANNER_MODEL}` function calling",
            "",
            "## 汇总对比",
            "",
            "| 指标 | 规则路线 | LLM 路线 |",
            "|---|---|---|",
        ]
        for label, fmt, key in rows:
            lines.append(
                f"| {label} | {render(rule, key, fmt)} | {render(llm, key, fmt)} |"
            )

        lines.extend(["", "## 按意图分组的必需工具召回率", "", "| 意图 | 规则路线 | LLM 路线 |", "|---|---|---|"])
        for intent, by_planner in results["by_intent"].items():
            rule_value = by_planner.get(PLANNER_RULE)
            llm_value = by_planner.get(PLANNER_LLM)
            lines.append(
                f"| {intent} | {'-' if rule_value is None else f'{rule_value:.4f}'} "
                f"| {'-' if llm_value is None else f'{llm_value:.4f}'} |"
            )

        lines.extend(["", "## 逐条明细", "",
                      "| 用例 | 意图 | 期望工具 | 规则路线 | LLM 路线 |",
                      "|---|---|---|---|---|"])
        by_id: dict[str, dict] = {}
        for record in results["records"]:
            by_id.setdefault(record["id"], {})[record["planner"]] = record
        for case_id, pair in by_id.items():
            rule_rec = pair.get(PLANNER_RULE, {})
            llm_rec = pair.get(PLANNER_LLM, {})
            expected = "、".join(rule_rec.get("expected", []))
            lines.append(
                f"| {case_id} | {rule_rec.get('intent', '')} | {expected} "
                f"| {_fmt_tools(rule_rec)} | {_fmt_tools(llm_rec)} |"
            )

        fallbacks = [
            f"- `{r['id']}`：{r['fallback_reason']}"
            for r in results["records"]
            if r.get("fallback") and r["fallback_reason"]
        ]
        lines.extend(["", "## 回退记录", ""])
        lines.extend(fallbacks or ["- 无"])

        lines.extend([
            "",
            "## 结论口径",
            "",
            "本表仅反映**规划阶段**的工具选择与开销差异，不代表最终回答质量。",
            f"数值由 `run_planner_eval.py` 于真实环境运行产出，用例数 {results['n_cases']} 条，"
            "不具备统计显著性，仅用于验证两条路线差异方向。",
            "",
        ])
        return "\n".join(lines)


def _fmt_tools(record: dict) -> str:
    if not record:
        return "-"
    if record.get("error"):
        return f"失败：{record['error'][:40]}"
    produced = record.get("produced") or []
    mark = "✓" if record.get("exact") else f"召回 {record.get('recall', 0):.2f}"
    suffix = "（已回退）" if record.get("fallback") else ""
    return f"{'、'.join(produced) or '无'} · {mark}{suffix}"
