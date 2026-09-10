"""规则 Planner 路线。

设计取舍：求职场景只有「岗位匹配 / 简历诊断 / 综合求职分析」三类固定意图，
关键词路由即可做到零 token、可复现、延迟稳定。代价是泛化性差——
复合意图（如「这份简历投大模型算法岗行不行」同时命中简历与岗位）只能命中一类分支，
这正是 LLM Planner 路线（``app/core/llm_agent.py``）要解决的问题。

两条路线的 A/B 对比由 ``run_planner_eval.py`` 产出。

注意：本模块的 plan / reflection 文案均为**规则产出**，不是模型推理结果。
对外展示时必须携带 ``planner="rule"`` 标记，不得表述为「LLM 自主规划」。
"""

from __future__ import annotations

import time

from app.core.agent_base import BaseCareerAgent, LongTermMemory, rule_based_reflection
from app.core.agent_types import PLANNER_RULE, AgentRun, AgentToolResult
from app.core.logger import logger

__all__ = [
    "RulePlannerAgent",
    "CareerAgentService",
    "LongTermMemory",
    "AgentRun",
    "AgentToolResult",
]


class RulePlannerAgent(BaseCareerAgent):
    """确定性 Planner：关键词判意图 → 固定计划模板 → 注册表调用工具 → 规则化反思。"""

    planner = PLANNER_RULE

    def prepare(self, query: str, session_uuid: str | None = None) -> AgentRun:
        started = time.perf_counter()

        turns, digest = self.load_history(session_uuid)
        mode = self._resolve_mode(query, turns)
        plan = self._build_plan(mode)
        memory_items = self.memory.load_recent()
        tool_names = self._select_tools(mode, query)

        tool_results: list[AgentToolResult] = []
        for name in tool_names:
            try:
                tool_results.append(self.tools.invoke(name, query, memory_items))
            except Exception as exc:
                logger.error(f"[Agent] 工具 {name} 调用失败: {exc}")
                tool_results.append(AgentToolResult(
                    name=name,
                    title=f"{name} 调用失败",
                    content=f"工具执行失败：{exc}",
                    metadata={"error": True},
                ))

        reflection = self._reflect(query, tool_results)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        return AgentRun(
            mode=mode,
            plan=plan,
            tool_results=tool_results,
            reflection=reflection,
            memory_updates=[],
            planner=PLANNER_RULE,
            fallback_reason="",
            metrics={
                "plan_latency_ms": latency_ms,
                "planner_llm_calls": 0,
                "llm_rounds": 0,
                "tool_rounds": 0,
                "invalid_tool_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "history_turns": len(turns),
            },
        )

    # -- 意图与计划 -------------------------------------------------------

    def _match_intent(self, text: str) -> str | None:
        """命不中任何意图关键词时返回 None（区别于「综合求职分析」这个兜底值）。"""
        if not text:
            return None
        if any(k in text for k in ("简历", "表达", "项目经历", "经历")):
            return "简历诊断"
        if any(k in text for k in ("JD", "岗位", "职位", "匹配", "要求")) or "job" in text.lower():
            return "岗位匹配"
        return None

    def _resolve_mode(self, query: str, turns: list[dict]) -> str:
        """优先看当前问题；命中不了再回看历史用户发言，用于解析省略式追问。"""
        mode = self._match_intent(query)
        if mode:
            return mode
        for turn in reversed(turns[-3:]):
            mode = self._match_intent(turn.get("user", ""))
            if mode:
                logger.info(f"[Agent] 当前问题无意图关键词，从历史推断为「{mode}」")
                return mode
        return "综合求职分析"

    def _detect_mode(self, query: str) -> str:
        """保留旧接口语义：只看当前问题，无命中即兜底。"""
        return self._match_intent(query) or "综合求职分析"

    def _build_plan(self, mode: str) -> list[str]:
        base = [
            "解析用户目标并判断当前求职场景",
            "读取长期记忆中的候选人偏好与历史短板",
            "调用知识库混合检索工具召回岗位、简历和项目证据",
        ]
        if mode == "岗位匹配":
            base.append("抽取岗位能力项并与候选人项目证据做匹配")
        elif mode == "简历诊断":
            base.append("定位简历表达中的空泛描述、证据缺口和可量化空间")
        else:
            base.append("综合岗位、简历和面试视角形成行动建议")
        base.extend([
            "执行反思检查，确认结论是否有证据支撑",
            "沉淀可复用的长期记忆或后续待办",
        ])
        return base

    def _select_tools(self, mode: str, query: str) -> list[str]:
        tools = ["memory_lookup", "hybrid_retrieval"]
        if mode == "岗位匹配":
            tools.append("job_match")
        elif mode == "简历诊断":
            tools.append("resume_diagnosis")
        else:
            tools.extend(["job_match", "resume_diagnosis"])
        return tools

    # -- 反思 -------------------------------------------------------------

    def _reflect(self, query: str, tool_results: list[AgentToolResult]) -> str:
        return rule_based_reflection(tool_results)


# 兼容既有引用（api_service 等处使用旧类名）
CareerAgentService = RulePlannerAgent
