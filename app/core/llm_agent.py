"""LLM Planner 路线（LangGraph plan → execute → reflect）。

与规则路线的差异只在 ``prepare()`` 如何决定「调用哪些工具」：

- 规则路线：关键词判意图 → 查固定工具表
- LLM 路线：``bind_tools`` + 真实 function calling → 模型自主决定工具与检索词

两者共享 ``ToolRegistry`` 与 ``LongTermMemory``，产出同一 ``AgentRun`` 契约。

健壮性约定（不允许静默降级）：
- 规划器调用异常 / 超时 → 回退规则路线，``fallback_reason`` 记录原因
- 模型幻觉出不存在的工具名 → 丢弃并计入 ``invalid_tool_calls``
- 一轮都没选到有效工具 → 回退规则路线
- 反思器失败 → 退回确定性反思文案
- 回退时**保留**本次尝试消耗的 token 与调用次数，保证账目真实
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.core import config_data as config
from app.core.agent_base import BaseCareerAgent, rule_based_reflection
from app.core.agent_types import (
    PLANNER_LLM,
    PLANNER_RULE,
    AgentRun,
    AgentToolResult,
    merge_token_usage,
)
from app.core.agent import RulePlannerAgent
from app.core.logger import logger
from app.core.prompts import planner_system_prompt, reflector_prompt

_MODES = ("岗位匹配", "简历诊断", "综合求职分析")


class PlannerState(TypedDict):
    """LangGraph 状态。顺序执行，节点直接返回替换值，不使用 reducer。"""

    query: str
    memory_items: list[dict]
    history_digest: str
    messages: list
    tool_results: list
    plan_steps: list
    executed: list
    iterations: int
    executions: int
    max_tool_rounds: int
    reflection: str
    token_usage: dict
    invalid_tool_calls: int
    duplicate_tool_calls: int


class LlmPlannerAgent(BaseCareerAgent):
    """由 LLM function calling 驱动的规划器。"""

    planner = PLANNER_LLM

    def __init__(
        self,
        rag_service,
        planner_llm=None,
        reflector=None,
        history_loader=None,
    ) -> None:
        super().__init__(rag_service, history_loader=history_loader)
        # 注入的是「未绑定工具」的裸模型，工具绑定统一由 _get_planner_llm 完成，
        # 保证注入路径与生产路径行为一致。
        self._planner_llm = planner_llm
        self._bound_planner = None
        self._reflector = reflector
        self._langchain_tools = self.tools.as_langchain_tools()


        # 回退目标复用同一工具层与记忆，避免两份状态漂移
        self._rule_agent = RulePlannerAgent(rag_service)
        self._rule_agent.tools = self.tools
        self._rule_agent.memory = self.memory

        self._graph = self._build_graph()

    # -- 依赖装配 ---------------------------------------------------------

    def _get_planner_llm(self):
        if self._planner_llm is None:
            from langchain_community.chat_models.tongyi import ChatTongyi

            self._planner_llm = ChatTongyi(model=config.LLM_PLANNER_MODEL, temperature=0)
        if self._bound_planner is None:
            self._bound_planner = self._planner_llm.bind_tools(self._langchain_tools)
        return self._bound_planner

    def _get_reflector(self):
        if self._reflector is None:
            from langchain_community.chat_models.tongyi import ChatTongyi

            self._reflector = ChatTongyi(model=config.REFLECTOR_MODEL, temperature=0)
        return self._reflector

    def _tools_description(self) -> str:
        return "\n".join(
            f"- {item.name}：{item.description}" for item in self._langchain_tools
        )

    # -- 状态图 -----------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(PlannerState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("reflect", self._reflect_node)

        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._route_after_plan, {
            "execute": "execute",
            "reflect": "reflect",
        })
        graph.add_edge("execute", "plan")
        graph.add_edge("reflect", END)
        return graph.compile()

    def _route_after_plan(self, state: PlannerState) -> str:
        """还有待执行的工具调用，且未超过最大工具轮次 → 继续执行，否则进入反思。"""
        last = state["messages"][-1] if state["messages"] else None
        calls = getattr(last, "tool_calls", None) or []
        if calls and state["executions"] < state["max_tool_rounds"]:
            return "execute"
        return "reflect"

    def _plan_node(self, state: PlannerState) -> dict:
        llm = self._get_planner_llm()
        system = planner_system_prompt.replace("{tools}", self._tools_description())

        conversation = [SystemMessage(content=system), HumanMessage(
            content=_build_planning_input(state["query"], state.get("history_digest", ""))
        )]
        conversation.extend(state["messages"])

        response = llm.invoke(conversation)
        calls = getattr(response, "tool_calls", None) or []

        steps: list[str] = []
        content = (getattr(response, "content", "") or "").strip()
        if content:
            # 规划器偶发会越界输出长解释，截断以免污染前端计划列表
            snippet = content[:60] + ("…" if len(content) > 60 else "")
            steps.append(f"LLM 说明：{snippet}")
        for call in calls:
            query = (call.get("args") or {}).get("query", state["query"])
            steps.append(f"LLM 规划：调用 {call.get('name')} → 「{query}」")
        if not steps:
            steps.append("LLM 规划：本轮未选择任何工具")

        return {
            "messages": state["messages"] + [response],
            "plan_steps": state["plan_steps"] + steps,
            "iterations": state["iterations"] + 1,
            "token_usage": merge_token_usage(
                state["token_usage"], _usage_of(response)
            ),
        }

    def _execute_node(self, state: PlannerState) -> dict:
        last = state["messages"][-1] if state["messages"] else None
        calls = getattr(last, "tool_calls", None) or []
        valid_names = set(self.tools.names())

        tool_results = list(state["tool_results"])
        new_messages = list(state["messages"])
        executed = list(state.get("executed") or [])
        invalid = state["invalid_tool_calls"]
        duplicated = state.get("duplicate_tool_calls", 0)

        for call in calls:
            name = call.get("name") or ""
            call_id = call.get("id") or name or "call"
            query = (call.get("args") or {}).get("query") or state["query"]

            if name not in valid_names:
                invalid += 1
                logger.warning(
                    f"[Planner] 丢弃无效工具调用（名称={name!r}，模型可能返回了空名或清单外工具）"
                )
                new_messages.append(ToolMessage(
                    content=f"工具 {name or '(未命名)'} 不存在。请只使用已提供的工具。",
                    tool_call_id=call_id,
                ))
                continue

            # prompt 已禁止同一工具重复调用；同参重复直接跳过执行，省一次检索开销
            signature = f"{name}::{query}"
            if signature in executed:
                duplicated += 1
                logger.info(f"[Planner] 跳过重复的工具调用: {name}")
                new_messages.append(ToolMessage(
                    content="该工具已用相同检索词调用过，结果见上，请勿重复调用。",
                    tool_call_id=call_id,
                ))
                continue

            executed.append(signature)
            try:
                result = self.tools.invoke(name, query, state["memory_items"])
            except Exception as exc:
                logger.error(f"[Planner] 工具 {name} 执行失败: {exc}")
                result = AgentToolResult(
                    name=name,
                    title=f"{name} 调用失败",
                    content=f"工具执行失败：{exc}",
                    metadata={"error": True},
                )

            tool_results.append(result)
            new_messages.append(ToolMessage(
                content=result.content,
                tool_call_id=call_id,
                name=name,
            ))

        return {
            "messages": new_messages,
            "tool_results": tool_results,
            "executed": executed,
            "executions": state["executions"] + 1,
            "invalid_tool_calls": invalid,
            "duplicate_tool_calls": duplicated,
        }

    def _reflect_node(self, state: PlannerState) -> dict:
        evidence = _format_evidence(state["tool_results"])
        text = ""
        usage: dict = {}

        try:
            reflector = self._get_reflector()
            response = reflector.invoke([HumanMessage(content=reflector_prompt.format(
                query=state["query"], evidence=evidence,
            ))])
            text = (getattr(response, "content", "") or "").strip()
            usage = _usage_of(response)
        except Exception as exc:
            logger.error(f"[Planner] 反思器调用失败，退回确定性反思: {exc}")

        if not text:
            text = rule_based_reflection(state["tool_results"])

        return {
            "reflection": text,
            "token_usage": merge_token_usage(state["token_usage"], usage),
        }

    # -- 对外接口 ---------------------------------------------------------

    def prepare(self, query: str, session_uuid: str | None = None) -> AgentRun:
        started = time.perf_counter()
        memory_items = self.memory.load_recent()
        turns, history_digest = self.load_history(session_uuid)

        try:
            final = self._graph.invoke({
                "query": query,
                "memory_items": memory_items,
                "history_digest": history_digest,
                "messages": [],
                "tool_results": [],
                "plan_steps": [],
                "executed": [],
                "iterations": 0,
                "executions": 0,
                "max_tool_rounds": max(1, config.MAX_TOOL_ROUNDS),
                "reflection": "",
                "token_usage": {},
                "invalid_tool_calls": 0,
                "duplicate_tool_calls": 0,
            })
        except Exception as exc:
            logger.error(f"[Planner] LLM 规划器执行失败，回退规则路线: {exc}")
            return self._fallback(query, f"LLM 规划器执行失败：{exc}", started)

        tool_results: list[AgentToolResult] = final.get("tool_results", [])
        token_usage = final.get("token_usage") or {}
        rounds = final.get("iterations", 0)
        invalid = final.get("invalid_tool_calls", 0)
        duplicated = final.get("duplicate_tool_calls", 0)

        attempt_metrics = {
            "planner_llm_calls": rounds,
            "llm_rounds": rounds,
            "tool_rounds": final.get("executions", 0),
            "invalid_tool_calls": invalid,
            "duplicate_tool_calls": duplicated,
            "history_turns": len(turns),
            **token_usage,
        }

        if not tool_results:
            # 零工具 + 无幻觉 = 模型主动判断无需检索，属于合法决策，不应降级
            if invalid:
                return self._fallback(
                    query,
                    "LLM 规划器未能产出有效工具调用（模型输出了不存在的工具名）",
                    started,
                    metrics=attempt_metrics,
                )
            logger.info("[Planner] LLM 主动判断本轮无需调用任何工具")
            return AgentRun(
                mode="综合求职分析",
                plan=final.get("plan_steps", []),
                tool_results=[],
                reflection=final.get("reflection", ""),
                memory_updates=[],
                planner=PLANNER_LLM,
                fallback_reason="",
                metrics={
                    "plan_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "fallback": 0,
                    "no_tool_decision": 1,
                    **attempt_metrics,
                },
            )

        tool_names = [item.name for item in tool_results]
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        return AgentRun(
            mode=self._derive_mode(tool_names),
            plan=final.get("plan_steps", []),
            tool_results=tool_results,
            reflection=final.get("reflection", ""),
            memory_updates=[],
            planner=PLANNER_LLM,
            fallback_reason="",
            metrics={
                "plan_latency_ms": latency_ms,
                "fallback": 0,
                "no_tool_decision": 0,
                **attempt_metrics,
            },
        )

    # -- 内部工具 ---------------------------------------------------------

    def _fallback(
        self,
        query: str,
        reason: str,
        started: float,
        metrics: dict | None = None,
    ) -> AgentRun:
        """回退到规则路线，同时保留本次 LLM 尝试的真实开销。"""
        run = self._rule_agent.prepare(query)
        run.planner = PLANNER_RULE
        run.fallback_reason = reason

        merged = dict(run.metrics)
        merged.update(metrics or {})
        merged["plan_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        merged["fallback"] = 1
        run.metrics = merged
        return run

    @staticmethod
    def _derive_mode(tool_names: list[str]) -> str:
        """由实际选中的工具反推意图标签（LLM 模式下 mode 是决策结果而非输入）。"""
        has_job = "job_match" in tool_names
        has_resume = "resume_diagnosis" in tool_names
        if has_job and has_resume:
            return "综合求职分析"
        if has_job:
            return "岗位匹配"
        if has_resume:
            return "简历诊断"
        return "综合求职分析"


def _build_planning_input(query: str, history_digest: str) -> str:
    """规划输入 = 可选的压缩历史 + 当前问题。历史用于解析省略式追问。"""
    if not history_digest:
        return query
    return (
        "历史对话摘要（最近轮次，仅供解析指代，不要当成本轮问题）：\n"
        f"{history_digest}\n\n"
        f"当前问题：{query}"
    )


def _usage_of(message) -> dict:
    from app.core.agent_types import extract_token_usage

    return extract_token_usage(message)


def _format_evidence(tool_results: list[AgentToolResult]) -> str:
    if not tool_results:
        return "本轮没有任何工具返回结果。"
    blocks = []
    for index, item in enumerate(tool_results, 1):
        flag = "（执行失败）" if item.metadata.get("error") else ""
        blocks.append(f"[{index}] {item.title}{flag}\n{item.content}")
    return "\n\n".join(blocks)
