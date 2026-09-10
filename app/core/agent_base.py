"""规划器基类与装配工厂。

两条 Planner 路线（规则 / LLM）共享长期记忆、工具层和上下文拼装逻辑，
差异只体现在 ``prepare()`` 如何决定「调用哪些工具」。
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from app.core import config_data as config
from app.core.agent_types import (
    PLANNER_LLM,
    PLANNER_RULE,
    AgentRun,
    AgentToolResult,
    normalize_planner_mode,
)
from app.core.logger import logger
from app.core.tools import ToolRegistry


# ---------------------------------------------------------------------------
# 对话历史上下文
#
# 规划器需要历史才能解析省略式追问（「那这个呢」「再展开说说」）。
# 但历史里的 AI 回答很长，直接全量注入会挤爆规划 prompt 的预算，
# 因此压缩成「每轮各截断 N 字」的摘要。
# ---------------------------------------------------------------------------


def parse_history_messages(messages) -> list[dict]:
    """把 LangChain 消息序列整理成 ``{"user": ..., "ai": ...}`` 轮次列表。"""
    turns: list[dict] = []
    pending_user: str | None = None

    for message in messages or []:
        if message is None:
            continue
        role = getattr(message, "type", None)
        if role is None and isinstance(message, str):
            content, role = message, "human"
        else:
            content = getattr(message, "content", "") or ""
        content = content if isinstance(content, str) else str(content)

        if role == "human":
            if pending_user is not None:
                turns.append({"user": pending_user, "ai": ""})
            pending_user = content
        elif role == "ai":
            turns.append({"user": pending_user or "", "ai": content})
            pending_user = None

    if pending_user is not None:
        turns.append({"user": pending_user, "ai": ""})
    return turns


def format_turn_digest(user_text: str, ai_text: str, max_chars: int) -> str:
    """单轮摘要：两侧各截断到 ``max_chars``。"""
    def clip(text: str) -> str:
        text = (text or "").strip().replace("\n", " ")
        if not text:
            return "(无回答)"
        return text if len(text) <= max_chars else text[:max_chars] + "…"

    return f"{clip(user_text)} ｜ {clip(ai_text)}"


def build_history_digest(
    turns: list[dict],
    max_turns: int | None = None,
    max_chars: int | None = None,
) -> str:
    """取最近 ``max_turns`` 轮，压缩成可注入 prompt 的摘要文本。"""
    max_turns = max_turns or config.PLANNER_HISTORY_TURNS
    max_chars = max_chars or config.PLANNER_HISTORY_CHARS
    if not turns:
        return ""

    recent = [t for t in turns[-max_turns:] if (t.get("user") or "").strip() or (t.get("ai") or "").strip()]
    if not recent:
        return ""

    lines = [
        f"[{index}] {format_turn_digest(t.get('user', ''), t.get('ai', ''), max_chars)}"
        for index, t in enumerate(recent, 1)
    ]
    return "\n".join(lines)


def default_history_loader(session_uuid: str | None) -> list[dict]:
    """默认历史来源：数据库中的会话消息记录。任何异常都吞掉并返回空。"""
    if not session_uuid:
        return []
    try:
        from app.utils.file_history_store import get_history

        return parse_history_messages(get_history(session_uuid).messages)
    except Exception as exc:
        logger.error(f"[History] 加载会话 {session_uuid} 历史失败: {exc}")
        return []


def rule_based_reflection(tool_results: list[AgentToolResult]) -> str:
    """确定性反思文案。

    规则路线直接使用；LLM 路线的反思器调用失败时作为兜底，
    因此两条路线的降级行为保持一致。
    """
    has_evidence = any(
        item.name == "hybrid_retrieval" and item.metadata.get("doc_count", 0) > 0
        for item in tool_results
    )
    failed_tools = [item.title for item in tool_results if item.metadata.get("error")]
    if failed_tools:
        return f"部分工具失败：{'、'.join(failed_tools)}；回答需要明确不确定性。"
    if has_evidence:
        return "已召回知识库证据，可以给出具体结论；仍需避免编造用户未提供的经历和量化指标。"
    return "知识库证据不足，回答应先给通用框架，并提醒用户补充真实 JD、简历或项目资料。"


class LongTermMemory:
    """Append-only 长期记忆存储，沉淀可复用的求职洞察。"""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or config.AGENT_MEMORY_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def load_recent(self, limit: int = 6) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        memories = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    memories.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return memories[-limit:]

    def update(self, query: str, answer: str, mode: str) -> list[str]:
        candidates = self._extract_memory_candidates(query, answer, mode)
        existing = {item.get("content", "") for item in self.load_recent(40)}
        new_items = [item for item in candidates if item and item not in existing]
        if not new_items:
            return ["本轮未发现新的稳定偏好或长期待办，保持现有记忆"]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            for content in new_items:
                f.write(json.dumps({
                    "time": now,
                    "mode": mode,
                    "content": content,
                }, ensure_ascii=False) + "\n")
        return new_items

    def _extract_memory_candidates(self, query: str, answer: str, mode: str) -> list[str]:
        text = f"{query}\n{answer}"
        candidates = []
        if any(word in text for word in ("RAG", "检索", "知识库", "Agent", "BM25", "RRF", "Chroma")):
            candidates.append("候选人重点项目标签：RAG/Agent/混合检索/求职知识库")
        if any(word in text for word in ("补强", "不足", "短板", "缺少", "需要")):
            candidates.append(f"{mode} 场景存在待补强项，后续回答应继续追踪能力差距")
        if any(word in text for word in ("简历", "量化", "表达")):
            candidates.append("简历优化偏好：优先给出可直接替换的项目表达")
        return candidates[:3]


class BaseCareerAgent:
    """两条 Planner 路线的共同基类。子类只需实现 ``prepare()``。"""

    planner: str = PLANNER_RULE

    def __init__(self, rag_service, history_loader=None) -> None:
        self.rag_service = rag_service
        self.memory = LongTermMemory()
        self.tools = ToolRegistry(rag_service, self.memory)
        self._history_loader = history_loader or default_history_loader

    # -- 子类实现 ---------------------------------------------------------

    def prepare(self, query: str, session_uuid: str | None = None) -> AgentRun:
        raise NotImplementedError

    # -- 共享逻辑 ---------------------------------------------------------

    def load_history(self, session_uuid: str | None) -> tuple[list[dict], str]:
        """返回 (轮次列表, 压缩摘要)。任何失败都降级为空历史，不阻断规划。"""
        if not session_uuid:
            return [], ""
        try:
            turns = self._history_loader(session_uuid) or []
        except Exception as exc:
            logger.error(f"[History] 历史加载器失败: {exc}")
            return [], ""
        return turns, build_history_digest(turns)

    def compose_agent_context(self, run: AgentRun) -> str:
        """把 AgentRun 拼成注入给回答模型的上下文。两条路线共用同一格式。"""
        sections = [
            "你正在以 CareerLens Agent 的身份工作。",
            f"本轮规划器：{run.planner_label}",
            "本轮自动任务计划：",
            *[f"- {step}" for step in run.plan],
            "",
            "工具调用结果：",
        ]
        for result in run.tool_results:
            sections.append(f"### {result.title}\n{result.content}")
        sections.extend([
            "",
            f"反思结论：{run.reflection}",
            "请基于以上 Agent 计划、工具结果和知识库证据回答用户。"
            "回答需要体现：任务规划、工具证据、结论、建议、下一步行动。",
        ])
        return "\n".join(sections)

    def finalize(self, query: str, answer: str, run: AgentRun) -> list[str]:
        updates = self.memory.update(query, answer, run.mode)
        run.memory_updates = updates
        return updates


def build_agent_service(rag_service, mode: str | None = None) -> BaseCareerAgent:
    """按 planner 标识装配对应的 Agent 实例。

    非法或未识别的标识一律回落到规则路线，保证接口层永远拿到可用实例。
    """
    resolved = normalize_planner_mode(mode or config.PLANNER_MODE, default=PLANNER_RULE)

    if resolved == PLANNER_LLM:
        try:
            from app.core.llm_agent import LlmPlannerAgent

            return LlmPlannerAgent(rag_service)
        except Exception as exc:  # 依赖缺失或初始化失败时不允许拖垮主链路
            logger.error(f"[Planner] LLM 规划器装配失败，回退规则路线: {exc}")

    from app.core.agent import RulePlannerAgent

    return RulePlannerAgent(rag_service)
