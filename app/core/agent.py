"""Agent orchestration layer for CareerLens.

Deterministic planner, tool executor, reflection, and long-term memory.
MVP: JD matching + resume diagnosis. Interview coach reserved for later.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable

from app.core import config_data as config
from app.core.logger import logger


@dataclass
class AgentToolResult:
    name: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentRun:
    mode: str
    plan: list[str]
    tool_results: list[AgentToolResult]
    reflection: str
    memory_updates: list[str]

    def status_lines(self) -> Iterable[str]:
        yield "[状态] Agent 正在识别任务意图...\n"
        yield f"[状态] Agent 已选择工作模式：{self.mode}\n"
        yield "[状态] Agent 已生成自动任务计划...\n"
        for step in self.plan:
            yield f"[Agent计划] {step}\n"
        for item in self.tool_results:
            yield f"[工具调用] {item.title}\n"
        yield "[状态] Agent 正在进行反思与自我修正...\n"
        yield f"[反思] {self.reflection}\n"
        for item in self.memory_updates:
            yield f"[记忆更新] {item}\n"


class LongTermMemory:
    """Append-only memory store for reusable career insights."""

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


class CareerAgentService:
    """Deterministic agent coordinator around the existing RagService."""

    def __init__(self, rag_service) -> None:
        self.rag_service = rag_service
        self.memory = LongTermMemory()
        self.tools: dict[str, Callable[[str, list[dict]], AgentToolResult]] = {
            "memory_lookup": self._tool_memory_lookup,
            "hybrid_retrieval": self._tool_hybrid_retrieval,
            "resume_diagnosis": self._tool_resume_diagnosis,
            "job_match": self._tool_job_match,
        }

    def prepare(self, query: str) -> AgentRun:
        mode = self._detect_mode(query)
        plan = self._build_plan(mode)
        memory_items = self.memory.load_recent()
        tool_names = self._select_tools(mode, query)
        tool_results = []
        for name in tool_names:
            try:
                tool_results.append(self.tools[name](query, memory_items))
            except Exception as exc:
                logger.error(f"[Agent] 工具 {name} 调用失败: {exc}")
                tool_results.append(AgentToolResult(
                    name=name,
                    title=f"{name} 调用失败",
                    content=f"工具执行失败：{exc}",
                    metadata={"error": True},
                ))
        reflection = self._reflect(query, tool_results)
        return AgentRun(
            mode=mode,
            plan=plan,
            tool_results=tool_results,
            reflection=reflection,
            memory_updates=[],
        )

    def compose_agent_context(self, run: AgentRun) -> str:
        sections = [
            "你正在以 CareerLens Agent 的身份工作。",
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
            "请基于以上 Agent 计划、工具结果和知识库证据回答用户。回答需要体现：任务规划、工具证据、结论、建议、下一步行动。",
        ])
        return "\n".join(sections)

    def finalize(self, query: str, answer: str, run: AgentRun) -> list[str]:
        updates = self.memory.update(query, answer, run.mode)
        run.memory_updates = updates
        return updates

    def _detect_mode(self, query: str) -> str:
        lower = query.lower()
        if any(k in query for k in ("简历", "表达", "项目经历", "经历")):
            return "简历诊断"
        if any(k in query for k in ("JD", "岗位", "职位", "匹配", "要求")) or "job" in lower:
            return "岗位匹配"
        return "综合求职分析"

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

    def _tool_memory_lookup(self, query: str, memory_items: list[dict]) -> AgentToolResult:
        if not memory_items:
            content = "暂无长期记忆，本轮将从知识库和用户输入中建立候选人画像。"
        else:
            content = "\n".join(f"- {item.get('content')}" for item in memory_items)
        return AgentToolResult("memory_lookup", "长期记忆检索", content)

    def _tool_hybrid_retrieval(self, query: str, memory_items: list[dict]) -> AgentToolResult:
        status, docs = self.rag_service.vector_service.hybrid_search_workflow(query)
        snippets = []
        for index, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "未知来源")
            text = re.sub(r"\s+", " ", doc.page_content).strip()
            snippets.append(f"{index}. 来源：{source}\n   证据：{text[:260]}")
        content = "\n".join(snippets) if snippets else "未召回到明确证据，回答时需要提示用户补充 JD、简历或项目资料。"
        return AgentToolResult(
            "hybrid_retrieval",
            "混合检索工具 Chroma+BM25+RRF",
            content,
            {"status": status, "doc_count": len(docs)},
        )

    def _tool_job_match(self, query: str, memory_items: list[dict]) -> AgentToolResult:
        keywords = self._keyword_hits(query, ["RAG", "Agent", "微调", "评估", "部署", "向量检索", "Prompt", "LoRA"])
        content = "岗位匹配关注项：" + ("、".join(keywords) if keywords else "大模型基础、检索增强、工程落地、评估闭环")
        content += "\n建议输出匹配矩阵：岗位要求 / 已有证据 / 风险缺口 / 补强动作。"
        return AgentToolResult("job_match", "岗位匹配分析工具", content)

    def _tool_resume_diagnosis(self, query: str, memory_items: list[dict]) -> AgentToolResult:
        content = (
            "简历诊断规则：优先检查项目是否包含问题定义、技术方案、指标结果、个人贡献和复盘改进。"
            '\n若出现「负责、参与、熟悉」等弱动词，应改成可验证动作和量化结果。'
        )
        return AgentToolResult("resume_diagnosis", "简历表达诊断工具", content)

    def _reflect(self, query: str, tool_results: list[AgentToolResult]) -> str:
        has_evidence = any(item.name == "hybrid_retrieval" and item.metadata.get("doc_count", 0) > 0 for item in tool_results)
        failed_tools = [item.title for item in tool_results if item.metadata.get("error")]
        if failed_tools:
            return f"部分工具失败：{'、'.join(failed_tools)}；回答需要明确不确定性。"
        if has_evidence:
            return "已召回知识库证据，可以给出具体结论；仍需避免编造用户未提供的经历和量化指标。"
        return "知识库证据不足，回答应先给通用框架，并提醒用户补充真实 JD、简历或项目资料。"

    def _keyword_hits(self, text: str, keywords: list[str]) -> list[str]:
        lower = text.lower()
        return [kw for kw in keywords if kw.lower() in lower or kw in text]
