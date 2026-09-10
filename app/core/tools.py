"""共享工具层。

``ToolRegistry`` 是 4 个求职分析工具的**唯一实现**，同时向两条 Planner 路线暴露：

- ``invoke(name, query)`` —— 返回结构化 ``AgentToolResult``，供规则路线直接调用
- ``as_langchain_tools()`` —— 返回 ``BaseTool`` 列表，供 LLM 路线做 function calling

工具描述（docstring）直接决定 LLM 的选型准确率，因此按
「做什么 / 何时用 / 何时不用」三段式编写，不要压缩成一句话。
"""

from __future__ import annotations

import re
from typing import Callable

from langchain_core.tools import BaseTool, tool

from app.core.agent_types import AgentToolResult

TOOL_NAMES = ("memory_lookup", "hybrid_retrieval", "resume_diagnosis", "job_match")

_JOB_KEYWORDS = ["RAG", "Agent", "微调", "评估", "部署", "向量检索", "Prompt", "LoRA"]


class ToolRegistry:
    """4 个工具的注册表。两条 Planner 路线共用，避免实现漂移。"""

    def __init__(self, rag_service, memory_store=None) -> None:
        self.rag_service = rag_service
        self.memory_store = memory_store
        # 供 LangChain 工具视图读取本轮已加载的长期记忆，构造时即就绪
        self._memory_holder: dict[str, list[dict]] = {}
        self._handlers: dict[str, Callable[[str, list[dict] | None], AgentToolResult]] = {
            "memory_lookup": self._memory_lookup,
            "hybrid_retrieval": self._hybrid_retrieval,
            "resume_diagnosis": self._resume_diagnosis,
            "job_match": self._job_match,
        }

    # -- 公共接口 ---------------------------------------------------------

    def names(self) -> list[str]:
        return list(TOOL_NAMES)

    def invoke(
        self,
        name: str,
        query: str,
        memory_items: list[dict] | None = None,
    ) -> AgentToolResult:
        if name not in self._handlers:
            raise KeyError(f"未注册的工具: {name}")
        return self._handlers[name](query, memory_items)

    def as_langchain_tools(self) -> list[BaseTool]:
        """导出 LangChain 工具视图，供 ``bind_tools`` 使用。"""
        registry = self
        memory_holder = self._memory_holder

        @tool
        def memory_lookup(query: str) -> str:
            """读取候选人的长期记忆，包括稳定偏好、历史短板与后续待办。

            何时使用：用户显式引用历史语境时，例如「上次说的那个」「之前提过」「还是按之前的方向」。
            何时不用：用户提出的是一个全新问题时，不要调用本工具。"""
            return registry.invoke("memory_lookup", query, memory_holder.get("items")).content

        @tool
        def hybrid_retrieval(query: str) -> str:
            """在求职知识库中执行混合检索，召回岗位 JD、个人简历、项目素材与面试题库证据。

            检索策略为 Chroma 语义检索 + BM25 关键词检索 + RRF 倒数秩融合，是本系统唯一的证据来源。
            何时使用：任何需要引用材料作答的问题都必须调用。query 需改写为检索友好的表述，
            保留原始实体（岗位名、技术名词、公司名），不要原样透传用户口语。
            何时不用：不要重复调用本工具；单轮内一次即可。"""
            result = registry.invoke("hybrid_retrieval", query, memory_holder.get("items"))
            return result.content

        @tool
        def resume_diagnosis(query: str) -> str:
            """诊断简历的写作质量，定位空泛描述、证据缺口与可量化空间。

            检查维度：是否包含问题定义、技术方案、指标结果、个人贡献、复盘改进。
            会识别「负责」「参与」「熟悉」等弱动词并给出可替换的强动作表达。
            何时使用：用户问及简历、项目经历、技能表达、成果量化、实习描述时。
            何时不用：纯岗位要求分析、与简历无关的技术概念答疑。"""
            return registry.invoke("resume_diagnosis", query, memory_holder.get("items")).content

        @tool
        def job_match(query: str) -> str:
            """分析岗位要求与候选人能力的匹配度，输出「已具备 / 需补强 / 可补充项目证据」三类结论。

            覆盖 RAG、Agent、微调、评估、部署、向量检索、Prompt、LoRA 等大模型岗核心能力项。
            何时使用：用户问及岗位、JD、职位要求、能力差距、匹配度、能否投递时。
            何时不用：用户只在讨论简历措辞，未涉及具体岗位要求。"""
            return registry.invoke("job_match", query, memory_holder.get("items")).content

        return [memory_lookup, hybrid_retrieval, resume_diagnosis, job_match]

    def bind_memory_items(self, items: list[dict] | None) -> None:
        """把本轮已加载的长期记忆注入 LangChain 工具视图。

        ``as_langchain_tools()`` 生成的工具不接收 memory_items 参数
        （function calling 的入参必须简洁），因此由调用方在执行前注入。
        未调用本方法时，工具会回落到 ``memory_store.load_recent()``。
        """
        if items is not None:
            self._memory_holder["items"] = items

    # -- 各工具实现（与重构前的 app/core/agent.py 行为保持一致） -----------

    def _memory_items(self, memory_items: list[dict] | None) -> list[dict]:
        if memory_items is not None:
            return memory_items
        if self.memory_store is None:
            return []
        return self.memory_store.load_recent()

    def _memory_lookup(self, query: str, memory_items: list[dict] | None) -> AgentToolResult:
        items = self._memory_items(memory_items)
        if not items:
            content = "暂无长期记忆，本轮将从知识库和用户输入中建立候选人画像。"
        else:
            content = "\n".join(f"- {item.get('content')}" for item in items)
        return AgentToolResult("memory_lookup", "长期记忆检索", content)

    def _hybrid_retrieval(self, query: str, memory_items: list[dict] | None) -> AgentToolResult:
        status, docs = self.rag_service.vector_service.hybrid_search_workflow(query)
        snippets = []
        for index, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "未知来源")
            text = re.sub(r"\s+", " ", doc.page_content).strip()
            snippets.append(f"{index}. 来源：{source}\n   证据：{text[:260]}")
        content = (
            "\n".join(snippets)
            if snippets
            else "未召回到明确证据，回答时需要提示用户补充 JD、简历或项目资料。"
        )
        return AgentToolResult(
            "hybrid_retrieval",
            "混合检索工具 Chroma+BM25+RRF",
            content,
            {"status": status, "doc_count": len(docs)},
        )

    def _resume_diagnosis(self, query: str, memory_items: list[dict] | None) -> AgentToolResult:
        content = (
            "简历诊断规则：优先检查项目是否包含问题定义、技术方案、指标结果、个人贡献和复盘改进。"
            "\n若出现「负责、参与、熟悉」等弱动词，应改成可验证动作和量化结果。"
        )
        return AgentToolResult("resume_diagnosis", "简历表达诊断工具", content)

    def _job_match(self, query: str, memory_items: list[dict] | None) -> AgentToolResult:
        keywords = self._keyword_hits(query, _JOB_KEYWORDS)
        content = "岗位匹配关注项：" + (
            "、".join(keywords) if keywords else "大模型基础、检索增强、工程落地、评估闭环"
        )
        content += "\n建议输出匹配矩阵：岗位要求 / 已有证据 / 风险缺口 / 补强动作。"
        return AgentToolResult("job_match", "岗位匹配分析工具", content)

    @staticmethod
    def _keyword_hits(text: str, keywords: list[str]) -> list[str]:
        lower = text.lower()
        return [kw for kw in keywords if kw.lower() in lower or kw in text]
