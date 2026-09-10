"""Planner A/B 评测数据集。

用例围绕「大模型算法岗求职」真实场景标注，``expected_tools`` 为人工判定的
**必需工具集合**：

- 非空集合 = 必须至少调用这些工具（多调记入「多余调用数」）
- **空集合 = 正确行为是不调用任何工具**（如纯寒暄）

分类意图：
- job_*   岗位匹配：应命中 job_match
- res_*   简历诊断：应命中 resume_diagnosis
- mix_*   复合意图：简历与岗位同时出现，规则路线（单分支路由）预计只命中一类
- mem_*   历史语境：应命中 memory_lookup
- edge_*  边界输入：只需基础检索，不应触发业务工具

注意：本数据集 20 条，仅用于验证两条路线的差异方向，不具备统计显著性。
"""

from __future__ import annotations

from app.core.tools import TOOL_NAMES

PLANNER_CASES: list[dict] = [
    # ---------- 岗位匹配 ----------
    {
        "id": "job_01",
        "intent": "岗位匹配",
        "query": "大模型算法岗的 JD 里最核心的能力要求是什么？",
        "expected_tools": ["hybrid_retrieval", "job_match"],
        "note": "纯岗位要求分析",
    },
    {
        "id": "job_02",
        "intent": "岗位匹配",
        "query": "这个岗位要求 LoRA 微调，我需要补强到什么程度？",
        "expected_tools": ["hybrid_retrieval", "job_match"],
        "note": "能力差距 + 技术名词",
    },
    {
        "id": "job_03",
        "intent": "岗位匹配",
        "query": "帮我对比一下大模型算法岗和大模型应用开发岗的要求差异。",
        "expected_tools": ["hybrid_retrieval", "job_match"],
        "note": "多岗位对比",
    },
    {
        "id": "job_04",
        "intent": "岗位匹配",
        "query": "JD 里写了要熟悉 RAG 和向量数据库，这算硬性要求吗？",
        "expected_tools": ["hybrid_retrieval", "job_match"],
        "note": "JD 条款解读",
    },
    {
        "id": "job_05",
        "intent": "岗位匹配",
        "query": "我还差哪些能力才能投这个职位？",
        "expected_tools": ["hybrid_retrieval", "job_match"],
        "note": "口语化能力差距",
    },
    {
        "id": "job_06",
        "intent": "岗位匹配",
        "query": "Do a gap analysis between my profile and this job posting.",
        "expected_tools": ["hybrid_retrieval", "job_match"],
        "note": "英文输入，验证 LLM 路线的跨语言泛化",
    },
    # ---------- 简历诊断 ----------
    {
        "id": "res_01",
        "intent": "简历诊断",
        "query": "帮我看看简历里专业技能这一段该怎么改。",
        "expected_tools": ["hybrid_retrieval", "resume_diagnosis"],
        "note": "简历措辞优化",
    },
    {
        "id": "res_02",
        "intent": "简历诊断",
        "query": "我的实习经历写得太空泛了，怎么量化？",
        "expected_tools": ["hybrid_retrieval", "resume_diagnosis"],
        "note": "量化表达",
    },
    {
        "id": "res_03",
        "intent": "简历诊断",
        "query": "项目经历里写「负责」是不是不太好？",
        "expected_tools": ["hybrid_retrieval", "resume_diagnosis"],
        "note": "弱动词识别",
    },
    {
        "id": "res_04",
        "intent": "简历诊断",
        "query": "简历上的技能表达怎么改成可验证的动作？",
        "expected_tools": ["hybrid_retrieval", "resume_diagnosis"],
        "note": "动作化改写",
    },
    {
        "id": "res_05",
        "intent": "简历诊断",
        "query": "我参与过一个知识库项目，应该怎么描述个人贡献？",
        "expected_tools": ["hybrid_retrieval", "resume_diagnosis"],
        "note": "个人贡献描述",
    },
    {
        "id": "res_06",
        "intent": "简历诊断",
        "query": "帮我诊断一下简历里的成果描述，有没有缺少指标？",
        "expected_tools": ["hybrid_retrieval", "resume_diagnosis"],
        "note": "成果指标缺失",
    },
    # ---------- 复合意图（规则路线的已知短板） ----------
    {
        "id": "mix_01",
        "intent": "复合意图",
        "query": "帮我看看这份简历投大模型算法岗行不行？",
        "expected_tools": ["hybrid_retrieval", "job_match", "resume_diagnosis"],
        "note": "简历 + 岗位同时出现",
    },
    {
        "id": "mix_02",
        "intent": "复合意图",
        "query": "我的项目经历能不能撑起这个 JD 的能力要求？",
        "expected_tools": ["hybrid_retrieval", "job_match", "resume_diagnosis"],
        "note": "项目经历 × 岗位要求",
    },
    {
        "id": "mix_03",
        "intent": "复合意图",
        "query": "简历里的 RAG 项目怎么写才能匹配岗位要求？",
        "expected_tools": ["hybrid_retrieval", "job_match", "resume_diagnosis"],
        "note": "改写方向由岗位倒推",
    },
    {
        "id": "mix_04",
        "intent": "复合意图",
        "query": "既要改简历又要看岗位匹配度，我应该先从哪一步开始？",
        "expected_tools": ["hybrid_retrieval", "job_match", "resume_diagnosis"],
        "note": "显式双诉求",
    },
    # ---------- 历史语境 ----------
    {
        "id": "mem_01",
        "intent": "历史语境",
        "query": "上次说我的短板是评估能力，那这次该怎么补？",
        "expected_tools": ["hybrid_retrieval", "memory_lookup"],
        "note": "显式引用历史结论",
    },
    {
        "id": "mem_02",
        "intent": "历史语境",
        "query": "还是按之前那个方向，帮我看下简历。",
        "expected_tools": ["hybrid_retrieval", "memory_lookup", "resume_diagnosis"],
        "note": "历史 + 简历双诉求",
    },
    # ---------- 边界输入 ----------
    {
        "id": "edge_01",
        "intent": "边界",
        "query": "你好",
        "expected_tools": [],
        "note": "纯寒暄，正确行为是不调用任何工具（空期望 = 不应调用）",
    },
    {
        "id": "edge_02",
        "intent": "边界",
        "query": "RAG 和 BM25 的区别是什么？",
        "expected_tools": ["hybrid_retrieval"],
        "note": "技术概念答疑，不需要简历/岗位工具",
    },
]


def validate_dataset(cases: list[dict] | None = None) -> list[str]:
    """返回数据集的问题清单，空列表表示通过。"""
    cases = cases if cases is not None else PLANNER_CASES
    problems: list[str] = []
    seen: set[str] = set()
    valid_tools = set(TOOL_NAMES)

    for index, case in enumerate(cases):
        for key in ("id", "query", "intent", "expected_tools"):
            if key not in case:
                problems.append(f"第 {index} 条缺少字段 {key}")
        case_id = case.get("id")
        if case_id in seen:
            problems.append(f"重复的用例 id: {case_id}")
        seen.add(case_id)

        expected = case.get("expected_tools")
        if not isinstance(expected, list):
            problems.append(f"{case_id} 的 expected_tools 必须是列表")
        else:
            unknown = [t for t in expected if t not in valid_tools]
            if unknown:
                problems.append(f"{case_id} 含未注册工具: {unknown}")
        if not (case.get("query") or "").strip():
            problems.append(f"{case_id} 的 query 为空")

    return problems
