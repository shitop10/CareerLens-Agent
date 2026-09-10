# 双 Planner 架构设计（规则路由 + LLM 规划）

- 日期：2026-09-10
- 状态：已确认，进入实施
- 关联：`docs/plans/2026-09-10-dual-planner.md`（实施计划）

---

## 1. 背景与问题

CareerLens 当前只有一条 Agent 编排路线。`app/core/agent.py` 中的
`_detect_mode` / `_build_plan` / `_select_tools` / `_tool_*` / `_reflect`
全部是本地确定性代码（if-else 与字符串拼接），不发起任何模型请求。

问题有两个层次：

**工程层面**

1. 规划能力对用户输入不敏感。`_select_tools` 只按 `mode` 三分类选工具，
   复合意图（「帮我看看这份简历投大模型算法岗行不行」）只能命中一类分支。
2. 前端 `html/index.html` 的 reasoning 面板把 `[状态] Agent 正在识别任务意图...`、
   `[Agent计划] ...`、`[反思] ...` 渲染成「Agent 工作流」，观感上等同模型思考过程，
   但实际是硬编码字符串。**这是表述风险**：演示时被要求打开 planner prompt 即会暴露。
3. 全链路唯一消耗 token 的环节是最后一步 `rag.chain.astream()`，规划本身零成本，
   但也没有任何可迁移的 Agent 能力沉淀。

**面试层面**

大模型岗面试官高频追问「你的 Agent 规划怎么实现的」。当前实现无法支撑
「LLM 自主规划」这一表述，但直接换成纯 LLM Planner 又会退化为
「又一个 LangGraph `AgentExecutor` 套壳」，且答不上「为什么两类固定意图还要上 LLM」。

## 2. 实现目标

| 编号 | 目标 | 验收方式 |
|---|---|---|
| G1 | 新增一条真实由 LLM function calling 驱动的规划路线 | 日志/SSE 可见真实 `tool_calls` 与 token 用量 |
| G2 | 两条路线并存、运行时切换，共享同一套工具实现 | 单测断言两条路线调用同一 `ToolRegistry` |
| G3 | 产出可量化 A/B 对比数据 | `run_planner_eval.py` 输出指标表 |
| G4 | 消除表述风险：规则路线不再伪装成模型思考 | SSE 携带 `planner` 标记，前端差异化展示 |
| G5 | 保持既有行为与测试不被破坏 | `tests/` 全绿 |

**非目标**：不做「Agent 全面智能化」的改造，不追求 LLM 路线在所有用例上优于规则路线。

## 3. 范围

### 3.1 范围内（IN）

| 模块 | 文件 | 说明 |
|---|---|---|
| 共享类型 | `app/core/agent_types.py`（新增） | `AgentToolResult` / `AgentRun` / planner 常量 / token 用量提取 |
| 共享工具层 | `app/core/tools.py`（新增） | `ToolRegistry`：4 个工具的唯一实现 + LangChain `@tool` 视图 |
| 规划基类 | `app/core/agent_base.py`（新增） | `LongTermMemory`、`BaseCareerAgent`、`build_agent_service` 工厂 |
| 规则路线 | `app/core/agent.py`（重构） | `RulePlannerAgent`，行为与现状一致，改用共享工具层 |
| LLM 路线 | `app/core/llm_agent.py`（新增） | `LlmPlannerAgent`，LangGraph plan→execute→reflect |
| 配置 | `app/core/config_data.py` | 新增 planner 相关配置项 |
| 接口 | `app/api/api_service.py` | `/chat` 支持 planner 覆盖；SSE 增加 `planner` / `metrics` 事件 |
| 评测 | `app/evaluation/`（新增）、`run_planner_eval.py`（新增） | A/B 对比与报告 |
| 前端 | `html/index.html` | planner 切换控件 + 诚实标注 + 指标展示 |
| 测试 | `tests/` | 新增零网络单测 |

### 3.2 范围外（OUT，YAGNI）

- 不引入新第三方依赖（`langgraph==1.1.3` 已在 `requirements.txt`）
- 不改动向量库 / 检索链路 / Embedding 模型
- 不做前端视觉改版，仅增加必要控件与事件渲染
- 不做多 Agent 协作、不做规划结果缓存
- 不迁移数据库 schema

## 4. 架构

```
                        ┌──────────────────┐
                        │  用户提问 (SSE)   │
                        └────────┬─────────┘
                                 │
                    PLANNER_MODE │ (rule | llm)
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      ┌────────────────────┐         ┌────────────────────────┐
      │ RulePlannerAgent   │         │ LlmPlannerAgent        │
      │ 关键词意图分类      │         │ LangGraph              │
      │ 固定计划模板        │         │  plan → execute →      │
      │ 规则化反思          │         │  plan → ... → reflect  │
      │ 0 token            │         │ qwen-max + bind_tools  │
      └─────────┬──────────┘         └───────────┬────────────┘
                │                                │
                └────────────┬───────────────────┘
                             ▼
                  ┌──────────────────────┐
                  │  ToolRegistry (共享)  │
                  │  memory_lookup        │
                  │  hybrid_retrieval     │
                  │  resume_diagnosis     │
                  │  job_match            │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ AgentRun (统一产出)   │
                  │  plan / tool_results  │
                  │  reflection / metrics │
                  │  planner / fallback   │
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ rag.chain.astream()  │  ← 回答生成（两条路线共用）
                  └──────────────────────┘
```

### 4.1 关键设计决策

**D1：不替换，而是并存。**
理由：两类固定意图上规则路由是更优解（零成本、可复现、延迟稳定），
LLM 路线的价值在于泛化与被验证。并存才能产出对比数据，而对比数据本身就是交付物。

**D2：工具实现唯一化。**
`ToolRegistry` 是 4 个工具的唯一实现，同时暴露
`invoke(name, query) -> AgentToolResult`（给规则路线）
与 `as_langchain_tools() -> list[BaseTool]`（给 LLM 路线）。
避免两条路线各写一份导致行为漂移。现有 4 个 `_tool_*` 方法已是
返回 `AgentToolResult` 的纯函数，迁移成本低。

**D3：`AgentRun` 作为唯一契约。**
两条路线产出同一个 dataclass，`api_service` 与前端无需分支处理。
新增字段全部带默认值，保证 `tests/test_agent_reasoning.py` 无需修改即可通过。

**D4：`mode` 在 LLM 路线下由实际选中的工具反推。**
规则路线用 `_detect_mode` 关键词分类；LLM 路线让规划器自主决定工具后，
从工具集合反推意图标签。两者的 `mode` 语义一致，但 LLM 路线的标签
是其决策的**结果**而非**输入**。

**D5：失败必须可见且可回退。**
LLM Planner 任何环节失败（网络异常、工具名幻觉、零有效工具）都回退到规则路线，
并在 `AgentRun.fallback_reason` 与 SSE 事件中显式说明。**不允许静默降级。**

## 5. 数据流

```
POST /chat {session_uuid, input_text, planner_mode?}
  → 解析 planner_mode（请求 > 配置默认）
  → get_agent_service(mode).prepare(query)
      ├─ rule: _detect_mode → _build_plan → registry.invoke(...) → _reflect
      └─ llm:  graph.invoke() → plan(LLM) → execute(registry) → ... → reflect(LLM)
              失败 → 回退 rule，写 fallback_reason
  → SSE: event:reasoning {type:"planner", planner, label, note}
  → SSE: event:reasoning run.reasoning_events()   ← 保持既有事件形状
  → compose_agent_context(run) → enhanced_input
  → rag.chain.astream(enhanced_input) → SSE: event:answer
  → SSE: event:reasoning {type:"metrics", ...}
  → finalize(query, answer, run) → SSE: event:reasoning {type:"memory"}
  → SSE: event:done
```

## 6. Prompt 设计

Prompt 是本方案相对「换个 LLM 库」的核心差异点，全部围绕
**大模型算法岗求职知识库**这一真实场景编写。

### 6.1 Planner 系统提示（`app/core/prompts.py`）

```text
你是 CareerLens 的 Agent 规划器（Planner），服务于「大模型算法岗求职」场景。

你的唯一职责：把用户的自然语言问题，拆解成对下方工具的调用计划。
你不负责回答问题，只负责决策"该调用哪些工具、用什么检索词"。

可用工具：
{tools}

规划规则（必须遵守）：
1. 任何问题都必须调用 hybrid_retrieval，它是证据来源；query 需改写成检索友好表述，
   保留原始实体（岗位名、技术名词、公司名）。
2. 问题涉及简历、项目经历、技能表达、成果量化 → 调用 resume_diagnosis。
3. 问题涉及岗位、JD、职位要求、能力差距、匹配度 → 调用 job_match。
4. 仅当用户显式引用历史语境（如"上次说的""之前提过""还是按那个"）时才调用 memory_lookup。
5. 单轮最多调用 3 个工具。禁止发明工具名，禁止调用清单外的工具。
6. 不要为同一个工具重复调用两次。若无法判断，只调用 hybrid_retrieval。

改写示例：
  用户：「帮我看看这份简历投大模型算法岗行不行」
  → hybrid_retrieval(query="大模型算法岗 岗位要求 简历匹配度")
  → job_match(query="大模型算法岗 能力匹配 差距")
  → resume_diagnosis(query="简历 项目经历 岗位匹配度 量化")
```

设计要点：
- **强制 `hybrid_retrieval`**：本项目回答质量强依赖知识库证据，与 `prompts.py`
  中 `rag_system_prompt` 第 1 条「优先基于参考资料回答」一致。
- **工具数量上限 3**：对应 `app/core/prompts.py` 中「结论/证据/建议/下一步」
  四段式输出，超过 3 个工具会挤占回答的上下文预算。
- **要求改写 query 而非原样透传**：这是 function calling 相对固定管道的实际增量，
  也是评测里可观察的差异点。

### 6.2 Reflector 提示

```text
你是 CareerLens 的反思评审器（Reflector）。

用户问题：{query}

本轮工具执行结果：
{evidence}

请输出一段不超过 120 字的反思结论，必须覆盖三点：
1. 证据是否足以支撑结论（充分 / 部分充分 / 不足）；
2. 是否存在风险（工具失败、召回为空、证据与问题不相关）；
3. 是否需要提醒用户补充材料（JD、简历原文、项目数据）。

约束：只能基于上方工具结果判断，禁止补充任何未出现的证据或量化指标。
只输出反思结论本身，不要输出标题、编号或前缀。
```

设计要点：与 `rag_system_prompt` 第 1、2 条（不编造、证据不足要说明）
对齐，把「不许编造」显式写进反思器而不是依赖模型自觉。

### 6.3 工具 description（决定规划质量的隐蔽关键）

Function calling 的选型准确率主要由 tool description 决定，因此 4 个工具的
docstring 按「做什么 + 何时用 + 何时不用」三段式编写，而非一句话概述：

| 工具 | description 要点 |
|---|---|
| `memory_lookup` | 读长期记忆中的偏好与历史短板；**仅在有历史语境时使用** |
| `hybrid_retrieval` | Chroma 语义 + BM25 关键词 + RRF 融合；**证据来源，必调** |
| `resume_diagnosis` | 检查问题定义/技术方案/指标结果/个人贡献；识别「负责/参与/熟悉」弱动词 |
| `job_match` | 抽取岗位能力项并对照候选人项目证据，产出「已具备/需补强/可补证据」 |

### 6.4 评测用 LLM-as-judge 提示（可选，`--with-judge`）

```text
你是严格的技术面试评委。请只依据下方材料评分，不要脑补。

用户问题：{query}
参考证据：
{evidence}
待评回答：
{answer}

按三个维度各打 0-2 分并给一句话理由：
A. 结论是否有证据支撑（无证据即 0 分）
B. 是否给出了可执行的具体动作（非泛泛鼓励）
C. 是否出现材料中不存在的编造信息（出现即 0 分）

最后输出一行：TOTAL=<0-6 的整数>
```

## 7. 错误处理

| 场景 | 处理 | 用户可见性 |
|---|---|---|
| 规划器 LLM 调用异常/超时 | 回退规则路线 | SSE `planner` 事件 `note` 写明原因 |
| 工具名幻觉（不在注册表） | 丢弃该调用并记日志 | `metrics.invalid_tool_calls` |
| 模型主动判断无需工具（0 调用且无幻觉） | **视为合法决策**，产出零工具 `AgentRun`，不降级 | `metrics.no_tool_decision=1` |
| 全部工具调用均为幻觉（0 有效且 invalid>0） | 回退规则路线 | `fallback_reason` |
| 同一工具以相同检索词重复调用 | 跳过执行，回复"已调用过" | `metrics.duplicate_tool_calls` |
| 工具执行抛错 | 捕获，产出 `metadata.error=True` 的 `AgentToolResult` | `reflection` 中提示不确定性 |
| 反思器失败 | 退回规则路线的确定性反思文案 | 无感 |
| 工具执行轮次达到 `MAX_TOOL_ROUNDS` | 强制进入 reflect，后续轮次的调用不再执行 | `metrics.tool_rounds` |

原则：**任何降级都必须留下痕迹**，不允许静默吞掉。

## 8. 测试策略

**单测（零网络，默认跑）** — 沿用 `tests/test_knowledge_files.py` 的
`_FakeEmbeddings` 注入模式，新增 `_FakeLLM` 桩：

| 测试 | 断言 |
|---|---|
| `test_tool_registry.py` | 4 个工具可调用、返回 `AgentToolResult`、LangChain 视图可 bind |
| `test_llm_planner.py` | 正常 tool_calls → 正确执行；幻觉工具名被丢弃；零工具 → 回退；LLM 抛错 → 回退 |
| `test_token_usage.py` | 兼容 `response_metadata.token_usage` 与 `usage_metadata` 两种格式 |
| `test_planner_sse.py` | `planner` / `metrics` 事件字段完备；`reasoning_events()` 形状不变 |
| `test_eval_metrics.py` | 召回率/多余调用数/精确匹配在构造输入上的计算正确 |

**回归（必须保持全绿）** — `tests/test_agent_reasoning.py`、
`tests/test_sse_format.py`、`tests/test_knowledge_files.py`、
`tests/test_delete_session.py`。

**集成（需网络与 token，手动触发）** — `run_planner_eval.py` 输出 A/B 指标表。

所有真实 LLM 调用在单测中一律用桩替换，**测试不得消耗 token**。

## 9. 预期产出

| 产出 | 路径 | 形式 |
|---|---|---|
| 共享类型 | `app/core/agent_types.py` | 代码 |
| 共享工具层 | `app/core/tools.py` | 代码 |
| 规划基类与工厂 | `app/core/agent_base.py` | 代码 |
| 规则路线（重构） | `app/core/agent.py` | 代码 |
| LLM 路线 | `app/core/llm_agent.py` | 代码 |
| Prompt 集 | `app/core/prompts.py` | 新增 3 段 prompt |
| 评测数据集（20 条） | `app/evaluation/dataset.py` | 代码 |
| A/B 评测器 | `app/evaluation/evaluator.py` | 代码 |
| 评测 CLI | `run_planner_eval.py` | 代码 |
| 前端控件与标注 | `html/index.html` | 代码 |
| 单测 | `tests/test_tool_registry.py` 等 5 个 | 代码 |
| 评测报告 | `docs/plans/2026-09-10-dual-planner-eval.md` | 运行后生成 |

**可量化验收指标** —— 由 `run_planner_eval.py` 产出，实测结果见第 11 节。

## 10. 已知限制

1. `MAX_TOOL_ROUNDS` 默认 2，即最多执行 2 轮工具调用、规划器最多被调用 3 次。
   更长的多跳任务（如「先查岗位→再查项目→再比对→再给学习路径」）覆盖不足，属成本约束下的取舍。
2. ~~规划器不接收历史对话上下文，多轮指代无法解析~~ **已解决，见第 12 节。**
3. 评测集的 `expected_tools` 为人工标注，规模 20 条，不具备统计显著性，
   仅用于验证路线差异方向。
4. 规则路线的 `mode` 与 LLM 路线的 `mode` 推算口径不同，跨路线比较 `mode` 无意义。
5. 规划延迟指标包含两条路线共享的检索耗时（`hybrid_retrieval` 需调用 Embedding 接口），
   因此"规则路线 456ms"并非纯路由开销。

## 11. 实测结论（2026-09-10，20 条用例）

由 `run_planner_eval.py` 真实运行产出，完整数据见 `docs/plans/2026-09-10-dual-planner-eval.md`。

| 指标 | 规则路线 | LLM 路线 |
|---|---|---|
| 必需工具召回率 | 0.9333 | 0.9500 |
| 多余调用数(均值) | 1.30 | 0.05 |
| 精确匹配率 | 0.0500 | 0.8000 |
| 规划延迟 P50 | 460 ms | 8054 ms |
| 规划延迟 P95 | 608 ms | 11799 ms |
| 规划 token/轮 | 0 | 4710 |
| 规划 LLM 调用次数 | 0 | 2.25 |
| 回退率 | — | 0.0000 |

**结论：两条路线各有胜负，不存在单方面更优。**

- LLM 路线在**精确匹配率**上碾压（0.80 vs 0.05）。根因是规则路线**固定调用
  `memory_lookup`**，20 条用例里 20 条都产生 1 个多余调用（`mem_01` 之外还额外
  调了 `job_match` + `resume_diagnosis`），均值 1.30。
- 规则路线在**复合意图**上是结构性短板：召回率 0.6667，因为关键词
  `_detect_mode` 只能命中单一分支；LLM 路线 0.8333。
- LLM 路线仍有漏召：`mix_02` 漏 `resume_diagnosis`、`mix_03` 漏 `job_match`、
  `mem_02` 漏 `hybrid_retrieval`（**违反 prompt 第 1 条强制检索**）。说明
  function calling 的稳定性是真实约束，不能默认"上了 LLM 就更好"。
- `edge_01`（"你好"）验证了零工具决策：LLM 正确判断无需检索（0 工具），
  而规则路线无差别调用 4 个工具。
- 代价明确：LLM 路线单轮多约 4710 token、延迟高约 17 倍（P50 8.1s vs 0.46s）。

这正是保留双路线而非替换的理由：**取舍需要数据支撑，而不是直觉。**
`PLANNER_MODE` 默认 `rule`，LLM 路线作为可切换的高泛化选项。

---

## 12. 迭代一：规划器接入对话历史（2026-09-10）

### 12.1 动机

原实现中 `prepare(query)` 只接收当前问题，`暂无历史` —— 用户问「那这个呢」时，
规划器无法判断指代对象，只能退化为单轮检索。多轮追问是求职咨询场景的**主流用法**，
这是上一轮 A/B 评测暴露出的最大体验缺口。

### 12.2 设计

**历史压缩而非全量注入。** AI 回答动辄 1000+ 字，全量塞进规划 prompt 会挤爆预算。
因此取最近 `PLANNER_HISTORY_TURNS`（默认 3）轮，每侧各截断 `PLANNER_HISTORY_CHARS`
（默认 80）字，压成编号摘要：

```
历史对话摘要（最近轮次，仅供解析指代，不要当成本轮问题）：
[1] 帮我看看简历里的项目经历该怎么写 ｜ 项目经历需具体量化，突出技术细节…
[2] 那这个呢 ｜ ### 任务规划 根据用户的需求，我将帮助优化…

当前问题：那这个呢
```

**两条路线共享历史，但用法不同（保持 A/B 可比）：**

| 路线 | 用法 |
|---|---|
| 规则路线 | 当前问题命中不了意图关键词时，回看最近 3 轮用户发言推断意图。**显式问题永远优先**，因此不改变既有行为。 |
| LLM 路线 | 摘要注入规划 prompt，由模型做真正的指代消解与检索词补全。 |

**新增 prompt 规则 9** —— 明确「摘要只用于理解指代，不要照抄，也不要把历史问题当成本轮问题」，
避免模型把历史问题当成新任务重复检索。

### 12.3 关键工程问题：历史落库竞态

首轮冒烟测试发现：**第 2 轮的 `history_turns` 恒为 0**，LLM 规划器对「那这个呢」
直接回「请提供更多信息」——历史没读到。

根因：`save_chat_history` 是流结束后才 `asyncio.create_task` 调度的后台任务，
且内部要先生成标题与摘要（2 次模型调用，约 1 秒）才写入消息行。
而用户在收到回答后**立刻**追问，此时上一轮的消息尚未提交。

修复：把落库拆成两段——

| 阶段 | 时机 | 内容 |
|---|---|---|
| `persist_message` | **`done` 事件之前 await** | 写入消息行（摘要先用 `clean_text[:50]` 占位），毫秒级 |
| `enrich_message` | `done` 之后 `create_task` | 生成标题与摘要并回写，慢但用户无感 |

这样历史可见性从「约 1 秒后」提前到「本轮响应结束前」，竞态窗口消失。

### 12.4 验证

真实 HTTP 多轮测试（同一会话，第 2 轮在第 1 轮响应结束后**立即**发出）：

| 路线 | 第 1 轮 history_turns | 第 2 轮 history_turns | 第 2 轮规划结果 |
|---|---|---|---|
| LLM | 0 | **1** | `hybrid_retrieval → 「简历 项目经历 写作指导」`、`resume_diagnosis → 「简历 项目经历 写作质量」` |
| 规则 | 0 | **1** | 由历史推断意图 = 简历诊断 |

修复前 LLM 路线第 2 轮返回「请提供更多信息」（0 工具）；修复后正确解析为简历项目经历主题。

### 12.5 新增测试（24 项，累计 214 项全绿）

| 文件 | 覆盖 |
|---|---|
| `tests/test_history_context.py` | 消息解析、摘要压缩与截断、空历史、隐式追问的意图继承、加载器异常降级 |
| `tests/test_llm_planner.py`（扩充） | 摘要注入 human 消息、无 session_uuid 时不注入、指标记录、两条路线签名一致 |
| `tests/test_message_persistence.py` | `persist_message` 返回后**另一条连接即可读到**（竞态回归）、`enrich_message` 回写标题与摘要、模型异常不丢消息、`done` 事件顺序的静态契约 |
