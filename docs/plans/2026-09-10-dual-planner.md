# 双 Planner 架构实施计划

- 日期：2026-09-10
- 设计依据：`docs/plans/2026-09-10-dual-planner-design.md`
- 执行方式：TDD，每个任务先写失败测试 → 实现 → 测试通过 → 提交

---

## 任务总览

| # | 任务 | 复杂度 | 产出文件 |
|---|---|---|---|
| T1 | 共享类型层 | 低 | `app/core/agent_types.py` |
| T2 | 共享工具层 | 中 | `app/core/tools.py` |
| T3 | 规划基类与工厂 | 低 | `app/core/agent_base.py` |
| T4 | 规则路线重构 | 中 | `app/core/agent.py` |
| T5 | LLM Planner | 高 | `app/core/llm_agent.py` |
| T6 | Prompt 集 | 中 | `app/core/prompts.py` |
| T7 | 配置项 | 低 | `app/core/config_data.py` |
| T8 | API 与 SSE 接入 | 中 | `app/api/api_service.py` |
| T9 | 评测数据集 | 中 | `app/evaluation/dataset.py` |
| T10 | A/B 评测器与 CLI | 高 | `app/evaluation/evaluator.py`、`run_planner_eval.py` |
| T11 | 前端标注与切换 | 中 | `html/index.html` |
| T12 | 全量回归与冒烟 | 中 | — |

---

## T1 共享类型层

**文件**：`app/core/agent_types.py`（新增）、`tests/test_agent_types.py`（新增）

**步骤**

1. 写测试 `tests/test_agent_types.py`：
   - `AgentRun` 仅用既有 5 个位置无关关键字参数构造成功（新字段有默认值）
   - `reasoning_events()` 首个事件 `type=="status"`、末个 `type=="reflection"`
   - `tool` 事件含 `title` / `content` 键
   - `planner_label` 对 `rule` / `llm` / 未知值均返回字符串
   - `extract_token_usage()` 对 `response_metadata.token_usage` 格式返回
     `{"input_tokens":18,"output_tokens":1,"total_tokens":19}`
   - `extract_token_usage()` 对 `usage_metadata` 格式返回同构 dict
   - `extract_token_usage()` 对无用量信息的对象返回全 0，不抛异常
2. 实现 `agent_types.py`：
   - 常量 `PLANNER_RULE="rule"` / `PLANNER_LLM="llm"` / `PLANNER_LABELS`
   - `@dataclass AgentToolResult(name, title, content, metadata={})`
   - `@dataclass AgentRun(mode, plan, tool_results, reflection, memory_updates,
     planner=PLANNER_RULE, fallback_reason="", metrics={})`
   - `status_lines()` / `reasoning_events()` 保持既有形状，事件内增加
     `"planner"` 字段
   - `extract_token_usage(message) -> dict` 双通道兼容
3. 运行 `tests/test_agent_types.py` + `tests/test_agent_reasoning.py`，全绿。

**验证**：`python -m unittest tests.test_agent_types tests.test_agent_reasoning -v`

---

## T2 共享工具层

**文件**：`app/core/tools.py`（新增）、`tests/test_tool_registry.py`（新增）

**步骤**

1. 写测试 `tests/test_tool_registry.py`（零网络，用假 `rag_service`）：
   - `ToolRegistry(fake_rag, fake_memory).names()` 返回 4 个工具名且顺序稳定
   - `invoke("hybrid_retrieval", q)` 返回 `AgentToolResult` 且
     `metadata["doc_count"]` 等于假检索返回条数
   - `invoke` 未知工具名抛 `KeyError`
   - `as_langchain_tools()` 返回 4 个 `BaseTool`，`name` 与注册表一致，
     且 `description` 非空
   - `resume_diagnosis` / `job_match` / `memory_lookup` 输出含设计文档
     约定的关键词（如 `量化`、`弱动词`、`匹配矩阵`）
2. 实现 `app/core/tools.py`：
   - `class ToolRegistry(rag_service, memory_store)`
   - 4 个 `_tool_*` 方法，逻辑从 `app/core/agent.py` 平移（**行为保持不变**）
   - `as_langchain_tools()` 用嵌套函数 + `@tool` 返回 `BaseTool` 列表，
     docstring 按设计文档 6.3 写作
3. 运行测试。

**验证**：`python -m unittest tests.test_tool_registry -v`

---

## T3 规划基类与工厂

**文件**：`app/core/agent_base.py`（新增）、`tests/test_agent_base.py`（新增）

**步骤**

1. 写测试：
   - `LongTermMemory` 按临时路径写入后可 `load_recent` 读回
   - `_extract_memory_candidates` 对含 `RAG`/`补强`/`简历` 的文本产出候选，
     且相同候选不重复写盘
   - `BaseCareerAgent.compose_agent_context(run)` 输出包含计划项、工具标题、
     反思结论三段
   - `build_agent_service(mode)` 对 `rule` / `llm` 返回对应类型实例
2. 实现 `agent_base.py`：
   - 平移 `LongTermMemory`（逻辑不变）
   - `BaseCareerAgent`：`__init__(rag_service)` 装配 `ToolRegistry` 与 `LongTermMemory`；
     提供 `compose_agent_context` / `finalize`；`prepare` 抛 `NotImplementedError`
   - `build_agent_service(rag_service, mode=None) -> BaseCareerAgent`，
     `llm` 分支延迟导入 `llm_agent` 避免循环依赖
3. 运行测试。

**验证**：`python -m unittest tests.test_agent_base -v`

---

## T4 规则路线重构

**文件**：`app/core/agent.py`（重构）、`tests/test_rule_planner.py`（新增）

**步骤**

1. 写测试（零网络，用假注册表断言调用顺序）：
   - 输入含「简历」→ `mode=="简历诊断"`，工具链含 `resume_diagnosis`
   - 输入含「JD」→ `mode=="岗位匹配"`，工具链含 `job_match`
   - 其他输入 → `mode=="综合求职分析"`，两条业务工具都出现
   - `plan` 长度在 5–6 之间，末两项固定为反思与记忆沉淀
   - `metrics["planner_llm_calls"] == 0` 且 `metrics["total_tokens"] == 0`
   - `planner == "rule"`
2. 重构 `agent.py`：
   - 删除本地 `AgentToolResult` / `AgentRun` / `LongTermMemory` 定义，改为
     `from app.core.agent_types import ...` 并保留 re-export
     （`from app.core.agent_base import LongTermMemory`）以满足
     `api_service` 的 `from app.core.agent import LongTermMemory`
   - `RulePlannerAgent(BaseCareerAgent)`：保留 `_detect_mode` / `_build_plan` /
     `_select_tools` / `_reflect`，工具调用改走 `self.tools.invoke(...)`
   - `CareerAgentService = RulePlannerAgent` 别名，保证外部引用不破
   - `metrics` 填 `{"planner_llm_calls": 0, "plan_latency_ms": <实测>, ...}`
3. 回归：`tests/test_agent_reasoning.py` 必须仍全绿。

**验证**：`python -m unittest tests.test_rule_planner tests.test_agent_reasoning -v`

---

## T5 LLM Planner

**文件**：`app/core/llm_agent.py`（新增）、`tests/test_llm_planner.py`（新增）

**步骤**

1. 写测试（用 `_FakeLLM` 桩注入，零网络）：
   - 桩返回 2 个合法 `tool_calls` → `run.plan` 含对应两行，`tool_results`
     长度为 2，`planner == "llm"`
   - 桩返回含幻觉工具名 `nonexistent_tool` 的 `tool_calls` → 该调用被丢弃，
     `metrics["invalid_tool_calls"] >= 1`
   - 桩返回空 `tool_calls` 且无幻觉 → 视为「模型主动判断无需工具」，`planner == "llm"`、
     `tool_names() == []`、`metrics["no_tool_decision"] == 1`，**不降级**
   - 桩返回的全是幻觉工具名 → 触发回退，`planner == "rule"` 且 `fallback_reason` 非空
   - 桩 `invoke` 抛异常 → 触发回退，`fallback_reason` 非空，不向外抛异常
   - 桩返回 `response_metadata.token_usage` → `metrics["total_tokens"] > 0`
   - `mode` 由实际工具集合反推：只调 `job_match` → `mode == "岗位匹配"`
2. 实现 `app/core/llm_agent.py`：
   - `LlmPlannerAgent(BaseCareerAgent)`
   - `TypedDict PlannerState(query, messages, tool_results, plan_steps,
     iterations, max_rounds, reflection, token_usage, invalid_tool_calls)`
   - `StateGraph`：`START → plan → (execute | reflect)`、`execute → plan`、
     `reflect → END`
   - `_plan_node`：`bind_tools(registry.as_langchain_tools())` 调用，
     记录 `plan_steps` 与 token 用量，`iterations += 1`
   - `_execute_node`：过滤非法工具名 → 逐个 `registry.invoke` → 追加
     `ToolMessage`，保证下一轮规划能看到结果
   - `_reflect_node`：调用 `REFLECTOR_MODEL`，异常时退回规则反思文案
   - `prepare()` 外层 `try/except`：失败 → `RulePlannerAgent.prepare()`
     并写 `fallback_reason`；零有效工具同理
   - 构造函数支持注入 `llm` / `reflector`，供测试替换
3. 运行测试。

**验证**：`python -m unittest tests.test_llm_planner -v`

---

## T6 Prompt 集

**文件**：`app/core/prompts.py`（追加）、`tests/test_prompts.py`（新增）

**步骤**

1. 写测试：
   - `planner_system_prompt` 含 `{tools}` 占位符，`format` 后含 4 个工具名
   - `reflector_prompt` 含 `{query}` 与 `{evidence}` 占位符
   - `judge_prompt` 含 `{query}` / `{evidence}` / `{answer}` 占位符
   - 三段 prompt 均含「禁止编造」或等价约束（防止后续被误改掉）
2. 按设计文档 6.1 / 6.2 / 6.4 写入 `prompts.py`，不改动既有 3 段 prompt。

**验证**：`python -m unittest tests.test_prompts -v`

---

## T7 配置项

**文件**：`app/core/config_data.py`

**追加**

```python
# Planner 配置
PLANNER_MODE = os.getenv("PLANNER_MODE", "rule")          # rule | llm
LLM_PLANNER_MODEL = os.getenv("LLM_PLANNER_MODEL", "qwen-max")
REFLECTOR_MODEL = os.getenv("REFLECTOR_MODEL", "qwen-turbo")
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "2"))  # 工具执行轮次上限
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen-turbo")      # 评测裁判
# 迭代一追加
PLANNER_HISTORY_TURNS = int(os.getenv("PLANNER_HISTORY_TURNS", "3"))
PLANNER_HISTORY_CHARS = int(os.getenv("PLANNER_HISTORY_CHARS", "80"))
```

> **实施偏差记录**：原计划中的 `PLANNER_LLM_TIMEOUT` **已删除**。原因是
> `ChatTongyi` 与 `dashscope.Generation.call` 均不支持透传 `timeout`，
> 无法真正生效；与其留一个撒谎的死参数，不如删掉并记入技术债
> （见架构文档 §9.2 第 5 条）。

**验证**：`python -c "from app.core import config_data as c; print(c.PLANNER_MODE)"` → `rule`

---

## T8 API 与 SSE 接入

**文件**：`app/api/api_service.py`、`tests/test_planner_sse.py`（新增）

**步骤**

1. 写测试：
   - `/health` 返回体包含 `planner_modes` 与默认 `planner`
   - `build_planner_event(run)` 产出 `{"type":"planner","planner","label","note"}` 四键
   - `build_metrics_event(run)` 产出 `{"type":"metrics", ...}` 且含
     `plan_latency_ms` / `total_tokens` / `planner_llm_calls`
   - `get_agent_service("llm")` 与 `get_agent_service("rule")` 返回不同实例类型
     （用 `PLANNER_MODE` 无关的显式传参路径，避免依赖环境）
2. 改 `api_service.py`：
   - `get_agent_service(mode=None)` 用 `build_agent_service` 工厂，
     按 mode 缓存到 `_agent_cache: dict[str, Any]`
   - `/chat` 增加可选 `planner_mode` Body 参数，非法值回落配置默认
   - `event_generator` 开头、事件序列与结尾分别插入 `planner` / `metrics` 事件
   - `/health` 增补 planner 能力描述
3. 运行 `tests/test_planner_sse.py` 与 `tests/test_sse_format.py`。

**验证**：`python -m unittest tests.test_planner_sse tests.test_sse_format -v`

---

## T9 评测数据集

**文件**：`app/evaluation/__init__.py`、`app/evaluation/dataset.py`（新增）

**步骤**

1. 定义 20 条用例，字段
   `{"id", "query", "expected_tools", "intent", "note"}`，覆盖：
   - 岗位匹配 6 条（含 1 条「只给 JD 问匹配度」）
   - 简历诊断 6 条（含 1 条「只给简历原文求改」）
   - 复合意图 4 条（简历 + 岗位同时出现，规则路线预计只命中一类）
   - 历史语境 2 条（应命中 `memory_lookup`）
   - 边界/无关输入 2 条
2. 加校验函数 `validate_dataset()`：断言 id 唯一、`expected_tools` 非空、
   工具名均在注册表内。测试 `tests/test_eval_dataset.py` 调用它。

**验证**：`python -m unittest tests.test_eval_dataset -v`

---

## T10 A/B 评测器与 CLI

**文件**：`app/evaluation/evaluator.py`、`run_planner_eval.py`、`tests/test_eval_metrics.py`（新增）

**步骤**

1. 写纯函数并测试（不涉及网络）：
   - `required_recall(expected, produced)` → 交集/期望
   - `extra_calls(expected, produced)` → 差集大小
   - `exact_match(expected, produced)` → 集合相等
   - `percentile(values, p)` → 正确处理空列表与单值
   - `aggregate(records)` → 产出两路线的汇总 dict，含 `n` / `recall_mean` /
     `extra_mean` / `exact_rate` / `latency_p50` / `latency_p95` /
     `tokens_mean` / `fallback_rate`
2. 实现 `PlannerABEvaluator`：
   - 复用 `RagService` 实例喂给两条路线，避免重复加载向量库
   - 每条用例分别跑 `rule` 与 `llm`，只测**规划阶段**（`prepare`），
     默认不生成最终回答（省 token、快）
   - `--with-answer` 才跑 `chain.invoke` 并统计端到端耗时
   - `--with-judge` 才调用 judge prompt 计算 0-6 分
   - `report()` 输出 markdown 表，写入
     `docs/plans/2026-09-10-dual-planner-eval.md`
3. `run_planner_eval.py` CLI 参数：`--planner`、`--limit`、
   `--with-answer`、`--with-judge`、`--out`。

**验证**：`python -m unittest tests.test_eval_metrics -v`
集成：`python run_planner_eval.py --limit 3`（消耗少量 token，手动触发）

---

## T11 前端标注与切换

**文件**：`html/index.html`

**步骤**

1. `state` 增加 `planner: "rule"`；侧栏增加 planner 切换组（复用 `.mode-chip` 样式）
2. `sendMessage` 请求体带上 `planner_mode: state.planner`
3. `renderReasoning` 增加 `planner` 分支：渲染为横幅
   「规划器：规则路由（无模型推理）」/「规划器：LLM 规划（Function Calling）」，
   有 `note` 时追加降级说明
4. 增加 `metrics` 分支：渲染为一行等宽小字「规划耗时 1.8s · 规划 token 1,240 ·
   LLM 轮次 2」
5. 面板标题由固定 `Agent 工作流` 改为按 `planner` 动态显示
6. 样式：仅新增 `.r-planner` / `.r-metrics` 两条规则，不改配色变量

**验证**：启动服务，两种 planner 各发一条消息，肉眼确认横幅与指标正确。

---

## T12 全量回归与冒烟

1. `python -m unittest discover -s tests -v` → 全部通过
2. `python run_planner_eval.py --limit 5` → 产出真实指标表
3. 启动 `uvicorn app.api.api_service:app --port 8000`，页面端到端验证两种模式
4. 填写设计文档第 9 节的真实指标表

---

## 提交节奏

每个任务测试转绿后提交一次，提交信息格式：

```
feat(planner): T5 新增 LangGraph LLM 规划器

- app/core/llm_agent.py: plan→execute→reflect 状态机
- 工具名幻觉过滤与失败回退
- tests/test_llm_planner.py: 6 项零网络单测
```

## 风险与回滚

| 风险 | 影响 | 应对 |
|---|---|---|
| `ChatTongyi.bind_tools` 在某模型上不稳定 | LLM 路线不可用 | 已验证 qwen-max 可用；异常统一走回退，不影响主链路 |
| 评测消耗 token | 成本 | 默认只测规划阶段；`--limit` 限制用例数 |
| 重构 `agent.py` 破坏既有引用 | 接口 500 | 保留 `CareerAgentService` 别名与 `LongTermMemory` re-export；T12 全量回归 |
| 前端改动影响现有交互 | UI 回归 | 仅新增分支与两条 CSS 规则，不动既有渲染路径 |
