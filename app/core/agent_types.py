"""共享类型与契约定义。

规则 Planner 与 LLM Planner 共用本模块，避免二者各写一份导致行为漂移，
同时避免 agent.py / llm_agent.py 之间的循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# ---------------------------------------------------------------------------
# Planner 标识
# ---------------------------------------------------------------------------

PLANNER_RULE = "rule"
PLANNER_LLM = "llm"

PLANNER_LABELS = {
    PLANNER_RULE: "规则路由（无模型推理）",
    PLANNER_LLM: "LLM 规划（Function Calling）",
}

PLANNER_MODES = (PLANNER_RULE, PLANNER_LLM)


def normalize_planner_mode(value: str | None, default: str = PLANNER_RULE) -> str:
    """把外部传入的 planner 标识收敛到合法值，非法值回落默认值。"""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in PLANNER_MODES:
            return normalized
    return default


# ---------------------------------------------------------------------------
# 运行时契约
# ---------------------------------------------------------------------------


@dataclass
class AgentToolResult:
    """单个工具的执行结果。"""

    name: str
    title: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentRun:
    """一次规划 + 工具执行的完整产出。两条 Planner 路线共用此契约。"""

    mode: str
    plan: list[str]
    tool_results: list[AgentToolResult]
    reflection: str
    memory_updates: list[str]
    planner: str = PLANNER_RULE
    fallback_reason: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def planner_label(self) -> str:
        return PLANNER_LABELS.get(self.planner, self.planner)

    def tool_names(self) -> list[str]:
        return [item.name for item in self.tool_results]

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

    def reasoning_events(self) -> list[dict]:
        """结构化思考过程事件（供 SSE reasoning 通道使用）。

        事件形状是对外契约：首个为 status、末个为 reflection，
        tool 事件必须含 title / content。新增字段一律追加，不删改既有键。
        """
        events = [
            {"type": "status", "content": "Agent 正在识别任务意图..."},
            {"type": "status", "content": f"Agent 已选择工作模式：{self.mode}"},
            {"type": "status", "content": "Agent 已生成自动任务计划..."},
            *[{"type": "plan", "content": f"- {step}"} for step in self.plan],
            *[{"type": "tool", "title": item.title, "content": item.content} for item in self.tool_results],
            {"type": "status", "content": "Agent 正在进行反思与自我修正..."},
            {"type": "reflection", "content": self.reflection},
        ]
        for event in events:
            event["planner"] = self.planner
        return events


def build_planner_event(run: AgentRun) -> dict:
    """SSE 首帧：显式声明本轮由哪条路线规划，规则路线不得伪装成模型推理。"""
    return {
        "type": "planner",
        "planner": run.planner,
        "label": run.planner_label,
        "note": run.fallback_reason,
    }


_METRIC_KEYS = (
    "plan_latency_ms",
    "planner_llm_calls",
    "llm_rounds",
    "tool_rounds",
    "history_turns",
    "invalid_tool_calls",
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


def build_metrics_event(run: AgentRun) -> dict:
    """SSE 尾帧：暴露规划阶段的真实开销，供前端与评测共用。"""
    metrics = {key: run.metrics.get(key, 0) for key in _METRIC_KEYS}
    metrics["type"] = "metrics"
    metrics["tool_count"] = len(run.tool_results)
    return metrics


# ---------------------------------------------------------------------------
# Token 用量提取
# ---------------------------------------------------------------------------

_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _coerce_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_token_usage(message) -> dict:
    """从模型响应中提取 token 用量。

    兼容两种来源：
    1. ChatTongyi 特有：``response_metadata["token_usage"]``
    2. LangChain 通用：``usage_metadata`` 或 ``response_metadata["usage_metadata"]``

    任何缺失或异常一律返回全 0，绝不抛出，避免拖垮主链路。
    """
    if message is None:
        return dict(_ZERO_USAGE)

    usage = None

    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        usage = usage_metadata

    if usage is None:
        response_metadata = getattr(message, "response_metadata", None)
        if isinstance(response_metadata, dict):
            candidate = response_metadata.get("token_usage")
            if not isinstance(candidate, dict):
                candidate = response_metadata.get("usage_metadata")
            if isinstance(candidate, dict):
                usage = candidate

    if not usage:
        return dict(_ZERO_USAGE)

    input_tokens = _coerce_int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
    output_tokens = _coerce_int(usage.get("output_tokens", usage.get("completion_tokens", 0)))
    total_tokens = _coerce_int(usage.get("total_tokens", 0)) or (input_tokens + output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def merge_token_usage(left: dict | None, right: dict | None) -> dict:
    """累加两份 token 用量。"""
    left = left or _ZERO_USAGE
    right = right or _ZERO_USAGE
    return {
        key: _coerce_int(left.get(key, 0)) + _coerce_int(right.get(key, 0))
        for key in _ZERO_USAGE
    }
