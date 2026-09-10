"""Planner A/B 评测入口。

用法：
    python run_planner_eval.py                    # 全量 20 条，只测规划阶段
    python run_planner_eval.py --limit 5          # 只跑前 5 条
    python run_planner_eval.py --planner llm      # 只跑 LLM 路线（调试用）
    python run_planner_eval.py --with-judge       # 额外调用裁判模型（消耗额度）
    python run_planner_eval.py --out docs/plans/2026-09-10-dual-planner-eval.md

默认只评测「规划阶段」，不生成最终回答，因此 token 开销很小。
"""

import argparse
import os
import sys

from app.core.agent_types import PLANNER_LLM, PLANNER_RULE
from app.evaluation.evaluator import PlannerABEvaluator

_PLANNER_SETS = {
    "both": (PLANNER_RULE, PLANNER_LLM),
    PLANNER_RULE: (PLANNER_RULE,),
    PLANNER_LLM: (PLANNER_LLM,),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="CareerLens Planner A/B 评测")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条用例")
    parser.add_argument("--planner", choices=list(_PLANNER_SETS), default="both",
                        help="只跑指定路线（默认两条都跑）")
    parser.add_argument("--with-judge", action="store_true",
                        help="额外调用裁判模型评分（会消耗额度）")
    parser.add_argument("--out", default="docs/plans/2026-09-10-dual-planner-eval.md",
                        help="报告输出路径")
    args = parser.parse_args()

    evaluator = PlannerABEvaluator()
    results = evaluator.evaluate(
        limit=args.limit,
        with_judge=args.with_judge,
        planners=_PLANNER_SETS[args.planner],
    )
    report = evaluator.report(results)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n[报告已写入] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
