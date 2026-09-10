# Planner A/B 评测报告

- 用例数：20
- 评测范围：`prepare()` 全流程（含两条路线共享的检索耗时，不含最终回答生成）
- 规则路线：`rule` · 零 token 确定性路由
- LLM 路线：`llm` · `qwen-max` function calling

## 汇总对比

| 指标 | 规则路线 | LLM 路线 |
|---|---|---|
| 用例数 | 20 | 20 |
| 必需工具召回率 | 0.9333 | 0.9500 |
| 多余调用数(均值) | 1.30 | 0.05 |
| 精确匹配率 | 0.0500 | 0.8000 |
| 规划延迟 P50 (ms) | 460.2 | 8053.7 |
| 规划延迟 P95 (ms) | 607.5 | 11799.0 |
| 规划 token/轮 | 0.0 | 4709.7 |
| 规划 LLM 调用次数 | 0.00 | 2.25 |
| 回退率 | 0.0000 | 0.0000 |

## 按意图分组的必需工具召回率

| 意图 | 规则路线 | LLM 路线 |
|---|---|---|
| 历史语境 | 1.0000 | 0.8333 |
| 复合意图 | 0.6667 | 0.8333 |
| 岗位匹配 | 1.0000 | 1.0000 |
| 简历诊断 | 1.0000 | 1.0000 |
| 边界 | 1.0000 | 1.0000 |

## 逐条明细

| 用例 | 意图 | 期望工具 | 规则路线 | LLM 路线 |
|---|---|---|---|---|
| job_01 | 岗位匹配 | hybrid_retrieval、job_match | memory_lookup、hybrid_retrieval、job_match · 召回 1.00 | hybrid_retrieval、job_match · ✓ |
| job_02 | 岗位匹配 | hybrid_retrieval、job_match | memory_lookup、hybrid_retrieval、job_match · 召回 1.00 | hybrid_retrieval、job_match · ✓ |
| job_03 | 岗位匹配 | hybrid_retrieval、job_match | memory_lookup、hybrid_retrieval、job_match · 召回 1.00 | hybrid_retrieval、hybrid_retrieval、job_match · ✓ |
| job_04 | 岗位匹配 | hybrid_retrieval、job_match | memory_lookup、hybrid_retrieval、job_match · 召回 1.00 | hybrid_retrieval、job_match · ✓ |
| job_05 | 岗位匹配 | hybrid_retrieval、job_match | memory_lookup、hybrid_retrieval、job_match · 召回 1.00 | hybrid_retrieval、job_match · ✓ |
| job_06 | 岗位匹配 | hybrid_retrieval、job_match | memory_lookup、hybrid_retrieval、job_match · 召回 1.00 | hybrid_retrieval、job_match、resume_diagnosis · 召回 1.00 |
| res_01 | 简历诊断 | hybrid_retrieval、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 1.00 | hybrid_retrieval、resume_diagnosis · ✓ |
| res_02 | 简历诊断 | hybrid_retrieval、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 1.00 | hybrid_retrieval、resume_diagnosis · ✓ |
| res_03 | 简历诊断 | hybrid_retrieval、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 1.00 | hybrid_retrieval、resume_diagnosis · ✓ |
| res_04 | 简历诊断 | hybrid_retrieval、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 1.00 | hybrid_retrieval、resume_diagnosis · ✓ |
| res_05 | 简历诊断 | hybrid_retrieval、resume_diagnosis | memory_lookup、hybrid_retrieval、job_match、resume_diagnosis · 召回 1.00 | hybrid_retrieval、resume_diagnosis · ✓ |
| res_06 | 简历诊断 | hybrid_retrieval、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 1.00 | hybrid_retrieval、resume_diagnosis · ✓ |
| mix_01 | 复合意图 | hybrid_retrieval、job_match、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 0.67 | hybrid_retrieval、job_match、resume_diagnosis · ✓ |
| mix_02 | 复合意图 | hybrid_retrieval、job_match、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 0.67 | hybrid_retrieval、job_match · 召回 0.67 |
| mix_03 | 复合意图 | hybrid_retrieval、job_match、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 0.67 | hybrid_retrieval、resume_diagnosis · 召回 0.67 |
| mix_04 | 复合意图 | hybrid_retrieval、job_match、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · 召回 0.67 | hybrid_retrieval、resume_diagnosis、job_match · ✓ |
| mem_01 | 历史语境 | hybrid_retrieval、memory_lookup | memory_lookup、hybrid_retrieval、job_match、resume_diagnosis · 召回 1.00 | memory_lookup、hybrid_retrieval · ✓ |
| mem_02 | 历史语境 | hybrid_retrieval、memory_lookup、resume_diagnosis | memory_lookup、hybrid_retrieval、resume_diagnosis · ✓ | memory_lookup、resume_diagnosis · 召回 0.67 |
| edge_01 | 边界 |  | memory_lookup、hybrid_retrieval、job_match、resume_diagnosis · 召回 1.00 | 无 · ✓ |
| edge_02 | 边界 | hybrid_retrieval | memory_lookup、hybrid_retrieval、job_match、resume_diagnosis · 召回 1.00 | hybrid_retrieval · ✓ |

## 回退记录

- 无

## 结论口径

本表仅反映**规划阶段**的工具选择与开销差异，不代表最终回答质量。
数值由 `run_planner_eval.py` 于真实环境运行产出，用例数 20 条，不具备统计显著性，仅用于验证两条路线差异方向。
