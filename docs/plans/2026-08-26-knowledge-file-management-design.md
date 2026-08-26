# 问题3：材料管理（删除 / 切换）— 设计文档

> 日期：2026-08-26 ｜ 流程：superpowers（design → plan → build → verify）
> 状态：待实施

## 1. 问题描述

上传的材料（如简历）没有任何管理能力：无法查看已上传了哪些材料、无法删除材料、无法切换生效材料。前端"本轮材料"列表是**内存假数据**（刷新即消失），知识库是全局堆叠的（新简历上传后旧简历仍参与检索，无法"切换"）。

## 2. 方案

### 2.1 核心机制：`active` 标记（实现"切换材料"）
- 上传时 Chroma metadata 增加：`active: True`（默认启用）、`uploaded_at`（ISO 时间）
- **检索只召回 `active=True` 的文档**（`vector_stores.search_chroma` 加 `where={"active": True}`）
- 材料可单独"停用/启用"（toggle）→ 停用后不参与检索 = 切换当前生效材料
- **旧数据迁移**：现有 chunks 无 `active` 字段，若直接过滤会被全部排除。提供 `ensure_active_metadata()` 惰性迁移（startup 时执行，为缺字段的旧 chunks 补 `active=True`）

### 2.2 BM25 语料与 Chroma 保持同步（删除一致性）
- 现状：`career_bm25_corpus.pkl` 是纯文本列表，无法按文件删除
- 方案：**重建制**——删除/切换后从 Chroma 读取 active 文档全量重建 pkl（chunk 量小，代价可忽略），pkl 结构保持 `list[str]` 不变，兼容旧数据

### 2.3 MD5 去重记录关联文件名
- 现状：`career_md5.text` 只存 md5 行，删除文件后重新上传会被"已在库中"跳过
- 方案：新行格式 `filename|md5`；`check_md5` 兼容旧纯 md5 行（取 `|` 后段比较）；删除文件时移除该 `filename|` 行（旧格式行无法关联，保留——边界可接受）

### 2.4 新增 API（均带 Cookie 鉴权）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/knowledge/files` | 文件列表（按 filename 聚合：chunk 数 / active / 上传时间） |
| POST | `/knowledge/files/toggle` | body `{filename}`，切换 active 并重建 BM25 |
| DELETE | `/knowledge/files?filename=xxx` | 删除该文件全部 chunks + 清理 MD5 + 重建 BM25 |

### 2.5 前端
- "本轮材料"侧边栏改为**真实知识库材料列表**（`GET /knowledge/files`，登录后加载、上传后刷新——不再用内存假数据）
- 每个材料项：文件名 / chunk 数 / 上传时间 / 状态徽标（启用·停用）+ 操作：**切换开关**、**删除按钮**（confirm 确认）
- 空状态提示"暂无材料，去上传页上传"

## 3. 组件变更

| 文件 | 变更 |
|------|------|
| `app/core/knowledge_base.py` | metadata 增强；`list_files` / `toggle_file` / `delete_file` / `_rebuild_bm25_corpus` / `ensure_active_metadata`；md5 关联 filename |
| `app/core/vector_stores.py` | `search_chroma` 增加 `where={"active": True}` |
| `app/api/api_service.py` | 3 个新端点；startup 迁移调用；上传接口 metadata 变更 |
| `html/index.html` | 材料列表真实化 + 切换/删除交互 |
| `tests/test_knowledge_files.py` | md5 工具函数 + toggle/delete 的 md5 与语料联动 |

## 4. 验证策略
- 单元：md5 读写/删除、`toggle_file`/`delete_file` 的联动（mock embedding 避免网络依赖）
- 端到端：上传 → 列表 → 停用 → 检索排除 → 删除 → 列表消失 → 重新上传同文件不再跳过
- 浏览器：材料列表真实显示、开关切换、删除确认
