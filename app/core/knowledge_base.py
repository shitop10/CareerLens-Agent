import os
import pickle
import hashlib

import chromadb
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from app.core import config_data as config
from app.core.logger import logger

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


def get_string_md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()


def check_md5(md5_str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding="utf-8").close()
        return False
    with open(config.md5_path, 'r', encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return md5_str in [line.split("|")[-1] for line in lines]


def save_md5(filename, md5_str):
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(f"{filename}|{md5_str}\n")


def remove_md5(filename):
    if not os.path.exists(config.md5_path):
        return
    with open(config.md5_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    remaining = [line for line in lines if not line.startswith(f"{filename}|")]
    with open(config.md5_path, 'w', encoding="utf-8") as f:
        f.writelines(remaining)


class KnowledgeBaseService:
    def __init__(self):
        logger.info("[System] 初始化 KnowledgeBaseService (Chroma)...")
        self.embeddings = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)

        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

        logger.info("[Splitter] 加载语义分割器...")
        self.splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=config.BREAKPOINT_TYPE,
            buffer_size=config.BUFFER_SIZE
        )
        self.bm25_corpus = self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            try:
                with open(config.BM25_CORPUS_PATH, 'rb') as f:
                    return pickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                logger.error(f"[Error] 加载 BM25 语料库失败: {e}")
                return []
        return []

    def _save_bm25_corpus(self):
        with open(config.BM25_CORPUS_PATH, 'wb') as f:
            pickle.dump(self.bm25_corpus, f)

    def upload_by_str(self, data, filename):
        logger.info(f"\n[Process] 开始处理文件: {filename}")
        md5_hex = get_string_md5(data)
        if check_md5(md5_hex):
            return "【跳过】内容已在库中"

        logger.info("[Semantic Split] 正在执行语义分割...")
        knowledge_chunks = self.splitter.split_text(data)

        logger.info("[Storage] 正在生成向量并写入 Chroma...")
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )

            vectors = self.embeddings.embed_documents(knowledge_chunks)

            existing_count = collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(knowledge_chunks))]
            from datetime import datetime
            uploaded_at = datetime.now().isoformat(timespec="seconds")
            metadatas = [
                {"filename": filename, "chunk_index": i, "active": True, "uploaded_at": uploaded_at}
                for i in range(len(knowledge_chunks))
            ]

            collection.add(
                ids=ids,
                embeddings=vectors,
                documents=knowledge_chunks,
                metadatas=metadatas
            )

            logger.info(f"[Storage] 成功存入 {len(knowledge_chunks)} 条数据到 Chroma！")
        except Exception as e:
            logger.error(f"[Error] 写入失败: {e}")
            return f"【失败】{e}"

        self.bm25_corpus.extend(knowledge_chunks)
        self._save_bm25_corpus()
        save_md5(filename, md5_hex)
        return "【成功】内容已载入知识库"

    # ========== 材料管理 ==========

    def _get_collection(self):
        try:
            return self.chroma_client.get_collection(name=config.COLLECTION_NAME)
        except Exception:
            return None

    def ensure_active_metadata(self):
        """惰性迁移：为缺少 active 字段的旧 chunk 补齐 active=True。"""
        collection = self._get_collection()
        if collection is None or collection.count() == 0:
            return
        res = collection.get(include=["metadatas"])
        missing_ids = [i for i, md in zip(res["ids"], res["metadatas"]) if "active" not in md]
        if missing_ids:
            collection.update(
                ids=missing_ids,
                metadatas=[{"active": True}] * len(missing_ids)
            )
            logger.info(f"[Migrate] 已为 {len(missing_ids)} 个旧 chunk 补齐 active=True 元数据")

    def list_files(self):
        """按 filename 聚合知识库文件：chunk 数 / active / 上传时间。"""
        collection = self._get_collection()
        if collection is None or collection.count() == 0:
            return []
        res = collection.get(include=["metadatas"])
        agg = {}
        for md in res["metadatas"]:
            fname = md.get("filename", "未知来源")
            item = agg.setdefault(fname, {
                "filename": fname, "chunks": 0, "active": True, "uploaded_at": ""
            })
            item["chunks"] += 1
            if not md.get("active", True):
                item["active"] = False
            if md.get("uploaded_at") and not item["uploaded_at"]:
                item["uploaded_at"] = md["uploaded_at"]
        return sorted(agg.values(), key=lambda x: x["uploaded_at"], reverse=True)

    def toggle_file(self, filename):
        """切换文件的启用/停用状态（active 取反），并重建 BM25 语料。"""
        collection = self._get_collection()
        if collection is None:
            raise ValueError(f"文件不存在: {filename}")
        res = collection.get(where={"filename": filename}, include=["metadatas"])
        if not res["ids"]:
            raise ValueError(f"文件不存在: {filename}")
        new_active = not bool(res["metadatas"][0].get("active", True))
        new_metadatas = []
        for md in res["metadatas"]:
            m = dict(md)
            m["active"] = new_active
            new_metadatas.append(m)
        collection.update(ids=res["ids"], metadatas=new_metadatas)
        self._rebuild_bm25_corpus()
        return {"filename": filename, "active": new_active, "chunks": len(res["ids"])}

    def delete_file(self, filename):
        """删除文件全部 chunks，清理 MD5 记录并重建 BM25 语料。"""
        collection = self._get_collection()
        if collection is None:
            raise ValueError(f"文件不存在: {filename}")
        res = collection.get(where={"filename": filename}, include=["metadatas"])
        if not res["ids"]:
            raise ValueError(f"文件不存在: {filename}")
        collection.delete(where={"filename": filename})
        remove_md5(filename)
        self._rebuild_bm25_corpus()
        logger.info(f"[Manage] 已删除文件 {filename}（{len(res['ids'])} 个 chunk）")
        return {"filename": filename, "deleted_chunks": len(res["ids"])}

    def _rebuild_bm25_corpus(self):
        """从 Chroma 全量重建 BM25 语料（仅 active 文档），保持与检索一致。"""
        collection = self._get_collection()
        if collection is None or collection.count() == 0:
            self.bm25_corpus = []
            self._save_bm25_corpus()
            return
        res = collection.get(where={"active": True}, include=["documents"])
        self.bm25_corpus = list(res["documents"])
        self._save_bm25_corpus()
        logger.info(f"[Manage] BM25 语料已重建（{len(self.bm25_corpus)} chunks）")


if __name__ == '__main__':
    service = KnowledgeBaseService()
    test_text = "大模型算法岗需要熟悉 RAG、Embedding、向量检索、BM25、RRF 融合和效果评估。"
    logger.info(service.upload_by_str(test_text, "career_rag_test"))
