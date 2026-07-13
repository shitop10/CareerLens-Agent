import os
import pickle
import re

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from typing import List

from app.core import config_data as config
from app.core.logger import logger

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


class VectorStoreService:
    def __init__(self, embedding):
        logger.info("[Retriever] 初始化检索服务 (Chroma + BM25 + RRF)...")
        self.embedding = embedding

        import chromadb
        self.chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    def _get_chroma_collection(self):
        try:
            return self.chroma_client.get_collection(name=config.COLLECTION_NAME)
        except Exception:
            return None

    def search_chroma(self, query, k=3):
        collection = self._get_chroma_collection()
        if collection is None or collection.count() == 0:
            logger.info("[Chroma] 集合为空或不存在，返回空结果")
            return []

        query_vector = self.embedding.embed_query(query)

        try:
            res = collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            docs = []
            if res["ids"] and res["ids"][0]:
                for i, doc_id in enumerate(res["ids"][0]):
                    doc = Document(
                        page_content=res["documents"][0][i],
                        metadata={
                            "filename": res["metadatas"][0][i].get("filename", "未知来源"),
                            "score": 1 - res["distances"][0][i] if res["distances"] else 0
                        }
                    )
                    docs.append(doc)

            return docs
        except Exception as e:
            logger.error(f"[Error] 搜索 Chroma 失败: {e}")
            return []

    def get_retriever(self):
        class ChromaRetriever(BaseRetriever):
            vector_service: 'VectorStoreService'
            k: int

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self.vector_service.search_chroma(query, k=self.k)

        dense_retriever = ChromaRetriever(vector_service=self, k=config.SIMILARITY_THRESHOLD)
        sparse_retriever = self._get_bm25_retriever()

        if sparse_retriever:
            logger.info("[Hybrid] 启动 RRF 倒数秩融合策略...")

            class RRFRetriever(BaseRetriever):
                retrievers: list
                k: int = 60

                def _get_relevant_documents(self, query: str) -> List[Document]:
                    all_results = []
                    for i, retriever in enumerate(self.retrievers):
                        results = retriever.invoke(query)
                        all_results.extend([(doc, i, rank) for rank, doc in enumerate(results, 1)])

                    doc_scores = {}
                    for doc, retriever_idx, rank in all_results:
                        doc_id = str(hash(doc.page_content))
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = {"doc": doc, "score": 0}
                        doc_scores[doc_id]["score"] += 1 / (rank + self.k)

                    sorted_items = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)

                    if sorted_items:
                        max_score = sorted_items[0]["score"]
                        relevant_docs = []
                        for item in sorted_items:
                            normalized_score = (item["score"] / max_score) * 100
                            if normalized_score >= 97:
                                relevant_docs.append(item["doc"])
                                if len(relevant_docs) >= 3:
                                    break
                        return relevant_docs
                    return []

            return RRFRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                k=60
            )

        return dense_retriever

    def _get_bm25_retriever(self):
        if os.path.exists(config.BM25_CORPUS_PATH):
            with open(config.BM25_CORPUS_PATH, 'rb') as f:
                corpus = pickle.load(f)
            if corpus:
                r = BM25Retriever.from_texts(corpus)
                r.k = config.SIMILARITY_THRESHOLD
                return r
        return None

    def hybrid_search_workflow(self, query):
        status_messages = [
            "[状态] 正在准备求职知识库检索器...\n",
            "[状态] 正在加载岗位语义向量检索器 (Chroma)...\n",
            "[状态] 正在加载关键词检索器 BM25...\n",
            "[状态] 正在启动 RRF 融合排序策略...\n",
            "[状态] 正在执行岗位资料混合搜索...\n",
        ]

        retriever = self.get_retriever()
        results = retriever.invoke(query)

        status_messages.append("[状态] 检索完成，正在整理证据...\n")
        status_messages.append("[状态] 正在生成求职建议...\n")

        return status_messages, results


if __name__ == "__main__":
    embeddings = DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
    service = VectorStoreService(embeddings)
    retriever = service.get_retriever()

    query = "大模型算法岗需要哪些 RAG 能力？"
    logger.info(f"\n[Search] 执行查询: {query}")

    try:
        results = retriever.invoke(query)
        for i, doc in enumerate(results):
            logger.info(f"结果 {i + 1}: {doc.page_content} (来源: {doc.metadata.get('filename')})")
    except Exception as e:
        logger.error(f"检索出错: {e}")
