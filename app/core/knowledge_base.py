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
        return md5_str in [line.strip() for line in f.readlines()]


def save_md5(md5):
    with open(config.md5_path, 'a', encoding="utf-8") as f:
        f.write(md5 + '\n')


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
            metadatas = [
                {"filename": filename, "chunk_index": i}
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
        save_md5(md5_hex)
        return "【成功】内容已载入知识库"


if __name__ == '__main__':
    service = KnowledgeBaseService()
    test_text = "大模型算法岗需要熟悉 RAG、Embedding、向量检索、BM25、RRF 融合和效果评估。"
    logger.info(service.upload_by_str(test_text, "career_rag_test"))
