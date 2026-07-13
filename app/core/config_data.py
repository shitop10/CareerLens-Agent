import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"))

# 基础配置
md5_path = "./database/career_md5.text"
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDINGS_MODEL = "text-embedding-v4"

# Chroma 配置 (替代 Milvus Lite)
CHROMA_PERSIST_DIR = "./database/chroma_db"
COLLECTION_NAME = "career_rag_collection"

# 语义分割配置
BREAKPOINT_TYPE = "percentile"
BUFFER_SIZE = 1

# 混合检索与召回配置
BM25_CORPUS_PATH = "./database/career_bm25_corpus.pkl"
AGENT_MEMORY_PATH = "./database/career_agent_memory.jsonl"
SIMILARITY_THRESHOLD = 3
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3

# 文本限制
MAX_SPLIT_CHAR_NUMBER = 1000

# API 服务
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./database/career_chat.db"
SALT_SUFFIX = "CAREER_RAG"
