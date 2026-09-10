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

# Planner 配置
# rule = 确定性关键词路由（零 token，可复现）；llm = LLM function calling 动态规划
PLANNER_MODE = os.getenv("PLANNER_MODE", "rule")
LLM_PLANNER_MODEL = os.getenv("LLM_PLANNER_MODEL", "qwen-max")
REFLECTOR_MODEL = os.getenv("REFLECTOR_MODEL", "qwen-turbo")
# 规划器 LLM 最多执行几轮工具调用；规划器调用次数上限为 MAX_TOOL_ROUNDS + 1
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "2"))
# 规划器接收的对话历史：最近几轮、每侧截断多少字
PLANNER_HISTORY_TURNS = int(os.getenv("PLANNER_HISTORY_TURNS", "3"))
PLANNER_HISTORY_CHARS = int(os.getenv("PLANNER_HISTORY_CHARS", "80"))
# 评测用裁判模型
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen-turbo")
