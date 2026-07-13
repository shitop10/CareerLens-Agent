import os
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from app.core.vector_stores import VectorStoreService
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from app.core import config_data as config
from app.core.prompts import rag_prompt_template
from langchain_community.chat_models.tongyi import ChatTongyi
from app.utils.file_history_store import get_history
from app.core.logger import logger

os.environ["DASHSCOPE_API_KEY"] = config.DASHSCOPE_API_KEY


class RagService(object):
    def __init__(self) -> None:
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.EMBEDDINGS_MODEL)
        )
        self.prompt_template = rag_prompt_template
        self.chat_model = ChatTongyi(model="qwen-max")
        self.chain = self.__get_chain()

    def __get_chain(self):
        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_docs = []
            for i, doc in enumerate(docs):
                content = f"资料[{i + 1}]: {doc.page_content}\n来源: {doc.metadata.get('filename')}"
                formatted_docs.append(content)
            return "\n\n".join(formatted_docs)

        def retrieve_context(input_data: dict):
            query = input_data["input"]
            status_messages, docs = self.vector_service.hybrid_search_workflow(query)
            for message in status_messages:
                print(message)
            return format_document(docs)

        chain = (
                {
                    "context": RunnableLambda(retrieve_context),
                    "input": lambda x: x["input"],
                    "history": lambda x: x["history"]
                }
                | self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain


if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "user002"
        }
    }

    service = RagService()
    res = service.chain.invoke({"input": "大模型算法岗 JD 中最核心的能力要求是什么？"}, session_config)

    logger.info("-" * 30)
    logger.info(f"AI 回复: {res}")
