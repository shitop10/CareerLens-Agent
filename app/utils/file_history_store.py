from typing import Sequence
from sqlalchemy import create_engine, select, delete
from sqlalchemy.orm import sessionmaker
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from app.core import config_data as config
from app.models.models import Base, ChatMessage, ChatSession
from app.core.logger import logger

_sync_url = config.ASYNC_DATABASE_URL.replace('sqlite+aiosqlite://', 'sqlite://')
sync_engine = create_engine(_sync_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

Base.metadata.create_all(bind=sync_engine)


class DatabaseChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id):
        self.session_id = session_id

    def _get_session_id(self):
        db = SessionLocal()
        try:
            result = db.execute(
                select(ChatSession.id).where(ChatSession.session_uuid == self.session_id)
            )
            session_id = result.scalars().first()
            return session_id
        finally:
            db.close()

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        pass

    @property
    def messages(self) -> Sequence[BaseMessage]:
        db = SessionLocal()
        try:
            session_id = self._get_session_id()
            if not session_id:
                return []

            result = db.execute(
                select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.create_time)
            )
            chat_messages = result.scalars().all()

            if not chat_messages:
                return []

            message_list = []
            total_messages = len(chat_messages)

            for i, msg in enumerate(chat_messages):
                message_list.append(HumanMessage(content=msg.user_input))
                if i == total_messages - 1:
                    ai_content = msg.output_uncode or msg.raw_output
                else:
                    ai_content = msg.streamline_input or msg.output_uncode or msg.raw_output
                message_list.append(AIMessage(content=ai_content))

            return message_list
        finally:
            db.close()

    def clear(self):
        db = SessionLocal()
        try:
            session_id = self._get_session_id()
            if session_id:
                db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
                db.commit()
        finally:
            db.close()


def get_history(session_id) -> DatabaseChatMessageHistory:
    messages = DatabaseChatMessageHistory(session_id).messages
    logger.debug(f"[History] 会话 {session_id} 的历史消息: {messages}")
    return DatabaseChatMessageHistory(session_id)
