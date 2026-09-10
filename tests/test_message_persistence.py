"""消息落库与历史可见性测试。

核心回归点：`persist_message` 必须在 SSE `done` 之前完成提交，
否则用户收到回答后立刻追问时，下一轮规划器读不到本轮历史（race）。

本测试直接操作真实 SQLite 库，用完即清理，沿用 tests/test_delete_session.py 的模式。
"""

import asyncio
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage
from sqlalchemy import delete, select

from app.api.api_service import (
    _split_output,
    enrich_message,
    get_password_hash,
    persist_message,
)
from app.api.api_service import AsyncSessionLocal
from app.core.config_data import SALT_SUFFIX
from app.models.models import ChatMessage, ChatSession, User


class TestSplitOutput(unittest.TestCase):
    def test_extracts_code_block(self):
        clean, code = _split_output("说明文字\n```python\nprint(1)\n```\n结尾")
        self.assertEqual(code, "print(1)")
        self.assertNotIn("print(1)", clean)
        self.assertIn("说明文字", clean)

    def test_no_code_block(self):
        clean, code = _split_output("纯文本回答")
        self.assertEqual(code, "")
        self.assertEqual(clean, "纯文本回答")

    def test_multiple_blocks_joined(self):
        _, code = _split_output("```\na\n```\n中间\n```\nb\n```")
        self.assertEqual(code, "a\n---\nb")


class _FakeTongyi:
    """替身模型：返回固定内容，避免测试发起真实调用。"""

    def __init__(self, *args, **kwargs):
        pass

    async def ainvoke(self, messages):
        return AIMessage(content="替身摘要")


class TestPersistAndEnrich(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.username = f"tst_persist_{uuid.uuid4().hex[:8]}"
        cls.user_id = asyncio.run(cls._create_user())

    @classmethod
    def tearDownClass(cls):
        asyncio.run(cls._cleanup())

    @classmethod
    async def _create_user(cls) -> int:
        async with AsyncSessionLocal() as db:
            user = User(username=cls.username, hashed_password=get_password_hash("x"))
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user.id

    @classmethod
    async def _cleanup(cls):
        async with AsyncSessionLocal() as db:
            sessions = await db.execute(
                select(ChatSession).where(ChatSession.user_id == cls.user_id)
            )
            for s in sessions.scalars().all():
                await db.execute(delete(ChatMessage).where(ChatMessage.session_id == s.id))
                await db.execute(delete(ChatSession).where(ChatSession.id == s.id))
            await db.execute(delete(User).where(User.id == cls.user_id))
            await db.commit()

    async def _new_session(self) -> int:
        async with AsyncSessionLocal() as db:
            session = ChatSession(
                session_uuid=str(uuid.uuid4()), user_id=self.user_id, title="测试会话"
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
            return session.id

    async def _count_messages(self, session_id: int) -> int:
        async with AsyncSessionLocal() as db:
            res = await db.execute(
                select(ChatMessage).where(ChatMessage.session_id == session_id)
            )
            return len(res.scalars().all())

    def test_message_is_visible_immediately_after_persist(self):
        async def scenario():
            sid = await self._new_session()
            self.assertEqual(await self._count_messages(sid), 0)

            msg_id = await persist_message(sid, "第一个问题", "第一个回答")
            self.assertIsNotNone(msg_id)

            # 关键断言：persist_message 返回后，另一条连接就能读到 —— 无 race
            self.assertEqual(await self._count_messages(sid), 1)

            async with AsyncSessionLocal() as db:
                res = await db.execute(select(ChatMessage).where(ChatMessage.id == msg_id))
                row = res.scalars().first()
                self.assertEqual(row.user_input, "第一个问题")
                self.assertEqual(row.output_uncode, "第一个回答")
                # 摘要尚未生成，先落占位符，保证历史读到的不是空串
                self.assertTrue(row.streamline_input)

        asyncio.run(scenario())

    @patch("langchain_community.chat_models.ChatTongyi", _FakeTongyi)
    def test_enrich_writes_summary_and_title(self):
        async def scenario():
            sid = await self._new_session()
            msg_id = await persist_message(sid, "帮我看看简历", "回答内容")

            await enrich_message(sid, msg_id, "帮我看看简历", "回答内容")

            async with AsyncSessionLocal() as db:
                res = await db.execute(select(ChatMessage).where(ChatMessage.id == msg_id))
                self.assertEqual(res.scalars().first().streamline_input, "替身摘要")

                s_res = await db.execute(select(ChatSession).where(ChatSession.id == sid))
                self.assertEqual(s_res.scalars().first().title, "替身摘要")

        asyncio.run(scenario())

    @patch("langchain_community.chat_models.ChatTongyi", _FakeTongyi)
    def test_enrich_does_not_overwrite_title_on_later_turns(self):
        async def scenario():
            sid = await self._new_session()
            first = await persist_message(sid, "第一问", "第一答")
            await enrich_message(sid, first, "第一问", "第一答")

            second = await persist_message(sid, "第二问", "第二答")
            await enrich_message(sid, second, "第二问", "第二答")

            async with AsyncSessionLocal() as db:
                res = await db.execute(select(ChatSession).where(ChatSession.id == sid))
                self.assertEqual(res.scalars().first().title, "替身摘要")
                rows = await db.execute(
                    select(ChatMessage).where(ChatMessage.session_id == sid).order_by(ChatMessage.id)
                )
                self.assertEqual([r.user_input for r in rows.scalars().all()], ["第一问", "第二问"])

        asyncio.run(scenario())

    def test_enrich_failure_does_not_lose_message(self):
        async def scenario():
            sid = await self._new_session()
            msg_id = await persist_message(sid, "问题", "回答")

            with patch("langchain_community.chat_models.ChatTongyi",
                       side_effect=RuntimeError("model down")):
                await enrich_message(sid, msg_id, "问题", "回答")

            # 模型挂了，消息本身必须还在
            self.assertEqual(await self._count_messages(sid), 1)

        asyncio.run(scenario())


class TestStreamOrdering(unittest.TestCase):
    """静态契约：done 事件必须在落库之后产出。"""

    def test_done_event_comes_after_persist(self):
        import inspect

        from app.api import api_service

        source = inspect.getsource(api_service.chat_stream)
        persist_at = source.index("await persist_message(")
        done_at = source.index('sse_event("done", {})')
        self.assertLess(persist_at, done_at, "落库必须早于 done 事件，否则历史存在 race")

    def test_enrich_is_backgrounded(self):
        import inspect

        from app.api import api_service

        source = inspect.getsource(api_service.chat_stream)
        self.assertIn("asyncio.create_task(enrich_message(", source)


if __name__ == "__main__":
    unittest.main()
