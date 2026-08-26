import datetime
import hashlib
import uuid
import re
import json
import asyncio
from typing import Optional, Any

import sys
import io
from contextlib import redirect_stdout
from fastapi import FastAPI, Response, HTTPException, Depends, Cookie, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update, func, desc
from app.core.config_data import ASYNC_DATABASE_URL, SALT_SUFFIX
from app.models.models import Base, User, ChatSession, ChatMessage
from app.core.logger import logger
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app.mount("/html", StaticFiles(directory=os.path.join(project_root, "html")), name="html")

async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
rag_service: Optional[Any] = None
agent_service: Optional[Any] = None
kb_service: Optional[Any] = None


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_password_hash(password: str) -> str:
    return hashlib.sha256((password + SALT_SUFFIX).encode('utf-8')).hexdigest()


def extract_upload_text(filename: str, content: bytes) -> str:
    suffix = os.path.splitext(filename.lower())[1]
    if suffix in {".txt", ".md"}:
        for encoding in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=400, detail="文本文件编码无法识别")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="缺少 pypdf 依赖") from exc

        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(text for text in pages if text.strip())
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"PDF 解析失败：{exc}") from exc

    raise HTTPException(status_code=400, detail="仅支持 txt、md、pdf 文件")


def sse_event(event: str, data: dict) -> str:
    """构造标准 SSE 事件帧：event 行 + data 行（JSON，中文原样输出）+ 空行。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def get_rag_service():
    global rag_service
    if rag_service is None:
        from app.core.rag import RagService
        rag_service = RagService()
    return rag_service


def get_agent_service():
    global agent_service
    if agent_service is None:
        from app.core.agent import CareerAgentService
        agent_service = CareerAgentService(get_rag_service())
    return agent_service


def get_kb_service():
    global kb_service
    if kb_service is None:
        from app.core.knowledge_base import KnowledgeBaseService
        kb_service = KnowledgeBaseService()
    return kb_service


async def save_chat_history(s_id: int, user_in: str, raw_out: str):
    logger.info(f"[Backend] 开始处理会话 {s_id} 的后台存储任务...")
    async with AsyncSessionLocal() as db:
        try:
            codes = re.findall(r"```[a-zA-Z0-9\+\#]*\n(.*?)\n```", raw_out, re.DOTALL)
            code_str = "\n---\n".join(codes) if codes else ""
            clean_text = re.sub(r"```.*?```", "", raw_out, flags=re.DOTALL).strip()

            count_res = await db.execute(select(func.count(ChatMessage.id)).where(ChatMessage.session_id == s_id))
            msg_count = count_res.scalar()

            if msg_count == 0:
                logger.info(f"[Backend] 检测到第一条消息，正在生成标题...")
                try:
                    from langchain_community.chat_models import ChatTongyi
                    from langchain_core.messages import HumanMessage
                    from app.core.prompts import title_generation_prompt

                    chat_model = ChatTongyi(model="qwen-turbo")
                    t_resp = await chat_model.ainvoke([HumanMessage(
                        content=title_generation_prompt.format(user_input=user_in))])
                    new_title = t_resp.content.strip().replace("“", "").replace("”", "").replace("标题：", "")

                    stmt = (
                        update(ChatSession)
                        .where(ChatSession.id == s_id)
                        .values(title=new_title)
                    )
                    await db.execute(stmt)
                    logger.info(f"[Backend] 标题已成功更新为: {new_title}")
                except Exception as e:
                    logger.error(f"[Backend] 标题生成过程出错: {str(e)}")

            summary = ""
            try:
                from langchain_community.chat_models import ChatTongyi
                from langchain_core.messages import HumanMessage
                from app.core.prompts import summary_generation_prompt

                chat_model = ChatTongyi(model="qwen-turbo")
                s_resp = await chat_model.ainvoke([HumanMessage(content=summary_generation_prompt.format(content=clean_text))])
                summary = s_resp.content.strip()
            except Exception as e:
                logger.error(f"[Backend] 生成总结过程出错: {str(e)}")
                summary = clean_text[:50]

            new_msg = ChatMessage(
                session_id=s_id,
                user_input=user_in,
                raw_output=raw_out,
                output_uncode=clean_text,
                code=code_str,
                streamline_input=summary
            )
            db.add(new_msg)

            await db.commit()
            logger.info(f"[Backend] 会话 {s_id} 数据存储完成。")

        except Exception as e:
            await db.rollback()
            logger.error(f"[Backend] 存储任务发生严重错误: {str(e)}")


@app.on_event("startup")
async def start_event():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.post("/auth/register")
async def register(username: str, password: str, db: AsyncSession = Depends(get_db)):
    try:
        res = await db.execute(select(User).where(User.username == username))
        if res.scalars().first():
            raise HTTPException(status_code=400, detail="用户名已被占用")
        db.add(User(username=username, hashed_password=get_password_hash(password)))
        await db.commit()
        return {"status": "success", "message": "注册成功"}
    except HTTPException as he:
        raise he
    except Exception:
        raise HTTPException(status_code=500, detail="服务器错误")


@app.post("/auth/login")
async def login(username: str, password: str, response: Response, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.username == username))
    user = res.scalars().first()
    if not user or get_password_hash(password) != user.hashed_password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    new_cookie = user.last_cookie or str(uuid.uuid4())
    user.last_cookie = new_cookie
    await db.commit()
    response.set_cookie(key="session_id", value=new_cookie, httponly=True, samesite="lax", secure=False)
    return {"status": "success", "message": "登录成功", "data": {"username": user.username}}


@app.post("/sessions")
async def create_session(session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")
    res = await db.execute(select(User).where(User.last_cookie == session_id))
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="无效会话")

    new_uuid = str(uuid.uuid4())
    new_s = ChatSession(session_uuid=new_uuid, user_id=user.id, title="新岗位分析")
    db.add(new_s)
    await db.commit()
    return {"status": "success", "data": {"session_id": new_uuid, "title": "新对话"}}


@app.get("/sessions")
async def get_sessions(session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    if not session_id:
        raise HTTPException(status_code=401)
    res = await db.execute(
        select(ChatSession).join(User).where(User.last_cookie == session_id).order_by(desc(ChatSession.update_time)))
    return {"status": "success", "data": [
        {"session_id": s.session_uuid, "title": s.title, "update_time": s.update_time.strftime("%m-%d %H:%M")} for s in
        res.scalars().all()]}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "project": "CareerLens",
        "retrieval": "Chroma + BM25 + RRF",
        "agent": "Planner + Tool Calling + Reflection + Memory",
    }


@app.get("/agent/memory")
async def get_agent_memory(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="请先登录")
    from app.core.agent import LongTermMemory
    return {
        "status": "success",
        "data": LongTermMemory().load_recent(12),
    }


@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile = File(...), session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="请先登录")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="文件为空")

    text = extract_upload_text(file.filename or "upload.txt", content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="没有解析到可写入的文本内容")

    result = await asyncio.to_thread(get_kb_service().upload_by_str, text, file.filename or "upload")
    return {
        "status": "success",
        "message": result,
        "data": {
            "filename": file.filename,
            "chars": len(text),
        },
    }


@app.get("/chat/{session_uuid}")
async def get_history(session_uuid: str, session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == session_uuid))
    curr = res.scalars().first()
    if not curr:
        raise HTTPException(status_code=404)
    msg_res = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == curr.id).order_by(ChatMessage.create_time))
    return {"status": "success",
            "data": [{"user_input": m.user_input, "raw_output": m.raw_output} for m in msg_res.scalars().all()]}


@app.post("/chat")
async def chat_stream(session_uuid: str = Body(..., embed=True), input_text: str = Body(..., embed=True),
                      session_id: str = Cookie(None), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == session_uuid))
    curr = res.scalars().first()
    if not curr:
        raise HTTPException(status_code=404)

    async def event_generator():
        current_agent = get_agent_service()
        current_rag = get_rag_service()
        agent_run = await asyncio.to_thread(current_agent.prepare, input_text)
        for evt in agent_run.reasoning_events():
            yield sse_event("reasoning", evt)
            await asyncio.sleep(0.08)

        agent_context = current_agent.compose_agent_context(agent_run)
        enhanced_input = f"{input_text}\n\n[Agent上下文]\n{agent_context}"

        stdout_capture = io.StringIO()
        full_out = ""

        yield sse_event("reasoning", {"type": "status", "content": "Agent 正在汇总工具结果并生成最终建议..."})
        await asyncio.sleep(0.1)

        with redirect_stdout(stdout_capture):
            async for chunk in current_rag.chain.astream({"input": enhanced_input},
                                                         config={"configurable": {"session_id": session_uuid}}):
                captured_output = stdout_capture.getvalue()
                if captured_output:
                    yield sse_event("reasoning", {"type": "status", "content": captured_output})
                    stdout_capture.truncate(0)
                    stdout_capture.seek(0)

                content = chunk if isinstance(chunk, str) else getattr(chunk, 'content', "")
                if not content:
                    continue
                full_out += content
                yield sse_event("answer", {"type": "token", "content": content})

        final_captured_output = stdout_capture.getvalue()
        if final_captured_output:
            yield sse_event("reasoning", {"type": "status", "content": final_captured_output})

        memory_updates = await asyncio.to_thread(current_agent.finalize, input_text, full_out, agent_run)
        for item in memory_updates:
            yield sse_event("reasoning", {"type": "memory", "content": item})
        yield sse_event("done", {})

        asyncio.create_task(save_chat_history(curr.id, input_text, full_out))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.delete("/delete/{session_uuid}")
async def delete_s(session_uuid: str, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(ChatSession).where(ChatSession.session_uuid == session_uuid))
    target = res.scalars().first()
    if target:
        await db.delete(target)
        await db.commit()
    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
