import unittest

from langchain_core.documents import Document
from langchain_core.tools import BaseTool

from app.core.agent_types import AgentToolResult
from app.core.tools import TOOL_NAMES, ToolRegistry


class _FakeVectorService:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def hybrid_search_workflow(self, query):
        self.calls.append(query)
        return ["[状态] 检索完成\n"], list(self.docs)


class _FakeRagService:
    def __init__(self, docs):
        self.vector_service = _FakeVectorService(docs)


class _FakeMemory:
    def __init__(self, items):
        self.items = items

    def load_recent(self, limit=6):
        return list(self.items)[-limit:]


def _docs(n=2):
    return [
        Document(page_content=f"证据内容{i}", metadata={"filename": f"file_{i}.txt"})
        for i in range(n)
    ]


class TestToolRegistryBasics(unittest.TestCase):
    def test_names_stable_and_complete(self):
        registry = ToolRegistry(_FakeRagService(_docs()), _FakeMemory([]))
        self.assertEqual(registry.names(), list(TOOL_NAMES))
        self.assertEqual(
            registry.names(),
            ["memory_lookup", "hybrid_retrieval", "resume_diagnosis", "job_match"],
        )

    def test_unknown_tool_raises_key_error(self):
        registry = ToolRegistry(_FakeRagService(_docs()), _FakeMemory([]))
        with self.assertRaises(KeyError):
            registry.invoke("nonexistent_tool", "q")


class TestToolResults(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry(_FakeRagService(_docs(2)), _FakeMemory([]))

    def test_invoke_returns_agent_tool_result(self):
        result = self.registry.invoke("resume_diagnosis", "帮我改简历")
        self.assertIsInstance(result, AgentToolResult)
        self.assertEqual(result.name, "resume_diagnosis")

    def test_hybrid_retrieval_reports_doc_count(self):
        result = self.registry.invoke("hybrid_retrieval", "RAG 能力")
        self.assertEqual(result.metadata["doc_count"], 2)
        self.assertIn("file_0.txt", result.content)
        self.assertEqual(result.name, "hybrid_retrieval")

    def test_hybrid_retrieval_empty_corpus_message(self):
        registry = ToolRegistry(_FakeRagService([]), _FakeMemory([]))
        result = registry.invoke("hybrid_retrieval", "随便问问")
        self.assertEqual(result.metadata["doc_count"], 0)
        self.assertIn("未召回到明确证据", result.content)

    def test_memory_lookup_without_memory(self):
        result = self.registry.invoke("memory_lookup", "q")
        self.assertIn("暂无长期记忆", result.content)

    def test_memory_lookup_with_memory(self):
        registry = ToolRegistry(
            _FakeRagService([]), _FakeMemory([{"content": "偏好量化表达"}])
        )
        result = registry.invoke("memory_lookup", "q")
        self.assertIn("偏好量化表达", result.content)

    def test_memory_items_can_be_injected(self):
        result = self.registry.invoke(
            "memory_lookup", "q", memory_items=[{"content": "注入的记忆"}]
        )
        self.assertIn("注入的记忆", result.content)

    def test_resume_diagnosis_keeps_rule_keywords(self):
        result = self.registry.invoke("resume_diagnosis", "q")
        self.assertIn("量化", result.content)
        self.assertIn("弱动词", result.content)

    def test_job_match_lists_matched_keywords(self):
        result = self.registry.invoke("job_match", "熟悉 RAG 与 Agent 部署")
        self.assertIn("RAG", result.content)
        self.assertIn("匹配矩阵", result.content)

    def test_job_match_defaults_when_no_keyword_hit(self):
        result = self.registry.invoke("job_match", "你好呀")
        self.assertIn("大模型基础", result.content)


class TestLangchainToolView(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry(_FakeRagService(_docs()), _FakeMemory([]))
        self.tools = self.registry.as_langchain_tools()

    def test_returns_base_tools(self):
        self.assertEqual(len(self.tools), 4)
        for item in self.tools:
            self.assertIsInstance(item, BaseTool)

    def test_names_match_registry(self):
        self.assertEqual([t.name for t in self.tools], list(TOOL_NAMES))

    def test_descriptions_are_substantive(self):
        for item in self.tools:
            self.assertGreaterEqual(len(item.description), 20, item.name)

    def test_tool_callable_delegates_to_registry(self):
        hybrid = next(t for t in self.tools if t.name == "hybrid_retrieval")
        text = hybrid.invoke({"query": "RAG 能力"})
        self.assertIn("file_0.txt", text)

    def test_tools_are_bindable(self):
        class _Sink:
            def bind_tools(self, tools):
                self.bound = list(tools)
                return self

        sink = _Sink()
        sink.bind_tools(self.tools)
        self.assertEqual(len(sink.bound), 4)

    def test_bind_memory_items_reaches_langchain_tool(self):
        registry = ToolRegistry(_FakeRagService([]), _FakeMemory([]))
        tools = registry.as_langchain_tools()
        registry.bind_memory_items([{"content": "绑定后的记忆"}])
        lookup = next(t for t in tools if t.name == "memory_lookup")
        self.assertIn("绑定后的记忆", lookup.invoke({"query": "q"}))

    def test_unbound_memory_falls_back_to_store(self):
        registry = ToolRegistry(
            _FakeRagService([]), _FakeMemory([{"content": "来自 memory_store"}])
        )
        tools = registry.as_langchain_tools()
        lookup = next(t for t in tools if t.name == "memory_lookup")
        self.assertIn("来自 memory_store", lookup.invoke({"query": "q"}))


if __name__ == "__main__":
    unittest.main()
