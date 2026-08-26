import unittest
import json
from app.api.api_service import sse_event


class TestSseEventFormat(unittest.TestCase):
    def test_frame_format(self):
        frame = sse_event("answer", {"type": "token", "content": "你好"})
        self.assertTrue(frame.startswith("event: answer\n"))
        self.assertTrue(frame.endswith("\n\n"))
        self.assertIn('"content": "你好"', frame)

    def test_json_data(self):
        frame = sse_event("reasoning", {"type": "status", "content": "x"})
        data_line = [l for l in frame.splitlines() if l.startswith("data: ")][0]
        payload = json.loads(data_line[6:])
        self.assertEqual(payload["type"], "status")
        self.assertEqual(payload["content"], "x")

    def test_unicode_safe(self):
        frame = sse_event("reasoning", {"type": "reflection", "content": "知识库证据不足，请补充材料"})
        self.assertNotIn("\\u", frame)  # ensure_ascii=False，中文原样输出


if __name__ == "__main__":
    unittest.main()
