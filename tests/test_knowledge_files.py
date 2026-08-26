import unittest
import tempfile
import os
import shutil

from app.core import config_data as config
from app.core import knowledge_base as kb


class _FakeEmbeddings:
    """不调用网络：embed_documents / embed_query 返回固定维度向量。"""

    def embed_documents(self, texts):
        return [[0.01] * 8 for _ in texts]

    def embed_query(self, text):
        return [0.01] * 8


class _FakeSplitter:
    def split_text(self, text):
        if len(text) > 100:
            return [text[:100], text[100:]]
        return [text] if text else []


def _tmpdir():
    return tempfile.mkdtemp(prefix="cl_kb_test_")


class TestMd5File(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp()
        os.close(fd)
        self._orig = config.md5_path
        config.md5_path = self.path

    def tearDown(self):
        config.md5_path = self._orig
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_save_and_check(self):
        kb.save_md5("a.txt", "md5a")
        self.assertTrue(kb.check_md5("md5a"))
        self.assertFalse(kb.check_md5("md5zzz"))

    def test_check_compat_old_format(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("oldmd5\n")
        self.assertTrue(kb.check_md5("oldmd5"))

    def test_remove_md5_by_filename(self):
        kb.save_md5("a.txt", "md5a")
        kb.save_md5("b.txt", "md5b")
        kb.remove_md5("a.txt")
        self.assertFalse(kb.check_md5("md5a"))
        self.assertTrue(kb.check_md5("md5b"))

    def test_remove_missing_ok(self):
        kb.remove_md5("not-exist.txt")


class TestKnowledgeFileManagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = _tmpdir()
        cls._orig = {
            "persist": config.CHROMA_PERSIST_DIR,
            "bm25": config.BM25_CORPUS_PATH,
            "collection": config.COLLECTION_NAME,
            "md5": config.md5_path,
        }
        config.CHROMA_PERSIST_DIR = os.path.join(cls.tmp, "chroma")
        config.BM25_CORPUS_PATH = os.path.join(cls.tmp, "bm25.pkl")
        config.COLLECTION_NAME = "cl_test_collection"
        config.md5_path = os.path.join(cls.tmp, "md5.text")

        cls.kb = kb.KnowledgeBaseService()
        cls.kb.embeddings = _FakeEmbeddings()
        cls.kb.splitter = _FakeSplitter()

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._orig.items():
            setattr(config, k, v)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _upload(self, text, filename):
        return self.kb.upload_by_str(text, filename)

    def test_upload_list_toggle_delete(self):
        text = ("大模型算法岗需要熟悉 RAG、Embedding、向量检索、BM25、RRF 融合和效果评估。" * 6)
        self.assertIn("成功", self._upload(text, "file_a.txt"))

        files = self.kb.list_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["filename"], "file_a.txt")
        self.assertTrue(files[0]["active"])
        self.assertGreater(files[0]["chunks"], 0)

        # 同内容重复上传 → 跳过（md5 去重）
        self.assertIn("跳过", self._upload(text, "file_a.txt"))

        # 停用（切换材料）
        toggled = self.kb.toggle_file("file_a.txt")
        self.assertFalse(toggled["active"])
        self.assertFalse(self.kb.list_files()[0]["active"])

        # 删除 → 列表空；md5 已清理，重新上传不再跳过
        self.kb.delete_file("file_a.txt")
        self.assertEqual(len(self.kb.list_files()), 0)
        self.assertIn("成功", self._upload(text, "file_a.txt"))

    def test_upload_second_file_keeps_first(self):
        self._upload("简历素材内容：" + "项目经历 指标 量化" * 20, "resume_b.txt")
        files = self.kb.list_files()
        names = {f["filename"] for f in files}
        self.assertIn("resume_b.txt", names)


if __name__ == "__main__":
    unittest.main()
