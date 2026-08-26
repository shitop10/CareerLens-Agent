import unittest
import uuid
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy import select
from app.api.api_service import app, AsyncSessionLocal
from app.models.models import User


def _uniq(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class TestDeleteSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.user_a = _uniq("tst_a")
        cls.user_b = _uniq("tst_b")
        with TestClient(app) as client:
            r = client.post("/auth/register", params={"username": cls.user_a, "password": "pass123"})
            assert r.status_code == 200, r.text
            r = client.post("/auth/register", params={"username": cls.user_b, "password": "pass123"})
            assert r.status_code == 200, r.text

    @classmethod
    def tearDownClass(cls):
        async def cleanup():
            async with AsyncSessionLocal() as db:
                res = await db.execute(select(User).where(User.username.in_([cls.user_a, cls.user_b])))
                for u in res.scalars().all():
                    await db.delete(u)
                await db.commit()
        asyncio.run(cleanup())

    def setUp(self):
        self.client_a = TestClient(app)
        r = self.client_a.post("/auth/login", params={"username": self.user_a, "password": "pass123"})
        assert r.status_code == 200
        r = self.client_a.post("/sessions")
        self.sid = r.json()["data"]["session_id"]

    def tearDown(self):
        self.client_a.delete(f"/delete/{self.sid}")

    def test_delete_requires_login(self):
        anon = TestClient(app)
        r = anon.delete(f"/delete/{self.sid}")
        self.assertEqual(r.status_code, 401)

    def test_delete_not_found(self):
        r = self.client_a.delete("/delete/no-such-uuid")
        self.assertEqual(r.status_code, 404)

    def test_delete_forbidden(self):
        client_b = TestClient(app)
        client_b.post("/auth/login", params={"username": self.user_b, "password": "pass123"})
        r = client_b.delete(f"/delete/{self.sid}")
        self.assertEqual(r.status_code, 403)

    def test_delete_own_session(self):
        r = self.client_a.delete(f"/delete/{self.sid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "success")


if __name__ == "__main__":
    unittest.main()
