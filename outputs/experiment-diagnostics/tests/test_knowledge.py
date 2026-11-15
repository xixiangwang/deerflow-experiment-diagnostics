import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from experiment_diagnostics.knowledge import build_index, retrieve


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "knowledge").mkdir()
        docs = []
        for version in ("1.0", "2.0"):
            raw = f"CUDA out of memory allocation guide for version {version}. Use memory profiling before choosing batch size.\n".encode()
            relative = f"knowledge/{version}.md"
            (self.root / relative).write_bytes(raw)
            docs.append({"status": "ok", "file": relative, "id": version,
                         "package": "pytorch", "version": version,
                         "url": "https://example.invalid/" + version,
                         "sha256": hashlib.sha256(raw).hexdigest()})
        (self.root / "knowledge/manifest.json").write_text(json.dumps(docs), encoding="utf-8")
        self.index = self.root / "index.sqlite3"
        build_index(self.root, self.index)

    def test_version_mismatch_cannot_leak(self):
        result = retrieve("CUDA memory", versions={"torch": "2.0+cu121"}, index=self.index)
        self.assertEqual([h["version"] for h in result["hits"]], ["2.0"])
        self.assertEqual(result["hits"][0]["applicability"], "exact_version")
        self.assertFalse(retrieve("CUDA memory", versions={"torch": "3.0"}, index=self.index)["hits"])

    def test_exact_line_citation(self):
        for hit in retrieve("CUDA memory", index=self.index)["hits"]:
            self.assertEqual(hit["source_line_start"], 1)
            self.assertIn("#L1", hit["citation"])

    def test_digest_rejects_changed_source(self):
        (self.root / "knowledge/1.0.md").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            build_index(self.root, self.root / "bad.sqlite3")

    def test_no_overlap_and_chitchat(self):
        self.assertFalse(retrieve("量子纠缠学校密码", index=self.index)["hits"])
        self.assertEqual(retrieve("你好", index=self.index)["status"], "skipped_chitchat")
        self.assertTrue(retrieve("你好显存不足", index=self.index)["hits"])

    def test_query_operators_do_not_break_index(self):
        retrieve('" OR * -- CUDA', index=self.index)
        with closing(sqlite3.connect(self.index)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM chunks").fetchone()[0], 2)

    def test_bad_versions_and_limits(self):
        for kwargs in ({"versions": {"torch": 2}}, {"top_k": True}, {"top_k": 100}):
            with self.assertRaises(ValueError):
                retrieve("CUDA", index=self.index, **kwargs)

    def test_long_source_cannot_occupy_the_first_diverse_evidence_slots(self):
        with closing(sqlite3.connect(self.index)) as db:
            row = db.execute("SELECT * FROM chunks WHERE id='1.0:0'").fetchone()
            # IDs are created by the reused chunker; read the actual first v1 chunk.
            if row is None:
                row = db.execute("SELECT * FROM chunks WHERE version='1.0' LIMIT 1").fetchone()
            terms = db.execute("SELECT terms FROM search WHERE id=?", (row[0],)).fetchone()[0]
            for number in range(5):
                identifier = row[0] + ":clone" + str(number)
                db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", (identifier, *row[1:]))
                db.execute("INSERT INTO search VALUES(?,?)", (identifier, terms))
            db.commit()
        plain = retrieve("CUDA", top_k=3, index=self.index)
        diverse = retrieve("CUDA", top_k=3, index=self.index, diversify_sources=True)
        self.assertEqual(len({hit["url"] for hit in plain["hits"]}), 1)
        self.assertEqual(len({hit["url"] for hit in diverse["hits"][:2]}), 2)
        self.assertEqual(len(diverse["hits"]), 3)


if __name__ == "__main__":
    unittest.main()
