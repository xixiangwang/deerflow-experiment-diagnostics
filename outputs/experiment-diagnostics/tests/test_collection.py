import base64
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import collect_docs


class CollectionTests(unittest.TestCase):
    def test_blob_integrity_failure_never_writes_a_document(self):
        content = b"official-looking content that is not actually the declared Git blob" * 4

        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"encoding": "base64", "content": base64.b64encode(content).decode(), "sha": "0" * 40}

        with tempfile.TemporaryDirectory() as directory, patch.object(collect_docs, "ROOT", Path(directory)), patch.object(collect_docs.requests, "get", return_value=Response()):
            result = collect_docs.fetch(("peft", "huggingface/peft", "v1.0", "docs/example.md", "a" * 40))
            self.assertEqual(result["status"], "failed")
            self.assertIn("digest mismatch", result["error"])
            self.assertFalse(list(Path(directory).rglob("*.md")))

    def test_success_uses_resolved_commit_for_citation_and_verifies_raw_blob(self):
        content = b"# Example\nThis is a versioned official source document.\n" * 4
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()

        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"encoding": "base64", "content": base64.b64encode(content).decode(), "sha": blob}

        with tempfile.TemporaryDirectory() as directory, patch.object(collect_docs, "ROOT", Path(directory)), patch.object(collect_docs.requests, "get", return_value=Response()):
            result = collect_docs.fetch(("peft", "huggingface/peft", "v1.0", "docs/example.md", "a" * 40))
            self.assertEqual(result["status"], "ok")
            self.assertIn("/blob/" + "a" * 40 + "/", result["url"])
            self.assertEqual((Path(directory) / result["file"]).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
