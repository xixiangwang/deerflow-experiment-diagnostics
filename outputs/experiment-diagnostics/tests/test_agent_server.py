import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from agent_server import JobStore, create_server, run_material_job


class AgentServerTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.received = []
        self.release = threading.Event()
        self.entered = threading.Event()

        def runner(data, directory):
            self.received.append((data, directory))
            self.entered.set()
            self.release.wait(timeout=5)
            return {"answer": {"status": "insufficient_evidence"}, "validation": {"executed": False}}

        self.jobs = JobStore(self.folder.name, runner=runner)
        self.server = create_server(port=0, token="test-only", jobs=self.jobs)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.jobs.close()
        self.thread.join(timeout=5)
        self.folder.cleanup()

    def request(self, path, data=None, token="test-only"):
        headers = {"Authorization": "Bearer " + token}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = Request(self.base + path, json.dumps(data).encode() if data is not None else None, headers)
        with urlopen(req, timeout=5) as response:
            return response.status, json.load(response)

    def test_real_http_submission_queue_and_redaction(self):
        status, first = self.request("/jobs", {"log": "api_key=abc123\nValueError: test", "code": "raise SystemExit"})
        self.assertEqual(status, 202)
        self.assertTrue(self.entered.wait(timeout=3))
        self.assertIn("[REDACTED]", self.received[0][0]["log"])
        self.assertEqual(self.received[0][0]["code"], "raise SystemExit")
        _, second = self.request("/jobs", {"question": "second"})
        self.assertEqual(second["state"], "queued")
        with self.assertRaises(HTTPError) as caught:
            self.request("/jobs", {"question": "third"})
        self.assertEqual(caught.exception.code, 429)
        self.release.set()
        for _ in range(100):
            _, record = self.request("/jobs/" + first["id"])
            if record["state"] == "completed":
                break
            time.sleep(.01)
        self.assertEqual(record["state"], "completed")
        self.assertFalse(record["validation_executed"])
        self.assertEqual(len(record["id"]), 32)

    def test_authentication_protects_submission_and_results(self):
        for path, data in (("/jobs", {"log": "x" * 4000}), ("/jobs/unknown", None)):
            with self.assertRaises(HTTPError) as caught:
                self.request(path, data, token="wrong")
            self.assertEqual(caught.exception.code, 401)
        self.assertFalse(self.received)

    def test_paths_and_model_selection_cannot_be_supplied(self):
        for field in ("path", "input", "model", "command", "api_key"):
            with self.assertRaises(HTTPError) as caught:
                self.request("/jobs", {field: "arbitrary"})
            self.assertEqual(caught.exception.code, 400)
        self.assertFalse(self.received)

    def test_health_and_static_assets(self):
        _, health = self.request("/health")
        self.assertEqual(health["mode"], "deerflow_read_only_materials")
        self.assertFalse(health["execution_enabled"])
        for path in ("/", "/app.js", "/style.css"):
            with urlopen(self.base + path, timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    def test_failed_worker_releases_capacity_and_redacts_errors(self):
        def fail(*args):
            raise RuntimeError("Authorization: Bearer deliberately-fake-secret")
        self.jobs.runner = fail
        _, job = self.request("/jobs", {})
        for _ in range(100):
            record = self.jobs.get(job["id"])
            if record["state"] == "failed":
                break
            time.sleep(.01)
        self.assertEqual(record["state"], "failed")
        self.assertNotIn("deliberately-fake-secret", record["error"])
        self.request("/jobs", {})

    def test_external_binding_requires_explicit_access_token(self):
        with self.assertRaises(ValueError):
            create_server(host="0.0.0.0", port=0)

    def test_worker_receipt_cannot_escape_report_root(self):
        class FakeProcess:
            returncode = 0
            stdout = json.dumps({"folder": str(Path(self.folder.name) / "other")})
        with patch("agent_server.subprocess.run", return_value=FakeProcess()) as execute:
            with self.assertRaisesRegex(RuntimeError, "unexpected report path"):
                run_material_job({"log": "redacted data"}, Path(self.folder.name) / "worker")
            arguments, options = execute.call_args
            self.assertIn("--preprocessed", arguments[0])
            self.assertNotIn("DIAG_ACCESS_TOKEN", options["env"])
            self.assertFalse(options.get("shell", False))


if __name__ == "__main__":
    unittest.main()
