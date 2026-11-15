import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from agent_server import JobStore
from experiment_diagnostics.job_storage import atomic_job_write


class JobRecoveryTests(unittest.TestCase):
    def test_completed_report_survives_restart_without_another_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def runner(data, folder):
                calls.append(data)
                return {"answer": {"observations": ["observed"]}, "validation": {"executed": False}}
            first = JobStore(directory, runner=runner)
            identifier = first.submit({"log": "api_key=do-not-save\nerror"})["id"]
            first.close()
            second = JobStore(directory, runner=runner)
            try:
                restored = second.get(identifier)
                self.assertEqual(restored["state"], "completed")
                self.assertEqual(restored["result"]["answer"]["observations"], ["observed"])
                self.assertEqual(len(calls), 1)
                self.assertNotIn("do-not-save", (Path(directory) / identifier / "job.json").read_text(encoding="utf-8"))
            finally:
                second.close()

    def test_inflight_jobs_are_interrupted_and_never_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            identifiers = []
            for state in ("queued", "running"):
                identifier = ("a" if state == "queued" else "b") * 32
                identifiers.append(identifier)
                atomic_job_write(Path(directory) / identifier,
                    {"id": identifier, "state": state, "created_at": time.time(), "validation_executed": False})
            called = []
            store = JobStore(directory, runner=lambda *args: called.append(args))
            try:
                for identifier in identifiers:
                    self.assertEqual(store.get(identifier)["state"], "interrupted")
                    self.assertIn("未自动重试", store.get(identifier)["error"])
                self.assertFalse(called)
            finally:
                store.close()

    def test_second_writer_is_rejected_and_lease_releases_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            first = JobStore(directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "Another server"):
                    JobStore(directory)
            finally:
                first.close()
            next_store = JobStore(directory)
            next_store.close()

    def test_corrupt_and_mismatched_records_do_not_block_good_results(self):
        with tempfile.TemporaryDirectory() as directory:
            good, bad, mismatched = "1" * 32, "2" * 32, "3" * 32
            atomic_job_write(Path(directory) / good, {"id": good, "state": "failed", "validation_executed": False, "error": "saved failure"})
            bad_file = Path(directory) / bad / "job.json"
            bad_file.parent.mkdir()
            bad_file.write_text("{incomplete", encoding="utf-8")
            atomic_job_write(Path(directory) / mismatched, {"id": "4" * 32, "state": "queued", "validation_executed": False})
            store = JobStore(directory)
            try:
                self.assertEqual(store.get(good)["error"], "saved failure")
                self.assertEqual(len(store.restore_issues), 2)
                self.assertIsNone(store.get(mismatched))
            finally:
                store.close()

    def test_disk_failure_before_running_never_calls_runner_and_releases_queue_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            called = []
            store = JobStore(directory, runner=lambda *args: called.append(args))
            try:
                with patch.object(store, "_save_locked", side_effect=OSError("disk unavailable")):
                    with self.assertRaises(OSError):
                        store.submit({})
                store.submit({})
            finally:
                store.close()
            self.assertEqual(len(called), 1)

    def test_final_write_failure_is_visible_and_does_not_hold_a_queue_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            store = JobStore(directory, runner=lambda *args: {"ok": True}, capacity=1)
            original = store._save_locked
            def save(identifier):
                if store.records[identifier]["state"] == "completed":
                    raise OSError("disk full")
                original(identifier)
            try:
                with patch.object(store, "_save_locked", side_effect=save):
                    identifier = store.submit({})["id"]
                    deadline = time.monotonic() + 3
                    while store.get(identifier)["state"] in {"queued", "running"} and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertIn("persistence_error", store.get(identifier))
                    self.assertIsNotNone(store.submit({}))
            finally:
                store.close()
