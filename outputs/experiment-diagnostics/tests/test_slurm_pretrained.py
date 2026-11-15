import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from deerflow_case.slurm_pretrained import SlurmPretrainedExecutor


class SlurmLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        ssh_file = self.root / "ssh"
        ssh_file.write_text("test placeholder")
        profile = {"host": "student@cluster.example", "identity_file": str(ssh_file),
                   "known_hosts": str(ssh_file), "account": "supervisor", "partition": "gpu",
                   "root": "/project/student/diag", "application": "/project/student/diag/source",
                   "python": "/project/student/diag/env/bin/python", "model": "/project/student/diag/model"}
        path = self.root / "profile.json"
        path.write_text(json.dumps(profile))
        self.executor = SlurmPretrainedExecutor(path)
        self.executor.receipts = self.root / "receipts"

    def receipt(self):
        return json.loads(next(self.executor.receipts.glob("*.json")).read_text())

    def test_wait_timeout_cancels_only_submitted_job_and_confirms_release(self):
        with patch.object(self.executor, "remote", return_value="123"), \
             patch.object(self.executor, "cancel_and_confirm", return_value=True) as cancel:
            with self.assertRaises(TimeoutError):
                self.executor.execute("inspect", max_wait=0)
            cancel.assert_called_once_with(123)
        self.assertTrue(self.receipt()["resources_released"])

    def test_unconfirmed_cancellation_is_recorded_as_unreleased(self):
        with patch.object(self.executor, "remote", return_value="123"), \
             patch.object(self.executor, "cancel_and_confirm", return_value=False):
            with self.assertRaises(TimeoutError):
                self.executor.execute("inspect", max_wait=0)
        self.assertFalse(self.receipt()["resources_released"])

    def test_ssh_failure_during_cleanup_preserves_unreleased_receipt(self):
        with patch.object(self.executor, "remote", return_value="123"), \
             patch.object(self.executor, "cancel_and_confirm", side_effect=RuntimeError("Connection lost")):
            with self.assertRaisesRegex(RuntimeError, "Could not confirm"):
                self.executor.execute("inspect", max_wait=0)
        self.assertFalse(self.receipt()["resources_released"])
        self.assertIn("Connection lost", self.receipt()["cleanup_error"])

    def test_lost_submission_response_recovers_by_unique_job_name(self):
        with patch.object(self.executor, "remote", side_effect=["unexpected response", "123"]), \
             patch.object(self.executor, "cancel_and_confirm", return_value=True) as cancel:
            with self.assertRaises(RuntimeError):
                self.executor.execute("inspect")
            cancel.assert_called_once_with(123)
        self.assertEqual(self.receipt()["recovery_job_ids"], [123])

    def test_finished_failed_worker_still_releases_allocation(self):
        with patch.object(self.executor, "remote", side_effect=["123", "123|FAILED|1:0|8|g0001|gres/gpu=1"]), \
             patch.object(self.executor, "active", return_value=False), \
             patch.object(self.executor, "cancel_and_confirm") as cancel:
            with self.assertRaisesRegex(RuntimeError, "trial failed"):
                self.executor.execute("inspect")
            cancel.assert_not_called()
        self.assertTrue(self.receipt()["resources_released"])

    def test_target_injection_rejected_before_any_ssh_or_submission(self):
        with patch.object(self.executor, "remote") as remote:
            with self.assertRaises(ValueError):
                self.executor.execute("validate", ["q_proj; touch /tmp/unwanted"])
            remote.assert_not_called()

    def test_model_api_secret_is_not_inherited_by_the_ssh_process(self):
        with patch.dict(os.environ, {"DIAG_API_KEY": "test-secret-value"}), \
             patch("deerflow_case.slurm_pretrained.subprocess.run",
                   return_value=SimpleNamespace(returncode=0, stdout=b"ok", stderr=b"")) as run:
            self.executor.remote("sbatch", "#!/bin/bash\necho test\n")
        self.assertNotIn("DIAG_API_KEY", run.call_args.kwargs["env"])
        self.assertNotIn("test-secret-value", str(run.call_args.args))
        self.assertEqual(run.call_args.kwargs["input"], b"#!/bin/bash\necho test\n")

    def test_completed_job_returns_real_worker_result_and_scheduler_receipt(self):
        responses = ["123", "123|COMPLETED|0:0|9|g0001|cpu=4,gres/gpu=1", '{"status":"passed"}']
        with patch.object(self.executor, "remote", side_effect=responses), \
             patch.object(self.executor, "active", return_value=False):
            result = self.executor.execute("validate", ["q_proj"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["slurm"]["job_id"], 123)
        self.assertTrue(result["slurm"]["resources_released"])

    def test_oom_pair_uses_one_a100_job_and_checks_all_three_source_files(self):
        self.executor.profile["partition"] = "a100x4"
        responses = ["123", "123|COMPLETED|0:0|9|g0001|cpu=4,gres/gpu=1", '{"passed":true}']
        with patch.object(self.executor, "remote", side_effect=responses) as remote, \
             patch.object(self.executor, "active", return_value=False):
            result = self.executor.execute("oom-pair")
        self.assertTrue(result["slurm"]["resources_released"])
        submit, script = remote.call_args_list[0].args
        self.assertIn("--partition=a100x4", submit)
        self.assertIn("--gres=gpu:1", submit)
        self.assertIn("--worker-profile a100", script)
        self.assertIn("/run_gpu_oom.py", script)
        self.assertIn("/pretrained_worker.py", script)
        self.assertIn("/model_profile.py", script)
