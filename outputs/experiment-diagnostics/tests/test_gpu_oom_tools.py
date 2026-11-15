import unittest
from unittest.mock import patch

from deerflow_case import gpu_oom_tools as tools


class GpuOomToolsTests(unittest.TestCase):
    def setUp(self):
        tools.TRACE.clear()
        self.addCleanup(tools.TRACE.clear)
        self.oom = {"status": "failed", "device": "cuda", "execution_started": True,
                    "error_type": "OutOfMemoryError", "stage": "training_forward", "validation_steps": 0}

    def test_last_attempt_is_available_for_the_smallest_micro_batch(self):
        with patch.object(tools, "execute", side_effect=[dict(self.oom), dict(self.oom), {"status": "passed"}]) as execute:
            tools.inspect_gpu_oom.invoke({})
            first = tools.verify_gpu_oom_repair.invoke({"micro_batch_size": 4})
            reserved = tools.verify_gpu_oom_repair.invoke({"micro_batch_size": 2})
            final = tools.verify_gpu_oom_repair.invoke({"micro_batch_size": 1})
        self.assertTrue(first["attempted"])
        self.assertEqual(reserved, {"status": "reserve_last_attempt", "execution_started": False,
                                    "remaining_micro_batch_size": 1})
        self.assertTrue(final["attempted"])
        self.assertEqual(execute.call_args_list[-1].args, ("repair", 1))
        self.assertEqual(execute.call_count, 3)

    def test_non_training_oom_cannot_start_a_repair(self):
        with patch.object(tools, "execute", return_value=dict(self.oom, stage="model_loading")) as execute:
            tools.inspect_gpu_oom.invoke({})
            result = tools.verify_gpu_oom_repair.invoke({"micro_batch_size": 1})
        self.assertFalse(result["execution_started"])
        execute.assert_called_once_with("baseline")
