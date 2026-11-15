import unittest

from compare_recipe_holdout import usage


class UsageAccountingTests(unittest.TestCase):
    def test_limit_notice_does_not_count_but_response_without_usage_does(self):
        response = {"type": "ai", "id": "a", "content": "Inspect the bound recipe.",
                    "usage_metadata": {"input_tokens": 20, "output_tokens": 3}}
        unmetered_response = {"type": "ai", "id": "b", "content": "Use the observed revision."}
        notice = {"type": "ai", "id": "c", "content": "Model call limits exceeded: run limit (10/10)"}
        local = {"type": "ai", "id": "d", "content": "{}",
                 "additional_kwargs": {"recipe_report_source": "worker_label_policy"}}
        events = [{"type": "values", "data": {"messages": messages}}
                  for messages in ([response], [response, unmetered_response], [response, unmetered_response, notice, local])]
        self.assertEqual(usage([], events), {"model_calls": 2, "calls_with_token_usage": 1,
                                              "input_tokens": 20, "output_tokens": 3})
