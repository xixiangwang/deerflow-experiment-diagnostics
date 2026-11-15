import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from server import create_server


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(port=0, token="test-only")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def post(self, path, data, token="test-only"):
        req = Request(self.base + path, json.dumps(data).encode(),
                      {"Content-Type": "application/json", "Authorization": "Bearer " + token})
        with urlopen(req, timeout=5) as result:
            return json.load(result)

    def test_health_is_explicitly_offline(self):
        with urlopen(self.base + "/health", timeout=5) as response:
            self.assertFalse(json.load(response)["llm_enabled"])

    def test_actual_diagnosis_over_http(self):
        result = self.post("/diagnose", {"log": "CUDA out of memory"})
        self.assertEqual(result["findings"][0]["code"], "cuda_oom")

    def test_authentication_and_bad_fields(self):
        with self.assertRaises(HTTPError) as caught:
            self.post("/diagnose", {"log": "x"}, token="wrong")
        self.assertEqual(caught.exception.code, 401)
        with self.assertRaises(HTTPError) as caught:
            self.post("/search", {"query": "x", "index": "/private.sqlite3"})
        self.assertEqual(caught.exception.code, 400)

    def test_external_binding_requires_token(self):
        with self.assertRaises(ValueError):
            create_server(host="0.0.0.0", port=0)

    def test_rejected_bodies_return_401_without_socket_reset(self):
        for _ in range(10):
            with self.assertRaises(HTTPError) as caught:
                self.post("/diagnose", {"log": "x" * 4000}, token="wrong")
            self.assertEqual(caught.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
