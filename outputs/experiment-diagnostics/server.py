"""Single-user development API for offline checks and retrieval, not an LLM gateway."""
import argparse
import asyncio
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from experiment_diagnostics import inspect_experiment, search_knowledge


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "mode": "offline_rules_with_retrieval",
                                 "llm_enabled": False, "deerflow_host_loaded": False})
        else:
            self.send_json(404, {"error": "unknown_endpoint"})

    def do_POST(self):
        token = self.server.access_token
        if token and not hmac.compare_digest(self.headers.get("Authorization", "").encode(), ("Bearer " + token).encode()):
            # Drain a bounded submitted body before closing. Otherwise Windows can
            # reset the socket and hide the 401 response from the client.
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if 0 < length <= 512000:
                    self.connection.settimeout(1)
                    self.rfile.read(length)
            except (ValueError, TimeoutError, OSError):
                pass
            self.send_json(401, {"error": "unauthorized"})
            return
        if self.path not in ("/diagnose", "/search"):
            self.send_json(404, {"error": "unknown_endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 512000:
                self.send_json(413, {"error": "body_size_must_be_1_to_512000_bytes"})
                return
            self.connection.settimeout(10)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            allowed = ({"log", "experiment_config", "environment"} if self.path == "/diagnose"
                       else {"query", "top_k", "versions", "package"})
            if not isinstance(payload, dict) or set(payload) - allowed:
                raise ValueError("Unexpected input fields")
            if self.path == "/search" and "query" not in payload:
                raise ValueError("query is required")
            if self.path == "/diagnose" and "log" not in payload:
                raise ValueError("log is required")
            handler = inspect_experiment if self.path == "/diagnose" else search_knowledge
            self.send_json(200, asyncio.run(handler(payload, None)))
        except (ValueError, TypeError, TimeoutError, UnicodeError):
            self.send_json(400, {"error": "invalid_request"})
        except Exception:
            self.send_json(500, {"error": "internal_error"})

    def log_message(self, format, *args):
        # Avoid logging request bodies or Authorization headers.
        pass


def create_server(host="127.0.0.1", port=8877, token=None):
    if host not in ("127.0.0.1", "localhost") and not token:
        raise ValueError("Set DIAG_ACCESS_TOKEN before binding a non-loopback address")
    server = ThreadingHTTPServer((host, port), Handler)
    server.access_token = token
    server.daemon_threads = True
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8877, type=int)
    args = parser.parse_args()
    server = create_server(args.host, args.port, os.environ.get("DIAG_ACCESS_TOKEN"))
    print(f"Offline diagnostic API: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
