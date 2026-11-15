"""Single-user asynchronous HTTP entry for the official read-only DeerFlow route.

Each request runs in a separate process because the framework config and scenario tool
bindings are process-global. No request can choose a path, command, model or executable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import re
import subprocess
import sys
from threading import BoundedSemaphore, RLock
import time
import uuid

from experiment_diagnostics.materials import MAX_BYTES, normalize, redact
from experiment_diagnostics.job_storage import DirectoryLease, atomic_job_write

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent / "work"


def run_material_job(materials, directory):
    """Only normalized, redacted material reaches disk or the existing CLI worker."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    selected = directory / "materials.json"
    selected.write_text(json.dumps(materials, ensure_ascii=False), encoding="utf-8")
    environment = {key: value for key, value in os.environ.items() if key != "DIAG_ACCESS_TOKEN"}
    process = subprocess.run([sys.executable, str(ROOT / "analyze_materials.py"), str(selected), "--preprocessed"],
        cwd=ROOT, env=environment, capture_output=True, encoding="utf-8", errors="replace", timeout=600)
    lines = process.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("Material worker returned no receipt (exit " + str(process.returncode) + ")")
    receipt = json.loads(lines[-1])
    folder = Path(receipt["folder"]).resolve()
    if not folder.is_relative_to((ROOT / "reports/materials").resolve()):
        raise RuntimeError("Worker returned an unexpected report path")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if process.returncode not in (0, 1) or (process.returncode == 0) != bool(report["contract_passed"]):
        raise RuntimeError("Worker exit code and report disagree")
    if report.get("submitted_code_executed") is not False or report.get("validation", {}).get("executed") is not False:
        raise RuntimeError("Read-only route returned contradictory execution flags")
    # Events remain in the local audit file. The HTTP response is a compact review view.
    return redact({key: report.get(key) for key in (
        "run_id", "deerflow_version", "model", "answer", "raw_answer", "validation",
        "submitted_code_executed", "contract_passed", "strict_contract_passed",
        "output_format", "answer_contract_errors", "citation_audit", "provenance", "error", "trace", "scope",
        "report_source", "normalized_fields")})


class JobStore:
    def __init__(self, directory, runner=run_material_job, capacity=2, record_limit=64):
        self.directory = Path(directory)
        self.lease = DirectoryLease(self.directory)
        self.runner = runner
        self.lock = RLock()
        self.slots = BoundedSemaphore(capacity)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="material-job")
        self.records = {}
        self.record_limit = record_limit
        self.restore_issues = []
        try:
            self._restore()
        except BaseException:
            self.lease.close()
            self.pool.shutdown(wait=False)
            raise

    def _save_locked(self, identifier):
        atomic_job_write(self.directory / identifier, self.records[identifier])

    def _restore(self):
        candidates = []
        for folder in self.directory.iterdir():
            if not re.fullmatch(r"[a-f0-9]{32}", folder.name) or folder.is_symlink() or not folder.is_dir():
                continue
            path = folder / "job.json"
            if path.is_file() and not path.is_symlink():
                candidates.append(path)
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for path in reversed(candidates[:self.record_limit]):
            try:
                if path.stat().st_size > 8000000:
                    raise ValueError("oversized state")
                record = json.loads(path.read_text(encoding="utf-8"))
                identifier = path.parent.name
                if (not isinstance(record, dict) or record.get("id") != identifier or
                    record.get("state") not in {"queued", "running", "completed", "failed", "interrupted"} or
                    record.get("validation_executed") is not False or
                    (record["state"] == "completed" and not isinstance(record.get("result"), dict))):
                    raise ValueError("invalid state")
                record = redact(record)
                if record["state"] in {"queued", "running"}:
                    record.update(state="interrupted", finished_at=time.time(),
                        error="服务在任务完成前中断；执行结果未知，未自动重试，避免重复调用模型。")
                self.records[identifier] = record
                self._save_locked(identifier)
            except (ValueError, OSError, RecursionError):
                self.records.pop(path.parent.name, None)
                self.restore_issues.append({"id": path.parent.name, "error": "unreadable_or_invalid_state"})

    def submit(self, data):
        materials = normalize(data)
        if not self.slots.acquire(blocking=False):
            return None
        identifier = uuid.uuid4().hex
        with self.lock:
            completed = [key for key, value in self.records.items() if value["state"] in {"completed", "failed", "interrupted"}]
            while len(self.records) >= self.record_limit and completed:
                del self.records[completed.pop(0)]
            self.records[identifier] = {"id": identifier, "state": "queued", "created_at": time.time(),
                "validation_executed": False}
        try:
            with self.lock:
                self._save_locked(identifier)
            self.pool.submit(self._run, identifier, materials)
        except Exception:
            with self.lock:
                del self.records[identifier]
            self.slots.release()
            raise
        return self.get(identifier)

    def _run(self, identifier, materials):
        with self.lock:
            self.records[identifier].update(state="running", started_at=time.time())
        outcome = {}
        try:
            with self.lock:
                self._save_locked(identifier)
            result = redact(self.runner(materials, self.directory / identifier))
            outcome = {"state": "completed", "result": result}
        except Exception as exc:
            outcome = {"state": "failed", "error": redact(type(exc).__name__ + ": " + str(exc))}
        finally:
            try:
                with self.lock:
                    self.records[identifier].update(outcome, finished_at=time.time())
                    try:
                        self._save_locked(identifier)
                    except (OSError, ValueError, TypeError, RecursionError):
                        self.records[identifier]["persistence_error"] = "Final state could not be saved; disk recovery is not guaranteed"
            finally:
                self.slots.release()

    def get(self, identifier):
        with self.lock:
            return copy.deepcopy(self.records.get(identifier))

    def close(self):
        try:
            self.pool.shutdown(wait=True)
        finally:
            self.lease.close()


class Handler(BaseHTTPRequestHandler):
    def send_body(self, status, body, content_type="application/json; charset=utf-8"):
        raw = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8") if not isinstance(body, bytes) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authenticated(self):
        expected = ("Bearer " + self.server.access_token).encode()
        return hmac.compare_digest(self.headers.get("Authorization", "").encode(), expected)

    def reject_auth(self):
        # Consume bounded rejected bodies so Windows clients can see the 401 reliably.
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if 0 < length <= MAX_BYTES:
                self.connection.settimeout(1)
                self.rfile.read(length)
        except (ValueError, TimeoutError, OSError):
            pass
        self.send_body(401, {"error": "unauthorized"})

    def do_GET(self):
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/app.js": ("app.js", "application/javascript; charset=utf-8"),
                  "/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path in assets:
            filename, kind = assets[self.path]
            self.send_body(200, (ROOT / "console" / filename).read_bytes(), kind)
        elif self.path == "/health":
            self.send_body(200, {"status": "ok", "mode": "deerflow_read_only_materials",
                "framework": "2.0.0", "api_configured": bool(os.environ.get("DIAG_API_KEY")),
                "execution_enabled": False, "authentication_required": True})
        elif self.path.startswith("/jobs/"):
            if not self.authenticated():
                self.reject_auth()
                return
            result = self.server.jobs.get(self.path[len("/jobs/"):])
            self.send_body(200 if result else 404, result or {"error": "unknown_job"})
        else:
            self.send_body(404, {"error": "unknown_endpoint"})

    def do_POST(self):
        if not self.authenticated():
            self.reject_auth()
            return
        if self.path != "/jobs":
            self.send_body(404, {"error": "unknown_endpoint"})
            return
        try:
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self.send_body(415, {"error": "use_content_length_and_application_json"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BYTES:
                self.send_body(413, {"error": "body_size_must_be_1_to_300000_bytes"})
                return
            self.connection.settimeout(10)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete body")
            payload = json.loads(raw.decode("utf-8"))
            job = self.server.jobs.submit(payload)
            self.send_body(202 if job else 429, job or {"error": "queue_full", "capacity": 2})
        except (ValueError, TypeError, RecursionError, UnicodeError, TimeoutError):
            self.send_body(400, {"error": "invalid_materials"})
        except Exception:
            self.send_body(500, {"error": "internal_error"})

    def log_message(self, format, *args):
        pass


def create_server(host="127.0.0.1", port=8890, token=None, jobs=None):
    if host not in ("127.0.0.1", "localhost") and not token:
        raise ValueError("Set DIAG_ACCESS_TOKEN for non-loopback access")
    selected_jobs = jobs if jobs is not None else JobStore(WORK / "agent-jobs")
    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except BaseException:
        if jobs is None:
            selected_jobs.close()
        raise
    server.daemon_threads = True
    server.access_token = token or secrets.token_urlsafe(32)
    server.jobs = selected_jobs
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8890, type=int)
    args = parser.parse_args()
    missing = [key for key in ("DIAG_API_KEY", "DIAG_BASE_URL", "DIAG_MODEL") if not os.environ.get(key)]
    if missing:
        parser.error("Missing: " + ", ".join(missing))
    server = create_server(args.host, args.port, os.environ.get("DIAG_ACCESS_TOKEN"))
    shown_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print("DeerFlow materials console: http://" + shown_host + ":" + str(server.server_port)
          + "/#access=" + server.access_token, flush=True)
    print("Single-user read-only route. Submitted materials go to the configured model API after common-secret redaction.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.jobs.close()


if __name__ == "__main__":
    main()
