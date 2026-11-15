"""Operator-configured SSH/Slurm transport for the fixed pretrained worker only."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import time
import uuid

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


class SlurmPretrainedExecutor:
    def __init__(self, profile_file):
        self.profile = json.loads(Path(profile_file).read_text(encoding="utf-8"))
        p = self.profile
        if not re.fullmatch(r"[A-Za-z0-9_]+@[A-Za-z0-9.-]+", p["host"]):
            raise ValueError("Expected user@host")
        if p["partition"] not in {"gpu", "a100x4"} or not re.fullmatch(r"[A-Za-z0-9_]+", p["account"]):
            raise ValueError("Unsupported Slurm account or partition")
        for field in ("root", "application", "python", "model"):
            path = PurePosixPath(p[field])
            if not path.is_absolute() or ".." in path.parts or "\n" in p[field]:
                raise ValueError("Expected absolute remote project paths")
            if not path.is_relative_to(PurePosixPath(p["root"])):
                raise ValueError("Remote paths must stay in the project")
        for field in ("identity_file", "known_hosts"):
            if not Path(p[field]).is_file():
                raise ValueError("Missing operator SSH file: " + field)
        self.ssh = ["ssh", "-i", p["identity_file"], "-o", "UserKnownHostsFile=" + p["known_hosts"],
                    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", "ConnectTimeout=8", "-o", "ConnectionAttempts=1", p["host"]]
        self.receipts = Path(os.environ.get("DIAG_SLURM_RECEIPTS", Path(__file__).resolve().parents[1] / "reports/slurm-jobs"))

    def remote(self, command, content=None):
        result = subprocess.run(self.ssh + [command], input=content.encode("utf-8") if content is not None else None,
                                capture_output=True, timeout=25,
                                env={key:value for key,value in os.environ.items()
                                     if not key.startswith(("DIAG_", "DEER_FLOW_"))})
        if result.returncode:
            raise RuntimeError("SSH operation failed: " + result.stderr.decode("utf-8", errors="replace")[-800:])
        return result.stdout.decode("utf-8", errors="replace").strip()

    def save(self, receipt):
        self.receipts.mkdir(parents=True, exist_ok=True)
        path = self.receipts / (receipt["request_id"] + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def active(self, job):
        return bool(self.remote("squeue -h -j " + str(job) + " -o %i"))

    def cancel_and_confirm(self, job):
        self.remote("scancel " + str(job))
        for _ in range(10):
            if not self.active(job):
                return True
            time.sleep(2)
        return False

    def execute(self, mode, targets=None, max_wait=300):
        if mode not in {"inspect", "validate", "oom-pair"}:
            raise ValueError("Unsupported GPU trial")
        if mode == "oom-pair" and (self.profile["partition"] != "a100x4" or targets is not None):
            raise ValueError("The OOM pair requires the A100 partition and fixed target modules")
        if targets is not None and (not isinstance(targets, list) or not 1 <= len(targets) <= 8
                                   or any(not isinstance(t, str) or t not in TARGETS for t in targets)):
            raise ValueError("Unsupported target modules")
        p, q = self.profile, shlex.quote
        request = "diag-" + uuid.uuid4().hex[:16]
        directory = p["root"] + "/state/trials/" + request
        application = PurePosixPath(p["application"]) / "outputs/experiment-diagnostics"
        worker = application / "deerflow_case/pretrained_worker.py"
        args = [p["python"], "-I", "-X", "utf8"]
        sources = ["deerflow_case/pretrained_worker.py", "deerflow_case/model_profile.py"]
        if mode == "oom-pair":
            args.extend([str(application / "run_gpu_oom.py"), "--worker-profile", "a100", "--worker-model", p["model"]])
            sources.append("run_gpu_oom.py")
        else:
            args.extend([str(worker), "--mode", mode, "--model", p["model"]])
        if targets is not None:
            args.extend(["--target-modules", json.dumps(targets)])
        hashes = []
        for name in sources:
            digest = hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            hashes.append(digest + "  " + str(application / name))
        script = "\n".join(["#!/bin/bash", "set -euo pipefail", "test -n \"${SLURM_JOB_ID:-}\"",
            "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
            "unset DIAG_API_KEY", "printf '%s\\n' " + " ".join(map(q, hashes)) + " | sha256sum -c -",
            " ".join(map(q, args)) + " > " + q(directory + "/result.json"), ""])
        receipt = {"request_id": request, "mode": mode, "account": p["account"], "partition": p["partition"],
                   "gpu_requested": 1, "time_limit_seconds": 180, "job_id": None, "resources_released": False,
                   "remote_directory": directory, "worker_sha256": hashes[0].split()[0]}
        self.save(receipt)
        try:
            submit = ["sbatch", "--parsable", "--job-name=" + request, "--account=" + p["account"],
                      "--partition=" + p["partition"], "--nodes=1", "--ntasks=1", "--cpus-per-task=4",
                      "--gres=gpu:1", "--mem=16G", "--time=00:03:00", "--output=" + directory + "/job.log",
                      "--chdir=" + p["root"]]
            output = self.remote("mkdir -p " + q(directory) + " && " + " ".join(map(q, submit)), script)
            if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_-]+)?", output):
                raise RuntimeError("Ambiguous Slurm submission response")
            receipt["job_id"] = int(output.split(";")[0])
            self.save(receipt)
            deadline = time.monotonic() + max_wait
            while time.monotonic() < deadline:
                if not self.active(receipt["job_id"]):
                    row = self.remote("sacct -X -n -P -j " + str(receipt["job_id"]) +
                                      " --format=JobID,State,ExitCode,ElapsedRaw,NodeList,AllocTRES").splitlines()
                    record = next((r.split("|") for r in row if r.split("|")[0] == str(receipt["job_id"])), [])
                    if len(record) >= 6 and record[1].split()[0].rstrip("+") in TERMINAL:
                        receipt.update(state=record[1], exit_code=record[2], elapsed_seconds=int(record[3]),
                                       node=record[4], allocated_tres=record[5], resources_released=True)
                        self.save(receipt)
                        if record[1] != "COMPLETED" or record[2] != "0:0":
                            raise RuntimeError("Slurm trial failed: " + record[1] + "; see " + directory + "/job.log")
                        result = json.loads(self.remote("cat " + q(directory + "/result.json")))
                        result["slurm"] = dict(receipt)
                        return result
                time.sleep(3)
            raise TimeoutError("Slurm trial exceeded the local wait budget")
        finally:
            try:
                if not receipt["resources_released"]:
                    # Recover a lost sbatch acknowledgement by this unique, operator-generated name.
                    ids = ([receipt["job_id"]] if receipt["job_id"] else
                           [int(s) for s in self.remote("squeue -u " + q(p["host"].split("@")[0]) +
                               " -n " + q(request) + " -h -o %i").splitlines() if s.isdigit()])
                    receipt["recovery_job_ids"] = ids
                    receipt["resources_released"] = all([self.cancel_and_confirm(job) for job in ids])
            except Exception as exc:
                receipt["cleanup_error"] = type(exc).__name__ + ": " + str(exc)
                raise RuntimeError("Could not confirm GPU resource release; check the saved Slurm receipt") from exc
            finally:
                self.save(receipt)
