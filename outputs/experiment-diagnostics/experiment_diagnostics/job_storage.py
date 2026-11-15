"""Atomic single-writer metadata storage; never resumes model work automatically."""
import json
import os
from pathlib import Path
import tempfile


class DirectoryLease:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise ValueError("Job directory cannot be a symbolic link")
        lock_path = self.directory / ".writer.lock"
        if lock_path.is_symlink():
            raise ValueError("Job lock cannot be a symbolic link")
        self.handle = lock_path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another server is using this job directory") from None

    def close(self):
        if not self.handle.closed:
            self.handle.close()


def atomic_job_write(directory, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Job directory cannot be a symbolic link")
    raw = json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".state-", delete=False) as output:
            temporary = Path(output.name)
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, directory / "job.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
