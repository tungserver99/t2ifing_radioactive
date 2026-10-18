"""Fixture integrity and cooperative cross-process publication primitives."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import time


def verify_sha256(path, expected):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    actual = h.hexdigest()
    if actual != expected:
        raise ValueError(f"Checkpoint SHA256 mismatch: expected {expected}, got {actual}. Refusing to load.")
    return actual


@contextmanager
def fixture_lock(out, timeout=3600):
    """Keep a persistent lock file: deleting it permits inode replacement races.

    OS releases the lock on process death. Linux flock and Windows byte locking
    coordinate callers of this helper; other filesystem writers are not trusted.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.parent / (out.name + ".lock"), "a+b") as f:
        f.seek(0, 2)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, PermissionError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for fixture lock: {out}")
                time.sleep(.1)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def publish_fixture(ready, out):
    """Reserve destination with atomic mkdir, never rename over a destination.

    Call under fixture_lock. Provenance is moved last as completion marker.
    A crash leaves an incomplete directory which reuse must reject explicitly.
    """
    ready, out = Path(ready), Path(out)
    if not (ready / "fixture_provenance.json").is_file():
        raise ValueError("Missing fixture completion marker")
    out.mkdir(exist_ok=False)
    for child in ready.iterdir():
        if child.name != "fixture_provenance.json":
            child.rename(out / child.name)
    (ready / "fixture_provenance.json").rename(out / "fixture_provenance.json")
