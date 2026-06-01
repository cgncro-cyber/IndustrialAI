"""I/O helpers for crash-tolerant artifact writing.

Used by the Phase-3 DoE sweep / analysis / confirmation drivers
(2026-06-01 hyperparameter DoE) to keep the sweep manifest and
result files safe against mid-write crashes — laptop sleep,
SIGINT, OS-level kills, etc.

Pattern: write to ``path.tmp``, then atomic rename onto ``path``.
The rename is POSIX-atomic on the same filesystem, so a reader
either sees the old file in full or the new file in full — never a
half-written one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_json"]


def atomic_write_json(path: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``path`` atomically.

    Writes to ``{path}.tmp`` first, ``fsync`` the file (so the
    contents reach disk before the rename), then atomically renames
    onto ``path``. If the rename fails, the temp file is left in
    place for inspection rather than silently swallowed.

    Parameters
    ----------
    path
        Final destination. Parent directory must exist.
    data
        JSON-serialisable payload.

    Raises
    ------
    OSError
        If the rename fails or the parent directory is missing.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(data, indent=2, default=str)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(serialized)
        fh.flush()
        # fsync ensures the bytes reach the disk before the rename;
        # without it, a power loss between write() and replace() can
        # leave the tmp file empty and the original gone.
        import os

        os.fsync(fh.fileno())
    tmp.replace(path)
