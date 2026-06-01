"""Tests for industrial_ai.io.atomic_write_json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_ai.io import atomic_write_json


def test_atomic_write_json_writes_and_reads_back(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    payload = {"k": 1, "list": [1, 2, 3], "nested": {"a": "b"}}
    atomic_write_json(target, payload)
    assert target.exists()
    assert json.loads(target.read_text()) == payload


def test_atomic_write_json_replaces_existing_file_atomically(tmp_path: Path) -> None:
    target = tmp_path / "out.json"
    atomic_write_json(target, {"v": 1})
    atomic_write_json(target, {"v": 2})
    assert json.loads(target.read_text()) == {"v": 2}
    # No leftover tmp file.
    assert not (tmp_path / "out.json.tmp").exists()


def test_atomic_write_json_tmp_left_on_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the rename fails, the tmp file is left for inspection rather than swallowed.

    Simulate a rename failure by monkey-patching ``Path.replace``.
    """
    target = tmp_path / "out.json"

    def _boom(self: Path, target: Path) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(Path, "replace", _boom)
    with pytest.raises(OSError):
        atomic_write_json(target, {"v": 1})
    tmp = tmp_path / "out.json.tmp"
    assert tmp.exists()
    assert json.loads(tmp.read_text()) == {"v": 1}


def test_atomic_write_json_creates_no_partial_on_serialization_crash(
    tmp_path: Path,
) -> None:
    """If json.dumps raises, the target file is unchanged (atomic by construction).

    Contract: ``target`` is either the old content or the new
    content, never a half-written truncated version. We trigger a
    serialization failure with a circular reference, which is the
    realistic failure mode (datetime objects etc. are handled via
    default=str).
    """
    target = tmp_path / "out.json"
    atomic_write_json(target, {"v": "old"})
    circular: dict[str, object] = {}
    circular["self"] = circular  # json.dumps → ValueError (circular ref)
    with pytest.raises(ValueError):
        atomic_write_json(target, circular)
    # Old content intact; the tmp was never opened (json.dumps fails
    # before any I/O happens).
    assert json.loads(target.read_text()) == {"v": "old"}
