"""Tests for ``tools/run_doe_sweep.py``.

The driver does the heavy lifting via subprocess.run; the tests focus
on the cell enumeration, the resilience state machine, and the
idempotent restart behavior. The actual smoke subprocess is only
exercised in the live --run-live-llm path against NIM (out of scope
for this file).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def doe_sweep() -> Any:
    return _load_module("run_doe_sweep", _REPO_ROOT / "tools" / "run_doe_sweep.py")


def test_enumerate_cells_full_factorial_count(doe_sweep: Any) -> None:
    cells = doe_sweep.enumerate_cells()
    assert len(cells) == 5 * 3 * 3 * 5 == 225


def test_enumerate_cells_have_stable_ids(doe_sweep: Any) -> None:
    cells = doe_sweep.enumerate_cells()
    ids = [c["cell_id"] for c in cells]
    assert len(set(ids)) == len(ids)
    assert all("T=" in c["cell_id"] and "p=" in c["cell_id"] for c in cells)


def test_enumerate_cells_parse_budget(doe_sweep: Any) -> None:
    cells = doe_sweep.enumerate_cells()
    by_cfg = {c["reasoning_config"] for c in cells}
    assert by_cfg == {"off", "on_budget_1024", "on_budget_4096"}
    # off → None budget; on_budget_NNNN → NNNN.
    off_cells = [c for c in cells if c["reasoning_config"] == "off"]
    on_1024 = [c for c in cells if c["reasoning_config"] == "on_budget_1024"]
    on_4096 = [c for c in cells if c["reasoning_config"] == "on_budget_4096"]
    assert all(c["reasoning_budget"] is None for c in off_cells)
    assert all(c["reasoning_budget"] == 1024 for c in on_1024)
    assert all(c["reasoning_budget"] == 4096 for c in on_4096)


def test_smoke_command_off_reasoning(doe_sweep: Any) -> None:
    cell = {
        "cell_id": "T=0.6_p=0.95_R=off_S=0",
        "temperature": 0.6,
        "top_p": 0.95,
        "reasoning_config": "off",
        "reasoning_budget": None,
        "seed": 0,
    }
    cmd = doe_sweep._smoke_command(cell, "nvidia/nemotron-3-super-120b-a12b", Path("/tmp/cell"))
    assert "--temperature" in cmd
    assert "0.6" in cmd
    assert "--top-p" in cmd
    assert "--reasoning-mode" in cmd
    assert cmd[cmd.index("--reasoning-mode") + 1] == "off"
    assert "--reasoning-budget" not in cmd


def test_smoke_command_on_reasoning_threads_budget(doe_sweep: Any) -> None:
    cell = {
        "cell_id": "T=0.6_p=0.95_R=on_budget_2048_S=0",
        "temperature": 0.6,
        "top_p": 0.95,
        "reasoning_config": "on_budget_2048",
        "reasoning_budget": 2048,
        "seed": 0,
    }
    cmd = doe_sweep._smoke_command(cell, "x", Path("/tmp/cell"))
    assert cmd[cmd.index("--reasoning-mode") + 1] == "on"
    assert "2048" in cmd


def test_dry_run_marks_all_cells_done(tmp_path: Path, doe_sweep: Any) -> None:
    """--dry-run + --max-cells writes a manifest where the touched cells are done."""
    manifest_path = tmp_path / "sweep_manifest.json"
    manifest = doe_sweep._build_manifest("model-x")
    from industrial_ai.io import atomic_write_json

    atomic_write_json(manifest_path, manifest)
    pending = list(doe_sweep._select_pending(manifest["cells"], retry_failed=False, max_cells=2))
    assert len(pending) == 2
    for cell in pending:
        doe_sweep._run_one_cell(
            cell,
            model="model-x",
            output_root=tmp_path,
            manifest_path=manifest_path,
            manifest=manifest,
            timeout_s=10.0,
            dry_run=True,
        )
    saved = json.loads(manifest_path.read_text())
    done = [c for c in saved["cells"] if c["status"] == "done"]
    assert len(done) == 2


def test_idempotent_restart_skips_done_cells(doe_sweep: Any) -> None:
    cells = doe_sweep.enumerate_cells()
    cells[0]["status"] = "done"
    cells[1]["status"] = "done"
    cells[2]["status"] = "pending"
    cells[3]["status"] = "failed"
    pending_default = list(doe_sweep._select_pending(cells, retry_failed=False, max_cells=None))
    assert all(c["status"] != "done" for c in pending_default)
    # Without --retry-failed, the failed cell is also skipped.
    assert not any(c["status"] == "failed" for c in pending_default)


def test_retry_failed_includes_failed_cells(doe_sweep: Any) -> None:
    cells = doe_sweep.enumerate_cells()
    cells[0]["status"] = "done"
    cells[1]["status"] = "failed"
    cells[2]["status"] = "pending"
    pending_retry = list(doe_sweep._select_pending(cells, retry_failed=True, max_cells=None))
    # Failed cell now in the list, done cell still excluded.
    assert any(c["status"] == "failed" for c in pending_retry)
    assert not any(c["status"] == "done" for c in pending_retry)


def test_corrupt_manifest_raises_loudly(tmp_path: Path, doe_sweep: Any) -> None:
    """ADR 010 §2: corrupt manifest is operator-visible, not silently overwritten."""
    manifest_path = tmp_path / "sweep_manifest.json"
    manifest_path.write_text("{not valid json")
    with pytest.raises(RuntimeError, match="corrupted"):
        doe_sweep._load_or_create_manifest(manifest_path, "model-x")


def test_manifest_model_mismatch_raises_loudly(tmp_path: Path, doe_sweep: Any) -> None:
    """A manifest for a different model is operator-visible, not auto-rebuilt."""
    manifest_path = tmp_path / "sweep_manifest.json"
    from industrial_ai.io import atomic_write_json

    atomic_write_json(manifest_path, doe_sweep._build_manifest("modelA"))
    with pytest.raises(RuntimeError, match="modelA"):
        doe_sweep._load_or_create_manifest(manifest_path, "modelB")


def test_classify_smoke_failure_recognises_rate_limit(doe_sweep: Any) -> None:
    err_class, _excerpt = doe_sweep._classify_smoke_failure(
        returncode=2, stdout="", stderr="HTTP 429 Rate Limit Exceeded"
    )
    assert err_class == "rate_limit"


def test_classify_smoke_failure_recognises_timeout(doe_sweep: Any) -> None:
    err_class, _excerpt = doe_sweep._classify_smoke_failure(
        returncode=124, stdout="", stderr="TimeoutExpired"
    )
    assert err_class == "timeout"


def test_to_str_handles_bytes_str_and_none(doe_sweep: Any) -> None:
    """Regression: TimeoutExpired exposes bytes despite text=True on subprocess.run.

    Crashed the live 120B sweep at cell T=0.8_p=1_R=on_budget_1024_S=3
    on 2026-06-02 with `TypeError: can't concat str to bytes`. The
    _to_str helper decodes bytes safely and passes str through.
    """
    assert doe_sweep._to_str(None) == ""
    assert doe_sweep._to_str("hello") == "hello"
    assert doe_sweep._to_str(b"hello") == "hello"
    assert doe_sweep._to_str(b"\xff invalid utf-8") == "� invalid utf-8"


def test_run_one_cell_handles_timeout_with_bytes_stdout(
    tmp_path: Path, doe_sweep: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cell handler must mark a timeout cell ``failed`` without crashing
    when ``subprocess.run`` exposes bytes on stdout/stderr (the actual
    Python stdlib behavior on ``TimeoutExpired`` even with text=True).
    """
    import subprocess

    def _fake_run(*_args: Any, **_kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(
            cmd=["fake"],
            timeout=10.0,
            output=b"partial stdout bytes",
            stderr=b"partial stderr bytes",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)

    cell = doe_sweep.enumerate_cells()[0]
    manifest = doe_sweep._build_manifest("test-model")
    manifest_path = tmp_path / "sweep_manifest.json"
    from industrial_ai.io import atomic_write_json

    atomic_write_json(manifest_path, manifest)
    doe_sweep._run_one_cell(
        cell,
        model="test-model",
        output_root=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        timeout_s=10.0,
        dry_run=False,
    )
    assert cell["status"] == "failed"
    assert cell["error_class"] == "timeout"
    # Classifier lowercases the excerpt for keyword matching, so
    # compare case-insensitively. The point of the test is that the
    # cell got marked failed without a TypeError crash.
    assert "timeoutexpired" in (cell["error_message_excerpt"] or "").lower()
