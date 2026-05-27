"""Tests that the persisted shootout JSON files are internally consistent.

These tests do not re-run the shootout (that lives in
``tools/run_pid_shootout.py`` and is expensive). They verify that
whatever the tool last wrote to ``data/reference/`` keeps a sane
shape so downstream code (``load_c0_tuning``, paper notebooks)
doesn't silently break.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SHOOTOUT = _REPO_ROOT / "data" / "reference" / "c0_pid_tuning_shootout.json"
_C0 = _REPO_ROOT / "data" / "reference" / "c0_pid_tuning.json"


pytestmark = pytest.mark.skipif(
    not _SHOOTOUT.exists() or not _C0.exists(),
    reason="shootout JSONs not yet produced (run tools/run_pid_shootout.py)",
)


def test_shootout_has_six_candidates() -> None:
    """The shootout JSON must record every candidate it scored."""
    with _SHOOTOUT.open() as fh:
        data = json.load(fh)
    assert len(data["candidates"]) == 6
    names = {c["name"] for c in data["candidates"]}
    assert len(names) == 6  # all distinct


def test_shootout_winner_matches_a_candidate() -> None:
    """The winner_name must appear in the candidate list."""
    with _SHOOTOUT.open() as fh:
        data = json.load(fh)
    candidate_names = {c["name"] for c in data["candidates"]}
    assert data["winner_name"] in candidate_names


def test_shootout_winner_has_finite_aggregate_iae() -> None:
    """The winner cannot be a failed run."""
    with _SHOOTOUT.open() as fh:
        data = json.load(fh)
    iae = data["winner_aggregate_iae"]
    assert np.isfinite(iae)
    assert iae >= 0.0


def test_shootout_winner_has_lowest_iae_among_finite_candidates() -> None:
    """winner_aggregate_iae must equal the minimum over candidates that completed."""
    with _SHOOTOUT.open() as fh:
        data = json.load(fh)
    finites = [
        c["results"]["aggregate_iae"] for c in data["candidates"] if not c["results"]["any_failure"]
    ]
    assert finites
    assert data["winner_aggregate_iae"] == pytest.approx(min(finites), rel=1e-12)


def test_robustness_block_present_with_three_ops() -> None:
    """The winner must have been spot-checked at three operating points."""
    with _SHOOTOUT.open() as fh:
        data = json.load(fh)
    assert "robustness" in data
    assert len(data["robustness"]) == 3


def test_robustness_non_skipped_ops_have_finite_iae() -> None:
    """Every robustness OP that *was* scored must produce a finite IAE.

    Skipped OPs (e.g., the F-perturbed ones that fall foul of the LV
    plant's RGA(1,1) ~ 36 numerical conditioning) are tolerated as
    long as they carry a documented reason.
    """
    with _SHOOTOUT.open() as fh:
        data = json.load(fh)
    saw_at_least_one_scored = False
    for label, payload in data["robustness"].items():
        if payload.get("skipped", False):
            assert "reason" in payload, f"{label}: skipped without a reason"
            continue
        saw_at_least_one_scored = True
        assert np.isfinite(payload["aggregate_iae"]), (
            f"{label} aggregate IAE non-finite — winner is not robust at this OP"
        )
    assert saw_at_least_one_scored, "no robustness OP was scored — block is unusable"


def test_c0_winner_json_has_winner_block_and_loops() -> None:
    """The runtime-loaded c0_pid_tuning.json carries the winner's PI gains."""
    with _C0.open() as fh:
        data = json.load(fh)
    assert "winner" in data
    assert "loops" in data
    assert "top" in data["loops"] and "bottom" in data["loops"]
    top = data["loops"]["top"]["tyreus_luyben"]
    bottom = data["loops"]["bottom"]["tyreus_luyben"]
    assert top["Kp"] > 0.0 and top["Ti_min"] > 0.0
    assert bottom["Kp"] > 0.0 and bottom["Ti_min"] > 0.0


def test_c0_winner_matches_shootout_winner() -> None:
    """The two JSONs must agree on which candidate won."""
    with _SHOOTOUT.open() as fh:
        shootout = json.load(fh)
    with _C0.open() as fh:
        c0 = json.load(fh)
    assert c0["winner"]["variant_name"] == shootout["winner_name"]
    assert c0["winner"]["aggregate_iae"] == pytest.approx(
        shootout["winner_aggregate_iae"], rel=1e-12
    )
