"""Unit tests for the off-nominal X0/MV dispatch helper in tools/run_c2_smoke.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from industrial_ai.agents.errors import InfeasibleSubmetricError

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_smoke() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_c2_smoke", _REPO_ROOT / "tools" / "run_c2_smoke.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_c2_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke() -> Any:
    return _load_smoke()


def test_target_acquisition_uses_lookup_lv_ss(smoke: Any) -> None:
    """X0 from lookup_lv_ss, LT/VB at nominal, x0_source labeled correctly."""
    X0, LT0, VB0, x0_source = smoke._load_op_initial_state(
        op_F=0.8, op_zF=0.45, submetric="target_acquisition"
    )
    assert isinstance(X0, np.ndarray)
    assert X0.shape == (82,)
    assert LT0 > 0 and VB0 > 0
    assert x0_source == "lookup_lv_ss"


def test_disturbance_rejection_uses_pre_stage_cache(smoke: Any) -> None:
    """X0 from off_nominal_on_spec_pre_stages, LT*/VB* from cache."""
    X0, LT0, VB0, x0_source = smoke._load_op_initial_state(
        op_F=0.8, op_zF=0.45, submetric="disturbance_rejection"
    )
    assert X0.shape == (82,)
    # LT*/VB* for the corner OP are documented in the cache; just verify
    # they're not the nominal defaults (which would indicate fall-back).
    assert LT0 == pytest.approx(2.1674, abs=1e-3)
    assert VB0 == pytest.approx(2.5266, abs=1e-3)
    assert x0_source == "off_nominal_on_spec_pre_stages"


def test_disturbance_rejection_raises_when_cache_missing_entry(smoke: Any) -> None:
    """Requesting an OP not in the cache raises InfeasibleSubmetricError."""
    with pytest.raises(InfeasibleSubmetricError):
        smoke._load_op_initial_state(op_F=99.9, op_zF=99.9, submetric="disturbance_rejection")


def test_unknown_submetric_raises_value_error(smoke: Any) -> None:
    with pytest.raises(ValueError):
        smoke._load_op_initial_state(op_F=0.8, op_zF=0.45, submetric="oops")
