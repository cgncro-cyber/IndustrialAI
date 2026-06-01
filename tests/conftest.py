"""Shared pytest fixtures for Column A twin tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live-llm",
        action="store_true",
        default=False,
        help="Run integration tests marked live_llm against the MLX server "
        "(ADR 005 amendment endpoint). Skipped by default to keep the unit "
        "suite independent of the Mac Studio runtime.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live-llm"):
        return
    skip_live = pytest.mark.skip(reason="needs --run-live-llm to run")
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip_live)


#: Path to the published Skogestad steady-state reference data, extracted
#: from ``cola_init.mat`` at
#: https://skoge.folk.ntnu.no/book/1st_edition/matlab_m/cola/cola_init.mat
_REPO_ROOT = Path(__file__).resolve().parent.parent
SKOGESTAD_REFERENCE_PATH = (
    _REPO_ROOT / "data" / "reference" / "skogestad_column_a_steady_state.json"
)


@pytest.fixture(scope="session")
def skogestad_reference() -> dict[str, Any]:
    """Load the published Skogestad Column A steady-state reference."""
    with SKOGESTAD_REFERENCE_PATH.open() as fh:
        data: dict[str, Any] = json.load(fh)
    return data


@pytest.fixture(scope="session")
def skogestad_reference_state(
    skogestad_reference: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Return the 82-element reference state vector (compositions + holdups)."""
    ss = skogestad_reference["steady_state"]
    return np.array(ss["compositions"] + ss["holdups_kmol"], dtype=np.float64)


@pytest.fixture(scope="session")
def skogestad_reference_inputs(
    skogestad_reference: dict[str, Any],
) -> npt.NDArray[np.float64]:
    """Return the 7-element nominal input vector from the reference data."""
    u = skogestad_reference["nominal_inputs"]
    return np.array(
        [
            u["reflux_LT_kmol_per_min"],
            u["boilup_VB_kmol_per_min"],
            u["distillate_D_kmol_per_min"],
            u["bottoms_B_kmol_per_min"],
            u["feed_F_kmol_per_min"],
            u["feed_composition_zF"],
            u["feed_liquid_fraction_qF"],
        ],
        dtype=np.float64,
    )
