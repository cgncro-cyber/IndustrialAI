"""Round-trip tests for the data-logging contract.

Verifies that a :class:`RunLogger` populates the eight contract files
defined in ``docs/figures.md`` and that the written artifacts are
readable back with the expected schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from industrial_ai.twin.data_logging import RunLogger, RunPaths


@pytest.fixture
def runs_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def _populate(logger: RunLogger, *, with_safety: bool = False) -> None:
    logger.set_config({"twin": {"NT": 41, "alpha": 1.5}})
    for k, t in enumerate(np.linspace(0.0, 10.0, 5)):
        logger.record_timeseries(
            t=float(t),
            y_D=0.99,
            x_B=0.01,
            L=2.706,
            V=3.206,
            D=0.5,
            B=0.5,
            F=1.0,
            zF=0.5,
            qF=1.0,
        )
        logger.record_tray_profile(
            t=float(t),
            compositions=np.linspace(0.01, 0.99, 41),
            holdups=np.full(41, 0.5),
        )
        logger.record_setpoint(t=float(t), channel="yD_setpoint", requested=0.99, applied=0.99)
        logger.record_latency(cycle_index=k, wall_clock_seconds=0.42)
        if with_safety:
            logger.record_safety_decision(
                t=float(t),
                anomaly_score=0.1,
                threshold=0.5,
                blocked=False,
                proposed_setpoint={"yD_setpoint": 0.99},
            )
    logger.set_kpis({"energy_per_kg_product": 7.3, "iae": 0.04})


def test_run_logger_writes_contract_files(runs_root: Path) -> None:
    logger = RunLogger.create(
        runs_root=runs_root, config="c0_pid_only", scenario="feed_step_+20pct"
    )
    _populate(logger)
    paths = logger.finalize(input_hashes={"reference.json": "deadbeef"})

    for p in (
        paths.timeseries,
        paths.tray_profile,
        paths.setpoints,
        paths.kpis,
        paths.latency,
        paths.config,
        paths.manifest,
    ):
        assert p.exists(), f"missing contract artifact: {p}"
    # safety_log is omitted when is_safety_run=False.
    assert not paths.safety_log.exists()


def test_run_logger_safety_run_writes_safety_log(runs_root: Path) -> None:
    logger = RunLogger.create(
        runs_root=runs_root,
        config="c3_agent_safety",
        scenario="reflux_step",
        is_safety_run=True,
    )
    _populate(logger, with_safety=True)
    paths = logger.finalize()
    assert paths.safety_log.exists()
    df = pd.read_parquet(paths.safety_log)
    assert {"t", "anomaly_score", "threshold", "blocked", "proposed_setpoint"} <= set(df.columns)


def test_timeseries_schema_round_trip(runs_root: Path) -> None:
    logger = RunLogger.create(runs_root=runs_root, config="c0_pid_only", scenario="zF_step_-10pct")
    _populate(logger)
    paths = logger.finalize()
    df = pd.read_parquet(paths.timeseries)
    assert len(df) == 5
    expected_cols = {"t", "y_D", "x_B", "L", "V", "D", "B", "F", "zF", "qF"}
    assert expected_cols <= set(df.columns)


def test_tray_profile_long_format(runs_root: Path) -> None:
    logger = RunLogger.create(runs_root=runs_root, config="c0_pid_only", scenario="nominal")
    _populate(logger)
    paths = logger.finalize()
    df = pd.read_parquet(paths.tray_profile)
    # 5 time points x 41 stages = 205 rows.
    assert len(df) == 5 * 41
    assert {"t", "stage", "composition", "holdup_kmol"} <= set(df.columns)
    assert df["stage"].min() == 0
    assert df["stage"].max() == 40


def test_manifest_records_input_hashes_and_versions(runs_root: Path) -> None:
    logger = RunLogger.create(runs_root=runs_root, config="c0_pid_only", scenario="nominal")
    _populate(logger)
    paths = logger.finalize(input_hashes={"foo.json": "abcd1234"})
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["input_hashes"] == {"foo.json": "abcd1234"}
    assert "industrial-ai" in manifest["package_versions"]
    assert "artifact_hashes" in manifest
    assert "timeseries.parquet" in manifest["artifact_hashes"]


def test_config_yaml_round_trips(runs_root: Path) -> None:
    logger = RunLogger.create(runs_root=runs_root, config="c0_pid_only", scenario="nominal")
    _populate(logger)
    paths = logger.finalize()
    with paths.config.open() as fh:
        cfg = yaml.safe_load(fh)
    assert cfg == {"twin": {"NT": 41, "alpha": 1.5}}


def test_record_safety_on_non_safety_logger_raises(runs_root: Path) -> None:
    logger = RunLogger.create(runs_root=runs_root, config="c0_pid_only", scenario="nominal")
    with pytest.raises(RuntimeError, match="is_safety_run=True"):
        logger.record_safety_decision(t=0.0, anomaly_score=0.0, threshold=0.5, blocked=False)


def test_run_paths_build() -> None:
    paths = RunPaths.build(
        runs_root=Path("/tmp/runs"),
        config="c0",
        scenario="s1",
        seed=42,
    )
    assert paths.root == Path("/tmp/runs/c0/s1/42")
    assert paths.timeseries.name == "timeseries.parquet"
    assert paths.manifest.name == "manifest.json"
