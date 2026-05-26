"""Data-logging contract for Column A runs.

Implements the run-directory contract defined in ``docs/figures.md``
section *Data-Logging Contract*. Every experimental run from Phase 1
onward writes its outputs into

    data/runs/<config>/<scenario>/<seed>/

with eight standard artifacts:

- ``timeseries.parquet``    high-frequency state for Figure 3
- ``tray_profile.parquet``  per-stage composition and holdup for Figure 2
- ``setpoints.parquet``     commanded setpoints with timestamps
- ``kpis.json``             scalar KPIs for Figures 4 / 7
- ``latency.json``          wall-clock per supervisory cycle for Figure 6
- ``safety_log.parquet``    anomaly scores and decisions for Figure 5 (C3 only)
- ``config.yaml``           full hyperparameter snapshot
- ``manifest.json``         input data hashes, model versions, seed

A run is built incrementally — observations are appended during
integration — and finalized with :meth:`RunLogger.finalize`, which
flushes the buffers to disk.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

__all__ = ["RunLogger", "RunPaths"]


# Packages whose installed versions are recorded in every manifest.
_RECORDED_PACKAGES = (
    "industrial-ai",
    "numpy",
    "scipy",
    "pandas",
    "pyarrow",
    "pyomo",
    "idaes-pse",
)


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Resolved filesystem paths for a single run.

    Attributes
    ----------
    root : pathlib.Path
        Top-level run directory ``data/runs/<config>/<scenario>/<seed>/``.
    timeseries, tray_profile, setpoints, safety_log : pathlib.Path
        Parquet artifact paths.
    kpis, latency, manifest : pathlib.Path
        JSON artifact paths.
    config : pathlib.Path
        YAML config snapshot.
    """

    root: Path
    timeseries: Path
    tray_profile: Path
    setpoints: Path
    safety_log: Path
    kpis: Path
    latency: Path
    config: Path
    manifest: Path

    @classmethod
    def build(
        cls,
        *,
        runs_root: Path,
        config: str,
        scenario: str,
        seed: int | str = 0,
    ) -> RunPaths:
        """Resolve all artifact paths for a ``config/scenario/seed`` triple."""
        root = runs_root / config / scenario / str(seed)
        return cls(
            root=root,
            timeseries=root / "timeseries.parquet",
            tray_profile=root / "tray_profile.parquet",
            setpoints=root / "setpoints.parquet",
            safety_log=root / "safety_log.parquet",
            kpis=root / "kpis.json",
            latency=root / "latency.json",
            config=root / "config.yaml",
            manifest=root / "manifest.json",
        )


@dataclass(slots=True)
class RunLogger:
    """Builder for a Phase 1+ Column A run artifact set.

    Buffers observations in memory and flushes them on
    :meth:`finalize`. Designed for the pattern

        logger = RunLogger.create(...)
        logger.set_config(...)
        for step in simulation:
            logger.record_timeseries(...)
            logger.record_tray_profile(...)
            logger.record_setpoint(...)
            logger.record_latency(...)
        logger.set_kpis(...)
        logger.finalize(input_hashes=...)
    """

    paths: RunPaths
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    _timeseries_rows: list[dict[str, Any]] = field(default_factory=list)
    _tray_profile_rows: list[dict[str, Any]] = field(default_factory=list)
    _setpoint_rows: list[dict[str, Any]] = field(default_factory=list)
    _safety_rows: list[dict[str, Any]] = field(default_factory=list)
    _latency_rows: list[dict[str, Any]] = field(default_factory=list)
    _kpis: dict[str, Any] = field(default_factory=dict)
    _is_safety_run: bool = False

    @classmethod
    def create(
        cls,
        *,
        runs_root: Path,
        config: str,
        scenario: str,
        seed: int | str = 0,
        is_safety_run: bool = False,
    ) -> RunLogger:
        """Create a logger and ensure its run directory exists."""
        paths = RunPaths.build(runs_root=runs_root, config=config, scenario=scenario, seed=seed)
        paths.root.mkdir(parents=True, exist_ok=True)
        return cls(paths=paths, _is_safety_run=is_safety_run)

    def set_config(self, snapshot: dict[str, Any]) -> None:
        """Set the YAML config snapshot to be written on finalize."""
        self.config_snapshot = snapshot

    def record_timeseries(
        self,
        *,
        t: float,
        y_D: float,
        x_B: float,
        L: float,
        V: float,
        D: float,
        B: float,
        F: float,
        zF: float,
        qF: float,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Append one row to the high-frequency timeseries buffer.

        Parameters match the canonical column names referenced in
        ``docs/figures.md`` Figure 3. ``extra`` is merged into the row
        for ad-hoc scalar channels (e.g., disturbance markers).
        """
        row: dict[str, Any] = {
            "t": t,
            "y_D": y_D,
            "x_B": x_B,
            "L": L,
            "V": V,
            "D": D,
            "B": B,
            "F": F,
            "zF": zF,
            "qF": qF,
        }
        if extra is not None:
            row.update(extra)
        self._timeseries_rows.append(row)

    def record_tray_profile(
        self,
        *,
        t: float,
        compositions: npt.NDArray[np.float64],
        holdups: npt.NDArray[np.float64],
    ) -> None:
        """Append one timestamp of per-stage compositions and holdups.

        Stored in long format (one row per stage per time) so it
        translates directly to the Figure 2 heatmaps.
        """
        if compositions.shape != holdups.shape:
            raise ValueError(
                "compositions and holdups must have the same shape; "
                f"got {compositions.shape} vs {holdups.shape}"
            )
        for stage_idx, (xi, mi) in enumerate(zip(compositions, holdups, strict=True)):
            self._tray_profile_rows.append(
                {
                    "t": t,
                    "stage": int(stage_idx),
                    "composition": float(xi),
                    "holdup_kmol": float(mi),
                }
            )

    def record_setpoint(
        self,
        *,
        t: float,
        channel: str,
        requested: float,
        applied: float,
    ) -> None:
        """Append one supervisory-setpoint event.

        ``requested`` is what the supervisor asked for; ``applied`` is
        what the rate-limiter allowed through.
        """
        self._setpoint_rows.append(
            {
                "t": t,
                "channel": channel,
                "requested": requested,
                "applied": applied,
            }
        )

    def record_safety_decision(
        self,
        *,
        t: float,
        anomaly_score: float,
        threshold: float,
        blocked: bool,
        proposed_setpoint: dict[str, float] | None = None,
    ) -> None:
        """Append one safety-gate decision (Phase 4 / configuration C3).

        Raises
        ------
        RuntimeError
            If the logger was not created with ``is_safety_run=True``.
        """
        if not self._is_safety_run:
            raise RuntimeError(
                "safety decisions can only be recorded on a safety-enabled logger; "
                "construct with is_safety_run=True"
            )
        self._safety_rows.append(
            {
                "t": t,
                "anomaly_score": anomaly_score,
                "threshold": threshold,
                "blocked": blocked,
                "proposed_setpoint": json.dumps(proposed_setpoint)
                if proposed_setpoint is not None
                else None,
            }
        )

    def record_latency(self, *, cycle_index: int, wall_clock_seconds: float) -> None:
        """Append one supervisory-cycle wall-clock measurement."""
        self._latency_rows.append(
            {"cycle_index": cycle_index, "wall_clock_seconds": wall_clock_seconds}
        )

    def set_kpis(self, kpis: dict[str, Any]) -> None:
        """Set scalar KPIs to be written on finalize."""
        self._kpis = dict(kpis)

    def finalize(self, *, input_hashes: dict[str, str] | None = None) -> RunPaths:
        """Flush all buffers and write the artifact set to disk.

        Parameters
        ----------
        input_hashes : dict of str to str, optional
            Optional dictionary of ``filename -> sha256-hex`` for input
            artifacts that should be recorded in the manifest. Hashes
            of files inside the run directory itself are computed
            automatically.

        Returns
        -------
        RunPaths
            The resolved paths of the written artifacts.
        """
        pd.DataFrame(self._timeseries_rows).to_parquet(self.paths.timeseries, index=False)
        pd.DataFrame(self._tray_profile_rows).to_parquet(self.paths.tray_profile, index=False)
        pd.DataFrame(self._setpoint_rows).to_parquet(self.paths.setpoints, index=False)
        if self._is_safety_run:
            pd.DataFrame(self._safety_rows).to_parquet(self.paths.safety_log, index=False)

        with self.paths.kpis.open("w") as fh:
            json.dump(self._kpis, fh, indent=2, default=_json_default)
        with self.paths.latency.open("w") as fh:
            json.dump(self._latency_rows, fh, indent=2)

        with self.paths.config.open("w") as fh:
            yaml.safe_dump(self.config_snapshot, fh, sort_keys=True)

        self._write_manifest(input_hashes or {})
        return self.paths

    def _write_manifest(self, input_hashes: dict[str, str]) -> None:
        artifact_files = [
            self.paths.timeseries,
            self.paths.tray_profile,
            self.paths.setpoints,
            self.paths.kpis,
            self.paths.latency,
            self.paths.config,
        ]
        if self._is_safety_run:
            artifact_files.append(self.paths.safety_log)

        artifact_hashes = {p.name: _sha256_of(p) for p in artifact_files if p.exists()}

        package_versions: dict[str, str] = {}
        for name in _RECORDED_PACKAGES:
            try:
                package_versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                package_versions[name] = "not-installed"

        manifest = {
            "created_at_utc": datetime.now(tz=UTC).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "input_hashes": input_hashes,
            "artifact_hashes": artifact_hashes,
            "package_versions": package_versions,
            "run_paths": {
                name: str(value.relative_to(self.paths.root.parents[2]))
                for name, value in _path_fields(self.paths).items()
                if value.exists() or name == "root"
            },
            "is_safety_run": self._is_safety_run,
        }
        with self.paths.manifest.open("w") as fh:
            json.dump(manifest, fh, indent=2)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_fields(paths: RunPaths) -> dict[str, Path]:
    return {f.name: getattr(paths, f.name) for f in dataclasses.fields(paths)}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON-serializable")
