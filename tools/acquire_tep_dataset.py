"""TEP dataset acquisition kickoff (Phase-4 prep).

Per ``data/external/tep/README.md`` the Phase-4 safety gate is trained
on the Rieth et al. (2017) Tennessee Eastman Process simulation data
from Harvard Dataverse (DOI ``10.7910/DVN/6C3JR1``, CC0 license).

Two modes:

- **Provenance-only (default).** Hits the Dataverse REST API to fetch
  the file index (file names, sizes, checksums, persistent IDs) and
  writes ``data/external/tep/file_listing.json`` plus
  ``data/external/tep/provenance.json``. Cheap (single API call,
  ~kB transfer). Idempotent. Run on every CI to confirm the upstream
  is still resolvable.

- **Bulk download** (``--download``). Iterates the file list and
  downloads each ``.RData`` to ``data/external/tep/`` after the
  upstream-reported SHA-256 matches the post-download local SHA-256.
  Ignored by git per ``.gitignore``. Required before Phase 4 detector
  training.

The acquisition is split this way so the provenance check is small and
fast enough to live in CI, while the multi-GB pull is opt-in. Both
modes are read-only against the project repo (the only writes are to
``data/external/tep/``).

Invocation:

    uv run python tools/acquire_tep_dataset.py                # provenance only
    uv run python tools/acquire_tep_dataset.py --download     # provenance + bulk pull
    uv run python tools/acquire_tep_dataset.py --verify-only  # re-verify cached files
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEP_DIR = _REPO_ROOT / "data" / "external" / "tep"
_FILE_LISTING = _TEP_DIR / "file_listing.json"
_PROVENANCE = _TEP_DIR / "provenance.json"

_DOI = "10.7910/DVN/6C3JR1"
_DATAVERSE_API = "https://dataverse.harvard.edu/api"
_DATASET_URL = f"{_DATAVERSE_API}/datasets/:persistentId?persistentId=doi:{_DOI}"


def _http_get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    """Single-shot JSON GET. ADR 010 §2: one attempt, named exception on failure."""
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "industrial_ai/tep-acquire"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Dataverse API request to {url!r} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _fetch_file_listing() -> dict[str, Any]:
    """Hit the Dataverse API and assemble a compact file index."""
    print(f"Querying Dataverse for dataset DOI {_DOI} ...")
    raw = _http_get_json(_DATASET_URL)
    if "data" not in raw or "latestVersion" not in raw["data"]:
        raise RuntimeError(f"unexpected Dataverse response shape: keys={list(raw.keys())}")
    version_block = raw["data"]["latestVersion"]
    files_payload = version_block.get("files", [])
    files: list[dict[str, Any]] = []
    for entry in files_payload:
        df = entry.get("dataFile", {})
        files.append(
            {
                "label": entry.get("label") or df.get("filename"),
                "filename": df.get("filename"),
                "directory_label": entry.get("directoryLabel"),
                "size_bytes": df.get("filesize"),
                "content_type": df.get("contentType"),
                "upstream_sha256": (df.get("checksum") or {}).get("value"),
                "checksum_type": (df.get("checksum") or {}).get("type"),
                "persistent_id": df.get("persistentId"),
                "id": df.get("id"),
                "version_number": version_block.get("versionNumber"),
                "version_state": version_block.get("versionState"),
                "download_url": (
                    f"{_DATAVERSE_API}/access/datafile/{df.get('id')}"
                    if df.get("id") is not None
                    else None
                ),
            }
        )
    listing = {
        "schema_version": 1,
        "doi": _DOI,
        "dataset_pid": raw["data"].get("persistentUrl"),
        "version": {
            "number": version_block.get("versionNumber"),
            "minor": version_block.get("versionMinorNumber"),
            "state": version_block.get("versionState"),
            "release_time": version_block.get("releaseTime"),
        },
        "fetched_at_utc": datetime.now(tz=UTC).isoformat(),
        "n_files": len(files),
        "files": files,
    }
    return listing


def _write_provenance(listing: dict[str, Any], bulk_downloaded: bool) -> None:
    provenance = {
        "schema_version": 1,
        "doi": _DOI,
        "title": "Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation",
        "authors": ["Cory A. Rieth", "Ben D. Amsel", "Randy Tran", "Maia B. Cook"],
        "year": 2017,
        "publisher": "Harvard Dataverse",
        "license": "CC0 1.0 (public domain dedication)",
        "dataset_pid": listing.get("dataset_pid"),
        "version": listing.get("version"),
        "accessed_at_utc": datetime.now(tz=UTC).isoformat(),
        "n_files_listed": listing.get("n_files"),
        "bulk_downloaded": bulk_downloaded,
        "tool_path": "tools/acquire_tep_dataset.py",
    }
    _PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")


def _download_one(file_entry: dict[str, Any], *, retry_pause: float = 5.0) -> dict[str, Any]:
    """Single-attempt download (ADR 010) with post-pull SHA-256 verify."""
    url = file_entry["download_url"]
    target = _TEP_DIR / file_entry["filename"]
    expected_upstream = file_entry.get("upstream_sha256")
    size = file_entry.get("size_bytes")
    if url is None:
        return {"filename": file_entry["filename"], "status": "skipped:no_url"}
    if target.exists():
        local_sha = _sha256_of(target)
        if expected_upstream and local_sha == expected_upstream:
            return {
                "filename": file_entry["filename"],
                "status": "cached:sha256_match",
                "size_bytes": target.stat().st_size,
                "local_sha256": local_sha,
            }
        # Upstream Rieth checksum is MD5 not SHA-256; if it doesn't
        # match our SHA-256 we just record both.
    print(f"  downloading {file_entry['filename']} (~{(size or 0) / 1e6:.1f} MB) ...")
    t0 = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "industrial_ai/tep-acquire"})
    try:
        with urllib.request.urlopen(request, timeout=600.0) as resp, target.open("wb") as fh:
            while True:
                buf = resp.read(1 << 20)
                if not buf:
                    break
                fh.write(buf)
    except urllib.error.URLError as exc:
        return {
            "filename": file_entry["filename"],
            "status": f"error:{type(exc).__name__}",
            "detail": str(exc),
        }
    elapsed = time.perf_counter() - t0
    local_sha = _sha256_of(target)
    md5 = hashlib.md5(target.read_bytes()).hexdigest()
    checksum_match = None
    if expected_upstream:
        checksum_match = expected_upstream in (local_sha, md5)
    return {
        "filename": file_entry["filename"],
        "status": "downloaded",
        "size_bytes": target.stat().st_size,
        "elapsed_seconds": elapsed,
        "local_sha256": local_sha,
        "local_md5": md5,
        "upstream_checksum": expected_upstream,
        "checksum_match": checksum_match,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="In addition to the provenance fetch, download the full file set.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-verify SHA-256 of files already on disk; skip download and API call.",
    )
    args = parser.parse_args()

    _TEP_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        if not _FILE_LISTING.exists():
            print("file_listing.json missing — run without --verify-only first")
            return 2
        with _FILE_LISTING.open() as fh:
            listing = json.load(fh)
        for entry in listing["files"]:
            target = _TEP_DIR / entry["filename"]
            if not target.exists():
                print(f"  MISSING  {entry['filename']}")
                continue
            local_sha = _sha256_of(target)
            md5 = hashlib.md5(target.read_bytes()).hexdigest()
            ok = entry.get("upstream_sha256") in {local_sha, md5}
            print(
                f"  {'OK ' if ok else 'BAD'}  {entry['filename']}  "
                f"sha256={local_sha[:12]}…  md5={md5[:12]}…"
            )
        return 0

    listing = _fetch_file_listing()
    _FILE_LISTING.write_text(json.dumps(listing, indent=2) + "\n")
    print(f"Wrote {_FILE_LISTING} — {listing['n_files']} files indexed.")

    download_results: list[dict[str, Any]] = []
    if args.download:
        print("Bulk download requested ...")
        for entry in listing["files"]:
            result = _download_one(entry)
            download_results.append(result)
            print(f"    -> {result['status']}")
    else:
        print("Provenance-only mode (default). Pass --download to also pull the bulk .RData files.")
    _write_provenance(listing, bulk_downloaded=bool(download_results))
    print(f"Wrote {_PROVENANCE}")

    if download_results:
        any_failed = any(r["status"].startswith("error") for r in download_results)
        any_mismatch = any(
            r.get("checksum_match") is False for r in download_results if "checksum_match" in r
        )
        if any_failed or any_mismatch:
            print("!! one or more files failed download or checksum verification.")
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
