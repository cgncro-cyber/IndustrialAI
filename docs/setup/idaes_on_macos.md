# Setting up IDAES on macOS (Apple Silicon)

This is the minimum reproducible path from a clean checkout to a working
IDAES + IPOPT environment on an Apple Silicon Mac. Reviewers should be
able to follow this verbatim.

## Prerequisites

- macOS 14 or later (Apple Silicon — M1/M2/M3/M4 series)
- [Homebrew](https://brew.sh/)
- [uv](https://docs.astral.sh/uv/) — install with `brew install uv`

Python itself is *not* a prerequisite: `uv` will fetch CPython 3.11
automatically based on the `requires-python` field in `pyproject.toml`.

## One-shot setup

```bash
git clone <repo-url>
cd IndustrialAI
make setup
```

`make setup` runs two steps:

1. `uv sync --extra dev` — creates `.venv/`, installs runtime + dev
   dependencies exactly as pinned in `uv.lock`.
2. `uv run idaes get-extensions --verbose` — downloads the pre-built
   IPOPT and auxiliary solver binaries into `~/.idaes/bin/`.

After `make setup` completes, verify with:

```bash
make smoke
```

Expected output:

```
IDAES + ipopt OK
```

## Known macOS pitfalls

### 1. Solver binaries blocked by Gatekeeper

`idaes get-extensions` downloads pre-built binaries. macOS may quarantine
them and block execution with a "cannot be opened because the developer
cannot be verified" dialog the first time `ipopt` runs.

If `make smoke` fails with a Gatekeeper-related error, clear the
quarantine attribute:

```bash
xattr -dr com.apple.quarantine ~/.idaes/bin
```

Then re-run `make smoke`.

### 2. Rosetta vs native binaries

Older IDAES releases shipped x86_64-only solver binaries that ran under
Rosetta 2 on Apple Silicon. IDAES ≥ 2.5 ships arm64 binaries directly.
This project pins `idaes-pse>=2.5` to avoid the Rosetta path.

If you ever see `Bad CPU type in executable`, your IDAES is older than
2.5 — re-run `make setup` to refresh.

### 3. System Python vs uv-managed Python

macOS ships Python 3.9 at `/usr/bin/python3`. Do not use it. `uv` keeps
its managed interpreters under `~/.local/share/uv/python/` and will pick
the right one automatically. Always invoke Python through `uv run …`,
not directly.

### 4. IDAES extensions cache

If a download is interrupted, `~/.idaes/bin/` can end up in an
inconsistent state. Clean and retry:

```bash
rm -rf ~/.idaes/bin
make setup
```

## Manual fallback: Homebrew IPOPT

If `idaes get-extensions` cannot fetch binaries (e.g. behind a corporate
proxy), IPOPT can be installed via Homebrew and pointed at by Pyomo:

```bash
brew install ipopt
```

Pyomo will discover the Homebrew-installed `ipopt` on `PATH`
automatically. Verify with:

```bash
uv run python -c "from pyomo.environ import SolverFactory; print(SolverFactory('ipopt').available())"
```

This is a fallback path only — the project's reproducibility story
assumes `idaes get-extensions` worked.

## Pre-commit hooks

After `make setup`:

```bash
uv run pre-commit install
```

This installs the git hooks declared in `.pre-commit-config.yaml`
(ruff format, ruff check, mypy strict on `src/`, plus standard
whitespace and YAML checks).
