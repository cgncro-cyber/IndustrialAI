# Tennessee Eastman Process (TEP) dataset — Phase 4 cross-domain anomaly training source

Per ADR 006 / `docs/kpis.md` §3, the C3 safety gate is an anomaly detector
**trained cross-domain on TEP** and applied to the Skogestad Column A twin
without per-plant retraining. This directory holds the locally-cached
training and evaluation data, plus the provenance metadata. The raw data
files are not version-controlled (see `.gitignore`); the provenance and
processing scripts are.

## Why TEP for the safety gate

- Canonical multivariable industrial benchmark with 52 process variables,
  12 manipulated variables, and 21 documented fault modes — the breadth of
  signal types (compositions, flows, levels, temperatures, pressures) that
  the methodology paper's transfer claim needs to be plausible across.
- Reviewer recognition for IFAC / `eess.SY`: TEP is the most-cited industrial
  control benchmark and `docs/figures.md` Figure 8 (cross-domain confusion
  matrix) cites it directly.
- License + reproducibility: the canonical Rieth et al. (2017) "Additional
  Tennessee Eastman Process Simulation Data" release is published on Harvard
  Dataverse under CC0 (no attribution required, but cited everywhere for
  scholarly integrity).

## Canonical source

| | |
|---|---|
| Persistent identifier | DOI `10.7910/DVN/6C3JR1` |
| Title | Additional Tennessee Eastman Process Simulation Data for Anomaly Detection Evaluation |
| Authors | Cory A. Rieth, Ben D. Amsel, Randy Tran, Maia B. Cook |
| Year | 2017 |
| Publisher | Harvard Dataverse |
| Direct URL | <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1> |
| License | CC0 1.0 (public domain dedication) |
| Citation key (BibTeX) | `RiethTEP2017` (entry in `paper/references.bib` when Phase 4 begins) |

### File set (per the Dataverse landing page)

The release ships four `.RData` files plus a metadata README:

| File | Rows | Purpose |
|---|---|---|
| `TEP_FaultFree_Training.RData` | 500 simulations × 500 obs × 55 vars | Normal-operation training signal |
| `TEP_Faulty_Training.RData` | 500 simulations × 21 faults × 480 obs × 55 vars | Fault-mode labelled training |
| `TEP_FaultFree_Testing.RData` | 500 simulations × 960 obs × 55 vars | Held-out normal eval |
| `TEP_Faulty_Testing.RData` | 500 simulations × 21 faults × 960 obs × 55 vars | Held-out fault eval |

Total uncompressed size ≈ 1.8 GB. The bulk binaries are git-ignored.

### Variables

The 52 process variables (XMEAS 1-41 + XMV 1-11) map across composition,
flow, level, temperature, pressure, and analyzer signals. Cross-domain
applicability to Skogestad Column A relies on the subset of compositions
(XMEAS 23-41) and the LV-analogue flow / level signals (XMEAS 1-22, XMV
1-11). Variable-by-variable mapping table will be added to
`docs/decisions/` as part of Phase 4 detector design.

## Companion sources (not bulk-downloaded by default)

- **Original Downs & Vogel (1993)**, *A plant-wide industrial process control
  problem*, Computers & Chemical Engineering 17(3):245-255. Defines the
  plant; the Fortran simulator is not strictly required for Phase 4 if the
  Rieth dataset suffices.
- **Bathelt, Ricker & Jelali (2015)**, *Revision of the Tennessee Eastman
  Process Model*, IFAC-PapersOnLine 48(8):309-314. Modernized state-space
  variant; not needed unless we extend to closed-loop fault injection.

These are referenced from the paper's Related Work, not pulled into the
repository.

## Acquisition procedure

The provenance-only path (default) is cheap and idempotent:

```bash
uv run python tools/acquire_tep_dataset.py
```

This writes:

- `data/external/tep/file_listing.json` — the full Dataverse file index
  with per-file SHA-256, byte-size, and download URL. Version-controlled.
- `data/external/tep/provenance.json` — DOI, accessed-at timestamp, the
  Dataverse-reported dataset version (e.g., V1).

For the bulk download (required before Phase 4 detector training):

```bash
uv run python tools/acquire_tep_dataset.py --download
```

This pulls all four `.RData` files, verifies each against the cached
SHA-256 from the file index, and writes them under `data/external/tep/`.
The files are git-ignored.

## Phase 4 detector pipeline (will be added in Phase 4)

- `src/industrial_ai/safety/tep_loader.py` — loads + parses the `.RData`
  files (via `rdata` package; pure Python, no R install required).
- `src/industrial_ai/safety/feature_pipeline.py` — maps the 52 TEP
  variables onto the cross-domain feature set used by both TEP and the
  Skogestad twin.
- `src/industrial_ai/safety/detector.py` — the anomaly detector itself
  (per ADR-005-style discussion to be opened in Phase 4).

## Data not committed

Per `.gitignore`:

```
data/external/**/*.RData
data/external/**/*.csv
data/external/**/*.dat
...
```

The provenance JSON files and this README are committed; everything else
is locally regenerable from the canonical source.
