# MAS4TE_ULIEGE

Digital twin of the Bressoux neighbourhood distribution network, developed as
part of the **MAS4TE** project. The core entry point is `run_digital_twin`
(`digital_twin/digital_twin.py`), which takes households, load and PV profiles as
`pandas` DataFrames and returns a network verdict (voltages, loadings,
violations).

## Using it as a dependency

`digital_twin` is a standard Python package (`pyproject.toml`), so other
projects — e.g. the MAS4TE multi-agent framework — can depend on it directly
from this repository:

```bash
pip install "git+https://github.com/bcornelusse/MAS4TE_ULIEGE.git@package-version"
```

```python
from digital_twin import run_digital_twin
```

## Documentation

Built locally with Sphinx:

```bash
cd docs
make html
open docs/_build/html/index.html
```

## Set up

1. Create or load a virtual environment for Python 3.10+.
2. Install the package (editable install, so edits to `digital_twin/` are
   picked up without reinstalling):

   ```bash
   pip install -e .
   ```

   To also pull the Sphinx documentation dependencies:

   ```bash
   pip install -e ".[docs]"
   ```

   This installs `digital_twin` as an importable package (`import
   digital_twin`) with pinned runtime dependencies, as declared in
   `pyproject.toml`. `requirements.txt` is kept for reference/CI parity but
   `pyproject.toml` is now the source of truth.

3. Create a `data/` folder at the repository root (**you have to create it** —
   it is git-ignored to avoid polluting the repo with large data files).

## Input data: sources vs. generated files

The `data/` folder is **not** distributed with the repository. It contains two
kinds of files:

- **Source files** you must provide yourself (the raw field data):
  - the raw smart-meter CSV (anonymised EAN readings, `part-*.csv`);
  - `coordonnées.csv` (building coordinates).
- **Generated files** produced from the sources by the scripts below — you do
  **not** provide these, you regenerate them.

See the *Data files* page of the documentation for the exact schema of each
generated file.

## Getting started: regenerating the data and running the twin

Starting from the two source files above placed in `data/`, run these steps
**in order** (each depends on the previous ones):

1. **Consumption bank + irradiance** (real-week bank from the meter data, plus
   the PVGIS irradiance series):

   ```bash
   python -m digital_twin.consumption_bank
   ```

   Writes `data/consumption_bank_real_weeks.parquet` and
   `data/pvgis_irradiance_quarter_hourly.parquet`.

2. **Households and MV passage profiles** — produced by the notebook
   `profil_conso.ipynb` (open it in Jupyter and run all cells). Writes
   `data/households.csv` and `data/mv_passage_profiles.csv`, which the twin
   needs.

   > Note: this notebook also writes an averaged `load_profiles.csv` (legacy
   > method). You can ignore it — at step 4, `main.py` regenerates
   > `load_profiles.csv` from the monthly bank, which is the version actually used.

3. **PV installation register** (which buses carry PV, and their size):

   ```bash
   python -m digital_twin.pv_metadata
   ```

   Writes `data/pv_installations.csv`.

4. **Run the twin.** `main.py` generates `load_profiles.csv` from the bank for
   the chosen month, then runs three scenarios (normal, load-stress ×2, and a
   PV run) and prints their verdicts:

   ```bash
   python main.py
   ```

Once the generated files exist, only step 4 needs to be repeated for day-to-day
runs; steps 1–3 are one-off (re-run them only if the sources change).

## Repository layout

- `digital_twin/` — the digital twin (`digital_twin.py`) and the upstream
  generators (`consumption_bank.py`, `pv_metadata.py`, `pv_production.py`) plus
  the multi-scenario PV tooling (`pv_injection.py`).
- `main.py` — orchestration/demo: generate profiles, then run the twin.
- `scripts/` — verification and visualisation scripts (checks, impact analysis,
  figures).
- `docs/` — Sphinx documentation.
- `archives/` — superseded scripts kept for reference.
