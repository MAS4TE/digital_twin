# MAS4TE_ULIEGE

Digital twin of a low-voltage distribution network, developed as part of the
**MAS4TE** project -- Bressoux (Liège) is the first site it was built for. The
core entry point is `run_digital_twin` (`digital_twin/digital_twin.py`), a
network-agnostic engine: it takes a network, household metadata, consumption
profiles and PV production and a battery schedule as arguments
(the network itself is a caller-supplied input, produced by a site-specific
`inputs/network.py`, not built into the package) and returns a network verdict
(voltages, loadings, violations). This is what makes it reusable for a
different LV network without touching `digital_twin/`.

The battery layer models **one centralised storage park on an MV node**, not
distributed household batteries: households hold contractual *shares* of that
single park. See the *Interfacing* page of the documentation for the schedule
format and its conventions.

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

## Scenario layout: `data/<site>/`

Each site/scenario (e.g. `data/bressoux/`) is a self-contained folder, split
in two:

```
data/bressoux/
  inputs/                  # TRUE inputs -- hand-provided or long-lived reference
                            # data, never overwritten by the pipeline itself
    network.py                 # site-specific topology builder -- exposes build_network(config),
                                # loaded dynamically by digital_twin.utils.load_site_network
    coordinates.csv             # building coordinates (read by network.py)
    part-*.csv                  # raw smart-meter readings (anonymised EAN)
    pv_installations.csv        # OPTIONAL -- PV installation register (bus_id, kwp, type, street);
                                # a scenario opts into PV simply by having this file
    bss_shares.csv              # OPTIONAL -- BSS contractual share register (bus_id, e_kwh, street);
                                # a scenario opts into the battery simply by having this file
  out/
    processed/                # DERIVED data, regenerated from inputs/ by the site's
                                # data pipeline (e.g. bressoux_setup.ipynb):
      metadata.csv                 # per-household attributes (profile_id, source_street, ...)
      load_profiles.csv            # household consumption timeseries -- the `households` argument
      mv_load_profiles.csv         # the MV load facility profiles
      pvgis_irradiance_quarter_hourly.parquet   # cached PVGIS irradiance, needed alongside
                                                 # pv_installations.csv to build `pv_profiles`
    results/                   # run reports (see save_run_report below)
  bressoux_setup.ipynb      # notebook that builds inputs/ -> out/processed/
```

`inputs/network.py` is what a new site replaces to reuse the digital twin
engine on a different neighbourhood -- everything under `digital_twin/` stays
network-agnostic.

## Input data: sources vs. generated files

The raw field data behind a site's `inputs/` files is **not** distributed with
the repository:

- **Source files** you must provide yourself, under `inputs/` (the raw field
  data): the raw smart-meter CSV (anonymised EAN readings, `part-*.csv`) and
  `coordinates.csv` (building coordinates).
- **Generated files**, under `out/processed/` (`metadata.csv`,
  `load_profiles.csv`, `mv_load_profiles.csv`, the consumption bank, the
  cached PVGIS irradiance, ...) produced from the sources by the setup
  notebook / scripts below — you do **not** provide these, you regenerate them.

See the *Data files* page of the documentation for the exact schema of each
generated file.

## Getting started: setting up the Bressoux scenario and running the twin

1. **Set up the Bressoux scenario** — run
   `data/bressoux/bressoux_setup.ipynb` in Jupyter (with the repository root
   as the working directory). It builds the network, visualizes it on a real
   map, and writes `out/processed/metadata.csv` and `load_profiles.csv` (from
   the 33 real EAN smart-meter profiles), `mv_load_profiles.csv` (the 4 MV
   load facility profiles), and the cached PVGIS irradiance reference needed
   for PV.

2. **Run the twin.**

   ```bash
   python main.py
   ```

   If `inputs/pv_installations.csv` exists, PV production is built and
   injected automatically -- nothing else to configure. Same rule for the
   battery: if `inputs/bss_shares.csv` exists, the park schedule is built from
   it and injected as a storage element, and the run report gains the `bss_*`
   metrics (bus, state of charge, energy charged/discharged). Either file
   missing simply means that layer is off. `config['save']` (already `True` in
   `main.py`) writes a Markdown run report (summary/violations/warnings) to
   `out/results/`.

The steps below are only needed for the separate monthly-bank / PV
impact-analysis tooling (`scripts/pv_impact_analysis.py`), not for the basic
`main.py` run above:

3. **Consumption bank** (every complete real calendar week from the meter
   data, tagged by month/season/street -- an alternative, month-conditioned
   source of `households`):

   ```bash
   python -m digital_twin.data_processing.profiles
   ```

   Writes `out/processed/consumption_bank_real_weeks.parquet` (the PVGIS
   irradiance reference is already written by step 1).

4. **PV installation register** (which buses carry PV, and their size --
   only needed to generate a *new* `pv_installations.csv`; the one already in
   `inputs/` is used as-is otherwise):

   ```bash
   python -m digital_twin.data_processing.pv
   ```

   Writes `inputs/pv_installations.csv`.

5. **BSS share register** (which buses hold a share of the central park, and
   how much energy each -- only needed to generate a *new* `bss_shares.csv`;
   the one already in `inputs/` is used as-is otherwise):

   ```bash
   python -c "from digital_twin.data_processing.bss import generate_bss_shares; generate_bss_shares({'data_path': 'data/bressoux'})"
   ```

   Writes `inputs/bss_shares.csv` -- one share per LV bus, of equal energy,
   summing to `config['bss_park_e_kwh']`.

Once the generated files exist, only step 2 needs to be repeated for
day-to-day runs; steps 1 and 3 to 5 are one-off (re-run them only if the
sources change).

## Repository layout

- `digital_twin/` — the network-agnostic engine (`digital_twin.py`,
  `run_digital_twin`, `save_run_report`), shared helpers (`utils.py`), and the
  upstream data generators (`data_processing/profiles.py`,
  `data_processing/pv.py`, `data_processing/bss.py`,
  `data_processing/process_data.py`).
- `data/<site>/inputs/network.py` — the site-specific topology builder (NOT
  part of the installable package — this is what a new site replaces).
- `main.py` — orchestration/demo: load a scenario's inputs, then run the twin.
- `scripts/` — impact analysis and figures (run from `scripts/`, not from the
  repository root).
- `docs/` — Sphinx documentation.
- `archives/` — superseded scripts, and one-off checks whose proof is done,
  kept for reference (see `archives/README.md`).
