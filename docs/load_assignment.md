# Load assignment

Consumption profiles reach the network in two decoupled stages. First, upstream
of the twin, a scenario's `out/processed/load_profiles.csv` is **generated**
from the real EAN smart-meter data. Then [`run_digital_twin`](interfacing.md)
**receives** that timeseries as its `households` argument, and its assignment
step maps the profiles onto the load buses. The twin never generates profiles
itself — it uses whatever `households` it is given, which in the MAS4TE
setting is how **we** supply realistic profiles from the Bressoux smart-meter
data (the agents may supply their own instead).

```{note}
The metered population is the project's target group of **social housing**
tenants. Consumption levels are accordingly modest and presence patterns
heterogeneous; the real profiles capture this faithfully, where generic
average-household profiles would not.
```

There are **two** generation mechanisms, used for different purposes:

- **Real-profile bank** (`generate_household_profiles`) — the primary
  mechanism, used by `bressoux_setup.ipynb` to produce `main.py`'s
  `households`. Agnostic to how many real EAN profiles exist; season-blind.
- **Monthly bank** (`build_consumption_bank` / `generate_load_profiles`) — an
  alternative, month-conditioned mechanism used where seasonal consistency
  with PV production matters (`pv.run_pv_scenario`,
  `scripts/pv_impact_analysis.py`).

## 1. Real-profile bank — `generate_household_profiles` (primary mechanism)

```{code-block} python
from digital_twin.data_processing.profiles import generate_household_profiles

households, metadata = generate_household_profiles(n_profiles=333, seed=42)
```

`generate_household_profiles(n_profiles, data_path="data/bressoux/", raw_file=..., seed=42)`
first calls `extract_real_weekly_profiles`, which averages each retained EAN's
readings (filtered on `total_cons >= 365` kWh) into ONE representative
672-step weekly profile (kW), keyed by day-of-week/time-of-day — this is the
33 real Bressoux profiles. It then generates `n_profiles` household profiles
from them, **agnostic to how many real profiles are actually available**:

- `n_profiles <= 33` (the real count) — draws **without** replacement, no
  perturbation: there's already enough real diversity.
- `n_profiles > 33` — draws **with** replacement, each draw shifted (±8 steps,
  ±2h, `SHIFT_MAX_STEPS`) and scaled (lognormal, σ=0.2, `SCALE_SIGMA`) so
  duplicated households don't look identical. This revives the historical
  `profil_conso.ipynb` `make_profile` mechanism, generalized to the
  agnostic case.

Returns `(households, metadata)` — `households` is the timeseries (index
`'Mon 00:00'..'Sun 23:45'`, columns `profile_{i}`, MW); `metadata` has one row
per profile, with the full MAS4TE household schema (see
[Data files](data_files.md#out-processed-metadata-csv)) defaulted to
`False`/`0.0`/unknown, plus `source_street`/`source_ean_index` for
traceability. `bressoux_setup.ipynb` writes both to
`out/processed/{load_profiles,metadata}.csv`.

```{warning}
This mechanism is season-blind (each EAN's profile is averaged over its whole
available history) and, in the `n_profiles > 33` branch, dilutes real peaks
through shift/scale perturbation (see "Why real weeks" below for why that
matters). It trades those for being trivially agnostic to how much real data
exists — reasonable for a general-purpose baseline scenario. Where seasonal
consistency with PV or undiluted peak fidelity matters more, use the monthly
bank instead (section 2).
```

## 2. Monthly bank — `build_consumption_bank` / `generate_load_profiles` (alternative mechanism)

### 2a. Building the bank — `build_consumption_bank`

Materialises the bank of real weekly profiles as a parquet file. Run once; the
result is reused by every simulation that uses this mechanism.

```{code-block} python
from digital_twin.data_processing.profiles import build_consumption_bank

bank = build_consumption_bank()   # writes data/bressoux/out/processed/consumption_bank_real_weeks.parquet
```

`build_consumption_bank(data_path="data/bressoux/", out_path="data/bressoux/out/processed/consumption_bank_real_weeks.parquet")`
extracts every complete calendar week (672 quarter-hourly steps, Monday→Sunday,
no gaps) from the retained meters (filtered on `total_cons >= 365` kWh) and tags
each week by EAN, month, season and street. See [Data files](data_files.md) for
the output schema.

The companion function `build_irradiance_reference(out_path=..., year=2023)`
fetches the matching PVGIS irradiance series used by the PV bricks (already
called by `bressoux_setup.ipynb`, since irradiance is needed regardless of
which load-generation mechanism is used).

### 2b. Generating `households` — `generate_load_profiles`

The upstream generator turns the bank into a ready-to-use
`out/processed/load_profiles.csv`.

```{code-block} python
from digital_twin.data_processing.profiles import generate_load_profiles

generate_load_profiles(6, metadata, data_path="data/bressoux/")  # June
```

`generate_load_profiles(mois_simule, metadata=None, config=None, out_path=None, data_path=...)`
writes `<data_path>/out/processed/load_profiles.csv` (index `'Mon 00:00'..'Sun 23:45'`,
columns `profile_{id}`, values in **MW**), which `process_data` then reads back
as `households`.

For each `profile_id`, a real week is drawn (with replacement, reproducible via
`seed`) from the sub-bank of the **same street** and of the month
`mois_simule`, so the consumption is both street-local and season-consistent.
No shift/scale perturbation is applied — load diversity comes from the natural
variety of the real weeks. A small anti-duplicate jitter (±1–2 quarter-hour
steps) is applied only when the same week is drawn more than once; if a street
has no week for that month, a global fallback is used and a warning is recorded.

```{note}
`mois_simule` keeps consumption seasonally consistent with PV production (which
reads the same month). Consumption and irradiance are matched at the **month**
level, not the exact calendar week, since the smart-meter and PVGIS reference
years differ.
```

## 3. Assigning profiles to buses (inside the twin)

Once `run_digital_twin` receives `households`, a **light street-based
association** maps a profile to each load bus (using `metadata`'s
`source_street`/`profile_id` columns): a bus on a street that has meters
receives a profile drawn at random among that street's profiles; a bus on a
street without meters (orphan) receives a profile from the whole set and is
flagged as a warning.

- Each load is modelled in **active and reactive power**: reactive power is
  derived from the active profile using a fixed residential power factor
  (cos φ = 0.95).
- Profiles are at a fixed **15-minute** step (672 steps per week) in MW/MVAr,
  matching pandapower's units. Near-zero loads are clipped to a small positive
  value to avoid power-flow convergence issues.

There is no runtime load-scaling knob: a different scenario (e.g. a 3x-load
stress test) is a different `data/bressoux/<scenario>/` folder with its own
generated `households`, produced via the setup notebook, not a multiplier
applied at execution time.

## Why real weeks (monthly bank rationale)

The monthly bank (section 2) replaces an earlier method based on
weekday/time-of-day **averaged** profiles perturbed by a random time shift and
lognormal scaling. Two problems motivated the change:

- the averaged method was **season-blind** — it produced identical profiles
  regardless of the simulated month;
- the shift/scale perturbation **flattened the aggregate peak**, because it
  desynchronised profiles as much as it diversified them, underestimating the
  winter peak.

Real weeks carry authentic diversity (each household's real peaks at their real
times), which reproduces a more faithful aggregate peak without any synthetic
perturbation.

```{note}
**Known limitation.** Because profiles are drawn **independently** per bus, the
aggregate load factor stays around 0.64, above the real ~0.25 — the independent
draw dilutes the natural coincidence of household peaks. Recovering the true
value would require a "simultaneous blocks" scheme (grouping real EANs from the
same calendar weeks). This becomes relevant when off-peak dynamics matter, e.g.
for detailed PV self-consumption studies.
```

## Regenerating the profiles

```{code-block} bash
jupyter execute data/bressoux/bressoux_setup.ipynb   # primary mechanism: metadata.csv,
                                                       # load_profiles.csv, mv_load_profiles.csv
python -m digital_twin.data_processing.profiles       # alternative: bank + irradiance parquets
```

`generate_load_profiles` (section 2b) is then called to produce a scenario's
`out/processed/load_profiles.csv` for the chosen month, if using the monthly
bank mechanism instead of the notebook's primary one.
