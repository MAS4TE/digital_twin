# Load assignment

Consumption profiles reach the network in two decoupled stages. First, upstream
of the twin, a `load_profiles.csv` is **generated** from a bank of real weeks.
Then [`run_digital_twin`](interfacing.md) **receives** that `load_profiles` and
its assignment step maps the profiles onto the load buses. The twin never
generates profiles itself — it uses whatever `load_profiles` it is given, which
in the MAS4TE setting is how **we** supply realistic profiles from the Bressoux
smart-meter data (the agents may supply their own instead).

```{note}
The metered population is the project's target group of **social housing**
tenants. Consumption levels are accordingly modest and presence patterns
heterogeneous; the real weeks capture this faithfully, where generic
average-household profiles would not.
```

## 1. Building the bank — `build_consumption_bank`

Materialises the bank of real weekly profiles as a parquet file. Run once; the
result is reused by every simulation.

```{code-block} python
from digital_twin.consumption_bank import build_consumption_bank

bank = build_consumption_bank()   # writes data/consumption_bank_real_weeks.parquet
```

`build_consumption_bank(data_path="data/", out_path="data/consumption_bank_real_weeks.parquet")`
extracts every complete calendar week (672 quarter-hourly steps, Monday→Sunday,
no gaps) from the retained meters (filtered on `total_cons >= 365` kWh) and tags
each week by EAN, month, season and street. See [Data files](data_files.md) for
the output schema.

The companion function `build_irradiance_reference(out_path=..., year=2023)`
fetches the matching PVGIS irradiance series used by the PV bricks.

## 2. Generating `load_profiles` — `generate_load_profiles`

The upstream generator turns the bank into a ready-to-use `load_profiles.csv`.
This is the function `main.py` calls before running the twin.

```{code-block} python
from digital_twin.consumption_bank import generate_load_profiles

generate_load_profiles(6, households, out_path="data/load_profiles.csv")  # June
```

`generate_load_profiles(mois_simule, households=None, config=None, out_path=..., data_path=...)`
writes `data/load_profiles.csv` (index `'Mon 00:00'..'Sun 23:45'`, columns
`profile_{id}`, values in **MW**), which the twin then reads back.

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

Once `run_digital_twin` receives `load_profiles`, a **light street-based
association** maps a profile to each load bus: a bus on a street that has meters
receives a profile drawn at random among that street's profiles; a bus on a
street without meters (orphan) receives a profile from the whole set and is
flagged as a warning.

- Each load is modelled in **active and reactive power**: reactive power is
  derived from the active profile using a fixed residential power factor
  (cos φ = 0.95).
- Profiles are at a fixed **15-minute** step (672 steps per week) in MW/MVAr,
  matching pandapower's units. Near-zero loads are clipped to a small positive
  value to avoid power-flow convergence issues.
- The stress multiplier `config['load_scale']` is applied to the received
  `load_profiles` before assignment, so it scales the actual simulated loads.

## Why real weeks

The bank replaces an earlier method based on weekday/time-of-day **averaged**
profiles perturbed by a random time shift and lognormal scaling. Two problems
motivated the change:

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

## Regenerating the bank and profiles

```{code-block} bash
python -m digital_twin.consumption_bank   # bank + irradiance parquets
```

`main.py` then calls `generate_load_profiles` to produce `load_profiles.csv`
for the chosen month before running the twin.