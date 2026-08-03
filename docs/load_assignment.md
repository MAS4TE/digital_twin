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

Generation is a **single** mechanism, the monthly bank of real EAN weeks:
`build_consumption_bank` materialises it once, then
`generate_household_metadata` + `generate_load_profiles` draw a scenario's
profiles from the month being simulated. `bressoux_setup.ipynb` runs both, and
`pv.run_pv_scenario` uses the same draw internally.

## 1. Building the bank — `build_consumption_bank`

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

The companion function `build_irradiance_reference(config)` fetches the matching
PVGIS irradiance series used by the PV bricks (also called by
`bressoux_setup.ipynb`). It takes the scenario config because the coordinates
are **derived** from the site's own network, not hard-coded.

## 2. Generating `metadata` and `households` — `generate_load_profiles`

The two files come out of **one** act, because the draw is what decides which
real week — hence which street and which month — backs each `profile_id`:

```{code-block} python
from digital_twin.data_processing.profiles import (
    generate_household_metadata, generate_load_profiles)
from digital_twin.utils import load_site_network, _config_to_dict

network = load_site_network(_config_to_dict({"data_path": "data/bressoux"}))
metadata = generate_household_metadata(network)               # one row per LV bus
households, metadata, warnings = generate_load_profiles(
    6, metadata, config=SCENARIO, data_path="data/bressoux/")   # June
```

`generate_household_metadata(network)` reads no meter data at all: it takes the
LV buses from the network and writes `profile_id`, `bus_id`, the MAS4TE
household schema at its defaults, and empty `source_*` columns. A site that can
supply a bank parquet therefore never needs to own a raw meter file.

`generate_load_profiles` then writes **both**
`out/processed/load_profiles.csv` (index `'Mon 00:00'..'Sun 23:45'`, columns
`profile_{id}`, values in **kW**) and an updated `metadata.csv` whose
`source_street`, `source_month`, `source_week_start` and `source_ean_index`
describe the weeks actually drawn. Writing one without the other is how the two
silently drift apart.

For each `profile_id` — hence for each LV bus, the two being 1:1 — a real week
is drawn from the **whole** sub-bank of month `mois_simule`, all streets
together. No shift/scale perturbation is applied: load diversity comes from the
natural variety of the real weeks.

The draw is **cyclic**, not with replacement (`_cyclic_draw`): the pool is
permuted, walked in order, and re-permuted only when exhausted. So every real
week is used before any is reused, and reuse is uniform at
`floor`/`ceil(n_buses / pool)`. On June that is 133/133 weeks used at a max
reuse of 3, where independent draws with replacement left 15 weeks unused while
another served 7 buses. When the pool is larger than the bus count, a random
subset is taken and nothing is reused.

A deterministic anti-duplicate jitter (`BANK_JITTER_LADDER`, ±1 step then ±2, up
to ±8) is applied only to the extra occurrences, and a `[LOAD/draw]` warning
always reports how many of the available weeks were used and the maximum reuse.

```{warning}
**The real ceiling is 33 households, not the week count.** The Bressoux bank
holds 1345 weeks, but from only 33 metered EANs over 50 calendar weeks. No
single month reaches 333 weeks (September peaks at 160, June has 133), so a
draw *without any reuse at all* is impossible at month granularity; only two of
the four seasons (Été 375, Automne 354) would suffice, at the cost of the month
conditioning. And even then those weeks would come from ~32 households, i.e.
~10 buses per real household. Different weeks of the same household are
genuinely different series — far better than a rolled copy — but the underlying
population is what it is. This is a limit of the field data, not of the code.
```

```{note}
**Why the draw is global, not per street.** Restricting it to the profile's own
street would cost a lot and buy nothing on the Bressoux bank: only 3 of the 7
network streets carry a metered EAN, so in June 83 Rue Ernest Malvoz buses would
share 8 candidate weeks (10.4× reuse) against a global monthly pool of 133 — and
a per-street draw guarantees the duplicates are *electrical neighbours*, so their
peaks add up on the very same LV feeder. Meanwhile street barely predicts
consumption here: dispersion of per-EAN weekly energy is ~7 kWh *between* streets
against ~21 kWh *within* one. Drawing globally keeps reuse at ~2.5× and scatters
the duplicates.
```

```{note}
`mois_simule` keeps consumption seasonally consistent with PV production (which
reads the same month). `metadata['source_month']` records it, and
`run_digital_twin` raises a `[LOAD/season]` warning when a run simulates a
different month than the one the profiles were drawn from — January consumption
under June sun is otherwise a plausible-looking run whose net load is wrong all
week. Consumption and irradiance are matched at the **month** level, not the
exact calendar week, since the smart-meter and PVGIS reference years differ.
```

## 3. Assigning profiles to buses (inside the twin)

There is **one household per LV charge bus**, and the assignment is *declared*,
not drawn: `generate_household_metadata(network)` writes a `bus_id` column, 1:1
with `profile_id` in ascending bus order, and `_assign_loads` simply looks it
up. `_validate_inputs` checks that column against the network with
`check_bus_ids(..., exhaustive=True)` — every LV bus exactly once, no
duplicates, no strays.

```{note}
Declaring the assignment upstream is what keeps the draw of section 2 intact:
all 333 profiles reach the network, one bus each, and the 133 June weeks are
reused at most 3×. Drawing a *second* time inside `_assign_loads` would undo
that — and since the only pool key available there is the street, the duplicates
would be **electrical neighbours** whose peaks add up on the very same LV feeder.
The anti-duplicate jitter cannot help in that case: it separates profiles from
each other, but a profile *copied* onto several buses reaches them all unchanged.
```

A caller supplying its own `metadata` without `bus_id` (a MAS4TE agent shipping
only profiles) falls back to a seeded street-filtered draw, with a
`[LOAD/assign]` warning naming how many buses it had to draw for.

- Each load is modelled in **active and reactive power**: reactive power is
  derived from the active profile using a fixed residential power factor
  (cos φ = 0.95).
- Profiles are at a fixed **15-minute** step (672 steps per week), in **kW**
  at the `households`/`load_profiles.csv` interface; `_assign_loads` converts
  to MW/MVAr internally, matching pandapower's units, right before assignment.
  Near-zero loads are clipped to a small positive value to avoid power-flow
  convergence issues.

```{note}
**The clipping floor erases the diversity of near-vacant dwellings.** In June,
29 of the 333 drawn weeks stay entirely below the 1e-4 MW (0.1 kW) floor — real
meters on dwellings consuming at most 8 kWh over the whole week. They are 29
*distinct* series before the clip and one constant series after it, so those
buses come out bit-identical and their `argmax` "peak" is meaningless. Harmless
for the network verdict (2.9 kW total on a 221 kW peak, and the bias is upward,
i.e. conservative), but worth knowing before reading any per-bus statistic.
```

There is no runtime load-scaling knob: a different scenario (e.g. a 3x-load
stress test) is a different `data/bressoux/<scenario>/` folder with its own
generated `households`, produced via the setup notebook, not a multiplier
applied at execution time.

## Why real weeks (monthly bank rationale)

Real weeks carry authentic diversity — each household's real peaks at their real
times — which reproduces a faithful aggregate peak without any synthetic
perturbation. Two properties follow directly, and neither survives a
weekday/time-of-day **averaged** profile perturbed by a random shift and scale:

- the draw is **season-aware**: profiles come from the sub-bank of the simulated
  month, so the loads can be seasonally consistent with the PV;
- the **aggregate peak is not flattened**: a shift/scale perturbation
  desynchronises profiles as much as it diversifies them, which understates the
  winter peak.

```{note}
**Known limitation.** Because profiles are drawn **independently** per bus, the
aggregate load factor stays around 0.59, well above the real ~0.25 — the
independent draw dilutes the natural coincidence of household peaks. Recovering
the true value would require a "simultaneous blocks" scheme (grouping real EANs
from the same calendar weeks). This becomes relevant when off-peak dynamics
matter, e.g. for detailed PV self-consumption studies.
```

## Regenerating the profiles

```{code-block} bash
jupyter execute data/bressoux/bressoux_setup.ipynb   # everything: bank, metadata.csv,
                                                       # load_profiles.csv, mv_load_profiles.csv,
                                                       # irradiance, equipment.csv
```

The notebook is the single setup path — it builds the bank if missing, draws the
profiles for its `MOIS_SIMULE`, and places the equipment. Keep that constant
equal to `main.py`'s `config['mois_simule']`; if they differ the run still works
but reports a `[LOAD/season]` warning.

`python -m digital_twin.data_processing.profiles` remains available to rebuild
just the bank and irradiance parquets without the rest.
