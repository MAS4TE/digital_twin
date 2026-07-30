# PV modelling

The digital twin simulates rooftop photovoltaic (PV) generation natively:
[`run_digital_twin`](interfacing.md) accepts a `pv_profiles` argument and, when
it is provided, injects the PV as `sgen` elements and runs the power flow with
loads **and** PV together. When `pv_profiles` is `None`, no PV is injected and
the behaviour is identical to a loads-only run.

The profiles themselves are produced **upstream**, by a small set of helpers
that turn the site's real data (an installation register, PVGIS irradiance)
into the `pv_profiles` that the twin consumes. In the MAS4TE setting these
helpers are what **we** provide to fill the input files, since the agents do
not hold the Bressoux field data themselves — the twin only ever *receives*
`pv_profiles`, it never generates it.

```{mermaid}
flowchart LR
    N[(inputs/network.py)] --> M[generate_equipment_table]
    M --> C[(out/processed/equipment.csv)]
    A[(out/processed/pvgis_irradiance...parquet)] --> P(build_pv_production)
    C --> P
    P -->|pv_profiles, kW -> MW| D(process_data)
    D --> T(run_digital_twin)
    L[(out/processed/load_profiles.csv)] --> D
    T --> R[verdict: vm, violations, ...]
```

```{note}
The exact split of responsibilities between the data we provide and what the
MAS4TE agents supply is still being finalised with the MAS4TE team. The
description below assumes we generate the input files from field data.
```

## 1. Placement — the equipment table

**Where** the panels are is not decided in `pv.py`. Placement lives in
`digital_twin/data_processing/equipment.py`, the single bus-level register that
also carries the batteries, and is written to `out/processed/equipment.csv` (see
[Data files](data_files.md)). `pv.py` takes that register as an input and turns
it into power.

The split matters: placement is a scenario decision about the grid, production is
physics about the sun. Keeping placement in one register — shared with the
battery layer — is what stops a second, independent register from drifting out
of agreement with it and with the network.

```{code-block} python
from digital_twin.data_processing.equipment import generate_equipment_table

table = generate_equipment_table({"data_path": "data/bressoux",
                                  "n_installations": 30, "seed": 42})
```

Placement is grouped by street (a realistic neighbourhood/installer effect that
concentrates PV on a few feeders) rather than spread uniformly. The knobs:

| Key                   | Default   | Description                                              |
|-----------------------|-----------|----------------------------------------------------------|
| `n_installations`     | `30`      | Number of residential installations to place.            |
| `kwp_residential`     | `3.0`     | Fixed size (kWp) of each residential installation.       |
| `mv_pv_installations` | `{}`      | PV on MV load facilities, `{name -> kWp}`.               |
| `seed`                | `42`      | Seed for the (reproducible) street/bus draw.             |
| `n_streets_range`     | `(2, 4)`  | Number of streets forming the clusters.                  |
| `min_street_size`     | `5`       | Minimum residential buses for a street to be eligible.   |
| `street_weights`      | `{}`      | Optional per-street weighting of the draw.               |

`mv_pv_installations` is named by **role**, not by building type: what the model
needs is "an MV load bus carrying a large array", not the fact that Bressoux's
happens to be a rest home. A dict rather than a name/size pair, because a
network may carry several such facilities — a school, a sports hall, a
horticulture centre — each at its own rating. An unknown name raises, listing
the facilities the network actually has.

```{note}
`street_weights` and `mv_pv_installations` default to **neutral** values, not to
Bressoux's. They describe one specific neighbourhood, so the scenario passes its
own (see `main.py`: `{"Rue Ernest Malvoz": 3.0}` and
`{"Paradis du Bouhay": 12.0}`). Left in the module defaults they would be dead
weight on any other site, silently matching nothing.
```

To study a higher penetration on the *same* neighbourhood, extend an existing
fleet instead of drawing a fresh one — this preserves the original buses so the
comparison isolates the effect of penetration rather than mixing it with
placement:

```{code-block} python
from digital_twin.data_processing.equipment import extend_pv

table50 = extend_pv(table.copy(), n_additional=20, config=SCENARIO)
```

`extend_pv(table, n_additional, config=None, rng=None)` mutates and returns the
table, filling already-equipped streets first before opening new ones. Pass a
`.copy()` to keep the base fleet intact.

## 2. Irradiance → production — `build_pv_production`

Turns an irradiance series into a per-installation production profile, using
`pv_system_size_kwp` from the equipment table. The conversion model is
deliberately simple for v1 and isolated in a single function so it can be
swapped later:

```{code-block} python
from digital_twin.data_processing.equipment import load_equipment_table
from digital_twin.data_processing.pv import build_pv_production
import pandas as pd

equipment = load_equipment_table({"data_path": "data/bressoux"})
irradiance = pd.read_parquet("data/bressoux/out/processed/pvgis_irradiance_quarter_hourly.parquet")

pv_profiles_kw = build_pv_production("juin", equipment, irradiance)
```

| Argument        | Type                | Description                                                 |
|-----------------|---------------------|-------------------------------------------------------------|
| `mois`          | `str` or `int`      | French month name or `1`–`12`. Selects the season.          |
| `equipment`     | `pandas.DataFrame`  | The equipment table; rows with `pv_system_size_kwp > 0` are kept. |
| `irradiance`    | `pandas.DataFrame`  | Must contain `timestamp` and `ghi_w_m2`.                    |
| `system_factor` | `float` (`0.8`)     | System efficiency / normalisation factor.                   |

A table with no equipped row yields an **empty** frame — which is what makes
"no PV in this scenario" expressible without a flag.

**Returns** a `pandas.DataFrame` indexed `'Mon 00:00'..'Sun 23:45'` (672
quarter-hourly steps, same layout as `households`), with columns
`pv_{bus_id}` in **kW**. The chosen calendar week is recorded in
`out.attrs['source_week_start']`.

The underlying model is `irradiance_to_power(irradiance_w_m2, kwp,
system_factor=0.8)`, i.e. `power_kW = max(ghi, 0) / 1000 * kwp * system_factor`.

```{warning}
`build_pv_production` returns **kW**, whereas `run_digital_twin` expects
`pv_profiles` in **MW** (same unit as `households`). The conversion is done
by the caller — `pv_profiles = build_pv_production(...) / 1000.0`.
```

```{warning}
The stored irradiance is **global horizontal** (GHI), not in the plane of the
tilted array (POA). A single constant factor therefore cannot match both the
midday peak and the daily energy across seasons: with `system_factor=0.8` the
summer peak is realistic but summer daily energy is on the high side. This is
acceptable for **voltage/overvoltage** studies, which depend on the production
**peak** (correctly calibrated), not on cumulative energy. For energy or
self-consumption studies, replace `irradiance_to_power` with a POA/pvlib model
(or fetch PVGIS directly in the array plane).
```

## 3. Consuming PV in the digital twin

`process_data` (`digital_twin.data_processing.process_data`) wires this up
automatically: it loads `<data_path>/out/processed/equipment.csv` together with
the site's cached PVGIS irradiance
(`<data_path>/out/processed/pvgis_irradiance_quarter_hourly.parquet`, see
`build_irradiance_reference` above), calls `build_pv_production` for
`config['mois_simule']`, converts kW to MW, and returns it as `pv_profiles`.

A scenario opts into PV by **content**: any row with `pv_system_size_kwp > 0`.
If the table is absent or carries no PV, `pv_profiles` is `None` (loads-only)
with nothing else needed. Activation is carried by the register rather than by
the presence of a file, so one register describes the whole scenario and it
cannot be half-described.

The produced `pv_profiles` (already in MW) is passed to `run_digital_twin`.
When present, the twin verifies temporal alignment against the loads, injects
one `sgen` per PV column (generator convention: **positive `p_mw` =
injection**), and runs the power flow with loads and PV together. The verdict
then reflects the PV effect — in particular the **maximum** bus voltage, which
is the quantity that matters for PV (overvoltage), as opposed to the minimum
voltage for load stress.

```{code-block} python
from digital_twin import run_digital_twin
from digital_twin.data_processing.process_data import process_data

config = {"data_path": "data/bressoux/"}
metadata, network, households, pv_profiles, schedules_BSS, equipment = process_data(config)

out = run_digital_twin(config, metadata, network, households, pv_profiles=pv_profiles)
```

```{note}
Loads and PV must share the same temporal index: the profiles are mapped step
by step, so a misalignment would silently pair the wrong instants. The twin
checks this before the power flow — keep that guard in place.
```

See [Interfacing with MAS4TE](interfacing.md) for the full `run_digital_twin`
signature and return format, and [Load assignment](load_assignment.md) for how
`households` is produced.

## 4. Impact analysis — multi-scenario study

Beyond a single run, a ready-made **multi-scenario** study compares several PV
penetrations. It does **not** go through `run_digital_twin`; it uses a separate
orchestration layer (`build_base_network` / `run_pv_scenario` in
`digital_twin/data_processing/pv.py`) designed to build the network once and run
many scenarios efficiently on deep copies.

```{code-block} python
from digital_twin.data_processing.equipment import generate_equipment_table
from digital_twin.data_processing.pv import build_base_network, run_pv_scenario
import pandas as pd

metadata = pd.read_csv("data/bressoux/out/processed/metadata.csv")
equipment = generate_equipment_table(SCENARIO, write=False)
irradiance = pd.read_parquet("data/bressoux/out/processed/pvgis_irradiance_quarter_hourly.parquet")

network = build_base_network(SCENARIO)  # built once (~10-15 s)
result = run_pv_scenario(network, metadata, equipment, irradiance,
                         config={"mois_simule": "juin"})
```

The script `scripts/pv_impact_analysis.py` uses this layer to compare 0 / 30 /
50 / 80 installations (summer, plus a winter check); the resulting metrics and
figures are produced by `scripts/viz/plot_pv_impact_analysis.py`. It reports,
per configuration, the maximum bus voltage, the number of buses above the 1.05
pu and 1.10 pu limits, the location of any overvoltages, and reverse-flow
indicators.

The nested fleets are built **in place** with `extend_pv` on every run, so the
50-fleet provably contains the 30-fleet and no stored file can drift out of sync
with the network.

```{note}
`run_pv_scenario` and `build_base_network` are **analysis tooling only** — the
nominal way to simulate with PV is `run_digital_twin(..., pv_profiles=...)`
(section 3). The injection primitives themselves live in
`digital_twin/digital_twin.py`; `data_processing/pv.py` imports them and adds
the multi-scenario orchestration.
```

```{note}
Overvoltage on this network is a **local density** constraint, not a global
capacity one: thermal loading stays low even at high penetration, while
clustered PV raises the voltage on its feeder — including on nearby streets
that carry no PV of their own. Placement geography therefore matters as much as
the number of installations.
```

## Generating the input files

The data files consumed above are produced by:

```{code-block} bash
python -m digital_twin.data_processing.profiles   # bank + PVGIS irradiance parquets
python -c "from digital_twin.data_processing.equipment import generate_equipment_table; generate_equipment_table({'data_path': 'data/bressoux'})"
```

See [Data files](data_files.md) for the exact schema of each file.