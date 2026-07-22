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
    M[generate_pv_metadata] --> C[(pv_installations.csv)]
    A[(PVGIS irradiance)] --> P(build_pv_production)
    C --> P
    P -->|pv_profiles, kW| K[main.py: kW -> MW]
    K --> T(run_digital_twin)
    L[(load_profiles)] --> T
    T --> R[verdict: vm, violations, ...]
```

```{note}
The exact split of responsibilities between the data we provide and what the
MAS4TE agents supply is still being finalised with the MAS4TE team. The
description below assumes we generate the input files from field data.
```

## 1. Installation metadata — `generate_pv_metadata`

Decides **which buses carry a PV installation and of what size**, and writes
`data/pv_installations.csv`. Placement is grouped by street (a realistic
neighbourhood/installer effect that concentrates PV on a few feeders) rather
than spread uniformly.

```{code-block} python
from sourcecode.pv_metadata import generate_pv_metadata

pv = generate_pv_metadata(config={"n_installations": 30, "seed": 42})
```

`config` is a dict; the defaults are:

| Key                | Default                          | Description                                              |
|--------------------|----------------------------------|----------------------------------------------------------|
| `n_installations`  | `30`                             | Number of residential installations to place.            |
| `kwp_residential`  | `3.0`                            | Fixed size (kWp) of each residential installation.       |
| `kwp_rest_home`    | `12.0`                           | Size (kWp) of the rest home, added separately.           |
| `seed`             | `42`                             | Seed for the (reproducible) street/bus draw.             |
| `n_streets_range`  | `(2, 4)`                         | Number of streets forming the clusters.                  |
| `min_street_size`  | `5`                              | Minimum residential buses for a street to be eligible.   |
| `street_weights`   | `{"Rue Ernest Malvoz": 3.0}`     | Optional per-street weighting of the draw.               |
| `rest_home_name`   | `"Paradis du Bouhay"`            | MV node receiving the rest-home installation.            |
| `out_path`         | `"data/pv_installations.csv"`    | Output CSV path.                                          |

**Returns** the written `pandas.DataFrame` (columns `bus_id, kwp, type, street`).

To study a higher penetration on the *same* neighbourhood, extend an existing
park instead of drawing a fresh one — this preserves the original buses so the
comparison isolates the effect of penetration:

```{code-block} python
from sourcecode.pv_metadata import extend_pv_metadata

pv50 = extend_pv_metadata(pv, n_additional=20, out_path="data/pv_installations_50.csv")
```

`extend_pv_metadata(base_installations, n_additional, config=None, out_path=None)`
keeps `base_installations` unchanged and adds `n_additional` installations,
filling already-equipped streets first before drawing new ones.

## 2. Irradiance → production — `build_pv_production`

Turns an irradiance series into a per-installation production profile, using
the `kwp` from the metadata. The conversion model is deliberately simple for
v1 and isolated in a single function so it can be swapped later:

```{code-block} python
from sourcecode.pv_production import build_pv_production
import pandas as pd

pv_installations = pd.read_csv("data/pv_installations.csv")
irradiance = pd.read_parquet("data/pvgis_irradiance_quarter_hourly.parquet")

pv_profiles_kw = build_pv_production("juin", pv_installations, irradiance)
```

| Argument           | Type                | Description                                              |
|--------------------|---------------------|----------------------------------------------------------|
| `mois`             | `str` or `int`      | French month name or `1`–`12`. Selects the season.       |
| `pv_installations` | `pandas.DataFrame`  | Must contain `bus_id` and `kwp`.                         |
| `irradiance`       | `pandas.DataFrame`  | Must contain `timestamp` and `ghi_w_m2`.                |
| `system_factor`    | `float` (`0.8`)     | System efficiency / normalisation factor.               |

**Returns** a `pandas.DataFrame` indexed `'Mon 00:00'..'Sun 23:45'` (672
quarter-hourly steps, same layout as `load_profiles`), with columns
`pv_{bus_id}` in **kW**. The chosen calendar week is recorded in
`out.attrs['source_week_start']`.

The underlying model is `irradiance_to_power(irradiance_w_m2, kwp,
system_factor=0.8)`, i.e. `power_kW = max(ghi, 0) / 1000 * kwp * system_factor`.

```{warning}
`build_pv_production` returns **kW**, whereas `run_digital_twin` expects
`pv_profiles` in **MW** (same unit as `load_profiles`). The conversion is done
by the caller — in `main.py`, `pv_profiles = build_pv_production(...) / 1000.0`.
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

The produced `pv_profiles` (converted to MW) is passed to `run_digital_twin`.
When present, the twin verifies temporal alignment against the loads, injects
one `sgen` per PV column (generator convention: **positive `p_mw` =
injection**), and runs the power flow with loads and PV together. The verdict
then reflects the PV effect — in particular the **maximum** bus voltage, which
is the quantity that matters for PV (overvoltage), as opposed to the minimum
voltage for load stress.

```{code-block} python
import pandas as pd
from sourcecode.digital_twin import run_digital_twin

households = pd.read_csv("data/households.csv")
load_profiles = pd.read_csv("data/load_profiles.csv", index_col=0)   # MW
# pv_profiles already in MW (see conversion above)

out = run_digital_twin({"load_scale": 1}, households, load_profiles,
                       pv_profiles=pv_profiles)
```

```{note}
Loads and PV must share the same temporal index: the profiles are mapped step
by step, so a misalignment would silently pair the wrong instants. The twin
checks this before the power flow — keep that guard in place.
```

See [Interfacing with MAS4TE](interfacing.md) for the full `run_digital_twin`
signature and return format, and [Load assignment](load_assignment.md) for how
`load_profiles` is produced.

## 4. Impact analysis — multi-scenario study

Beyond a single run, a ready-made **multi-scenario** study compares several PV
penetrations. It does **not** go through `run_digital_twin`; it uses a separate
orchestration layer (`build_base_network` / `run_pv_scenario` in
`sourcecode/pv_injection.py`) designed to build the network once and run many
scenarios efficiently on deep copies.

```{code-block} python
from sourcecode.pv_injection import build_base_network, run_pv_scenario
import pandas as pd

households = pd.read_csv("data/households.csv")
pv_installations = pd.read_csv("data/pv_installations.csv")
irradiance = pd.read_parquet("data/pvgis_irradiance_quarter_hourly.parquet")

net_base, buses, mv_nodes, validation = build_base_network()  # built once (~10-15 s)
result = run_pv_scenario(net_base, buses, mv_nodes, households,
                         pv_installations, irradiance,
                         config={"mois_simule": "juin"})
```

The script `scripts/pv_impact_analysis.py` uses this layer to compare 0 / 30 /
50 / 80 installations (summer, plus a winter check). It reports, per
configuration, the maximum bus voltage, the number of buses above the 1.05 pu
and 1.10 pu limits, the location of any overvoltages, and reverse-flow
indicators.

```{note}
`run_pv_scenario` and `build_base_network` are **analysis tooling only** — the
nominal way to simulate with PV is `run_digital_twin(..., pv_profiles=...)`
(section 3). The injection primitives themselves live in `digital_twin.py`;
`pv_injection.py` imports them and adds the multi-scenario orchestration.
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
python -m sourcecode.consumption_bank   # bank + PVGIS irradiance parquets
python -m sourcecode.pv_metadata        # data/pv_installations.csv
```

See [Data files](data_files.md) for the exact schema of each file.