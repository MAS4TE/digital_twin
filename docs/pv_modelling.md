# PV modelling
 
The digital twin can simulate rooftop photovoltaic (PV) generation on top of
the household loads. PV is **not** built into
[`run_digital_twin`](interfacing.md); instead, a small set of helpers turns
the site's real data (PVGIS irradiance, an installation register) into the
`pv_profiles` and metadata that a simulation consumes. In the MAS4TE setting
these helpers are what **we** provide to fill the input files, since the
agents do not hold the Bressoux field data themselves.
 
The volume is organised as a pipeline of independent bricks:
 
```{mermaid}
flowchart LR
    A[(PVGIS irradiance)] --> P(build_pv_production)
    M[generate_pv_metadata] --> C[(pv_installations.csv)]
    C --> P
    C --> S(run_pv_scenario)
    P --> S
    H[(households)] --> S
    N[build_base_network] --> S
    S --> R[timeseries: vm, line_load, ...]
```
 
Each brick is usable on its own, which makes the intermediate results easy to
inspect and verify.
 
```{note}
The exact split of responsibilities between the data we provide and what the
MAS4TE agents supply is still being finalised with the MAS4TE team. The
description below assumes we generate the input files from field data.
```
 
## 1. Installation metadata — `generate_pv_metadata`
 
Decides **which buses carry a PV installation and of what size**, and writes
`data/pv_installations.csv`. The placement is grouped by street (a realistic
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
 
pv_profiles = build_pv_production("juin", pv_installations, irradiance)
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
The stored irradiance is **global horizontal** (GHI), not in the plane of the
tilted array (POA). A single constant factor therefore cannot match both the
midday peak and the daily energy across seasons: with `system_factor=0.8` the
summer peak is realistic but summer daily energy is on the high side. This is
acceptable for **voltage/overvoltage** studies, which depend on the production
**peak** (correctly calibrated), not on cumulative energy. For energy or
self-consumption studies, replace `irradiance_to_power` with a POA/pvlib model
(or fetch PVGIS directly in the array plane).
```
 
## 3. Base network and scenarios — `pv_injection`
 
PV is injected as pandapower `sgen` elements (generator convention: **positive
`p_mw` = injection**). The network is expensive to build (it reads the raw
335 MB CSV, ~10–15 s) but stable across scenarios, so it is built **once** and
each scenario runs on a deep copy.
 
```{code-block} python
import pandas as pd
from sourcecode.pv_injection import build_base_network, run_pv_scenario
 
households = pd.read_csv("data/households.csv")
pv_installations = pd.read_csv("data/pv_installations.csv")
irradiance = pd.read_parquet("data/pvgis_irradiance_quarter_hourly.parquet")
 
# Built once (~10-15 s); reuse across scenarios.
net_base, buses, mv_nodes, validation = build_base_network()
 
# Each scenario runs on a deepcopy of net_base (never mutates it).
result = run_pv_scenario(
    net_base, buses, mv_nodes, households, pv_installations, irradiance,
    config={"mois_simule": "juin"},   # drives BOTH consumption and PV month
)
 
print(result["vm"].shape)         # (672, n_bus) -- voltages per step, pu
print(result["vm"].values.max())  # max voltage over the week
```
 
`build_base_network(config=None)` returns `(net_base, buses, mv_nodes,
validation)`, where `buses` maps `(street, num) -> bus_id` (333 residential
buses) and `mv_nodes` maps name `-> bus_id` (slack + substations + MV pass-through).
 
`run_pv_scenario(net_base, buses, mv_nodes, households, pv_installations,
irradiance, config=None, run_powerflow=True)` deep-copies the base network,
assigns the monthly-bank loads for `config['mois_simule']`, injects the PV
`sgen`, checks temporal alignment, and (by default) runs the power flow.
 
**Returns** a dict always containing `net`, `sgen_bus`, `bus_to_profile`,
`df_load_p`, `df_load_q`, `df_sgen_p`, `warnings`; when `run_powerflow=True` it
also contains the time-series below.
 
| Key           | Shape / unit                | Description                              |
|---------------|-----------------------------|------------------------------------------|
| `vm`          | 672 × n_bus, pu             | Bus voltage magnitudes.                  |
| `line_load`   | 672 × n_line, %             | Line loading.                            |
| `line_p_from` | 672 × n_line, MW (signed)   | Line power flow (sign gives direction).  |
| `trafo_load`  | 672 × n_trafo, %            | Transformer loading.                     |
| `trafo_p_hv`  | 672 × n_trafo, MW (signed)  | Transformer HV-side flow.                |
| `ext_grid_p`  | 672 × 1, MW (signed)        | Flow at the source (negative = export).  |
| `sgen_p_out`  | 672 × n_sgen, MW            | PV power actually injected.              |
 
```{note}
`mois_simule` is the single knob that keeps consumption and PV **seasonally
consistent**: it selects both the load bank month and the irradiance week. The
two are matched at the month level, not the exact calendar week (the smart-meter
and PVGIS years differ).
```
 
```{note}
`run_pv_scenario` calls `verify_temporal_alignment(df_load_p, df_sgen_p)`
before the power flow. This guard is deliberate: the load assignment reindexes
profiles positionally, so a silent load/PV misalignment could otherwise pass
unnoticed. Keep it active.
```
 
## 4. Impact analysis
 
A ready-made multi-scenario study lives in `scripts/pv_impact_analysis.py`
(0 / 30 / 50 / 80 installations, summer and a winter check). It reports, per
configuration, the **maximum** bus voltage (the relevant quantity for PV, as
opposed to the minimum voltage for load stress), the number of buses above the
1.05 pu and 1.10 pu limits, the location of any overvoltages, and reverse-flow
indicators.
 
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