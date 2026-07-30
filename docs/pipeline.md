# Pipeline

## 1. Data ingestion

- **Smart-meter data** (`data/bressoux/inputs/part-*.csv`): anonymized EAN
  (metering point) readings, including the street address and the substation
  ("cabine") each meter is attached to.
- **Coordinates** (`data/bressoux/inputs/coordinates.csv`): GPS coordinates of
  the buildings in Bressoux, matched to house numbers.
- EANs with less than one year of metered consumption are discarded to avoid
  building the network from too short a history.
- From the retained meters, the `households` timeseries is drawn from a bank of
  **real weekly consumption profiles** built at
  `data/bressoux/out/processed/consumption_bank_real_weeks.parquet`, conditioned
  on the simulated month. See [Load assignment](load_assignment.md).

## 2. Network reconstruction

This step is a **caller-supplied input** to `run_digital_twin` (the
`network` argument), built by a site-specific `inputs/network.py` (e.g.
[data/bressoux/inputs/network.py](../data/bressoux/inputs/network.py)) rather
than by the engine itself — see [Interfacing](interfacing.md#the-network-input).
The description below is specific to the Bressoux site.

- The network is organised in two voltage levels: **LV buses** (0.4 kV,
  one per household) and **MV buses** (15 kV).
- MV buses are of two kinds:
  - **Substations** ("cabines": Troulouette, Cimetière Bressoux, Malvoz
    Bressoux, Grottes), each connected to its LV side through a single
    **MV/LV transformer** that feeds the surrounding LV households.
  - **MV load nodes** (Paradis du Bouhay, Plantations, Omnisport BX, Groupe
    Scolaire), which structure the MV feeder and carry their own load directly
    at the MV level (large consumers connected in medium voltage, without a
    dedicated transformer). Three of them are extracted from the coordinate
    dataset by their OSM `@id`; Groupe Scolaire is added from manual
    coordinates.
- A single **MV slack bus** models the external grid connection.
- The **MV feeder** is reconstructed as a radial topology matching the DSO
  one-line diagram: slack → Paradis du Bouhay → Plantations → Troulouette,
  which then branches towards Cimetière Bressoux and Malvoz Bressoux, the
  latter feeding Grottes and Omnisport BX → Groupe Scolaire. MV line lengths
  are derived from bus coordinates and cross-checked against the reference
  lengths of the one-line diagram.
- On the **LV side**, buses are connected into **lines** using a
  street/house-number based heuristic (nodes on the same street, sorted by
  house number and parity, are linked together). Each substation's LV
  secondary feeds the nearby LV buses.
- Generic line parameters (length, R/X/C, thermal limit) are used as a first
  approximation; transformers use typical distribution ratings (configurable
  `sn_mva`).
- The result is a [pandapower](http://www.pandapower.org/) network (`net`)
  representing the Bressoux LV/MV grid: radial, fully connected, with one
  transformer per substation.

## 3. Load assignment

Consumption profiles reach the buses in two decoupled stages: a scenario's
`out/processed/load_profiles.csv` is **generated upstream** (see
[Load assignment](load_assignment.md)), then `run_digital_twin` **receives** it
(as `households`) and its assignment step maps the profiles onto the load buses.
The mapping is a lookup, not a draw: `metadata['bus_id']` declares it 1:1, and a
caller supplying no such column falls back to a seeded draw with a warning.
Reactive power uses a fixed residential power factor (cos φ = 0.95); profiles
are at a 15-minute step in MW/MVAr.

## 4. Time-series power flow

- Each load is attached to its profile through a pandapower `ConstControl`
  (fed by a `DFData` data source); when PV is present, one `ConstControl` per
  `sgen` is added for the PV injection.
- An `OutputWriter` logs bus voltage magnitudes, line and transformer loadings
  for every time step; with PV, it additionally logs the signed line, trafo,
  external-grid and sgen powers (to observe reverse flows).
- `pandapower.timeseries.run_timeseries` runs an AC power flow (Newton-Raphson)
  over the full simulation horizon.
- The power-flow helper returns loads-only results when no PV is given, and an
  extended result (including the flow signals above) when PV is injected.

## 5. Results and visualisation

- The aggregated verdict (summary + violations + warnings) is returned by
  `run_digital_twin` as a single long-format DataFrame.
- `save_run_report(out, out_path)` (`digital_twin/digital_twin.py`) renders
  that DataFrame as a human-readable Markdown report; `main.py` writes it to
  `data/bressoux/out/results/` when `config['save']` is `True`.
- Results can be inspected at any time step and plotted on a map to visualise
  voltage levels and line loading across the neighbourhood.

## 6. Packaging as a digital twin function

Steps 3-5 are consolidated into `run_digital_twin`
(`digital_twin/digital_twin.py`), a network-agnostic engine that takes its
inputs (configuration, household metadata, the network bundle from step 2,
consumption/PV profiles, battery schedules) and returns a single DataFrame
with the resulting summary verdict, violations and warnings — see
[Interfacing](interfacing.md).