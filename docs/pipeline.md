# Pipeline

## 1. Data ingestion

- **Smart-meter data** (`data/part-*.csv`): anonymized EAN (metering point)
  readings, including the street address and the substation ("cabine") each
  meter is attached to.
- **Coordinates** (`data/coordonnées.csv`): GPS coordinates of the buildings
  in Bressoux, matched to house numbers.
- EANs with less than one year of metered consumption are discarded to avoid
  building the network from too short a history.
- **Load profile** (`data/load_profiles.csv`): Consumption profile bank built based on the EANs and Stochastic variation over time and power (+/-2 hours and 0-20% increase)

## 2. Network reconstruction

- The network is organised in two voltage levels: **LV buses** (0.4 kV,
  one per household) and **MV buses** (15 kV).
- MV buses are of two kinds:
  - **Substations** ("cabines": Troulouette, Cimetière Bressoux, Malvoz
    Bressoux, Grottes), each connected to its LV side through a single
    **MV/LV transformer** that feeds the surrounding LV households.
  - **Passage nodes** (Paradis du Bouhay, Plantations, Omnisport BX, Groupe
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

Load buses are assigned consumption profiles in two decoupled stages: a
`load_profiles` structure is first filled from a bank of real weeks, then a
common assignment step maps those profiles onto the load buses.

### Filling `load_profiles` — monthly real-week bank

- Profiles are **real recorded weeks** drawn from a bank built from the
  smart-meter data (`data/consumption_bank_real_weeks.parquet`: 1346 complete
  weeks across 33 EAN, tagged by EAN, street, month and season). Each profile
  is a genuine recorded week, not an average.
- The simulated month is set by `config['mois_simule']`. For each street,
  `load_profiles` is filled with real weeks of that month drawn from the EAN(s)
  of that same street, preserving both **seasonal consistency** (matching the
  PVGIS irradiance window, `data/pvgis_irradiance_quarter_hourly.parquet`) and
  **street locality**. The draw is independent per bus and reproducible via the
  exposed `seed`.
- No shift/scale perturbation is applied: load diversity comes from the natural
  variety of the real weeks rather than from synthetic desynchronisation.

### Assigning profiles to buses

- Once `load_profiles` is filled, a **light street-based association** maps a
  profile to each load bus: a bus on a street that has meters receives a
  profile drawn at random among the profiles of that same street; a bus on a
  street without meters (orphan) receives a profile drawn at random from the
  whole set, and is flagged as a warning.
- Each load is modelled in **active and reactive power**: reactive power is
  derived from the active profile using a fixed residential power factor
  (cos φ = 0.95).
- Profiles are expressed at a fixed **15-minute** time step (672 steps per
  week) and converted to MW/MVAr to match pandapower's units. Near-zero loads
  are clipped to a small positive value to avoid power-flow convergence issues.
- Because the stress multiplier in `main.py` scales `load_profiles`, it applies
  directly to the assigned loads.

### Known limitation

- The aggregate load factor stays around 0.64, above the real ~0.25, because
  profiles are drawn independently per bus, which dilutes the natural
  coincidence of household peaks. Recovering the true value would require a
  "simultaneous blocks" scheme (grouping real EAN from the same calendar weeks)
  — relevant later, once PV/EV make the off-peak dynamics matter.

## 4. Time-series power flow

- Each load is attached to its profile through a pandapower `ConstControl`.
- An `OutputWriter` logs bus voltage magnitudes, line loadings and load
  active/reactive power for every time step.
- `pandapower.timeseries.run_timeseries` runs an AC power flow (Newton-Raphson)
  over the full simulation horizon.

## 5. Results and visualisation

- Time-series results (voltages, loadings) are written to CSV under
  `outputs/`.
- Results can be inspected at any time step and plotted on a map
  (`pf_res_plotly`, `simple_plotly`) to visualise voltage levels and line
  loading across the neighbourhood.

## 6. Packaging as a digital twin function

The steps above are being consolidated into `run_digital_twin`
(`sourcecode/digital_twin.py`), which takes the same kind of inputs
(configuration, households, load/PV profiles, battery schedules) as plain
`pandas` DataFrames and returns a single DataFrame with the resulting
time-series and warnings — see [Interfacing](interfacing.md).
