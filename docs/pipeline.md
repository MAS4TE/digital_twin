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

## 3. Load assignment

- Each load bus is given a `load` element, fed by a weekly-type consumption
  profile drawn from a **profile bank** built from the smart-meter data
  (see `profil_conso`). Each profile is a 15-minute, one-week average built
  by aggregating the meter readings per weekday and time of day.
- Profiles are assigned with a **light street-based association**: a bus on a
  street that has meters receives a profile drawn at random among the profiles
  of that same street; a bus on a street without meters (orphan) receives a
  profile drawn at random from the whole bank, and is flagged as a warning.
- Each assigned profile is **stochastically perturbed** to reproduce load
  diversity (foisonnement): a random circular time shift desynchronises the
  peaks between buses, and a random lognormal scaling factor reflects the
  spread of household sizes. This avoids an unrealistic simultaneous peak
  across all buses.
- Both **active and reactive power** are modelled: reactive power is derived
  from the active profile using a fixed residential power factor
  (cos φ = 0.95).
- Profiles are expressed at a fixed 15-minute time step and converted to
  MW/MVAr to match pandapower's units. Near-zero loads are clipped to a small
  positive value to avoid power-flow convergence issues.

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
