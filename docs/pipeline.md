# Pipeline

## 1. Data ingestion

- **Smart-meter data** (`data/part-*.csv`): anonymized EAN (metering point)
  readings, including the street address and the substation ("cabine") each
  meter is attached to.
- **Coordinates** (`data/coordonnées.csv`): GPS coordinates of the buildings
  in Bressoux, matched to house numbers.
- EANs with less than one year of metered consumption are discarded to avoid
  building the network from too short a history.

## 2. Network reconstruction

- One electrical **bus** is created per household (LV, 0.4 kV) and one per
  substation ("cabine", MV, 15 kV), plus a single MV slack bus for the
  external grid connection.
- Buses are connected into **lines** using a street/house-number based
  heuristic (nodes on the same street, sorted by house number and parity, are
  linked together), with generic line parameters (length, R/X/C, thermal
  limit) as a first approximation.
- The result is a [pandapower](http://www.pandapower.org/) network (`net`)
  representing the Bressoux LV/MV grid.

## 3. Load assignment

- Each bus is given a `load` element, fed by a real household consumption
  profile taken from the smart-meter data for the corresponding street. When
  no meter can be matched to a street, a profile is picked at random among
  all available households (and flagged as a warning).
- Profiles are resampled to a fixed time step (15 minutes) and converted to
  MW to match pandapower's units.

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
