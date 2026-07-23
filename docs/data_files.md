# Data files

The digital twin exchanges data through a handful of files with well-defined
schemas. They are the **contract between the bricks**: the load bank fills
`load_profiles`, the PV helpers produce `pv_profiles`, and everything flows
through these files. In the MAS4TE setting, these are the files **we** generate
from the Bressoux field data before handing them to the simulation.

This page documents the schema of each generated file. See
[Load assignment](load_assignment.md) and [PV modelling](pv_modelling.md) for
the functions that read and write them.

## `pv_installations.csv`

The PV installation register: which bus carries a PV system and its size.
Produced by `generate_pv_metadata` (see [PV modelling](pv_modelling.md)).

| Column   | Type            | Description                                          |
|----------|-----------------|------------------------------------------------------|
| `bus_id` | `int64`         | Target bus in the pandapower network.                |
| `kwp`    | `float64`       | Installed peak power (kWp).                           |
| `type`   | `str`           | `"residential"` or `"rest_home"`.                    |
| `street` | `str` or `NaN`  | Street of the installation; `NaN` for the rest home. |

```{code-block} text
bus_id,kwp,type,street
99,3.0,residential,Rue des Primeroses
111,3.0,residential,Rue des Primeroses
5,12.0,rest_home,
```

## `consumption_bank_real_weeks.parquet`

The bank of real weekly consumption profiles (1346 rows × 678 columns), one row
per complete calendar week of a retained EAN. Produced by
`build_consumption_bank` (see [Load assignment](load_assignment.md)).

| Column          | Type                     | Description                                        |
|-----------------|--------------------------|----------------------------------------------------|
| `ean_idx`       | `int64`                  | Integer index of the meter.                        |
| `ean_anon`      | `str`                    | Anonymised meter hash.                             |
| `street`        | `str`                    | Street the meter belongs to.                       |
| `month`         | `int64` (1–12)           | Month of the week's start.                         |
| `season`        | `str`                    | Season label (e.g. `"Automne"`).                   |
| `week_start`    | `str` (ISO, UTC)         | Start timestamp of the calendar week.              |
| `t0000`…`t0671` | `float64` (MW)           | 672 quarter-hourly consumption values (Mon→Sun).   |

```{note}
The `t0000..t0671` columns are the week's 672 quarter-hourly steps (7 days ×
96 steps), in **MW**, ordered Monday 00:00 to Sunday 23:45 — the same temporal
layout used everywhere else in the pipeline.
```

## `pvgis_irradiance_quarter_hourly.parquet`

The reference irradiance series (35040 rows × 3 columns), one full year at a
quarter-hourly step. Produced by `build_irradiance_reference` from a PVGIS
(SARAH3) query for the Bressoux coordinates (lat 50.639, lon 5.605).

| Column      | Type                    | Description                                          |
|-------------|-------------------------|------------------------------------------------------|
| `timestamp` | `datetime64[ns, UTC]`   | Quarter-hourly timestamp.                            |
| `ghi_w_m2`  | `float64`               | Global horizontal irradiance (W/m²).                 |
| `year`      | `int64`                 | Reference year of the series.                        |

```{warning}
`ghi_w_m2` is **global horizontal** irradiance, not in the plane of a tilted
array. The hourly PVGIS values are expanded to quarter-hourly by forward-fill
(each hourly value repeated over its four quarters), which conserves energy but
produces a step profile. See the conversion note in
[PV modelling](pv_modelling.md).
```

## Regenerating the files

```{code-block} bash
python -m digital_twin.consumption_bank   # bank + irradiance parquets
python -m digital_twin.pv_metadata        # pv_installations.csv
```

```{note}
`households.csv` and `coordonnées.csv` (network topology and household-to-street
mapping) are inputs to the pipeline rather than generated artefacts; they are
documented in [Network reconstruction](pipeline.md).
```