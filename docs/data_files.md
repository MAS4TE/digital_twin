# Data files

The digital twin exchanges data through a handful of files with well-defined
schemas. They are the **contract between the bricks**: the load bank fills
`load_profiles`, the PV helpers produce `pv_profiles`, and everything flows
through these files. In the MAS4TE setting, these are the files **we** generate
from the Bressoux field data before handing them to the simulation.

This page documents the schema of each generated file. See
[Load assignment](load_assignment.md) and [PV modelling](pv_modelling.md) for
the functions that read and write them.

## `out/processed/equipment.csv`

The **bus-level equipment register**: one row per bus that can carry equipment,
whether it carries any or not. Produced by `generate_equipment_table`
(`digital_twin/data_processing/equipment.py`).

One file carries both optional layers, and two of its properties are
load-bearing:

- It is **exhaustive by construction** — every equipable bus of the network is a
  row. That is what makes the bus id contract checkable by *set* comparison
  rather than by cardinality: 333 rows for 333 buses can coincide while hiding a
  placement disagreement.
- It lives under `out/processed/`, not `inputs/`, because it is **derived from
  the network**. Regenerate it rather than hand-editing it.

Activation is by CONTENT, not by file presence: a scenario has PV when some row
has `pv_system_size_kwp > 0`, and batteries when some row has
`battery_size_kwh > 0`. One register describes the whole scenario, so it cannot
be half-described.

| Column               | Type              | Description                                                       |
|----------------------|-------------------|-------------------------------------------------------------------|
| `bus_id`             | `int64`           | Target bus in the pandapower network.                             |
| `bus_type`           | `str`             | `"lv_residential"` or `"mv_load"` (a facility connected at MV).    |
| `name`               | `str`             | Bus name in the network, or the facility name.                     |
| `street`, `num`      | `str` or `NA`     | Address of the LV bus; `NA` for MV load facilities.                |
| `lon`, `lat`         | `float64`         | Coordinates, read from the network's GeoJSON `geo` column.         |
| `pv_system_size_kwp` | `Float64`         | Installed PV peak power (kWp); `0` when the bus carries none.       |
| `battery_id`         | `string` or `NA`  | `b1`…`bn`, in ascending `bus_id` order (see below).                 |
| `battery_size_kwh`   | `Float64`         | Domestic battery capacity (kWh); `0` when the bus carries none.     |
| `heat_pump`          | `boolean`         | MAS4TE schema field, defaults to `False`.                          |
| `electric_vehicle`   | `boolean`         | MAS4TE schema field, defaults to `False`.                          |

```{code-block} text
bus_id,bus_type,name,street,num,lon,lat,pv_system_size_kwp,battery_id,battery_size_kwh,heat_pump,electric_vehicle
9,lv_residential,Malvoz 12,Rue Ernest Malvoz,12,5.6058,50.6393,3.0,,0.0,False,False
13,lv_residential,Malvoz 20,Rue Ernest Malvoz,20,5.6061,50.6395,3.0,b1,10.0,False,False
5,mv_load,Paradis du Bouhay,,,5.6042,50.6401,12.0,,0.0,False,False
```

`battery_id` is assigned deterministically in ascending `bus_id` order, so the
same table always yields the same identifiers — that is what the schedule's
`b1`…`bn` are matched against. **`b0` is never a row of this table**: it is the
central battery, sits on an MV node, and its placement and size come from
`config['bss_bus']`, `config['bss_park_e_kwh']` and `config['bss_park_p_kw']`.

```{note}
The equipment columns deliberately reuse the MAS4TE **household** schema names
(`battery_size_kwh`, `pv_system_size_kwp`, `heat_pump`, `electric_vehicle`), so
the join between this bus-level table and `metadata.csv` is obvious.
```

```{warning}
Two columns must be pinned to `str` when reading this file, and
`load_equipment_table` does it for you. `num`, because `read_csv` would infer it
back as `float64` (written `"93.0"`, read `93.0`), which then fails to match the
`(street, num)` key of `network['buses']`. And `battery_id`, because it is the
join key the schedule's `b1`…`bn` are matched against.
```

## `out/processed/metadata.csv`

Per-household attributes -- the `metadata` argument of `run_digital_twin`.
Created by `generate_household_metadata` and completed by
`generate_load_profiles`, which writes it **in lockstep** with
`load_profiles.csv` (same `profile_id`s, same draw) -- see
[Load assignment](load_assignment.md).

| Column                  | Type              | Description                                                  |
|--------------------------|-------------------|----------------------------------------------------------------|
| `profile_id`             | `int64`           | Matches a `profile_{id}` column in `load_profiles.csv`.        |
| `bus_id`                 | `int64`           | The LV bus this household sits on -- the assignment itself, 1:1 with `profile_id`. |
| `constellation`          | `Int64` or `NA`   | MAS4TE schema field, unused for now.                           |
| `building_type`          | `string` or `NA`  | MAS4TE schema field, unused for now.                           |
| `heat_pump`              | `boolean`         | Defaults to `False` -- flip per household to define scenarios. |
| `electric_water_heating` | `boolean`         | Defaults to `False`.                                           |
| `electric_vehicle_type`  | `string` or `NA`  | MAS4TE schema field, unused for now.                           |
| `n_persons`              | `Int64` or `NA`   | MAS4TE schema field, unused for now.                           |
| `pv_system_size_kwp`     | `Float64`         | Projected from `equipment.csv` on this household's bus.        |
| `battery_size_kwh`       | `Float64`         | Projected from `equipment.csv` on this household's bus.        |
| `source_street`          | `string`          | Street of the real week drawn for this profile.                |
| `source_month`           | `Int64`           | Month it was drawn from -- checked against `config['mois_simule']`. |
| `source_week_start`      | `string`          | Start date of the drawn calendar week (traceability).          |
| `source_ean_index`       | `Int64`           | Index of the source EAN in the bank (traceability).            |

```{warning}
The `source_*` columns describe **the week that was drawn**, and nothing else.
`source_street` is what `_assign_loads` matches against bus streets, and
`source_month` is what makes the seasonal coherence with the simulated month
checkable at all -- so `metadata.csv` must never be rewritten independently of
`load_profiles.csv`. `generate_load_profiles` writes both together for exactly
that reason.
```

```{note}
`pv_system_size_kwp` and `battery_size_kwh` also appear in `equipment.csv`. The
name reuse is deliberate (one MAS4TE schema), but the two are at different
levels: here per household, there per **bus**. The **bus-level table is the
authority** — it is where placement is decided — and
`equipment.sync_household_equipment` projects it onto the household rows
(joined on `bus_id`). Left unprojected these columns keep their `0.0` default
and read as "this household has no battery" for buses that do have one, which
is why `_validate_inputs` raises a `[META/equip]` warning when the two files
disagree.
```

## `out/processed/load_profiles.csv`

Household consumption timeseries -- the `households` argument of
`run_digital_twin`. Produced by `generate_load_profiles`, which draws one real
EAN week per profile from the sub-bank of the simulated month (see
[Load assignment](load_assignment.md)).

| Column            | Type      | Description                                        |
|--------------------|-----------|------------------------------------------------------|
| `week_step`        | `str`     | Index: `'Mon 00:00'..'Sun 23:45'`, 672 quarter-hourly steps. |
| `profile_{id}`     | `float64` | Active power (MW), one column per `metadata.csv` row. |

## `out/processed/pv_profiles.csv` and `schedules_bss.csv`

**Written for inspection, never read back.** Both are rebuilt from
`equipment.csv` + the irradiance + `load_profiles.csv` on every `process_data`
call (~3 s), and that is precisely what makes them incapable of contradicting
the equipment table. They exist so you can open them, sort them, plot a column —
not as a cache.

| File | Shape | Unit | Contents |
|---|---|---|---|
| `pv_profiles.csv` | 672 × n_equipped | **kW** | one column `pv_{bus_id}` per bus carrying PV |
| `schedules_bss.csv` | (672 × n_batteries) × 2 | **kW** | long format: `battery_id`, `p_kw` |

```{warning}
`pv_profiles.csv` is in **kW**, while `run_digital_twin` receives `pv_profiles`
in **MW**. The `/1000` is applied on the way out of `_load_pv_profiles`, just
before injection into pandapower. The asymmetry is deliberate — kW is the
readable unit and the one `build_pv_production` works in — but it means the file
does *not* show the numbers the power flow sees, off by a factor 1000.
`schedules_bss.csv` has no such gap: kW is the twin's own interface unit for it.
```

```{note}
When a layer is switched off (no bus with `pv_system_size_kwp > 0`, or no
domestic battery), any previous file is **deleted** rather than left behind — a
stale `pv_profiles.csv` showing 31 installations for a scenario that now has
none would be worse than no file at all.

Do not be tempted to read them back to skip the recomputation: that turns an
output into an input, and gives the scenario a second register free to disagree
with `equipment.csv`.
```

## `out/processed/mv_load_profiles.csv`

The 4 MV load facility profiles (rest home, horticultural centre, sports
hall, school) -- deterministic day/time profiles, NOT drawn from the
residential bank. Generated inline in `bressoux_setup.ipynb` (ported from the
historical `profil_conso.ipynb`).

| Column                | Type      | Description                                  |
|------------------------|-----------|-------------------------------------------------|
| `week_step`            | `str`     | Index: `'Mon 00:00'..'Sun 23:45'`, 672 steps.   |
| `Paradis du Bouhay`, `Plantations`, `Omnisport BX`, `Groupe Scolaire` | `float64` | Active power (MW) per facility. |

## `out/processed/consumption_bank_real_weeks.parquet`

The bank of real weekly consumption profiles (1345 rows × 678 columns), one row
per complete calendar week of a retained EAN. Produced by
`build_consumption_bank` (see [Load assignment](load_assignment.md)) -- the
month-conditioned source every `households` draw reads from, including
`pv.run_pv_scenario` / `scripts/pv_impact_analysis.py`.

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

## `out/processed/pvgis_irradiance_quarter_hourly.parquet`

The reference irradiance series (35040 rows × 3 columns), one full year at a
quarter-hourly step. Produced by `build_irradiance_reference` from a PVGIS
(SARAH3) query for the site's coordinates, DERIVED from the network's own bus
geodata (`utils.site_centroid`) rather than hard-coded — 50.639 / 5.606 on
Bressoux. Required alongside `equipment.csv` for `process_data` to build
`pv_profiles`.

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
jupyter execute data/bressoux/bressoux_setup.ipynb   # metadata, load_profiles,
                                                       # mv_load_profiles, irradiance
python -m digital_twin.data_processing.profiles       # (alternative) bank + irradiance parquets
python -c "from digital_twin.data_processing.equipment import generate_equipment_table; generate_equipment_table({'data_path': 'data/bressoux'})"
```

```{note}
`inputs/coordinates.csv` and `inputs/part-*.csv` (network topology and raw
smart-meter readings) are inputs to the pipeline rather than generated
artefacts; they are documented in [Network reconstruction](pipeline.md).
```