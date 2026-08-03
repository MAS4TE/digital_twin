# Electrification scenarios

Section 9 of `bressoux_setup.ipynb` builds several **self-contained scenario
folders** under `data/bressoux/scenarios/<name>/`, each a full `data_path`
that `process_data`/`run_digital_twin` can load exactly like the base
`data/bressoux/` scenario -- no code changes needed to switch between them.

Each one answers the same question at a different intensity: *what happens to
the network as households electrify (heat pumps, water heaters, EVs) and PV
adoption grows?*

## Why `resflex`, and why it stays outside `digital_twin/`

Deciding **which** households get a heat pump / water heater / EV, and
simulating the power draw of the ones that do, is not something the base
Bressoux data can answer on its own -- it comes from `resflex`, a
StROBe-based residential load generator maintained separately (a local dev
dependency of the notebook's own conda environment, not published on PyPI).

`resflex` is deliberately **not** a dependency of the installable
`digital_twin` package:

- MAS4TE agents only ever call `run_digital_twin` with already-generated
  `metadata`/`load_profiles`/`equipment` files -- they never see how those
  were produced, and `resflex` has nothing to do with running the twin.
- `data/bressoux/resflex_scenarios.py` imports `resflex` **lazily**, inside
  the one function that calls it (`simulate_flex_assets`), so importing the
  module -- or running any other part of the notebook -- never requires
  `resflex` to be installed.

```{mermaid}
flowchart LR
    B[(out/processed/metadata.csv,<br/>load_profiles.csv,<br/>equipment.csv)] --> S(resflex_scenarios.build_scenario)
    R[resflex] --> S
    S --> M[(scenarios/name/.../metadata.csv)]
    S --> L[(scenarios/name/.../load_profiles.csv)]
    S --> E[(scenarios/name/.../equipment.csv)]
```

## The mechanism: an additive overlay, not a replacement

`resflex` decides, per household and per scenario, which houses get a heat
pump / water heater / EV -- driven by per-household equipment
**probabilities** (`P_HP`, `P_WB`, `P_EV`, resflex's own mechanism, a
Bernoulli draw per household, *not* a count or a target penetration rate).

Their simulated power draw is **added on top of** the household's existing
real-smart-meter consumption profile:

- Only the asset-specific columns of resflex's output are kept (`P_HP`,
  `P_WB`, `P_EV`, identified by the `P_` prefix) -- resflex's own StROBe
  baseload simulation is discarded, since the household's real EAN-based
  consumption already covers baseload. Keeping both would double-count it.
- No flexibility or optimization is exploited: the raw simulated draw is
  added as-is, at the household's own local time. resflex's `df_Flex` output
  (state-of-charge/setpoint detail meant for an optimizer) is unused.
- The household's `metadata.csv` row is updated to match: `heat_pump`,
  `electric_water_heating`, `electric_vehicle_type` and `n_persons` are
  filled in from resflex's own per-household parameter draw -- previously
  always `False`/`NA` for every household (see [Load assignment](load_assignment.md)).

This is implemented by `simulate_flex_assets` / `apply_flex_scenario` in
`data/bressoux/resflex_scenarios.py`. Time alignment matters here: resflex is
run with `year=2024` (2024-01-01 is a Monday) and `nb_days=7`, which is what
makes its 672 resampled rows land positionally on the same `'Mon 00:00'`..
`'Sun 23:45'` index `load_profiles.csv` uses -- resflex's own `DatetimeIndex`
is *not* trustworthy for this (it is anchored to a hardcoded reference date
inside resflex, regardless of `year`), so the overlay is placed positionally,
never by relabelling on that index.

## PV and batteries stay on the digital twin's own pipeline

The PV/battery side of each scenario does **not** go through resflex at all
(resflex's own `P_PV`/`P_BSS` sampling is non-functional in the installed
version -- it draws sizing parameters per household but never actually
simulates PV or battery power). Instead, each scenario's `equipment.csv` is
built by reusing [`digital_twin.data_processing.equipment`](bss_modelling.md)
directly, via `resflex_scenarios._build_equipment_table`, in one of three
modes:

| `pv_spec['mode']` | What it does |
|---|---|
| `"reuse"` | The base `equipment.csv` as-is -- same PV/battery placement, unchanged. |
| `"extend"` | `equipment.extend_pv` off the base table: `n_additional` more residential PV installations, keeping every existing one exactly where it is (isolates the *penetration rate* effect from *placement*). Battery placement is not redone. |
| `"generate"` | `generate_equipment_table` from scratch, `base_config` (the scenario's own `SCENARIO`) layered under the mode's own overrides -- placement AND battery siting are redrawn, central battery included. |

## The presets

| Scenario | P(heat pump) | P(water heater) | P(EV) | PV |
|---|---|---|---|---|
| **Low** | 0.0 | 0.0 | 0.0 | reused as-is (30 installations) -- a control: reproduces today's scenario exactly |
| **Medium** | 0.3 | 0.5 | 0.15 | +20 installations (50 total) |
| **High** | 0.6 | 1.0 | 0.4 | +50 installations (80 total) |

`Low` is intentionally a no-op overlay (`apply_flex_scenario` adds a
zero-valued profile to every household when every `P_*` is 0), so its
`load_profiles.csv` matches the base scenario to floating-point precision --
a control scenario to compare the other two against.

These numbers are starting points, easy to retune in the notebook's
`ELECTRIFICATION_SCENARIOS` dict -- there is nothing special about exactly
three presets or these exact probabilities.

## Scenario folder layout

```{code-block} text
data/bressoux/scenarios/<name>/
├── inputs/
│   ├── network.py            -- symlink to the base scenario (shared topology)
│   └── coordinates.csv       -- symlink
└── out/processed/
    ├── metadata.csv          -- scenario-specific (electrification overlay applied)
    ├── load_profiles.csv     -- scenario-specific (additive overlay applied)
    ├── equipment.csv         -- scenario-specific (PV/battery per pv_spec)
    ├── mv_load_profiles.csv  -- symlink to the base scenario (deterministic, unaffected)
    └── pvgis_irradiance_quarter_hourly.parquet  -- symlink (unaffected)
```

Only what actually varies per scenario is written fresh; everything shared
(network topology, MV facility loads, cached irradiance) is symlinked from
`data/bressoux/` rather than duplicated -- see
`resflex_scenarios._symlink_static_files`. The raw 320 MB smart-meter CSV is
never needed inside a scenario folder: only the one-time base
`generate_load_profiles` call touches it.

## Running one

Every scenario folder is a complete, valid `data_path` -- `main.py` is set up
to run the **Medium** scenario as its worked example:

```{code-block} python
config = {'data_path': "data/bressoux/scenarios/medium", ...}
metadata, network, households, pv_profiles, schedules_BSS, equipment = process_data(config)
out = run_digital_twin(config, metadata, network, households,
                       pv_profiles=pv_profiles, schedules_BSS=schedules_BSS,
                       equipment=equipment)
```

Comparing the three verdicts (see `bressoux_setup.ipynb`, section 9) shows
grid stress increasing monotonically with electrification: more peak load,
lower minimum voltage, higher maximum line loading, from `Low` to `High`.
