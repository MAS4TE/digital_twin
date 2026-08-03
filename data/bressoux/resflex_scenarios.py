"""
Notebook-local resflex integration for the Bressoux case.

Decides, per scenario, which households get a heat pump / electric vehicle /
water boiler -- via resflex's own equipment-probability mechanism (P_HP,
P_EV, P_WB, each a per-household Bernoulli draw, NOT a count) -- extracts
their simulated power draw, and layers it on top of the household's existing
real-smart-meter consumption profile (load_profiles.csv). This is a pure
additive "electrification overlay": no flexibility/optimization is exploited
(resflex's df_Flex output, and its P_PV/P_BSS sampling, are intentionally
unused -- PV and domestic batteries stay on the digital twin's own bus-level
equipment table, see digital_twin.data_processing.equipment). Also builds the
paired PV/battery equipment table for the scenario, and assembles a
self-contained scenario folder that
digital_twin.data_processing.process_data.process_data can load unmodified.

Deliberately NOT part of digital_twin/: resflex is a local dev dependency
(pip-installed in the `bressoux` conda env, not on PyPI, not needed to run
the twin itself) -- MAS4TE agents only ever see the already-generated
metadata/load_profiles/equipment files through run_digital_twin, never
resflex. Import only from data/bressoux/ notebooks/scripts running with
data/bressoux itself as the working directory (bressoux_setup.ipynb's own
directory -- resflex is imported lazily, inside simulate_flex_assets, so
importing this module doesn't require it to be installed unless a scenario
is actually built).
"""
import copy
import os

import numpy as np
import pandas as pd

from digital_twin.utils import _WEEK_TIME_INDEX
from digital_twin.data_processing.profiles import MAS4TE_METADATA_COLUMNS
from digital_twin.data_processing.equipment import (
    equipment_table_path, extend_pv, generate_equipment_table,
    load_equipment_table, sync_household_equipment,
)

# resflex config shared by every scenario -- only P_HP/P_WB/P_EV (equipment
# probabilities) vary per scenario, see _build_resflex_config. year=2024 is
# required: 2024-01-01 is a Monday, which is what makes the 672 resampled
# rows land positionally on Mon 00:00 .. Sun 23:45 (see module docstring of
# simulate_flex_assets for why the resulting DatetimeIndex itself can't be
# trusted for this). P_PV/P_BSS are kept at 0 -- resflex samples PV/BSS sizing
# per household but never actually simulates their power (no add_PV/add_BSS
# in the installed version); *_data blocks below are still required by
# resflex.utils.create_params/get_list_param even when the matching P_* is 0.
RESFLEX_DEFAULTS = {
    "year": 2024,
    "start_day": 0,
    "nb_days": 7,
    "timestep": 15,
    "flexibility": True,

    "appliances": {
        "P_WashingMachine": 0.65,
        "P_TumbleDryer_given_WM": 0.6,
        "P_DishWasher": 0.6,
    },

    "P_PV": 0.0,
    "PV_data": {
        "Pmax": [2, 5, 10, 15, 20],
        "P_Pmax": [0.1, 0.3, 0.4, 0.1, 0.1],
        "id": [1],
        "P_id": [1],
    },

    "P_BSS": 0.0,
    "BSS_data": {
        "Capacity": [5, 10, 15],
        "P_Capacity": [0.3, 0.4, 0.3],
        "Pmax": [2, 5, 7],
        "P_Pmax": [0.3, 0.4, 0.3],
        "SoC_min": [0.2],
        "P_SoC_min": [1],
        "SoC_max": [0.8],
        "P_SoC_max": [1],
        "eta": [0.95],
        "P_eta": [1],
    },

    "WB_data": {
        "Pmax": [1.5, 3, 5],
        "P_Pmax": [0.3, 0.4, 0.3],
        "Volume": [100, 200, 250],
        "P_Volume": [0.2, 0.6, 0.2],
        "T_set": [60],
        "P_T_set": [1],
    },

    "EV_data": {
        "Consumption": [15, 20, 25],
        "P_Consumption": [0.3, 0.4, 0.3],
        "Capacity": [30, 50, 60],
        "P_Capacity": [0.3, 0.4, 0.3],
        "Pmax": [7, 11, 15],
        "P_Pmax": [0.3, 0.4, 0.3],
        "eta": [0.95],
        "P_eta": [1],
        "SoC_target": [0.80],
        "P_SoC_target": [1],
        "Usage": [7000, 14000, 25000],
        "P_Usage": [0.25, 0.5, 0.25],
    },

    "HP_data": {
        "Year": [1969, 1989, 2006, 2010],
        "P_Year": [0.25, 0.25, 0.25, 0.25],
        "Size": [70, 90, 100, 150],
        "P_Size": [0.2, 0.4, 0.2, 0.2],
        "Floors": [1, 2, 3],
        "P_Floors": [0.10, 0.45, 0.45],
        "P_nom": [2, 3, 4, 5],
        "P_P_nom": [0.25, 0.25, 0.25, 0.25],
        "COP": [5],
        "P_COP": [1],
    },

    "inhabitants": [2, 3, 4, 5],
    "P_inhabitants": [0.25, 0.25, 0.25, 0.25],
}

_ASSET_METADATA_COLS = ["heat_pump", "electric_water_heating", "electric_vehicle_type", "n_persons"]


def _build_resflex_config(n_households, probabilities):
    cfg = copy.deepcopy(RESFLEX_DEFAULTS)
    cfg["nb_households"] = n_households
    cfg["P_HP"] = probabilities.get("P_HP", 0.0)
    cfg["P_WB"] = probabilities.get("P_WB", 0.0)
    cfg["P_EV"] = probabilities.get("P_EV", 0.0)
    return cfg


def simulate_flex_assets(n_households, probabilities, seed=42):
    """Run resflex for `n_households` households and extract, per household,
    the asset-only (HP/WB/EV) power addition. Baseload/appliance columns from
    resflex's own StROBe simulation are discarded -- the household's real
    EAN-based consumption (load_profiles.csv) already covers baseload; only
    the flexible-asset columns (reliably identified by the 'P_' prefix, e.g.
    'P_HP'/'P_WB'/'P_EV') are added on top.

    Returns
    -------
    asset_profiles : pd.DataFrame   # index _WEEK_TIME_INDEX, columns profile_{id}, kW
    metadata_updates : pd.DataFrame # profile_id, heat_pump, electric_water_heating,
                                     # electric_vehicle_type, n_persons
    """
    import resflex  # local dev dependency -- only needed here, not by digital_twin/MAS4TE

    resflex.set_seed(seed)
    config = _build_resflex_config(n_households, probabilities)
    dic_df_P, _dic_df_Flex, dic_params = resflex.simulate_all(
        config, save=False, plot_res=False, print_res=False)

    asset_data = {}
    meta_rows = []
    for i in range(1, n_households + 1):
        profile_id = i - 1
        df_p = dic_df_P[f"household_{i}"]
        params = dic_params[f"household_{i}"]

        asset_cols = [c for c in df_p.columns if c.startswith("P_")]
        asset_kw = df_p[asset_cols].sum(axis=1).to_numpy() if asset_cols else np.zeros(len(df_p))
        if len(asset_kw) != len(_WEEK_TIME_INDEX):
            raise ValueError(
                f"resflex household_{i}: expected {len(_WEEK_TIME_INDEX)} steps, got "
                f"{len(asset_kw)} -- check nb_days/timestep/year alignment (RESFLEX_DEFAULTS)."
            )
        asset_data[f"profile_{profile_id}"] = asset_kw  # kW, like load_profiles.csv

        meta_rows.append({
            "profile_id": profile_id,
            "heat_pump": bool(params.get("HP", False)),
            "electric_water_heating": bool(params.get("WB", False)),
            "electric_vehicle_type": "EV" if params.get("EV", False) else pd.NA,
            "n_persons": params.get("family"),
        })

    asset_profiles = pd.DataFrame(asset_data, index=_WEEK_TIME_INDEX)
    asset_profiles.index.name = "week_step"
    metadata_updates = pd.DataFrame(meta_rows)
    return asset_profiles, metadata_updates


def apply_flex_scenario(metadata, households, probabilities, seed=42):
    """Additive electrification overlay: adds the resflex-simulated HP/WB/EV
    power on top of the existing household consumption, and fills in the
    corresponding metadata columns. No flexibility/optimization -- the raw
    simulated draw is added as-is.

    `metadata` is expected one row per profile_id (see
    digital_twin.data_processing.profiles.generate_household_metadata) --
    any other columns it carries (bus_id, source_*, pv_system_size_kwp,
    battery_size_kwh, ...) pass through untouched.

    Returns
    -------
    (new_metadata, new_households) -- same shapes/columns/dtypes as the inputs.
    """
    n_households = len(metadata)
    asset_profiles, metadata_updates = simulate_flex_assets(n_households, probabilities, seed)

    new_households = households.copy()
    for col in asset_profiles.columns:
        new_households[col] = new_households[col] + asset_profiles[col]

    new_metadata = metadata.drop(columns=_ASSET_METADATA_COLS).merge(
        metadata_updates, on="profile_id", how="left")
    for col in _ASSET_METADATA_COLS:
        dtype, _default = MAS4TE_METADATA_COLUMNS[col]
        new_metadata[col] = new_metadata[col].astype(dtype)
    new_metadata = new_metadata[metadata.columns]

    return new_metadata, new_households


_STATIC_INPUT_FILES = ["network.py", "coordinates.csv"]
_STATIC_PROCESSED_FILES = ["mv_load_profiles.csv", "pvgis_irradiance_quarter_hourly.parquet"]


def _symlink_static_files(base_data_path, scenario_data_path):
    """Symlink the files shared by every scenario (network topology, MV
    facility loads, cached irradiance) instead of duplicating them -- only
    metadata.csv/load_profiles.csv/equipment.csv are scenario-specific.
    """
    base_data_path = os.path.abspath(base_data_path)
    for fname in _STATIC_INPUT_FILES:
        dst = os.path.join(scenario_data_path, "inputs", fname)
        if not os.path.exists(dst):
            os.symlink(os.path.join(base_data_path, "inputs", fname), dst)
    for fname in _STATIC_PROCESSED_FILES:
        dst = os.path.join(scenario_data_path, "out", "processed", fname)
        if not os.path.exists(dst):
            os.symlink(os.path.join(base_data_path, "out", "processed", fname), dst)


def _build_equipment_table(base_data_path, pv_spec, seed, base_config=None):
    """pv_spec['mode']: 'reuse' (base equipment.csv as-is, PV+battery
    placement unchanged), 'extend' (equipment.extend_pv off the base table,
    pv_spec['n_additional'] more residential PV installations -- battery
    placement is NOT redone, so the extended buses do not automatically gain
    a battery), or 'generate' (generate_equipment_table from scratch, the
    remaining pv_spec keys passed through as config overrides -- placement
    AND battery siting are redrawn).

    `base_config` (mode='generate' only): the scenario's own config (e.g.
    bressoux_setup.ipynb's SCENARIO), layered UNDER pv_spec's overrides. This is
    what makes 'bss_bus'/'bss_park_e_kwh'/'bss_park_p_kw' survive into a
    'generate' scenario unless it explicitly overrides them -- omitting it would
    silently generate a table with no central battery.
    """
    mode = pv_spec.get("mode", "reuse")
    base_equipment = load_equipment_table({"data_path": base_data_path})
    if base_equipment is None:
        raise FileNotFoundError(
            f"No equipment.csv found under {base_data_path} -- run "
            "generate_equipment_table(SCENARIO) once for the base scenario first "
            "(bressoux_setup.ipynb, section 'Place the equipment')."
        )

    if mode == "reuse":
        return base_equipment

    if mode == "extend":
        return extend_pv(base_equipment.copy(), n_additional=pv_spec["n_additional"],
                         config={"seed": seed})

    if mode == "generate":
        overrides = {k: v for k, v in pv_spec.items() if k != "mode"}
        cfg = {**(base_config or {}), "data_path": base_data_path, "seed": seed, **overrides}
        return generate_equipment_table(cfg, write=False)

    raise ValueError(f"Unknown pv_spec mode: {mode!r} (expected 'reuse', 'extend' or 'generate')")


def build_scenario(name, flex_probabilities, pv_spec, seed=42,
                    base_data_path="data/bressoux", scenarios_root=None,
                    base_config=None):
    """Build a full scenario folder at `<scenarios_root>/<name>/`: the
    electrification overlay (flex_probabilities, via apply_flex_scenario) and
    the paired PV/battery equipment table (pv_spec, via
    _build_equipment_table), written next to symlinks of the shared static
    files -- ready for `process_data({'data_path': <returned path>, ...})` /
    `run_digital_twin(...)` unmodified, exactly as MAS4TE would call them.

    `scenarios_root` defaults to `<base_data_path>/scenarios`, DERIVED from
    `base_data_path` rather than hardcoded -- so a caller that runs with the
    site's own directory as cwd (base_data_path="", as bressoux_setup.ipynb
    does) gets "scenarios", not a path that re-prepends "data/bressoux" onto
    an already-local cwd (which used to nest a stray data/bressoux/ under
    data/bressoux/).

    `base_config`: the scenario's own config (e.g. bressoux_setup.ipynb's
    SCENARIO) -- only used by pv_spec['mode'] == 'generate' (see
    _build_equipment_table), harmless to always pass.

    Returns
    -------
    scenario_data_path : str
    """
    base_data_path = base_data_path.rstrip("/")
    if scenarios_root is None:
        scenarios_root = os.path.join(base_data_path, "scenarios") if base_data_path else "scenarios"
    base_processed = os.path.join(base_data_path, "out", "processed")

    metadata = pd.read_csv(os.path.join(base_processed, "metadata.csv"))
    households = pd.read_csv(os.path.join(base_processed, "load_profiles.csv"), index_col=0)

    new_metadata, new_households = apply_flex_scenario(metadata, households, flex_probabilities, seed)
    equipment = _build_equipment_table(base_data_path, pv_spec, seed, base_config=base_config)
    # equipment is the AUTHORITY for pv_system_size_kwp/battery_size_kwh (see
    # equipment.sync_household_equipment) -- re-sync after the resflex overlay
    # so both layers of the scenario end up reflected in metadata.csv.
    new_metadata = sync_household_equipment(new_metadata, equipment)

    scenario_data_path = os.path.join(scenarios_root, name)
    os.makedirs(os.path.join(scenario_data_path, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(scenario_data_path, "out", "processed"), exist_ok=True)
    _symlink_static_files(base_data_path, scenario_data_path)

    new_metadata.to_csv(os.path.join(scenario_data_path, "out", "processed", "metadata.csv"), index=False)
    new_households.to_csv(os.path.join(scenario_data_path, "out", "processed", "load_profiles.csv"))
    equipment.to_csv(equipment_table_path(scenario_data_path), index=False)

    return scenario_data_path
