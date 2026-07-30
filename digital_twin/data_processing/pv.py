"""
PV: irradiance -> production conversion, and the multi-scenario tooling used to
compare PV penetrations (impact analysis).

WHERE the panels are is NOT decided here -- placement lives in
data_processing.equipment, the single bus-level register that also carries the
batteries. This module takes that register as an input and turns it into power.
The split matters: placement is a scenario decision about the grid, production
is physics about the sun.

Brick 1 -- build_pv_production / irradiance_to_power
    PURE computation (no network, no pandapower): converts a PVGIS irradiance
    series into per-installation production profiles for a given month.

Brick 2 -- build_base_network / run_pv_scenario
    Multi-scenario orchestration for injecting PV as sgen on the network:
    "pre-built base network + PV layer" architecture -- build_base_network()
    is called ONCE, each scenario works on a copy.deepcopy() of it, never on
    the base directly, so sgen/load never accumulate across scenarios. The
    injection primitives themselves (sgen injection, alignment check, loads+PV
    power flow) live in digital_twin.digital_twin -- this module imports them
    (single source, no duplication) and keeps only the multi-scenario
    orchestration used by scripts/pv_impact_analysis.py.

    sgen convention: p_mw POSITIVE = injection (NOT a negative load) --
    pandapower's native behaviour for `sgen` elements (unlike `load`, where
    positive p_mw = consumption). q_mvar=0 for all PV sgens -- documented
    assumption: residential/tertiary inverters inject essentially active power
    at the nominal operating point.

    Bus ids are not verified here: the contract lives in utils.check_bus_ids,
    called from digital_twin._validate_inputs, on the path every caller takes.
"""
import copy

import numpy as np
import pandas as pd

from ..digital_twin import _assign_loads, _inject_pv_sgens, _run_powerflow, _verify_temporal_alignment
from ..utils import (STEPS_PER_WEEK, _WEEK_TIME_INDEX, _config_to_dict,
                     _normalize_month, irradiance_path_for, load_site_network)
from .equipment import PV_SIZE_COLUMN
from .profiles import _build_load_profiles_banque_mensuelle

# Bressoux paths, used by this module's __main__ demo only. Everything on the
# execution path resolves its own from config['data_path'] (see
# utils.irradiance_path_for) -- these are not a fallback for another site.
SCENARIO_DATA_PATH = "data/bressoux"
IRRADIANCE_PATH = irradiance_path_for(SCENARIO_DATA_PATH)

DEFAULT_SYSTEM_FACTOR = 0.8


# ============================================================
#  Brick 1 -- irradiance -> production
# ============================================================
def _select_representative_week(irradiance, mois):
    """Real calendar week (672 quarter-hourly steps) representative of the
    requested month: the first Monday 00:00 whose month matches `mois` AND that
    has a full week of data following (may spill into the next month, like the
    real weeks of the load bank -- not an average, a real week).

    Returns
    -------
    irr_week : np.ndarray (672,)  -- irradiance [W/m2]
    monday : pd.Timestamp         -- start of the selected week
    """
    month_num = _normalize_month(mois)
    ts = pd.to_datetime(irradiance["timestamp"])
    s = pd.Series(irradiance["ghi_w_m2"].to_numpy(dtype=float), index=ts).sort_index()

    mondays = s.index[(s.index.weekday == 0) & (s.index.hour == 0)
                      & (s.index.minute == 0) & (s.index.month == month_num)]
    for monday in mondays:
        week = s.loc[monday: monday + pd.Timedelta(minutes=15 * (STEPS_PER_WEEK - 1))]
        if len(week) == STEPS_PER_WEEK:
            return week.to_numpy(dtype=float), monday
    raise ValueError(
        f"No complete {STEPS_PER_WEEK}-step week found for month {month_num} in the "
        f"irradiance -- check the coverage of {IRRADIANCE_PATH}."
    )


def irradiance_to_power(irradiance_w_m2, kwp, system_factor=DEFAULT_SYSTEM_FACTOR):
    """Linear model: power [kW] = irradiance [W/m2] / 1000 * kWp * system_factor.
    Isolated here so it can be replaced by a full pvlib model (temperature,
    orientation, inverter) without touching the rest of the brick.

    `system_factor` (default 0.8) encapsulates, IN A SINGLE NUMBER, both the
    real system efficiency (inverter ~97%, wiring ~98%, soiling/mismatch ~95%)
    AND the fact that we use GLOBAL HORIZONTAL irradiance (GHI, tilt=0) rather
    than plane-of-array irradiance (POA, tilt ~30-35 typical in Belgium).

    Clips input irradiance at 0 (defensive: guarantees strictly zero, never
    negative, production at night, even if the source irradiance contained a
    negative numerical residue).
    """
    irr = np.maximum(np.asarray(irradiance_w_m2, dtype=float), 0.0)
    return irr / 1000.0 * kwp * system_factor


def build_pv_production(mois, equipment, irradiance, system_factor=DEFAULT_SYSTEM_FACTOR):
    """Build the PV production profiles (kW) of every equipped bus, for the
    representative week of month `mois`.

    Reads the EQUIPMENT TABLE (data_processing.equipment) and keeps the rows
    whose `pv_system_size_kwp` is > 0 -- residential buses and MV load facilities
    alike, the distinction mattering at placement and to the bus_id contract, not
    here. A table with no equipped row yields an EMPTY frame, which is what makes
    "no PV in this scenario" expressible without a flag.

    Parameters
    ----------
    mois : str | int          -- target month (French name or 1-12)
    equipment : pd.DataFrame  -- the equipment table (bus_id, pv_system_size_kwp, ...)
    irradiance : pd.DataFrame -- pvgis parquet (timestamp, ghi_w_m2, year)
    system_factor : float     -- see irradiance_to_power

    Returns
    -------
    pd.DataFrame : index 'Mon 00:00'..'Sun 23:45' (672 steps, SAME format as
    households), columns 'pv_{bus_id}' (kW), one per equipped bus.
    """
    if PV_SIZE_COLUMN not in equipment.columns:
        raise ValueError(
            f"The equipment table has no '{PV_SIZE_COLUMN}' column: columns received "
            f"{sorted(equipment.columns)[:10]}."
        )
    irr_week, monday = _select_representative_week(irradiance, mois)

    equipped = equipment[equipment[PV_SIZE_COLUMN].fillna(0.0) > 0]
    data = {
        f"pv_{int(row['bus_id'])}": irradiance_to_power(irr_week, row[PV_SIZE_COLUMN],
                                                        system_factor)
        for _, row in equipped.iterrows()
    }

    out = pd.DataFrame(data, index=_WEEK_TIME_INDEX)
    out.index.name = "week_step"
    out.attrs["source_week_start"] = monday.isoformat()
    return out


# ============================================================
#  Brick 2 -- multi-scenario PV injection tooling (analysis only)
# ============================================================
def build_base_network(config=None):
    """Build the network ONCE -- reuse it via copy.deepcopy() for each scenario
    (never modified directly).

    Returns
    -------
    network : dict  -- same bundle as digital_twin.utils.load_site_network
        ("net", "buses", "mv_nodes", "mv_load_names", "validation", "warnings").
        "buses"/"mv_nodes" stay valid for ANY copy of network['net']: they are
        integer pandapower bus indices, unchanged by a deepcopy.
    """
    net_cfg = _config_to_dict(config)
    return load_site_network(net_cfg)


def run_pv_scenario(network, metadata, equipment, irradiance,
                     config=None, run_powerflow=True):
    """Run a full scenario (loads + PV) on a COPY of network['net'] -- the
    base network is NEVER modified.

    `config['mois_simule']` drives BOTH the load sub-bank
    (`_build_load_profiles_banque_mensuelle`) and the PV production month
    (`build_pv_production`) -- a single parameter, seasonal consistency
    guaranteed by construction (no two settings to keep in sync by hand).

    Parameters
    ----------
    network : dict -- from build_base_network(); network['net'] is deep-copied,
        never mutated.
    equipment : pd.DataFrame -- the equipment table for THIS scenario; its
        pv_system_size_kwp column is what defines the penetration being tested
        (see equipment.extend_pv for building nested fleets).
    run_powerflow : bool -- if False, prepares everything (loads, sgen, profiles,
        alignment check) but does NOT run the power flow -- useful for pure
        injection checks without paying the timeseries cost each time.

    Returns
    -------
    dict : net, sgen_bus, bus_to_profile, df_load_p, df_load_q, df_sgen_p,
           and if run_powerflow=True: vm, line_load, trafo_load, sgen_p_out.
    """
    cfg = _config_to_dict(config)
    net = copy.deepcopy(network['net'])
    scenario_network = {**network, "net": net}

    # Loads via the monthly bank. run_digital_twin receives households as-is;
    # run_pv_scenario relies on the bank for the impact analysis (pass a
    # households CSV directly to run_digital_twin if needed).
    households_for_assignment, w_bank = _build_load_profiles_banque_mensuelle(metadata, cfg)

    df_load_p, df_load_q, bus_to_profile, warnings = _assign_loads(
        scenario_network, metadata, households_for_assignment, cfg)

    prod_kw = build_pv_production(cfg['mois_simule'], equipment, irradiance)
    # kW -> MW, then onto a positional RangeIndex matching df_load_p (which lost
    # its text index via `.values` in _assign_loads). The positional index is
    # what matters at runtime: DFData/ConstControl index by position via
    # time_steps=range(n_timesteps).
    df_sgen_p = (prod_kw / 1000.0).reset_index(drop=True)
    df_load_p_aligned = df_load_p.reset_index(drop=True)

    _verify_temporal_alignment(df_load_p_aligned, df_sgen_p)
    sgen_bus = _inject_pv_sgens(net, df_sgen_p)

    result = {
        "net": net, "sgen_bus": sgen_bus, "bus_to_profile": bus_to_profile,
        "df_load_p": df_load_p, "df_load_q": df_load_q, "df_sgen_p": df_sgen_p,
        "warnings": warnings,
    }

    if run_powerflow:
        pf_out = _run_powerflow(net, df_load_p, df_load_q, cfg, df_pv_p=df_sgen_p)
        result.update(pf_out)

    return result


if __name__ == "__main__":
    import time

    from ..utils import BUS_DOMAIN_EQUIPABLE, check_bus_ids
    from .equipment import load_equipment_table

    scenario = {"data_path": SCENARIO_DATA_PATH, "mois_simule": "juin"}
    metadata = pd.read_csv(f"{SCENARIO_DATA_PATH}/out/processed/metadata.csv")
    equipment = load_equipment_table(scenario)
    irradiance = pd.read_parquet(IRRADIANCE_PATH)

    t0 = time.time()
    network = build_base_network(scenario)
    print(f"network built in {time.time() - t0:.2f}s")

    equipped = equipment[equipment[PV_SIZE_COLUMN].fillna(0.0) > 0]
    check_bus_ids(equipped["bus_id"], network, BUS_DOMAIN_EQUIPABLE, "equipment table")
    print(f"bus_id of the {len(equipped)} installations checked against the network: OK")

    t0 = time.time()
    net_copy = copy.deepcopy(network["net"])
    print(f"copy.deepcopy(network['net']): {time.time() - t0:.4f}s")

    result = run_pv_scenario(network, metadata, equipment, irradiance, config=scenario)
    print(f"sgens created: {len(result['net'].sgen)} (expected {len(equipped)})")
    print(f"vm shape: {result['vm'].shape}")
