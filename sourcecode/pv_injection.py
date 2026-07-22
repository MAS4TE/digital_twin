"""
PV volet -- Brick 3: multi-scenario orchestration for injecting PV as sgen on
the network, from pv_installations.csv (brick 1) and the production profiles of
build_pv_production() (brick 2).

"Pre-built base network + PV layer" architecture: _build_network() is called
ONCE (net_base) -- the cost is reading the 335 MB raw CSV (~14s), NOT the
pandapower in-memory build itself (the deepcopy is a few hundredths of a second).
Each scenario works on a copy.deepcopy(net_base), never on net_base directly, so
sgen/load never accumulate across scenarios.

The injection logic proper (sgen injection, alignment check, loads+PV power flow)
lives in digital_twin.py -- _inject_pv_sgens / _verify_temporal_alignment /
_run_powerflow. This module IMPORTS them (single source, no duplication) and
keeps only the multi-scenario orchestration (build_base_network + deepcopy,
verify_bus_ids on the metadata, run_pv_scenario) used by the impact analysis
(scripts/pv_impact_analysis.py). Also reuses _build_network, _assign_loads,
_build_load_profiles_banque_mensuelle, _config_to_dict.

sgen convention: p_mw POSITIVE = injection (NOT a negative load) -- pandapower's
native behaviour for `sgen` elements (unlike `load`, where positive p_mw =
consumption): no sign inversion to do ourselves, just do not use `create_load`
with a negative p_mw.

q_mvar=0 for all PV sgens -- documented assumption: residential/tertiary
inverters inject essentially active power at the nominal operating point; a Q
control (variable cos phi, Q(V), ...) would be a later refinement, not done here.
"""
import copy

import pandas as pd

from sourcecode.digital_twin import (
    _build_network, _assign_loads, _build_load_profiles_banque_mensuelle, _config_to_dict,
    _inject_pv_sgens, _verify_temporal_alignment, _run_powerflow,
)
from sourcecode.pv_production import build_pv_production

PV_INSTALLATIONS_PATH = "data/pv_installations.csv"
IRRADIANCE_PATH = "data/pvgis_irradiance_quarter_hourly.parquet"


def build_base_network(config=None):
    """Build the network ONCE -- reuse it via copy.deepcopy() for each scenario
    (never modified directly).

    `buses` (dict (street,num)->bus_id) and `mv_nodes` (dict name->bus_id) stay
    valid for ANY copy of net_base: they are integer pandapower bus indices,
    unchanged by a deepcopy (the deepcopy clones net's DataFrames, not the keys/
    values of these external dicts -- no need to copy or rebuild them per scenario).
    """
    net_cfg = _config_to_dict(config)
    net_base, buses, mv_nodes, validation = _build_network(net_cfg)
    return net_base, buses, mv_nodes, validation


def verify_bus_ids(pv_installations, buses, mv_nodes):
    """Check that each bus_id of the PV CSV matches a real network bus:
    residential via `buses` (333 buses), OR an MV passage bus via `mv_nodes`
    (the rest home case, outside the residential coordinates). Raise a clear
    error rather than a silent failure or an opaque pandapower crash later in
    the injection.
    """
    valid_bus_ids = set(int(b) for b in buses.values()) | set(int(b) for b in mv_nodes.values())
    pv_bus_ids = set(int(b) for b in pv_installations["bus_id"])
    missing = sorted(pv_bus_ids - valid_bus_ids)
    if missing:
        raise ValueError(f"bus_id du CSV PV introuvables dans le reseau reel : {missing}")

    rest_home_rows = pv_installations[pv_installations["type"] == "rest_home"]
    for _, row in rest_home_rows.iterrows():
        bus_id = int(row["bus_id"])
        if bus_id not in set(int(b) for b in mv_nodes.values()):
            raise ValueError(
                f"Maison de repos : bus_id={bus_id} n'est pas un bus MV de passage connu "
                f"({sorted(mv_nodes.keys())}) -- verifier la coherence avec sourcecode/pv_metadata.py."
            )
    return True


def run_pv_scenario(net_base, buses, mv_nodes, households, pv_installations, irradiance,
                     config=None, run_powerflow=True):
    """Run a full scenario (loads + PV) on a COPY of net_base -- net_base is
    NEVER modified.

    `config['mois_simule']` drives BOTH the load sub-bank
    (`_build_load_profiles_banque_mensuelle`) and the PV production month
    (`build_pv_production`) -- a single parameter, seasonal consistency
    guaranteed by construction (no two settings to keep in sync by hand).

    Parameters
    ----------
    run_powerflow : bool -- if False, prepares everything (loads, sgen, profiles,
        alignment check) but does NOT run the power flow -- useful for pure
        injection checks without paying the timeseries cost each time.

    Returns
    -------
    dict : net, sgen_bus, bus_to_profile, df_load_p, df_load_q, df_sgen_p,
           and if run_powerflow=True: vm, line_load, trafo_load, sgen_p_out.
    """
    cfg = _config_to_dict(config)
    net = copy.deepcopy(net_base)

    # Loads via the monthly bank. run_digital_twin receives load_profiles as-is;
    # run_pv_scenario relies on the bank for the impact analysis (pass a
    # load_profiles CSV directly to run_digital_twin if needed).
    load_profiles_for_assignment, w_bank = _build_load_profiles_banque_mensuelle(households, cfg)
    load_profiles_for_assignment = load_profiles_for_assignment * cfg['load_scale']

    df_load_p, df_load_q, bus_to_profile, warnings = _assign_loads(
        net, buses, mv_nodes, households, load_profiles_for_assignment, cfg)

    prod_kw = build_pv_production(cfg['mois_simule'], pv_installations, irradiance)
    # kW -> MW. build_pv_production() keeps the text index 'Mon 00:00'..; put it
    # on a positional RangeIndex, identical to df_load_p (which lost the text
    # index via `.values` in _assign_loads). That positional index is what
    # matters at runtime (DFData/ConstControl index by position via
    # time_steps=range(n_timesteps)).
    df_sgen_p = (prod_kw / 1000.0).reset_index(drop=True)
    df_load_p_aligned = df_load_p.reset_index(drop=True)

    _verify_temporal_alignment(df_load_p_aligned, df_sgen_p)   # index guardrail
    sgen_bus = _inject_pv_sgens(net, df_sgen_p)                # sgen from the 'pv_{bus_id}' columns

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

    households = pd.read_csv("data/households.csv")
    pv_installations = pd.read_csv(PV_INSTALLATIONS_PATH)
    irradiance = pd.read_parquet(IRRADIANCE_PATH)

    t0 = time.time()
    net_base, buses, mv_nodes, validation = build_base_network()
    print(f"net_base construit en {time.time() - t0:.2f}s")

    verify_bus_ids(pv_installations, buses, mv_nodes)
    print("bus_id du CSV PV verifies contre le reseau reel : OK")

    t0 = time.time()
    net_copy = copy.deepcopy(net_base)
    print(f"copy.deepcopy(net_base) : {time.time() - t0:.4f}s")

    result = run_pv_scenario(net_base, buses, mv_nodes, households, pv_installations, irradiance,
                              config={"mois_simule": "juin"})
    print(f"sgen crees : {len(result['net'].sgen)} (attendu {len(pv_installations)})")
    print(f"vm shape : {result['vm'].shape}")
