"""
PV generation: installation metadata, irradiance -> production conversion, and
the multi-scenario tooling used to compare PV penetrations (impact analysis).

Brick 1 -- generate_pv_metadata / extend_pv_metadata
    Decides which buses carry a PV installation and of what size, and writes
    pv_installations.csv. Reuses digital_twin.utils.load_site_network
    READ-ONLY to obtain the real topology (bus_id <-> street), so the bus_ids
    generated here match exactly those of the real network.

Brick 2 -- build_pv_production / irradiance_to_power
    PURE computation (no network, no pandapower): converts a PVGIS irradiance
    series into per-installation production profiles for a given month.

Brick 3 -- build_base_network / verify_bus_ids / run_pv_scenario
    Multi-scenario orchestration for injecting PV as sgen on the network:
    "pre-built base network + PV layer" architecture -- build_base_network()
    is called ONCE, each scenario works on a copy.deepcopy() of it, never on
    the base directly, so sgen/load never accumulate across scenarios. The
    injection primitives themselves (sgen injection, alignment check, loads+PV
    power flow) live in digital_twin.digital_twin -- this module imports them
    (single source, no duplication) and keeps only the multi-scenario
    orchestration used by scripts/pv_impact_analysis.py and the check scripts.

    sgen convention: p_mw POSITIVE = injection (NOT a negative load) --
    pandapower's native behaviour for `sgen` elements (unlike `load`, where
    positive p_mw = consumption). q_mvar=0 for all PV sgens -- documented
    assumption: residential/tertiary inverters inject essentially active power
    at the nominal operating point.
"""
import copy
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from ..digital_twin import _assign_loads, _inject_pv_sgens, _run_powerflow, _verify_temporal_alignment
from ..utils import STEPS_PER_DAY, STEPS_PER_WEEK, _WEEK_TIME_INDEX, _config_to_dict, _normalize_month, load_site_network
from .profiles import _build_load_profiles_banque_mensuelle

PV_INSTALLATIONS_PATH = "data/bressoux/inputs/pv_installations.csv"
IRRADIANCE_PATH = "data/bressoux/out/processed/pvgis_irradiance_quarter_hourly.parquet"

DEFAULT_SYSTEM_FACTOR = 0.8

DEFAULTS = {
    "n_installations": 30,
    "kwp_residential": 3.0,
    "kwp_rest_home": 12.0,
    "seed": 42,
    "n_streets_range": (2, 4),     # number of equipped "cluster" streets, drawn in this range
    "min_street_size": 5,          # excludes streets too small from the draw (guardrail;
                                    # never triggered with the 7 real streets, 19-106 buses each)
    # Relative per-street draw weight (default 1.0 for any street absent from
    # this dict). QUALITATIVE calibration from a field report (site visit): PV
    # observed more often on Rue Ernest Malvoz than on Plateau du Bouhay -- not
    # a precise statistic (no exhaustive count provided), just a tendency to
    # reflect. Weight=3.0 makes this street ~3x more likely to be drawn than a
    # neutral one, without ever forcing it (the draw stays probabilistic,
    # seed-driven) -- adjust here if a better estimate (real count) becomes available.
    "street_weights": {"Rue Ernest Malvoz": 3.0},
    "rest_home_name": "Paradis du Bouhay",  # see the site network's MV_LOADS
    "out_path": PV_INSTALLATIONS_PATH,
}


# ============================================================
#  Brick 1 -- PV installation metadata
# ============================================================
def _allocate_across_streets(sizes, total, rng):
    """Distribute `total` installations across streets of sizes `sizes`, in
    proportion to their size, never exceeding a street's size (avoids the
    absurdity of "3 installations on a street of 2 houses"). The (rounded)
    remainder is handed out one by one, in a random permuted order, to the
    streets that still have room.
    """
    sizes = np.asarray(sizes, dtype=float)
    if total > sizes.sum():
        raise ValueError(
            f"n_installations={int(total)} depasse la capacite totale des rues "
            f"choisies ({int(sizes.sum())} bus) -- augmenter n_streets_range."
        )
    weights = sizes / sizes.sum()
    alloc = np.minimum(np.floor(weights * total).astype(int), sizes.astype(int))
    remaining = int(total - alloc.sum())
    order = rng.permutation(len(sizes))
    guard = 0
    while remaining > 0:
        g = order[guard % len(order)]
        if alloc[g] < sizes[g]:
            alloc[g] += 1
            remaining -= 1
        guard += 1
        if guard > 100 * len(sizes) + 1000:
            raise RuntimeError(
                "Repartition des installations par rue : boucle anormale, "
                "capacite insuffisante malgre la verification initiale."
            )
    return alloc


def generate_pv_metadata(config=None):
    """Generate `pv_installations.csv` (or cfg['out_path']).

    Residential placement: GROUPED BY STREET (clusters), not uniform -- draw
    `n_streets` streets (count drawn in n_streets_range, seed-driven), then
    distribute n_installations across them in proportion to their real size
    (see _allocate_across_streets), before drawing, WITHIN each chosen street,
    the exact buses to equip (without replacement, seed-driven). Intent: a few
    heavily equipped streets rather than a homogeneous spread -- neighbourhood/
    installer effect, localized network stress on a single LV feeder.

    If the first-drawn streets do not have enough cumulative capacity for
    n_installations, the selection extends to other eligible streets (permuted
    order) until demand is met -- never triggered with the 7 real streets and the
    defaults, but avoids a silent crash for extreme parameters.

    The rest home (`rest_home_name`) is added explicitly, OUTSIDE the residential
    draw, with kwp_rest_home and type='rest_home'. Its bus_id is that of its
    existing MV load bus (the same bus as its consumption load in
    digital_twin._assign_loads) -- no associated street ('street' column empty),
    since it does not belong to the residential street system.

    Parameters (cfg, all optional -- defaults in DEFAULTS)
    ----------
    n_installations : int    -- number of residential installations (30)
    kwp_residential : float  -- fixed residential size [kWp] (3.0)
    kwp_rest_home : float    -- rest home size [kWp] (12.0)
    seed : int               -- draw reproducibility (42)

    Returns
    -------
    pd.DataFrame : the content written to the CSV (bus_id, kwp, type, street).
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    net_cfg = _config_to_dict(config)  # fixed topology, independent of PV parameters
    network = load_site_network(net_cfg)
    buses, mv_nodes = network['buses'], network['mv_nodes']

    buses_by_street = defaultdict(list)
    for (street, num), b in buses.items():
        buses_by_street[street].append(b)

    rng = np.random.default_rng(cfg["seed"])

    eligible_streets = [s for s, bs in buses_by_street.items() if len(bs) >= cfg["min_street_size"]]
    lo, hi = cfg["n_streets_range"]
    n_streets = int(rng.integers(lo, hi + 1))
    n_streets = min(n_streets, len(eligible_streets))

    # WEIGHTED draw order (without replacement) rather than a uniform
    # permutation: streets with a weight > 1.0 (see street_weights) are more
    # likely to appear early in the order, hence to be chosen as "cluster"
    # streets -- without ever guaranteeing their presence (stays probabilistic).
    street_weights = cfg.get("street_weights") or {}
    weights = np.array([street_weights.get(s, 1.0) for s in eligible_streets], dtype=float)
    probs = weights / weights.sum()
    shuffled = rng.choice(eligible_streets, size=len(eligible_streets), replace=False, p=probs).tolist()

    chosen_streets = shuffled[:n_streets]
    idx_next = n_streets
    while (sum(len(buses_by_street[s]) for s in chosen_streets) < cfg["n_installations"]
           and idx_next < len(shuffled)):
        chosen_streets.append(shuffled[idx_next])
        idx_next += 1

    sizes = [len(buses_by_street[s]) for s in chosen_streets]
    alloc = _allocate_across_streets(sizes, cfg["n_installations"], rng)

    rows = []
    for street, n_take in zip(chosen_streets, alloc):
        if n_take == 0:
            continue
        chosen_buses = rng.choice(buses_by_street[street], size=int(n_take), replace=False)
        for b in chosen_buses:
            rows.append({"bus_id": int(b), "kwp": cfg["kwp_residential"],
                         "type": "residential", "street": street})

    rest_home_bus = mv_nodes[cfg["rest_home_name"]]
    rows.append({"bus_id": int(rest_home_bus), "kwp": cfg["kwp_rest_home"],
                 "type": "rest_home", "street": None})

    out = pd.DataFrame(rows, columns=["bus_id", "kwp", "type", "street"])
    os.makedirs(os.path.dirname(cfg["out_path"]), exist_ok=True)
    out.to_csv(cfg["out_path"], index=False)
    return out


def extend_pv_metadata(base_installations, n_additional, config=None, out_path=None):
    """Extend an existing PV fleet by `n_additional` extra residential
    installations, PRESERVING the existing installations IDENTICALLY (same
    bus_id) -- NOT an independent draw. Needed to compare penetration rates
    (30 vs 50) without mixing rate effect and placement effect.

    Logic (same "clusters by street" spirit as generate_pv_metadata): extend the
    already-equipped streets FIRST (in proportion to their REMAINING capacity,
    not their total size), to reinforce existing clusters rather than seed new
    ones. If their residual capacity is not enough, pick new streets via the same
    weighted mechanism (`street_weights`) as generate_pv_metadata -- never
    triggered with the defaults (30->50 on 2 streets of 106+30 buses, ample
    margin), but a guardrail for larger extensions (80, 100).

    Parameters
    ----------
    base_installations : pd.DataFrame -- existing fleet (e.g. pv_installations.csv)
    n_additional : int                -- number of residential installations to add
    config : dict, optional           -- same keys as DEFAULTS (seed, street_weights, ...)
    out_path : str, optional          -- if provided, writes the extended CSV

    Returns
    -------
    pd.DataFrame : full fleet (existing + new + rest home), same schema.
    """
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    net_cfg = _config_to_dict(config)
    network = load_site_network(net_cfg)
    buses = network['buses']
    buses_by_street = defaultdict(list)
    for (street, num), b in buses.items():
        buses_by_street[street].append(b)

    base_res = base_installations[base_installations["type"] == "residential"].copy()
    rest_home_rows = base_installations[base_installations["type"] == "rest_home"].copy()
    existing_bus_ids = set(int(b) for b in base_res["bus_id"])
    existing_streets = [s for s in base_res["street"].dropna().unique()]

    rng = np.random.default_rng(cfg["seed"])

    # 1) Extend the already-equipped streets first, in proportion to their
    #    REMAINING capacity (buses not yet equipped).
    remaining_in_existing = {
        s: [b for b in buses_by_street[s] if int(b) not in existing_bus_ids]
        for s in existing_streets
    }
    total_remaining_existing = sum(len(v) for v in remaining_in_existing.values())

    n_from_existing = min(n_additional, total_remaining_existing)
    new_rows = []
    if n_from_existing > 0:
        sizes = [len(remaining_in_existing[s]) for s in existing_streets]
        alloc = _allocate_across_streets(sizes, n_from_existing, rng)
        for street, n_take in zip(existing_streets, alloc):
            if n_take == 0:
                continue
            chosen = rng.choice(remaining_in_existing[street], size=int(n_take), replace=False)
            for b in chosen:
                new_rows.append({"bus_id": int(b), "kwp": cfg["kwp_residential"],
                                 "type": "residential", "street": street})

    # 2) If not enough, pick new streets via the same weighted mechanism.
    still_needed = n_additional - n_from_existing
    if still_needed > 0:
        eligible_streets = [s for s, bs in buses_by_street.items()
                            if len(bs) >= cfg["min_street_size"] and s not in existing_streets]
        street_weights = cfg.get("street_weights") or {}
        weights = np.array([street_weights.get(s, 1.0) for s in eligible_streets], dtype=float)
        probs = weights / weights.sum()
        shuffled = rng.choice(eligible_streets, size=len(eligible_streets), replace=False, p=probs).tolist()

        chosen_new_streets = []
        idx_next = 0
        while (sum(len(buses_by_street[s]) for s in chosen_new_streets) < still_needed
               and idx_next < len(shuffled)):
            chosen_new_streets.append(shuffled[idx_next])
            idx_next += 1

        sizes = [len(buses_by_street[s]) for s in chosen_new_streets]
        alloc = _allocate_across_streets(sizes, still_needed, rng)
        for street, n_take in zip(chosen_new_streets, alloc):
            if n_take == 0:
                continue
            chosen = rng.choice(buses_by_street[street], size=int(n_take), replace=False)
            for b in chosen:
                new_rows.append({"bus_id": int(b), "kwp": cfg["kwp_residential"],
                                 "type": "residential", "street": street})

    new_res = pd.DataFrame(new_rows, columns=["bus_id", "kwp", "type", "street"])
    out = pd.concat([base_res, new_res, rest_home_rows], ignore_index=True)[
        ["bus_id", "kwp", "type", "street"]
    ]

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.to_csv(out_path, index=False)
    return out


# ============================================================
#  Brick 2 -- irradiance -> production
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
        f"Aucune semaine complete de {STEPS_PER_WEEK} pas trouvee pour le mois {month_num} "
        f"dans l'irradiance -- verifier la couverture de {IRRADIANCE_PATH}."
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


def build_pv_production(mois, pv_installations, irradiance, system_factor=DEFAULT_SYSTEM_FACTOR):
    """Build the PV production profiles (kW) of all installations in
    `pv_installations` (residential + rest home), for the representative week of
    month `mois`.

    Parameters
    ----------
    mois : str | int              -- target month (French name or 1-12)
    pv_installations : pd.DataFrame -- pv_installations.csv (bus_id, kwp, type, street)
    irradiance : pd.DataFrame       -- pvgis_irradiance_quarter_hourly.parquet (timestamp, ghi_w_m2, year)
    system_factor : float           -- see irradiance_to_power

    Returns
    -------
    pd.DataFrame : index 'Mon 00:00'..'Sun 23:45' (672 steps, SAME format as
    households), columns 'pv_{bus_id}' (kW), one per installation.
    """
    irr_week, monday = _select_representative_week(irradiance, mois)

    data = {}
    for _, row in pv_installations.iterrows():
        col = f"pv_{int(row['bus_id'])}"
        data[col] = irradiance_to_power(irr_week, row["kwp"], system_factor)

    out = pd.DataFrame(data, index=_WEEK_TIME_INDEX)
    out.index.name = "week_step"
    out.attrs["source_week_start"] = monday.isoformat()
    return out


# ============================================================
#  Brick 3 -- multi-scenario PV injection tooling (analysis only)
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


def verify_bus_ids(pv_installations, network):
    """Check that each bus_id of the PV CSV matches a real network bus:
    residential via `network['buses']` (333 buses), OR an MV load bus via
    `network['mv_nodes']` (the rest home case, outside the residential
    coordinates). Raise a clear error rather than a silent failure or an
    opaque pandapower crash later in the injection.
    """
    buses, mv_nodes = network['buses'], network['mv_nodes']
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
                f"Maison de repos : bus_id={bus_id} n'est pas un bus de charge MV connu "
                f"({sorted(mv_nodes.keys())}) -- verifier la coherence avec generate_pv_metadata."
            )
    return True


def run_pv_scenario(network, metadata, pv_installations, irradiance,
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

    metadata = pd.read_csv("data/bressoux/out/processed/metadata.csv")
    pv_installations = pd.read_csv(PV_INSTALLATIONS_PATH)
    irradiance = pd.read_parquet(IRRADIANCE_PATH)

    t0 = time.time()
    network = build_base_network()
    print(f"network construit en {time.time() - t0:.2f}s")

    verify_bus_ids(pv_installations, network)
    print("bus_id du CSV PV verifies contre le reseau reel : OK")

    t0 = time.time()
    net_copy = copy.deepcopy(network['net'])
    print(f"copy.deepcopy(network['net']) : {time.time() - t0:.4f}s")

    result = run_pv_scenario(network, metadata, pv_installations, irradiance,
                              config={"mois_simule": "juin"})
    print(f"sgen crees : {len(result['net'].sgen)} (attendu {len(pv_installations)})")
    print(f"vm shape : {result['vm'].shape}")

    n_res = int((pv_installations["type"] == "residential").sum())
    n_rest = int((pv_installations["type"] == "rest_home").sum())
    print(f"{PV_INSTALLATIONS_PATH} : {len(pv_installations)} lignes "
          f"({n_res} residentielles + {n_rest} maison de repos)")
