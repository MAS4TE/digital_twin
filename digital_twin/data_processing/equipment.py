"""
Bus-level equipment register: ONE row per bus that can carry equipment, derived
from the network itself rather than read from a hand-maintained file.

This is the object the whole scenario definition hangs on. It is EXHAUSTIVE by
construction: every equipable bus of the network is a row, equipped or not. That
is what makes the bus_id contract checkable by SET comparison rather than by
cardinality -- 333 rows for 333 buses can coincide while hiding a placement
disagreement.

WHAT COMES FROM WHERE
    * the skeleton (bus_id, bus_type, street, num, lon, lat) is DERIVED from the
      network -- no draw, no scenario choice, reproducible from inputs/network.py
      alone. Nothing here is Bressoux-specific: it reads only network['buses'],
      network['mv_nodes'], network['mv_load_names'] and net.bus.
    * the equipment columns (pv_system_size_kwp, battery_size_kwh, heat_pump,
      electric_vehicle) are a scenario DECISION, filled by the placement layer
      below. The network cannot know them.

COLUMN NAMING. The equipment columns deliberately reuse the MAS4TE household
schema names (see profiles.MAS4TE_METADATA_COLUMNS): `battery_size_kwh` and
`pv_system_size_kwp` are the SAME names metadata.csv carries. The join between
this bus-level table and the household file is then obvious, which is what the
metadata<->schedule coherence check needs.

BATTERY IDENTIFIERS. `battery_id` is b1..bn, assigned in ascending bus_id order
so the mapping is stable across runs. **b0 is NOT in this table**: it is the
CENTRAL battery, sitting on an MV node, and its placement and size already have
their own config keys (bss_bus / bss_park_e_kwh / bss_park_p_kw). This table is
about batteries that live on a bus of the distribution grid; the central one is
injected "in addition to" them (see digital_twin._inject_bss_storage).

ROWS ARE NOT ALL EQUIPABLE THE SAME WAY. `bus_type` separates:
    * 'lv_residential' -- the 333 LV charge buses. PV and domestic batteries.
    * 'mv_load'        -- the 4 MV load facilities (rest home, school, sports
                          hall, horticulture centre). PV only: the rest home
                          carries a large installation in the current scenario,
                          and it has no LV bus of its own. A DOMESTIC battery on
                          an MV bus is a placement error, which is exactly the
                          distinction the bus_id contract has to enforce.
"""
import os

import numpy as np
import pandas as pd

from ..utils import (BSS_ID_COLUMN as BATTERY_ID_COLUMN, BSS_SIZE_COLUMN,
                     _config_to_dict, bus_lonlat, load_site_network)

EQUIPMENT_TABLE_FILE = "equipment.csv"

# Column names shared with the MAS4TE household schema, exported so consumers
# (pv.build_pv_production, the BSS layer) name them once instead of repeating a
# string literal that could drift from the table. The two BSS ones are DEFINED in
# utils and re-exported here: the engine has to name them too, and it cannot
# import this package back (see utils.BSS_ID_COLUMN for why).
PV_SIZE_COLUMN = 'pv_system_size_kwp'

# Equipment columns and their "not equipped" default, in table order.
# dtypes mirror profiles.MAS4TE_METADATA_COLUMNS so the two tables stay joinable.
EQUIPMENT_COLUMNS = {
    PV_SIZE_COLUMN:     ('Float64', 0.0),
    BATTERY_ID_COLUMN:  ('string', pd.NA),
    BSS_SIZE_COLUMN:    ('Float64', 0.0),
    'heat_pump':        ('boolean', False),
    'electric_vehicle': ('boolean', False),
}

BUS_TYPE_LV = 'lv_residential'
BUS_TYPE_MV_LOAD = 'mv_load'

# SITE-CALIBRATED values do NOT belong here. `street_weights` (a field-report
# tendency observed on one street of Bressoux) and `mv_pv_installations` (named
# facilities of that network) describe a specific neighbourhood, so they default
# to neutral and the scenario passes its own. In DEFAULTS they would be dead
# weight on any other site -- silently matching nothing, which reads as "no
# clustering preference" instead of "this default was never meant for you".
#
# Bressoux's own values, for reference:
#     street_weights      = {"Rue Ernest Malvoz": 3.0}
#     mv_pv_installations = {"Paradis du Bouhay": 12.0}
DEFAULTS = {
    "seed": 42,
    # --- PV placement ---
    "n_installations": 30,
    "kwp_residential": 3.0,
    # PV on the MV LOAD FACILITIES, as {facility name -> kWp}. Named by ROLE
    # rather than by building type: what matters to the model is "an MV load bus
    # carrying a large array", not that Bressoux's happens to be a rest home. A
    # dict rather than a name/size pair, because a network may have several such
    # facilities (a school, a sports hall, a horticulture centre, ...).
    "mv_pv_installations": {},
    "n_streets_range": (2, 4),
    "min_street_size": 5,
    "street_weights": {},       # scenario-supplied: {street -> relative weight}
    # --- Domestic battery placement ---
    "n_batteries": 20,          # domestic batteries, i.e. b1..bn
    "battery_kwh": 10.0,        # uniform size [kWh], residential order of magnitude
    # Domestic batteries are drawn AMONG THE PV-EQUIPPED BUSES by default: a home
    # battery is almost always installed alongside a PV system, and the BSS
    # schedule of a bus is driven by its net load (load - PV), so a battery
    # without PV degenerates into night-valley/evening-peak arbitrage only.
    # Set False to place them independently over all LV buses.
    "batteries_follow_pv": True,
}


# ============================================================
#  Brick 1 -- skeleton, derived from the network (no draw)
# ============================================================
def build_equipment_skeleton(network):
    """One row per equipable bus, with every equipment column at its "off" value.

    Purely derived: same network in, same table out. The scenario layer
    (place_pv / place_bss) fills the equipment columns afterwards.

    Parameters
    ----------
    network : dict   -- the bundle from utils.load_site_network

    Returns
    -------
    pd.DataFrame  # bus_id, bus_type, name, street, num, lon, lat + EQUIPMENT_COLUMNS
    """
    net = network['net']
    rows = []

    for (street, num), b in network['buses'].items():
        lon, lat = bus_lonlat(net, b)
        rows.append({
            'bus_id': int(b), 'bus_type': BUS_TYPE_LV,
            'name': str(net.bus.loc[b, 'name']), 'street': street, 'num': str(num),
            'lon': lon, 'lat': lat,
        })

    # MV load facilities: equipable (PV), but never with a DOMESTIC battery.
    for facility in network['mv_load_names']:
        b = network['mv_nodes'][facility]
        lon, lat = bus_lonlat(net, b)
        rows.append({
            'bus_id': int(b), 'bus_type': BUS_TYPE_MV_LOAD,
            'name': facility, 'street': pd.NA, 'num': pd.NA,
            'lon': lon, 'lat': lat,
        })

    table = pd.DataFrame(rows).sort_values('bus_id').reset_index(drop=True)
    for col, (dtype, default) in EQUIPMENT_COLUMNS.items():
        table[col] = pd.array([default] * len(table), dtype=dtype)
    return table


# ============================================================
#  Brick 2 -- equipment placement (the scenario decision)
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
            f"{int(total)} requested installations exceed the total capacity of the "
            f"chosen streets ({int(sizes.sum())} buses) -- widen n_streets_range."
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
                "Per-street allocation: runaway loop, capacity insufficient despite "
                "the initial check."
            )
    return alloc


def _draw_clustered_buses(candidate_buses_by_street, n_total, cfg, rng, label="placement"):
    """Draw `n_total` buses GROUPED BY STREET (clusters), not spread uniformly.

    One placement rule shared by PV and batteries: draw a few streets (count in
    n_streets_range, weighted by street_weights), distribute n_total across them
    in proportion to their real size, then draw the exact buses within each.
    Intent: a few heavily equipped streets rather than a homogeneous spread --
    neighbourhood/installer effect, localized stress on a single LV feeder.

    Streets below `min_street_size` are excluded, UNLESS that would leave nothing
    eligible (the case when drawing from an already-small pool, e.g. batteries
    among PV buses) -- then every street of the pool is eligible.

    Parameters
    ----------
    candidate_buses_by_street : dict {street -> [bus_id, ...]}
    n_total : int    -- buses to equip; MUST fit in the pool (see below)
    cfg, rng         -- merged config and seeded generator
    label : str      -- what is being placed, for the error message

    Returns
    -------
    list[int] : chosen bus_ids

    Raises
    ------
    ValueError
        When `n_total` exceeds the candidate pool. Deliberately NOT capped
        silently: on a smaller network than the one the defaults were tuned for,
        a silent cap turns "equip 30 of 333 buses" (9% penetration) into "equip
        all 19" (100%) with nothing to warn the caller that the scenario they
        described is not the scenario they got.
    """
    pool_size = sum(len(v) for v in candidate_buses_by_street.values())
    n_total = int(n_total)
    if n_total > pool_size:
        raise ValueError(
            f"{label}: {n_total} requested for only {pool_size} candidate bus(es). "
            "Lower the request, or widen the pool (on a network smaller than the one "
            "the defaults were tuned for, those defaults need readjusting)."
        )
    if n_total <= 0:
        return []

    eligible = [s for s, bs in candidate_buses_by_street.items()
                if len(bs) >= cfg["min_street_size"]]
    if not eligible or sum(len(candidate_buses_by_street[s]) for s in eligible) < n_total:
        # Small pool: the size guardrail would starve the draw, so drop it rather
        # than fail. Not a fallback that hides a problem -- the guardrail only
        # exists to avoid "3 installations on a street of 2 houses".
        eligible = list(candidate_buses_by_street)

    lo, hi = cfg["n_streets_range"]
    n_streets = min(int(rng.integers(lo, hi + 1)), len(eligible))

    # WEIGHTED draw order (without replacement): a street with weight > 1 is more
    # likely to come out early, hence to be picked as a cluster -- never
    # guaranteed, the draw stays probabilistic and seed-driven.
    weights = np.array([cfg.get("street_weights", {}).get(s, 1.0) for s in eligible],
                       dtype=float)
    shuffled = rng.choice(eligible, size=len(eligible), replace=False,
                          p=weights / weights.sum()).tolist()

    chosen_streets = shuffled[:n_streets]
    idx_next = n_streets
    while (sum(len(candidate_buses_by_street[s]) for s in chosen_streets) < n_total
           and idx_next < len(shuffled)):
        chosen_streets.append(shuffled[idx_next])
        idx_next += 1

    sizes = [len(candidate_buses_by_street[s]) for s in chosen_streets]
    alloc = _allocate_across_streets(sizes, n_total, rng)

    chosen = []
    for street, n_take in zip(chosen_streets, alloc):
        if n_take == 0:
            continue
        chosen += [int(b) for b in
                   rng.choice(candidate_buses_by_street[street], size=int(n_take), replace=False)]
    return chosen


def _buses_by_street(table, mask=None):
    """{street -> [bus_id, ...]} over the LV rows of `table` (optionally masked)."""
    lv = table[table['bus_type'] == BUS_TYPE_LV]
    if mask is not None:
        lv = lv[mask.reindex(lv.index, fill_value=False)]
    return {street: grp['bus_id'].astype(int).tolist()
            for street, grp in lv.groupby('street')}


def place_pv(table, config=None, rng=None):
    """Fill `pv_system_size_kwp`: cfg['n_installations'] residential installations
    clustered by street, plus each facility named in cfg['mv_pv_installations'].

    MV load facilities are placed OUTSIDE the residential draw: they do not
    belong to the street system the clustering works on, and they sit on MV load
    buses. Their sizes come straight from the dict, so a network can carry
    several of them at different ratings.

    An unknown facility name RAISES rather than being skipped: the dict defaults
    to empty, so a site with no such installation simply says nothing -- and a
    name that matches nothing is then a typo or a stale scenario, not a portable
    default. Silently skipping it would report "12 kWp placed" while placing
    none.

    Mutates and returns `table`.
    """
    cfg = {**DEFAULTS, **(config or {})}
    rng = rng if rng is not None else np.random.default_rng(cfg["seed"])

    chosen = _draw_clustered_buses(_buses_by_street(table), cfg["n_installations"],
                                   cfg, rng, label="PV installations")
    table.loc[table['bus_id'].isin(chosen), PV_SIZE_COLUMN] = cfg["kwp_residential"]

    for facility, kwp in (cfg.get("mv_pv_installations") or {}).items():
        target = table['name'] == facility
        if not target.any():
            known = sorted(table.loc[table['bus_type'] == BUS_TYPE_MV_LOAD, 'name'])
            raise ValueError(
                f"mv_pv_installations: no facility named {facility!r} in this network. "
                f"Available names (MV load buses): {known}."
            )
        table.loc[target, PV_SIZE_COLUMN] = float(kwp)
    return table


def extend_pv(table, n_additional, config=None, rng=None):
    """Add `n_additional` residential PV installations to an ALREADY PLACED table,
    keeping every existing installation exactly where it is.

    This is what makes penetration rates comparable: a 30 -> 50 comparison must
    isolate the RATE effect, so the 50-fleet has to contain the 30-fleet. An
    independent re-draw would mix rate and placement and make the delta
    uninterpretable. Used by scripts/pv_impact_analysis.py.

    Same clustering spirit as place_pv, in two stages:
      1. fill the already-equipped streets first, in proportion to their
         REMAINING free buses -- reinforcing existing clusters rather than
         seeding new ones;
      2. only if that is not enough, open new streets through the same weighted
         mechanism.

    Works on the table alone -- street membership and free capacity are already
    columns, so nothing has to be recomputed from the network.

    Mutates and returns `table` (pass a .copy() to keep the base fleet intact).
    """
    cfg = {**DEFAULTS, **(config or {})}
    rng = rng if rng is not None else np.random.default_rng(cfg["seed"])
    if n_additional <= 0:
        return table

    lv = table[table['bus_type'] == BUS_TYPE_LV]
    has_pv = lv['pv_system_size_kwp'].fillna(0.0) > 0
    equipped_streets = sorted(lv.loc[has_pv, 'street'].dropna().unique())

    free_by_street = {
        street: grp.loc[grp['pv_system_size_kwp'].fillna(0.0) <= 0, 'bus_id'].astype(int).tolist()
        for street, grp in lv.groupby('street')
    }
    total_free = sum(len(v) for v in free_by_street.values())
    if n_additional > total_free:
        raise ValueError(
            f"PV extension: {n_additional} installation(s) requested for "
            f"{total_free} LV bus(es) still free."
        )

    chosen = []

    # 1) already-equipped streets, in proportion to their remaining free buses
    free_in_equipped = {s: free_by_street.get(s, []) for s in equipped_streets}
    free_in_equipped = {s: v for s, v in free_in_equipped.items() if v}
    n_from_existing = min(n_additional, sum(len(v) for v in free_in_equipped.values()))
    if n_from_existing > 0:
        streets = list(free_in_equipped)
        alloc = _allocate_across_streets([len(free_in_equipped[s]) for s in streets],
                                         n_from_existing, rng)
        for street, n_take in zip(streets, alloc):
            if n_take:
                chosen += [int(b) for b in
                           rng.choice(free_in_equipped[street], size=int(n_take), replace=False)]

    # 2) new streets, same weighted mechanism as place_pv
    still_needed = n_additional - n_from_existing
    if still_needed > 0:
        fresh = {s: v for s, v in free_by_street.items()
                 if v and s not in equipped_streets}
        chosen += _draw_clustered_buses(fresh, still_needed, cfg, rng,
                                        label="PV extension (new streets)")

    table.loc[table['bus_id'].isin(chosen), 'pv_system_size_kwp'] = cfg["kwp_residential"]
    return table


def place_bss(table, config=None, rng=None):
    """Fill `battery_size_kwh` and `battery_id` for the DOMESTIC batteries.

    Candidates are the PV-equipped LV buses when cfg['batteries_follow_pv'] (the
    default, see DEFAULTS for why), all LV buses otherwise. MV load buses are
    never candidates: a domestic battery on an MV bus is a placement error.

    `battery_id` is assigned b1..bn in ascending bus_id order -- deterministic,
    so the same table always yields the same identifiers, and b0 stays free for
    the central battery (which is not a row of this table).

    Mutates and returns `table`.
    """
    cfg = {**DEFAULTS, **(config or {})}
    rng = rng if rng is not None else np.random.default_rng(cfg["seed"])

    if cfg["batteries_follow_pv"]:
        mask = table[PV_SIZE_COLUMN].fillna(0.0) > 0
        candidates = _buses_by_street(table, mask=mask)
    else:
        candidates = _buses_by_street(table)

    chosen = sorted(_draw_clustered_buses(
        candidates, cfg["n_batteries"], cfg, rng,
        label=("Domestic batteries (pool = PV-equipped buses)"
               if cfg["batteries_follow_pv"] else "Domestic batteries")))
    equipped = table['bus_id'].isin(chosen)
    table.loc[equipped, BSS_SIZE_COLUMN] = cfg["battery_kwh"]
    # b1..bn: b0 is reserved for the central MV battery, which this table does
    # not describe (see the module docstring). Assigned in ascending bus_id order
    # (`chosen` is sorted), so the same table always yields the same identifiers
    # -- the schedule's b1..bn are matched against them by SET in
    # digital_twin._validate_inputs.
    table.loc[equipped, BATTERY_ID_COLUMN] = pd.array(
        [f"b{i}" for i in range(1, len(chosen) + 1)], dtype='string')
    return table


def sync_household_equipment(metadata, equipment):
    """Project the BUS-level equipment onto the HOUSEHOLD rows of `metadata`.

    `metadata` carries `pv_system_size_kwp` and `battery_size_kwh` because the
    MAS4TE household schema does -- but the AUTHORITY for both is this bus-level
    table, since that is where placement is decided. Left unfilled they keep
    their 0.0 default, which reads as "this household has no battery" for buses
    that do have one.

    Joined on `bus_id`, which metadata carries 1:1 with profile_id (see
    profiles.generate_household_metadata). A household whose bus is absent from
    the table keeps its current value rather than being zeroed -- absence of a
    row is "unknown", not "nothing installed".

    Returns a COPY; `metadata` is not mutated.
    """
    if metadata is None or 'bus_id' not in getattr(metadata, 'columns', []):
        raise ValueError(
            "`metadata` must carry a 'bus_id' column to be joined with the equipment "
            "table (see profiles.generate_household_metadata)."
        )
    if equipment is None or 'bus_id' not in getattr(equipment, 'columns', []):
        raise ValueError("`equipment` must carry a 'bus_id' column.")

    out = metadata.copy()
    by_bus = equipment.drop_duplicates('bus_id').set_index('bus_id')
    for col in (PV_SIZE_COLUMN, BSS_SIZE_COLUMN):
        if col not in equipment.columns:
            continue
        mapped = out['bus_id'].map(pd.to_numeric(by_bus[col], errors='coerce'))
        out[col] = mapped.fillna(pd.to_numeric(out[col], errors='coerce')).astype('Float64')
    return out


# ============================================================
#  Brick 3 -- orchestration and I/O
# ============================================================
def equipment_table_path(data_path):
    """<data_path>/out/processed/equipment.csv -- DERIVED data, like metadata.csv
    and load_profiles.csv. Deliberately NOT in inputs/, which is documented as
    holding true inputs the pipeline never overwrites (see process_data)."""
    return os.path.join(str(data_path).rstrip('/'), "out", "processed", EQUIPMENT_TABLE_FILE)


def generate_equipment_table(config=None, out_path=None, write=True):
    """Build the scenario's equipment table end to end and write it.

    Skeleton from the network, then PV placement, then battery placement (which
    reads the PV column, hence the order). ONE seeded generator for both draws,
    so the whole placement is reproducible from cfg['seed'] alone.

    Returns
    -------
    pd.DataFrame : the table (also written to `out_path` when write=True)
    """
    # Two-level config split: the PLACEMENT parameters take their defaults from
    # this module (DEFAULTS wins over the engine-wide config, whose own 'seed'
    # means something else), while the network is resolved through the engine
    # config so data_path behaves as everywhere else.
    net_cfg = _config_to_dict(config)
    cfg = {**DEFAULTS, **(config or {})}
    network = load_site_network(net_cfg)

    rng = np.random.default_rng(cfg["seed"])
    table = build_equipment_skeleton(network)
    table = place_pv(table, cfg, rng=rng)
    table = place_bss(table, cfg, rng=rng)

    if write:
        if out_path is None:
            out_path = equipment_table_path(net_cfg['data_path'])
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        table.to_csv(out_path, index=False)
    return table


def load_equipment_table(config=None, path=None):
    """Read the scenario's equipment table, or None when it has none.

    Returns None rather than raising, so a scenario that has not generated one
    yet degrades to "no equipment" instead of failing -- the same opt-in-by-
    presence idiom the PV and BSS layers already follow.

    The two IDENTIFIER columns are pinned to str: read_csv would otherwise infer
    'num' back as float64 (written as "93.0", read as 93.0), which then fails to
    match the (street, num) key of network['buses'] -- a silent join failure.
    'battery_id' is pinned for the same reason: it is the join key of the
    metadata<->schedule coherence check. Numeric and boolean columns round-trip
    correctly and are left to inference.
    """
    cfg = _config_to_dict(config)
    if path is None:
        path = equipment_table_path(cfg['data_path'])
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={'num': str, 'battery_id': str})
