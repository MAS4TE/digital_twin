"""
PV volet -- Brick 1: PV installation metadata generation.

Generates data/pv_installations.csv: which residential bus carries a PV
installation and of what size, plus one fixed installation on the rest home.
Performs NO pandapower injection and NO production computation -- only the
metadata, consumed by the following bricks (production, injection).

Reuses _build_network() and _config_to_dict() READ-ONLY to obtain the real
topology (bus_id <-> street), so the bus_ids generated here match exactly those
of the real network (no reimplementation that could drift from the true
pandapower bus_id).

The bus -> street link must exist and be usable for the 333 residential buses:
  - households.csv: NOT usable for this -- it only knows 3 streets (Avenue
    Joseph Merlot / Rue Ernest Malvoz / Plateau du Bouhay), a simplified split
    used for load-profile assignment, unrelated to the full physical topology.
  - 'street' column of consumption_bank_real_weeks.parquet: NOT usable either
    -- limited to the streets of the 33 measured EANs (same subset as above,
    see sourcecode/consumption_bank.py).
  - _build_network()['buses'] (digital_twin.py): USABLE and authoritative --
    dict (street, num) -> bus_id, covering exactly the 333 real residential
    buses across 7 real streets (Avenue Joseph Merlot 106, Rue Ernest Malvoz
    83, Plateau du Bouhay 54, Rue des Primeroses 30, Rue des Bluets 21, Rue
    Simon Burnet 20, Rue Joseph Lambremont 19). It is the source of truth
    because it is literally what the network reconstruction uses (derived from
    coordonnées.csv). No uniform random fallback needed: the exact link exists.

Field calibration: a user report (site visit) indicates PV observed more often
on Rue Ernest Malvoz than on Plateau du Bouhay. Reflected via `street_weights`
(per-street draw weight, default 1.0 everywhere except Rue Ernest Malvoz = 3.0)
-- a probabilistic bias, not a deterministic guarantee (see DEFAULTS and
generate_pv_metadata).
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from sourcecode.digital_twin import _build_network, _config_to_dict

OUT_PATH = "data/pv_installations.csv"

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
    "rest_home_name": "Paradis du Bouhay",  # see digital_twin.MV_PASSAGE
    "out_path": OUT_PATH,
}


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
    """Generate data/pv_installations.csv (or cfg['out_path']).

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
    existing MV passage bus (the same bus as its consumption load in
    _assign_loads) -- no associated street ('street' column empty), since it does
    not belong to the residential street system.

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

    net_cfg = _config_to_dict(None)  # fixed topology, independent of PV parameters
    net, buses, mv_nodes, validation = _build_network(net_cfg)

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
    """PV volet -- Brick 4: extend an existing PV fleet by `n_additional` extra
    residential installations, PRESERVING the existing installations IDENTICALLY
    (same bus_id) -- NOT an independent draw. Needed to compare penetration rates
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

    net_cfg = _config_to_dict(None)
    net, buses, mv_nodes, validation = _build_network(net_cfg)
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


if __name__ == "__main__":
    df = generate_pv_metadata()

    n_res = int((df["type"] == "residential").sum())
    n_rest = int((df["type"] == "rest_home").sum())
    print(f"{OUT_PATH} genere : {len(df)} lignes "
          f"({n_res} residentielles + {n_rest} maison de repos)")

    print("\nRepartition par rue (residentiel) :")
    print(df[df["type"] == "residential"].groupby("street").size()
          .sort_values(ascending=False).to_string())

    kwp_res = df.loc[df["type"] == "residential", "kwp"].sum()
    kwp_rest = df.loc[df["type"] == "rest_home", "kwp"].sum()
    print(f"\nPuissance PV totale installee : {df['kwp'].sum():.1f} kWp "
          f"(residentiel : {kwp_res:.1f} kWp, maison de repos : {kwp_rest:.1f} kWp)")

    print("\nReproductibilite (2 appels, meme seed) :", end=" ")
    df2 = generate_pv_metadata()
    print("OK, CSV identique" if df.equals(df2) else "ECHEC, CSV different !")
