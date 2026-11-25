"""Shared, network-agnostic helpers used across the engine and the data
processing bricks: config handling, the weekly time index, the loader that turns
a site's `data_path` into a built network bundle, the bus id contract, and the
BSS quantities both the engine and the schedule generator must agree on
(C-rate, state-of-charge band).
"""

import importlib.util
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

RANDOM_SEED = 369  # default seed, exposed for reproducible comparisons

STEPS_PER_DAY = 96         # 15-min steps
STEPS_PER_WEEK = 672

CONSUMPTION_BANK_FILE = "consumption_bank_real_weeks.parquet"


IRRADIANCE_FILE = "pvgis_irradiance_quarter_hourly.parquet"


def irradiance_path_for(data_path):
    """Path of a scenario's cached PVGIS irradiance, DERIVED from its `data_path`.

    Same single-source discipline as bank_path_for: the generator
    (profiles.build_irradiance_reference) and the consumers
    (process_data._load_pv_profiles, the PV scenario tooling) all resolve it
    here, so a new site never has to be told about it twice.
    """
    return os.path.join(str(data_path).rstrip('/'), "out", "processed", IRRADIANCE_FILE)


def bus_lonlat(net, bus_id):
    """(lon, lat) of a bus, from the pandapower 3.x GeoJSON 'geo' column.

    Buses get their geodata at creation in a site's build_network, so this works
    for LV buses and MV nodes alike. Returns (nan, nan) when a bus carries none,
    rather than raising: a site may legitimately leave some bus un-located.
    """
    geo = net.bus.loc[bus_id, "geo"]
    if geo is None or (isinstance(geo, float) and pd.isna(geo)):
        return (float('nan'), float('nan'))
    lon, lat = json.loads(geo)["coordinates"]
    return (float(lon), float(lat))


def site_centroid(network):
    """(lat, lon) at the centre of a site's LV charge buses.

    Lets anything that needs "where is this site" derive it from the network
    instead of carrying a hard-coded pair -- which is how a new scenario would
    otherwise end up silently fetching Bressoux's irradiance. Falls back to the
    MV nodes when a site declares no LV bus.

    Returns
    -------
    (lat, lon) : tuple[float, float]
    """
    net = network['net']
    ids = [int(b) for b in network.get('buses', {}).values()]
    if not ids:
        ids = [int(b) for b in network.get('mv_nodes', {}).values()]
    if not ids:
        raise ValueError(
            "Cannot locate the site: network['buses'] and network['mv_nodes'] are "
            "both empty."
        )
    pts = [bus_lonlat(net, b) for b in ids]
    lons = [p[0] for p in pts if not np.isnan(p[0])]
    lats = [p[1] for p in pts if not np.isnan(p[1])]
    if not lons:
        raise ValueError(
            "Cannot locate the site: no bus carries coordinates (empty 'geo' column). "
            "Pass lat/lon explicitly."
        )
    return (float(np.mean(lats)), float(np.mean(lons)))


def bank_path_for(data_path):
    """Path of a scenario's consumption bank, DERIVED from its `data_path`.

    Single source of truth for that path: the generator
    (data_processing.profiles.build_consumption_bank) and the consumer
    (_build_load_profiles_banque_mensuelle, via config['bank_path']) both go
    through here, so they cannot disagree.

    The bank is DERIVED data of a scenario, hence <data_path>/out/processed/,
    like metadata.csv and load_profiles.csv (see data_processing.process_data).
    """
    return os.path.join(str(data_path).rstrip('/'), "out", "processed", CONSUMPTION_BANK_FILE)

# BSS naming: b0 (the CENTRAL battery, on an MV node) and b1..bn (DOMESTIC
# batteries, one per equipped LV bus) are now both ROWS of the equipment table
# (see data_processing.equipment.place_central_bss / place_bss) -- generated
# once, at data-generation time, like PV. Nothing constrains their sum: a
# domestic battery is not a share of the park. What IS checked, in
# digital_twin._validate_inputs, is that the schedule's b1..bn and the
# equipment table's non-zero capacities designate THE SAME batteries.
#
# Central battery identifier, shared by utils.bss_c_rate (derives the domestic
# C-rate from b0's row) and data_processing.bss (schedules b0 alongside b1..bn).
BSS_CENTRAL_ID = 'b0'

# State-of-charge band, shared by the engine (digital_twin._track_bss_soc) and the
# reference schedule generator (data_processing.bss). Defined HERE rather than in
# either of them because both must plan and simulate against the same band: two
# copies drifting apart would raise [BSS/soc] warnings on feasible schedules.
# These are only the DEFAULTS of config['bss_soc_min'] / config['bss_soc_init'];
# both modules read the config keys, so one override moves both together.
BSS_SOC_MIN_FRAC = 0.1    # unusable reserve: min_e_mwh = 10% of capacity
BSS_SOC_INIT = 0.5        # state of charge at step 0

# Equipment-table columns the ENGINE also has to name: the schedule's b0..bn are
# matched against them, and their bus_id is where the storage elements land.
# Defined HERE rather than in data_processing.equipment (their natural owner)
# only because of the import direction: data_processing.pv imports the engine, so
# the engine importing data_processing back would close a cycle. utils is the
# layer both sides already depend on. equipment.py re-exports them under its own
# names, so its consumers are unaffected.
BSS_ID_COLUMN = 'battery_id'
BSS_SIZE_COLUMN = 'battery_size_kwh'
# Explicit power rating -- meaningful ONLY on the central battery's row (b0): a
# park cannot derive its own power from itself, unlike a domestic battery, whose
# rating is DERIVED (battery_size_kwh * bss_c_rate(...)) rather than stored, so
# the two numbers can never silently drift apart. Domestic rows keep this column
# at its 0.0 default.
BSS_P_MAX_COLUMN = 'battery_p_max_kw'

# Weekly time index 'Mon 00:00'..'Sun 23:45'. This is the exact index format of
# load_profiles.csv; every generator that fills the same shape reuses it so a
# bank of real weeks lands on EXACTLY the same index.
_WEEK_DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
_WEEK_TIME_INDEX = [
    f"{_WEEK_DAYS[step // STEPS_PER_DAY]} "
    f"{(step % STEPS_PER_DAY) * 15 // 60:02d}:{(step % STEPS_PER_DAY) * 15 % 60:02d}"
    for step in range(STEPS_PER_WEEK)
]

_MONTH_NAMES_FR = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "aout": 8, "août": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
}


def _normalize_month(mois):
    """Accept a number (1-12) or a French month name; return 1-12."""
    if isinstance(mois, (int, np.integer)):
        m = int(mois)
    else:
        key = str(mois).strip().lower()
        if key in _MONTH_NAMES_FR:
            m = _MONTH_NAMES_FR[key]
        else:
            try:
                m = int(key)
            except ValueError:
                raise ValueError(
                    f"month ({mois!r}) not recognised (expected an integer 1-12 or a "
                    "French month name)."
                )
    if not (1 <= m <= 12):
        raise ValueError(f"month ({mois!r}) must map to a month between 1 and 12.")
    return m


def _config_to_dict(config):
    """Accept config as dict or None; return a plain dict with defaults."""
    defaults = {
        'n_timesteps': STEPS_PER_WEEK,
        'v_min': 0.90,
        'v_max': 1.1,
        'loading_max': 90.0,
        'cos_phi': 0.95,
        'trafo_sn_mva': 0.22,
        'data_path': 'data/',
        'seed': RANDOM_SEED,                  # exposed for reproducible comparisons
        # mois_simule / bank_path: GENERATION parameters (monthly bank). The twin
        # does not read them itself; they are defaults for
        # _build_load_profiles_banque_mensuelle, called upstream by
        # digital_twin.data_processing.profiles.generate_load_profiles (CSV
        # generator) and by digital_twin.data_processing.pv.run_pv_scenario.
        'mois_simule': 1,                     # target month of the bank at generation time
        # 'bank_path' is NOT listed here: it derives from the FINAL data_path
        # (after the caller's overrides are merged), see below.
        'export_aggregate_profile': False,    # opt-in 672-row profile_agg export
        # BSS: the fleet's PHYSICAL facts (which batteries exist, where, how big --
        # central b0 included) live in the equipment table now, generated once at
        # data-generation time (see data_processing.equipment.place_central_bss /
        # place_bss), NOT here. What's left here are pure engine-level injection
        # choices with defensible universal defaults. No 'bss_enabled' flag -- the
        # layer is active exactly when schedules_BSS is passed, the same
        # activation-by-presence idiom as pv_profiles.
        'bss_eta': 0.95,                       # one-way efficiency, 1.0 = lossless
        'bss_enforce_limits': True,           # clip the schedule to the power and SoC limits
        'bss_soc_init': BSS_SOC_INIT,          # SoC at step 0, fraction of capacity
        'bss_soc_min': BSS_SOC_MIN_FRAC,       # lower SoC bound (unusable reserve), fraction
        'bss_c_rate': None,                   # None -> DERIVED from the park, see bss_c_rate()
        'export_bss_soc': False,              # opt-in per-battery SoC timeseries export
        'verbose': False,                     # print scenario/result info + progress bar during the run
    }
    cfg = dict(defaults) if config is None else {**defaults, **config}
    # Derived AFTER the merge, so a caller that overrides only data_path gets a
    # bank_path pointing at its own scenario. An explicit bank_path still wins.
    cfg.setdefault('bank_path', bank_path_for(cfg['data_path']))
    return cfg


# ============================================================
#  BSS: the equipment-table reader and derived quantities
# ============================================================
def bss_register(equipment):
    """{battery_id: {'bus_id', 'e_kwh', 'p_max_kw'}} for EVERY battery row of
    `equipment` -- 'b0' (the central battery, see
    data_processing.equipment.place_central_bss) and 'b1'..'bn' (domestic,
    place_bss), now both rows of the SAME table.

    'p_max_kw' is the row's own battery_p_max_kw for 'b0' (a park cannot derive
    its own power from itself, so it is stored explicitly) and None for every
    domestic id: their rating is DERIVED (e_kwh * bss_c_rate(...)) at the one
    place that actually needs a number (digital_twin._bss_fleet_specs /
    data_processing.bss.build_bss_schedule) -- storing a placeholder here would
    let the two silently drift.

    Returns {} when `equipment` is None or empty.

    Raises
    ------
    ValueError
        A required column is missing (battery_p_max_kw included -- this is what
        makes a stale, un-regenerated equipment.csv fail loudly rather than
        silently omitting the central battery), a battery_size_kwh > 0 row has
        no battery_id ("orphan"), or two rows share a battery_id.
    """
    if equipment is None or len(equipment) == 0:
        return {}
    required = {'bus_id', BSS_ID_COLUMN, BSS_SIZE_COLUMN, BSS_P_MAX_COLUMN}
    missing = required - set(equipment.columns)
    if missing:
        raise ValueError(
            f"`equipment` has no {sorted(missing)} column(s), required to locate the "
            "batteries (see data_processing.equipment) -- regenerate equipment.csv if "
            "it predates the central-battery-as-a-row schema."
        )

    sizes = pd.to_numeric(equipment[BSS_SIZE_COLUMN], errors='coerce').fillna(0.0)
    p_max = pd.to_numeric(equipment[BSS_P_MAX_COLUMN], errors='coerce').fillna(0.0)
    equipped = equipment[sizes > 0]
    ids = equipped[BSS_ID_COLUMN]

    orphan = equipped.loc[ids.isna(), 'bus_id'].astype(int).tolist()
    if orphan:
        raise ValueError(
            f"`equipment`: {len(orphan)} bus(es) carry a {BSS_SIZE_COLUMN} > 0 capacity "
            f"without a {BSS_ID_COLUMN}: {orphan[:10]}"
            f"{' ...' if len(orphan) > 10 else ''}. A battery without an identifier "
            "cannot be tied to its schedule."
        )
    labels = [str(i) for i in ids]
    dups = sorted(i for i, n in Counter(labels).items() if n > 1)
    if dups:
        raise ValueError(
            f"`equipment`: duplicate {BSS_ID_COLUMN}: {dups[:10]}"
            f"{' ...' if len(dups) > 10 else ''}. An identifier names a single battery."
        )

    equipped_p_max = p_max[sizes > 0]
    return {
        label: {'bus_id': int(bus), 'e_kwh': float(e),
               'p_max_kw': float(p) if label == BSS_CENTRAL_ID else None}
        for label, bus, e, p in zip(labels, equipped['bus_id'], sizes[sizes > 0], equipped_p_max)
    }


def bss_c_rate(config, equipment=None):
    """C-rate [1/h] applied to the DOMESTIC batteries: config['bss_c_rate'] when
    the scenario sets one, otherwise DERIVED from the central battery's own row
    of `equipment` (battery_p_max_kw / battery_size_kwh of 'b0').

    Deriving rather than defaulting to a literal keeps the fleet physically
    coherent: b0 and b1..bn are the same battery TECHNOLOGY at two scales, and a
    scenario that re-specs the park drags its domestic batteries along instead
    of silently pairing a fast park with slow homes.

    A domestic battery's power rating is therefore battery_size_kwh * c_rate --
    computed in digital_twin._bss_fleet_specs, the single place that turns
    capacities into ratings.

    Raises
    ------
    ValueError
        No explicit config['bss_c_rate'] AND `equipment` carries no central
        battery row -- a domestic-only fleet has no park to derive a rate from
        and must set config['bss_c_rate'] itself. Also raised on a non-positive
        park capacity, which would make the derived rate infinite.
    """
    explicit = config.get('bss_c_rate')
    if explicit is not None:
        c = float(explicit)
        if c <= 0:
            raise ValueError(f"config['bss_c_rate'] ({c}) must be strictly positive.")
        return c

    central = bss_register(equipment).get(BSS_CENTRAL_ID) if equipment is not None else None
    if central is None:
        raise ValueError(
            "C-rate not derivable: `equipment` carries no central battery (b0) row "
            "(see data_processing.equipment.place_central_bss), and config['bss_c_rate'] "
            "is not set. A domestic-only battery fleet needs an explicit C-rate to size "
            "its batteries' power -- pass config['bss_c_rate']."
        )
    e_kwh = central['e_kwh']
    if e_kwh <= 0:
        raise ValueError(
            f"C-rate not derivable: the central battery's capacity ({e_kwh} kWh) must be "
            "strictly positive, or pass config['bss_c_rate'] explicitly."
        )
    return central['p_max_kw'] / e_kwh


# ============================================================
#  Bus id contract
#
#  Every layer that places something on the grid (PV sgens, domestic batteries,
#  the central battery) names its target by bus_id. Checking those ids against
#  the REAL network stops a placement disagreement from surfacing much later as
#  an opaque pandapower error -- or worse, from not surfacing at all.
#
#  Deliberately checked by SET, never by cardinality: "31 installations for 31
#  equipped buses" can match while pointing at the wrong buses.
#
#  DOMAINS matter as much as existence. net.bus holds more than equipable buses
#  (the slack, the MV substations, each substation's LV secondary), so "the bus
#  exists" is a weaker statement than "something may be installed there". A
#  domestic battery on an MV node, or a PV sgen on the slack, are placement
#  errors that a plain existence check lets through.
# ============================================================
BUS_DOMAIN_LV = 'lv'                  # LV charge buses -- households: PV, domestic batteries
BUS_DOMAIN_MV_LOAD = 'mv_load'        # MV load facilities -- PV only (e.g. the rest home)
BUS_DOMAIN_MV_NODE = 'mv_node'        # any MV node -- the CENTRAL battery
BUS_DOMAIN_EQUIPABLE = 'equipable'    # lv + mv_load: anything a consumer may carry


def network_bus_domain(network, domain):
    """Set of bus ids belonging to `domain` (one of the BUS_DOMAIN_* constants).

    Reads only the site-agnostic keys of the network bundle ('buses',
    'mv_nodes', 'mv_load_names'), so it works on any site's build_network.
    """
    lv = {int(b) for b in network.get('buses', {}).values()}
    mv_nodes = {int(b) for b in network.get('mv_nodes', {}).values()}
    mv_load = {int(network['mv_nodes'][n]) for n in network.get('mv_load_names', [])
               if n in network.get('mv_nodes', {})}

    if domain == BUS_DOMAIN_LV:
        return lv
    if domain == BUS_DOMAIN_MV_LOAD:
        return mv_load
    if domain == BUS_DOMAIN_MV_NODE:
        return mv_nodes
    if domain == BUS_DOMAIN_EQUIPABLE:
        return lv | mv_load
    raise ValueError(
        f"unknown bus domain: {domain!r} (expected one of "
        f"{[BUS_DOMAIN_LV, BUS_DOMAIN_MV_LOAD, BUS_DOMAIN_MV_NODE, BUS_DOMAIN_EQUIPABLE]})."
    )


def check_bus_ids(bus_ids, network, domain, label, exhaustive=False):
    """Verify `bus_ids` against the network, and raise on any disagreement.

    Parameters
    ----------
    bus_ids : iterable[int]  -- the ids the caller wants to place something on
    network : dict           -- the bundle from load_site_network
    domain : str             -- BUS_DOMAIN_* the ids must belong to
    label : str              -- what is being checked, for the error messages
                                (e.g. "pv_profiles", "schedules_BSS")
    exhaustive : bool        -- also require the domain to be FULLY covered.
        False for a sparse layer (not every bus carries PV); True for a register
        meant to describe every bus, where a missing row is a silent hole.

    Returns
    -------
    set[int] : the validated ids

    Raises
    ------
    ValueError
        Ids absent from the network, ids outside `domain`, duplicated ids, or
        (when exhaustive) domain buses the caller never mentions.
    """
    ids = [int(b) for b in bus_ids]
    dups = sorted(b for b, n in Counter(ids).items() if n > 1)
    if dups:
        raise ValueError(
            f"`{label}` lists {len(dups)} duplicate bus_id: {dups[:10]}"
            f"{' ...' if len(dups) > 10 else ''}. A bus may carry only one row."
        )

    ids = set(ids)
    allowed = network_bus_domain(network, domain)
    all_known = network_bus_domain(network, BUS_DOMAIN_MV_NODE) | network_bus_domain(
        network, BUS_DOMAIN_LV)

    unknown = sorted(ids - all_known)
    if unknown:
        raise ValueError(
            f"`{label}` targets {len(unknown)} bus_id absent from the network: "
            f"{unknown[:10]}{' ...' if len(unknown) > 10 else ''}."
        )

    # Exists, but not somewhere this layer may be installed. Reported apart from
    # the previous case: the fix is different (bad placement vs bad id).
    out_of_domain = sorted(ids - allowed)
    if out_of_domain:
        raise ValueError(
            f"`{label}` targets {len(out_of_domain)} bus_id outside the '{domain}' "
            f"domain: {out_of_domain[:10]}{' ...' if len(out_of_domain) > 10 else ''}. "
            "These buses exist but cannot carry this equipment (slack, MV substation "
            "or a cabin's LV secondary, depending on the case)."
        )

    if exhaustive:
        missing = sorted(allowed - ids)
        if missing:
            raise ValueError(
                f"`{label}` does not cover {len(missing)} bus(es) of the '{domain}' "
                f"domain: {missing[:10]}{' ...' if len(missing) > 10 else ''}. This "
                "register must describe EVERY bus of the domain (one row per bus, "
                "equipped or not) -- equal cardinality does not guarantee it."
            )

    return ids


def load_site_network(config):
    """Dynamically load `<config['data_path']>/inputs/network.py` and call its
    `build_network(config)`.

    This is what makes the network a caller-supplied INPUT rather than
    something `run_digital_twin` builds itself: each site ships its own
    `inputs/network.py` describing its nodes/lines, and this loader is the
    single place that knows how to find and call it -- reused by
    `process_data.process_data`, `data_processing.pv` (base network for the
    multi-scenario tooling and for PV placement), and digital_twin/plots/.

    Returns
    -------
    dict : {"net", "buses", "mv_nodes", "mv_load_names", "validation"}
        See the site's `build_network` docstring for the exact contents.
    """
    network_path = os.path.join(config['data_path'], "inputs", "network.py")
    if not os.path.exists(network_path):
        raise FileNotFoundError(
            f"Site network file not found ({network_path}). Each scenario's "
            "data_path must ship an inputs/network.py exposing build_network(config)."
        )
    spec = importlib.util.spec_from_file_location("site_network", network_path)
    site_network = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site_network)   # defines build_network, MV_LOADS, ...
    return site_network.build_network(config)