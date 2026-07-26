"""Shared, network-agnostic helpers used across the engine and the data
processing bricks: config handling, the weekly time index, and the loader
that turns a site's `data_path` into a built network bundle.
"""

import importlib.util
import os

import numpy as np

RANDOM_SEED = 369  # default seed, exposed for reproducible comparisons

STEPS_PER_DAY = 96         # 15-min steps
STEPS_PER_WEEK = 672

CONSUMPTION_BANK_FILE = "consumption_bank_real_weeks.parquet"

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
                    f"mois ({mois!r}) non reconnu (attendu un entier 1-12 ou un nom de mois francais)."
                )
    if not (1 <= m <= 12):
        raise ValueError(f"mois ({mois!r}) doit correspondre a un mois entre 1 et 12.")
    return m


def _config_to_dict(config):
    """Accept config as dict or None; return a plain dict with defaults."""
    defaults = {
        'n_timesteps': STEPS_PER_WEEK,
        'v_min': 0.95,
        'v_max': 1.05,
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
        'bank_path': 'data/' + CONSUMPTION_BANK_FILE,  # bank of real weeks
        'export_aggregate_profile': False,    # opt-in 672-row profile_agg export
    }
    if config is None:
        return defaults
    return {**defaults, **config}


def load_site_network(config):
    """Dynamically load `<config['data_path']>/inputs/network.py` and call its
    `build_network(config)`.

    This is what makes the network a caller-supplied INPUT rather than
    something `run_digital_twin` builds itself: each site (e.g. Bressoux)
    ships its own `inputs/network.py` describing its nodes/lines, and this
    loader is the single place that knows how to find and call it -- reused by
    `process_data.process_data`, `data_processing.pv` (base network for the
    multi-scenario tooling and for PV placement), and the check/viz scripts.

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
    spec.loader.exec_module(site_network) # Run the code inside network.py and defines build_network, MV_LOADS, etc
    return site_network.build_network(config)
