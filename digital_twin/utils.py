"""Shared, network-agnostic helpers used across the engine and the data
processing bricks: config handling, the weekly time index, the loader
that turns a site's `data_path` into a built network bundle, and the upstream
BSS (battery park) consistency checks.
"""

import importlib.util
import os

import numpy as np
import pandas as pd

RANDOM_SEED = 369  # default seed, exposed for reproducible comparisons

STEPS_PER_DAY = 96         # 15-min steps
STEPS_PER_WEEK = 672

CONSUMPTION_BANK_FILE = "consumption_bank_real_weeks.parquet"

# BSS naming, two levels kept apart:
#   bss_park_*        : physical specs of the central park (config keys)
#   battery_size_kwh  : per-household contractual slice (metadata column). Keeps
#                       its schema name -- possibly fixed by the MAS4TE contract,
#                       don't rename unilaterally.
# Unlike v_min/v_max/cos_phi, park specs describe one specific park, so silently
# defaulting them is a scenario error -- hence check_bss_config_specs.
BSS_PHYSICAL_SPEC_KEYS = ('bss_park_e_kwh', 'bss_park_p_kw')

# Other recognised bss_* keys. These drive the INJECTION rather than the park's
# physical size, so a default is legitimate and they are exempt from the fallback
# warning above. Listed only so check_bss_config_specs doesn't read them as typos.
BSS_OPTIONAL_KEYS = ('bss_bus', 'bss_eta', 'bss_enforce_limits')

# metadata.csv column: each household's contractual slice of the park [kWh].
BSS_SLICE_COLUMN = 'battery_size_kwh'

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
        'bank_path': 'data/' + CONSUMPTION_BANK_FILE,  # bank of real weeks
        'export_aggregate_profile': False,    # opt-in 672-row profile_agg export
        # Central park specs -- scenario-tunable, not universal constants. The
        # per-household slices (battery_size_kwh) must fit inside them, see
        # check_bss_energetic_feasibility.
        'bss_park_e_kwh': 2600.0,             # park energy capacity [kWh] (FZJ high-energy park)
        'bss_park_p_kw': 595.0,               # park power rating [kW]     (FZJ high-energy park)
        # Injection parameters (see digital_twin._inject_bss_storage): engine-level
        # choices with defensible defaults, hence not in BSS_PHYSICAL_SPEC_KEYS.
        # No 'bss_enabled' flag -- the layer is active exactly when schedules_BSS
        # is passed, the same activation-by-presence idiom as pv_profiles.
        'bss_bus': 'MALVOZ BRESSOUX',         # MV node NAME (a network['mv_nodes'] key) or bus id
        'bss_eta': 0.95,                       # one-way efficiency, 1.0 = lossless
        'bss_enforce_limits': True,           # clip the schedule to the power and SoC limits
    }
    if config is None:
        return defaults
    return {**defaults, **config}


# ============================================================
#  BSS (central battery park) upstream checks
#
#  Both RETURN their warnings rather than raising or printing, per the engine's
#  convention -- a caller that discards the list gets no output.
#
#  Neither is wired into run_digital_twin: they check the SCENARIO DEFINITION
#  (config + metadata), and belong where the slices and schedule are prepared.
# ============================================================
def check_bss_config_specs(config, bss_active=True):
    """Warn when the park specs in use are the FZJ defaults because the caller
    never set them.

    `_config_to_dict` merges as `{**defaults, **config}`, so an absent or
    misspelled key falls back with no trace: the merged dict looks the same
    whether the caller set 2600.0 or never mentioned it. The fallback must
    therefore be detected on the ORIGINAL config, by key presence, before the
    merge -- inspecting the merged dict cannot work.

    A typo like 'bss_park_e_khw' is caught twice: the real key reads as absent,
    and the stray bss_* key is named so it isn't left to hunt.

    Parameters
    ----------
    config : dict or None    -- the caller's config, BEFORE _config_to_dict
    bss_active : bool        -- no warning when the scenario has no BSS

    Returns
    -------
    warnings : list[str]     -- '[BSS/config] ...' messages, empty if every spec
                                was set explicitly
    """

    warnings = []
    if not bss_active:
        return warnings

    provided = set(config.keys()) if config else set()
    cfg = _config_to_dict(config)

    for key in BSS_PHYSICAL_SPEC_KEYS:
        if key not in provided:
            warnings.append(
                f"[BSS/config] '{key}' absent from the caller's config -- running on "
                f"the FZJ default ({cfg[key]}). This is a physically meaningful park "
                "spec, not a universal constant: set it explicitly to confirm it "
                "matches this scenario's battery park."
            )

    known = set(BSS_PHYSICAL_SPEC_KEYS) | set(BSS_OPTIONAL_KEYS)
    unknown = sorted(
        k for k in provided
        if str(k).startswith('bss_') and k not in known
    )
    if unknown:
        warnings.append(
            f"[BSS/config] unrecognized BSS key(s) in config: {unknown} -- ignored by "
            f"the defaults merge (expected {sorted(known)}). Likely a typo."
        )

    return warnings


def check_bss_energetic_feasibility(metadata, bss_park_e_kwh):
    """Verify the contractual slices fit inside the park:
    `sum(metadata['battery_size_kwh']) <= bss_park_e_kwh`.

    Energy only -- `bss_park_p_kw` constrains the schedule, not the allocation.
    Warns rather than raises: an oversubscribed park is the caller's scenario to
    fix, not a reason to abort data preparation.

    Parameters
    ----------
    metadata : pd.DataFrame  -- per-household attributes, must carry
                                'battery_size_kwh' [kWh/household]
    bss_park_e_kwh : float   -- park physical energy capacity [kWh]

    Returns
    -------
    warnings : list[str]     -- '[BSS/contract] ...' messages, empty when slices fit
    """
    warnings = []

    if metadata is None or BSS_SLICE_COLUMN not in getattr(metadata, 'columns', []):
        # Distinguish "verified, fits" from "could not verify": an empty list
        # would read as the former.
        warnings.append(
            f"[BSS/contract] energetic feasibility NOT checked: `metadata` has no "
            f"'{BSS_SLICE_COLUMN}' column (or is None). Expected the per-household "
            "slice capacities in kWh."
        )
        return warnings

    # metadata.csv round-trips through a bare read_csv: the generator's nullable
    # Float64 comes back as float64, an empty column as all-NaN. to_numeric keeps
    # the sum meaningful in both cases.
    slices = pd.to_numeric(metadata[BSS_SLICE_COLUMN], errors='coerce')
    total_kwh = float(slices.fillna(0.0).sum())

    if total_kwh > bss_park_e_kwh+1e-3:  # allow a tiny epsilon for float rounding
        warnings.append(
            f"[BSS/contract] the per-household BSS slices oversubscribe the park: "
            f"sum({BSS_SLICE_COLUMN}) = {total_kwh:.1f} kWh > bss_park_e_kwh = "
            f"{float(bss_park_e_kwh):.1f} kWh (excess {total_kwh - float(bss_park_e_kwh):.1f} "
            f"kWh, {len(metadata)} households). The allocation is not energetically "
            "feasible on this park."
        )

    return warnings


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