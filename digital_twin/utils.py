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

# BSS naming convention, two levels kept apart:
#   bss_park_*        : physical specs of the aggregated central park (config keys).
#   battery_size_kwh  : per-household contractual slice (metadata column, see
#                       BSS_SLICE_COLUMN). Keeps its schema name -- may be fixed by
#                       the MAS4TE contract, don't rename unilaterally (pending
#                       consortium clarification).
# Unlike v_min/v_max/cos_phi, park specs describe one specific park: silently
# defaulting them is a scenario error, hence check_bss_config_specs.
BSS_PHYSICAL_SPEC_KEYS = ('bss_park_e_kwh', 'bss_park_p_kw')

# Other recognised bss_* config keys. These drive the INJECTION (placement,
# efficiency, clipping policy -- see digital_twin._inject_bss_storage /
# _track_bss_soc) rather than describing the park's physical size, so they are
# NOT subject to the "was it set explicitly?" fallback warning above: a sensible
# default is legitimate for them. They are listed here only so
# check_bss_config_specs does not mistake them for typos.
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
        # Physical capacity/power of the central battery park (bss_park_*) -- scenario-tunable,
        # not universal constants; defaults describe the FZJ high-energy park.
        # Per-household slices (battery_size_kwh) must fit inside these: see check_bss_energetic_feasibility.
        'bss_park_e_kwh': 2600.0,             # park energy capacity [kWh] (FZJ high-energy park)
        'bss_park_p_kw': 595.0,               # park power rating [kW]     (FZJ high-energy park)
        # Injection parameters (see digital_twin._inject_bss_storage). Unlike the
        # two park specs above, these are engine-level choices with defensible
        # defaults, so they are not part of BSS_PHYSICAL_SPEC_KEYS.
        # There is NO 'bss_enabled' flag: the BSS layer is active exactly when
        # `schedules_BSS` is passed to run_digital_twin, the same activation-by-
        # presence idiom as `pv_profiles`.
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
#  Both functions follow the engine's warning convention: they RETURN a list of
#  warning strings rather than raising or printing, and the caller surfaces them
#  (see _validate_inputs / save_run_report). A caller that discards the returned
#  list gets no output -- surface it.
#
#  Neither is wired into run_digital_twin: they are UPSTREAM checks on the
#  scenario definition (config + metadata), to be run where the BSS slices and
#  schedule are prepared, before the twin is called.
# ============================================================
def check_bss_config_specs(config, bss_active=True):
    """Warn when the physical park specs (`bss_park_e_kwh`, `bss_park_p_kw`) in use
    are the FZJ defaults because the caller never set them.

    `_config_to_dict` merges as `{**defaults, **config}`, so an ABSENT or
    MISSPELLED key falls back to the default with no trace -- the merged dict
    looks the same whether the caller set 2600.0 or never mentioned it. The
    fallback is therefore detected on the ORIGINAL config by key presence,
    before the merge; inspecting the merged dict cannot work.

    A misspelled key (e.g. 'bss_park_e_khw') is caught twice: the correct key is
    absent (fallback warning), and the stray `bss_*` key is named so the typo
    isn't left to hunt.

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
    """Verify the contractual BSS slices fit inside the physical park:
    `sum(metadata['battery_size_kwh']) <= bss_park_e_kwh`.
    Energetic feasibility only -- the power rating `bss_park_p_kw` constrains
    the schedule, not the allocation, and is checked elsewhere.
    Warns rather than raises: an oversubscribed park is a scenario-definition
    problem for the caller (resize the park or the slices), not a reason to
    abort data preparation.

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

    # metadata.csv round-trips through a bare read_csv, so the nullable Float64
    # of the generator comes back as float64 -- and an empty column as all-NaN.
    # to_numeric keeps the sum meaningful in both cases.
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