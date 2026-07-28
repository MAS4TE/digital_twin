"""
Load a scenario's prepared inputs and assemble the DataFrames run_digital_twin
expects.

A scenario folder (`config['data_path']`) is split in two:
    <data_path>/inputs/       : TRUE inputs -- hand-provided or long-lived
                                reference data, never overwritten by the
                                pipeline itself.
        - network.py              : site-specific topology builder, see
                                     digital_twin.utils.load_site_network and
                                     e.g. data/bressoux/inputs/network.py
        - coordinates.csv          : building coordinates (read by network.py)
        - part-*.csv               : raw smart-meter readings
        - pv_installations.csv     : OPTIONAL -- PV installation register
                                     (bus_id, kwp, type, street, see
                                     pv.generate_pv_metadata). A scenario opts
                                     into PV simply by having this file.
        - bss_shares.csv           : OPTIONAL -- BSS contractual share register
                                     (bus_id, e_kwh, street, see
                                     bss.generate_bss_shares). A scenario opts
                                     into BSS simply by having this file.
    <data_path>/out/processed/ : DERIVED data, regenerated from inputs/ by the
                                site's data pipeline (e.g. bressoux_setup.ipynb,
                                digital_twin.data_processing.profiles/.pv):
        - metadata.csv             : per-household attributes (profile_id,
                                     source_street, ...)
        - load_profiles.csv        : household consumption timeseries
                                     (index=week_step, columns=profile_{id})
                                     -- this is the `households` argument of
                                     run_digital_twin
        - pvgis_irradiance_quarter_hourly.parquet : required alongside
                                     pv_installations.csv to build `pv_profiles`

process_data itself only loads already-materialized files, it does not
regenerate them on every call.
"""
import os

import pandas as pd

from ..utils import _config_to_dict, load_site_network
from .bss import load_bss_schedule
from .pv import build_pv_production


def process_data(config):
    """Load a scenario's network, household metadata, consumption profiles and,
    if the scenario has them, PV production profiles and the BSS schedule.

    Parameters
    ----------
    config : dict with the simulation parameters, including 'data_path' (path to the scenario folder).

    Returns
    -------
    (metadata, network, households, pv_profiles, schedules_BSS) : as expected by
    digital_twin.run_digital_twin(config, metadata, network, households,
    pv_profiles=pv_profiles, schedules_BSS=schedules_BSS)

        metadata      : pd.DataFrame          -- per-household attributes
        network       : dict                  -- network bundle (net, buses, mv_nodes, ...)
        households    : pd.DataFrame          -- consumption timeseries, MW
        pv_profiles   : pd.DataFrame or None  -- PV production, MW; None when the
                                                 scenario has no pv_installations.csv
        schedules_BSS : pd.DataFrame or None  -- BSS park schedule, kW; None when the
                                                 scenario has no bss_shares.csv

    The two optional layers follow the SAME opt-in rule: a scenario enables them
    by having the corresponding file in inputs/, and gets None here otherwise.
    """
    cfg = _config_to_dict(config)
    data_path = cfg['data_path'].rstrip('/')
    inputs_dir = f"{data_path}/inputs"
    processed_dir = f"{data_path}/out/processed"

    network = load_site_network(cfg)
    metadata = pd.read_csv(f"{processed_dir}/metadata.csv")
    households = pd.read_csv(f"{processed_dir}/load_profiles.csv", index_col=0)
    pv_profiles = _load_pv_profiles(cfg, inputs_dir, processed_dir)
    # Built from households and pv_profiles, so it must come after both.
    schedules_BSS = load_bss_schedule(households, pv_profiles, cfg, inputs_dir)

    # POSITIONAL CONTRACT -- callers unpack this tuple by position, so the order
    # below IS the interface. Any new element goes at the END, never in the
    # middle: inserting one re-binds every position after it, and since four of
    # the five are DataFrames a wrong binding does not raise at unpack time --
    # it surfaces much later, as a confusing error deep inside the twin.
    #   0 metadata | 1 network | 2 households | 3 pv_profiles | 4 schedules_BSS
    return metadata, network, households, pv_profiles, schedules_BSS


def _load_pv_profiles(cfg, inputs_dir, processed_dir):
    """Build `pv_profiles` (MW, see run_digital_twin) from
    `<inputs_dir>/pv_installations.csv` and the site's cached PVGIS irradiance
    (`<processed_dir>/pvgis_irradiance_quarter_hourly.parquet`), for the month
    `cfg['mois_simule']`.

    Returns None when the scenario has no PV installation register (no PV in
    this scenario) -- a scenario with PV is opted into simply by having a
    pv_installations.csv file, nothing else to configure.
    """
    pv_installations_path = f"{inputs_dir}/pv_installations.csv"
    if not os.path.exists(pv_installations_path):
        return None

    irradiance_path = f"{processed_dir}/pvgis_irradiance_quarter_hourly.parquet"
    if not os.path.exists(irradiance_path):
        raise FileNotFoundError(
            f"{pv_installations_path} exists but {irradiance_path} is missing -- "
            "run digital_twin.data_processing.profiles.build_irradiance_reference() "
            "once to materialize it before running the twin with PV."
        )

    pv_installations = pd.read_csv(pv_installations_path)
    irradiance = pd.read_parquet(irradiance_path)
    pv_profiles_kw = build_pv_production(cfg['mois_simule'], pv_installations, irradiance)
    return pv_profiles_kw / 1000.0  # kW -> MW, same unit as households (see run_digital_twin)
