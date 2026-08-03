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
    <data_path>/out/processed/ : DERIVED data, regenerated from inputs/ by the
                                site's data pipeline (e.g. bressoux_setup.ipynb,
                                digital_twin.data_processing.profiles/.pv):
        - equipment.csv            : bus-level equipment register -- one row per
                                     equipable bus, carrying the PV sizes, the
                                     battery ids/capacities (central b0 AND
                                     domestic b1..bn alike), and the coordinates
                                     (see data_processing.equipment). Derived
                                     from the network; it is what decides
                                     whether the PV and BSS layers run at all,
                                     and it is now the ONLY place a battery's
                                     bus/capacity/power is defined -- never a
                                     runtime config key.
        - metadata.csv             : per-household attributes (profile_id,
                                     source_street, ...)
        - load_profiles.csv        : household consumption timeseries
                                     (index=week_step, columns=profile_{id})
                                     -- this is the `households` argument of
                                     run_digital_twin
        - pvgis_irradiance_quarter_hourly.parquet : required alongside
                                     equipment.csv to build `pv_profiles`
        - pv_profiles.csv          : WRITTEN FOR INSPECTION, NEVER READ BACK.
                                     PV production per equipped bus, in kW (same
                                     unit the twin itself receives). Rebuilt on
                                     every call; deleted when the layer is off.
        - schedules_bss.csv        : WRITTEN FOR INSPECTION, NEVER READ BACK.
                                     The battery fleet schedule, long format, kW.
                                     Same rule -- see _write_for_inspection for
                                     why these two are outputs and not caches.

process_data itself only loads already-materialized files, it does not
regenerate them on every call.
"""
import os

import pandas as pd

from ..utils import _config_to_dict, irradiance_path_for, load_site_network
from .bss import load_bss_schedule
from .equipment import PV_SIZE_COLUMN, load_equipment_table
from .pv import build_pv_production

# WRITTEN FOR INSPECTION, NEVER READ BACK. See _write_for_inspection.
PV_PROFILES_FILE = "pv_profiles.csv"
BSS_SCHEDULE_FILE = "schedules_bss.csv"


def _write_for_inspection(frame, cfg, filename):
    """Write a DERIVED frame to <data_path>/out/processed/ so it can simply be
    opened and looked at.

    THIS FILE IS NEVER READ BACK. `pv_profiles` and `schedules_BSS` are rebuilt
    from equipment.csv + irradiance + households on every call (~3 s), which is
    what makes them incapable of contradicting the equipment table. Caching them
    here would turn an output into an input, and give the scenario two registers
    free to disagree. Read this before adding any recomputation shortcut.

    When the corresponding layer is OFF, any previous file is DELETED rather
    than left behind: a stale pv_profiles.csv showing 31 installations for a
    scenario that now has none would be worse than no file at all.

    Returns the path written, or None.
    """
    path = os.path.join(str(cfg['data_path']).rstrip('/'), "out", "processed", filename)
    if frame is None or len(frame) == 0:
        if os.path.exists(path):
            os.remove(path)
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path)
    return path


def process_data(config):
    """Load a scenario's network, household metadata, consumption profiles and,
    if the scenario has them, PV production profiles and the BSS schedule.

    Parameters
    ----------
    config : dict with the simulation parameters, including 'data_path' (path to the scenario folder).

    Returns
    -------
    (metadata, network, households, pv_profiles, schedules_BSS, equipment) : as
    expected by digital_twin.run_digital_twin(config, metadata, network,
    households, pv_profiles=..., schedules_BSS=..., equipment=...)

        metadata      : pd.DataFrame          -- per-household attributes
        network       : dict                  -- network bundle (net, buses, mv_nodes, ...)
        households    : pd.DataFrame          -- consumption timeseries, kW
        pv_profiles   : pd.DataFrame or None  -- PV production, kW; None when no bus
                                                 of the equipment table carries PV
        schedules_BSS : pd.DataFrame or None  -- battery fleet schedule, kW, long
                                                 format; None when the table carries
                                                 no battery at all (central or domestic)
        equipment     : pd.DataFrame or None  -- the bus-level register itself; the
                                                 twin needs it to place b0..bn and to
                                                 check the schedule's identifiers

    The two optional layers follow the SAME opt-in rule, on CONTENT rather than
    on files: a scenario enables PV by having a bus with pv_system_size_kwp > 0,
    and BSS by having one with battery_size_kwh > 0.
    """
    cfg = _config_to_dict(config)
    data_path = cfg['data_path'].rstrip('/')
    processed_dir = f"{data_path}/out/processed"

    network = load_site_network(cfg)
    metadata = pd.read_csv(f"{processed_dir}/metadata.csv")
    households = pd.read_csv(f"{processed_dir}/load_profiles.csv", index_col=0)
    # ONE register drives both optional layers: which buses carry PV, and which
    # carry a domestic battery. None when the scenario never generated one.
    equipment = load_equipment_table(cfg)
    pv_profiles = _load_pv_profiles(cfg, equipment)
    # Built from households, pv_profiles and the declared household -> bus
    # assignment (metadata['bus_id']), so it must come after all three: each
    # domestic battery is scheduled against the load of its OWN bus.
    schedules_BSS = load_bss_schedule(households, pv_profiles, equipment, metadata, cfg)
    # kW already at this interface, so the file matches what the twin receives.
    _write_for_inspection(schedules_BSS, cfg, BSS_SCHEDULE_FILE)

    # POSITIONAL CONTRACT -- callers unpack this tuple by position, so the order
    # below IS the interface. Any new element goes at the END, never in the
    # middle: inserting one re-binds every position after it, and since five of
    # the six are DataFrames a wrong binding does not raise at unpack time --
    # it surfaces much later, as a confusing error deep inside the twin.
    #   0 metadata | 1 network | 2 households | 3 pv_profiles | 4 schedules_BSS
    #   5 equipment
    return metadata, network, households, pv_profiles, schedules_BSS, equipment


def _load_pv_profiles(cfg, equipment):
    """Build `pv_profiles` (kW, see run_digital_twin) from the scenario's
    EQUIPMENT TABLE and its cached PVGIS irradiance (utils.irradiance_path_for),
    for the month `cfg['mois_simule']`.

    Returns None when the scenario has no equipment table, or when no bus in it
    carries PV. ACTIVATION BY PRESENCE, on content: "does any row of the table
    have pv_system_size_kwp > 0".

    Also writes `pv_profiles.csv` for inspection, in the same kW build_pv_production
    already works in and the twin now receives directly -- no conversion needed
    between the two, the file matches what run_digital_twin sees.
    """
    pv_profiles_kw = None
    if (equipment is not None and len(equipment) > 0
            and PV_SIZE_COLUMN in equipment.columns
            and (equipment[PV_SIZE_COLUMN].fillna(0.0) > 0).any()):
        irradiance_path = irradiance_path_for(cfg['data_path'])
        if not os.path.exists(irradiance_path):
            raise FileNotFoundError(
                f"The equipment table carries PV but {irradiance_path} is missing -- "
                "run digital_twin.data_processing.profiles.build_irradiance_reference(config) "
                "once to materialize it before running the twin with PV."
            )
        irradiance = pd.read_parquet(irradiance_path)
        pv_profiles_kw = build_pv_production(cfg['mois_simule'], equipment, irradiance)

    _write_for_inspection(pv_profiles_kw, cfg, PV_PROFILES_FILE)
    return pv_profiles_kw
