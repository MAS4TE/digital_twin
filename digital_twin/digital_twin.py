"""
Digital twin engine -- network-agnostic.

Exposes a single entry point `run_digital_twin` callable by the MAS4TE
framework. Given a network, household metadata and consumption profiles, it
assigns loads, runs a time-series power flow, and returns a verdict on whether
the network keeps voltages and line/transformer loadings within acceptable
limits.

Pipeline (see docs/pipeline.md):
    1. Data ingestion / input validation
    2. Load assignment
    3. Time-series power flow
    4. Results evaluation (verdicts)
    5. Output formatting

Scope: LOADS + PV + BSS. Both optional layers are activated BY PRESENCE of their
argument -- no enable flag for either:
  * `pv_profiles`   -> sgen injection (_inject_pv_sgens).
  * `schedules_BSS` -> storage injection (_bss_fleet_specs -> _read_bss_schedule
    -> _track_bss_soc -> _inject_bss_storage): a long-format schedule
    (battery_id, p_kw) describing a fleet -- b0, the central battery, and
    b1..bn, one domestic battery per equipped LV bus. Each becomes its own
    pandapower storage element with its own capacity, power rating and state of
    charge. A battery is present exactly when the schedule names it, so
    central-only, domestic-only and both are all expressible without a flag.

`equipment` (the bus-level register, see data_processing.equipment) is where
EVERY battery's placement and physical specs live now -- b0 (generated once by
place_central_bss) exactly like b1..bn (place_bss): the schedule carries
identifiers, the table carries placement and sizing, and _validate_inputs
requires the two to designate the SAME set of batteries. The engine holds no
placement or config-level sizing knowledge of its own -- `config` only tunes
injection behaviour (bss_eta, bss_enforce_limits, an optional bss_c_rate
override), never a battery's bus or capacity.

Two OPPOSITE sign conventions live in this file:
  * sgen (PV):     positive p_mw = INJECTION.
  * storage (BSS): positive p_mw = CHARGING (consumption), negative = discharge.
The BSS schedule follows the storage one; _inject_bss_storage documents it for
the whole BSS path.

Units: households, pv_profiles and schedules_BSS all arrive in kW. MW is a
pandapower-internal detail: the conversion happens at _KW_TO_MW, right before
injection into net.load/net.sgen/net.storage, and nowhere else.

`network` is caller-supplied (see digital_twin.utils.load_site_network and
data/bressoux/inputs/network.py for how a site builds one), never built here:
this is what makes the engine reusable for another LV network without touching
this package. A caller reusing the SAME `network` object across repeated
run_digital_twin calls must copy it first -- net.load/net.sgen are cleared and
repopulated in place every call. See
digital_twin.data_processing.pv.run_pv_scenario for the copy.deepcopy pattern.
"""

import datetime
import os
from collections import Counter

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

from .utils import (BSS_CENTRAL_ID, BSS_SIZE_COLUMN, BSS_SOC_INIT,
                    BSS_SOC_MIN_FRAC, BUS_DOMAIN_EQUIPABLE, BUS_DOMAIN_LV,
                    BUS_DOMAIN_MV_NODE, STEPS_PER_DAY, _config_to_dict,
                    _normalize_month, bss_c_rate, bss_register, check_bus_ids)

_REMOVED_BSS_CONFIG_KEYS = frozenset({'bss_bus', 'bss_park_e_kwh', 'bss_park_p_kw'})

# THE kW -> MW conversion point for every one of households/pv_profiles/
# schedules_BSS -- applied right before injection into pandapower
# (net.load/net.sgen/net.storage), and nowhere else.
_KW_TO_MW = 1e-3

# ============================================================
#  1. DATA INGESTION / INPUT VALIDATION
# ============================================================
def _validate_inputs(config, metadata, network, households, pv_profiles, schedules_BSS,
                     equipment=None):
    """Check that injected DataFrames have the expected structure.
    """
    warnings = []

    # --- metadata ---
    if metadata is None or len(metadata) == 0:
        raise ValueError("`metadata` is empty or None: at least one household is required.")
    missing_cols = {'profile_id', 'source_street'} - set(metadata.columns)
    if missing_cols:
        raise ValueError(f"`metadata` is missing required column(s): {sorted(missing_cols)}")

    # --- households (consumption timeseries) ---
    if households is None or len(households) == 0:
        raise ValueError("`households` is empty or None: at least one timestep of data is required.")
    missing_profiles = sorted(
        pid for pid in metadata['profile_id'].unique()
        if f"profile_{pid}" not in households.columns
    )
    if missing_profiles:
        preview = missing_profiles[:10]
        raise ValueError(
            "`households` is missing column(s) for profile_id(s) referenced by "
            f"`metadata`: {preview}{' ...' if len(missing_profiles) > 10 else ''}"
        )

    # --- BUS ID CONTRACT for the household -> bus assignment ---
    # EXHAUSTIVE, unlike the sparse PV and BSS layers: every LV charge bus must
    # carry exactly one household, or it silently ends up with no load at all.
    # check_bus_ids also rejects a bus_id listed twice, which would mean two
    # households on one bus. Absent column -> no contract to enforce,
    # _assign_loads falls back to a draw and warns there.
    if 'bus_id' in metadata.columns:
        assigned = metadata.dropna(subset=['bus_id'])
        check_bus_ids(assigned['bus_id'], network, BUS_DOMAIN_LV,
                      "metadata['bus_id'] (household -> bus assignment)",
                      exhaustive=True)

    # --- metadata <-> equipment agreement on the equipment columns ---
    # Both files carry pv_system_size_kwp and battery_size_kwh (the MAS4TE
    # household schema reuses the names), but only the bus-level table decides
    # them: a metadata generated independently of the equipment table reads 0.0
    # everywhere while buses do carry equipment. Nothing in the engine consumes
    # metadata's copies, so this is a WARNING -- it protects the reader of the
    # file, not the run.
    if equipment is not None and 'bus_id' in metadata.columns:
        by_bus = equipment.drop_duplicates('bus_id').set_index('bus_id')
        for col in (BSS_SIZE_COLUMN, 'pv_system_size_kwp'):
            if col not in metadata.columns or col not in equipment.columns:
                continue
            expected = metadata['bus_id'].map(pd.to_numeric(by_bus[col], errors='coerce'))
            got = pd.to_numeric(metadata[col], errors='coerce')
            mismatch = int((expected.fillna(0.0) - got.fillna(0.0)).abs().gt(1e-9).sum())
            if mismatch:
                warnings.append(
                    f"[META/equip] `metadata` column '{col}' disagrees with the equipment "
                    f"table on {mismatch} household(s) (metadata total "
                    f"{got.fillna(0.0).sum():.1f} against {expected.fillna(0.0).sum():.1f} "
                    "at bus level). The table is authoritative; regenerate metadata via "
                    "data_processing.equipment.sync_household_equipment."
                )

    # --- SEASONAL COHERENCE: loads vs the simulated month ---
    # config['mois_simule'] drives the PV production month, but the loads were
    # drawn upstream and shipped as a CSV, so nothing mechanically ties them to
    # it. metadata['source_month'] records the month they were drawn from, which
    # makes the mismatch detectable: January consumption under June sun is a
    # plausible-looking run whose net load is wrong all week.
    #
    # A WARNING, not an error: deliberately mismatched months are a legitimate
    # experiment. It is silence that is not acceptable.
    if 'source_month' in metadata.columns:
        drawn = pd.to_numeric(metadata['source_month'], errors='coerce').dropna().unique()
        drawn = sorted(int(m) for m in drawn)
        try:
            simulated = _normalize_month(config['mois_simule'])
        except ValueError:
            simulated = None
        if simulated is not None and drawn and drawn != [simulated]:
            warnings.append(
                f"[LOAD/season] load profiles were drawn from month(s) {drawn} "
                f"(metadata['source_month']) while the simulation targets month "
                f"{simulated} (config['mois_simule']), which drives PV production. "
                "Load and sunlight are not on the same season -- regenerate "
                "load_profiles.csv for that month, or accept the gap."
            )
    else:
        warnings.append(
            "[LOAD/season] seasonal coherence NOT checked: `metadata` has no "
            "'source_month' column. Regenerate metadata.csv and load_profiles.csv via "
            "data_processing.profiles.generate_load_profiles, which writes it."
        )

    # --- config sanity ---
    # Required keys are guaranteed present: _config_to_dict merges the caller's
    # config with the defaults before this runs. Only the VALUES are checked.
    if config['v_min'] >= config['v_max']:
        raise ValueError(f"config['v_min'] ({config['v_min']}) must be < config['v_max'] ({config['v_max']}).")
    if config['loading_max'] <= 0:
        raise ValueError(f"config['loading_max'] ({config['loading_max']}) must be positive.")
    if not (0 < config['cos_phi'] <= 1):
        raise ValueError(f"config['cos_phi'] ({config['cos_phi']}) must be in (0, 1].")
    if config['n_timesteps'] > len(households):
        raise ValueError(
            f"config['n_timesteps'] ({config['n_timesteps']}) exceeds the number of "
            f"timesteps available in `households` ({len(households)})."
        )

    # --- pv_profiles (optional) ---
    # Expected format: columns 'pv_{bus_id}' (integer bus_id), time index aligned
    # on households ('Mon 00:00'..'Sun 23:45'), values in kW (same unit as
    # households, see run_digital_twin). Column shape, BUS ID CONTRACT and time
    # coverage are checked here; index alignment is checked at injection time
    # (_verify_temporal_alignment).
    if pv_profiles is not None and len(pv_profiles) > 0:
        bad_cols = [c for c in pv_profiles.columns
                    if not (str(c).startswith('pv_') and str(c)[3:].isdigit())]
        if bad_cols:
            raise ValueError(
                "`pv_profiles` has columns outside the 'pv_{bus_id}' format (integer "
                f"bus_id): {bad_cols[:10]}{' ...' if len(bad_cols) > 10 else ''}"
            )
        # NOT exhaustive: PV is a sparse layer, most buses carry none. The domain
        # is EQUIPABLE rather than LV alone, because a facility connected at MV
        # can carry an installation too (the rest home, 12 kWp). Stricter than
        # _inject_pv_sgens' own guard, which only asks that the bus exist in
        # net.bus and so accepts the slack, an MV substation or a cabin's LV
        # secondary. Both are kept: this one fails early with a clear message,
        # the other covers callers bypassing run_digital_twin (pv.run_pv_scenario).
        check_bus_ids(
            (int(str(c).split('_', 1)[1]) for c in pv_profiles.columns),
            network, BUS_DOMAIN_EQUIPABLE, 'pv_profiles',
        )
        if config['n_timesteps'] > len(pv_profiles):
            raise ValueError(
                f"config['n_timesteps'] ({config['n_timesteps']}) exceeds the number of "
                f"steps available in `pv_profiles` ({len(pv_profiles)})."
            )


    # --- schedules_BSS (optional) ---
    # LONG format: index = time steps repeated once per battery, columns
    # ['battery_id', 'p_kw'] in kW. The identifier column DECLARES which battery
    # each row is about, which is what makes a bus id contract expressible at all.
    #
    # Malformed (bad identifiers, non-numeric power, too short for n_timesteps),
    # or disagreeing with the equipment table -> ValueError, as for pv_profiles.
    # Infeasible -> warning downstream, power as [BSS/schedule] and energy as
    # [BSS/soc]; both are clipped only when config['bss_enforce_limits'] is set,
    # otherwise applied as-is.
    if schedules_BSS is not None and len(schedules_BSS) > 0:
        battery_ids = _bss_battery_ids(schedules_BSS)          # validates the syntax
        if BSS_POWER_COLUMN not in schedules_BSS.columns:
            raise ValueError(
                f"`schedules_BSS` has no power column '{BSS_POWER_COLUMN}' (kW). "
                f"Columns received: {list(schedules_BSS.columns)}."
            )

        # Every battery must cover the same number of steps. Checked on the row
        # COUNTS here, which yields a readable message; _read_bss_schedule then
        # checks the index LABELS, the stronger statement but only phrasable once
        # the counts agree.
        counts = Counter(str(b) for b in schedules_BSS[BSS_BATTERY_ID_COLUMN])
        if len(set(counts.values())) > 1:
            ragged = sorted(counts.items(), key=lambda kv: kv[1])
            raise ValueError(
                f"`schedules_BSS`: batteries do not all cover the same number of time "
                f"steps (from {ragged[0][0]} with {ragged[0][1]} to "
                f"{ragged[-1][0]} with {ragged[-1][1]}). Long format expected: one row "
                "per (time step, battery)."
            )
        n_steps = counts[battery_ids[0]]
        if config['n_timesteps'] > n_steps:
            raise ValueError(
                f"config['n_timesteps'] ({config['n_timesteps']}) exceeds the number of "
                f"steps available per battery in `schedules_BSS` ({n_steps})."
            )

        # --- IDENTIFIER CONTRACT: schedule <-> equipment table ---
        # By SET, never by count: "20 batteries scheduled for 20 equipped buses"
        # can match while designating different batteries. Both directions are
        # reported, because they mean opposite things -- a schedule for a battery
        # that is not installed, or an installed battery nobody scheduled.
        scheduled_domestic = {b for b in battery_ids if b != BSS_CENTRAL_ID}
        register = _bss_domestic_register(equipment)
        if scheduled_domestic != set(register):
            only_schedule = sorted(scheduled_domestic - set(register),
                                   key=lambda i: int(i[1:]))
            only_table = sorted(set(register) - scheduled_domestic,
                                key=lambda i: int(i[1:]))
            detail = []
            if only_schedule:
                detail.append(
                    f"{len(only_schedule)} scheduled but absent from the table: "
                    f"{only_schedule[:10]}{' ...' if len(only_schedule) > 10 else ''}")
            if only_table:
                detail.append(
                    f"{len(only_table)} installed but unscheduled: "
                    f"{only_table[:10]}{' ...' if len(only_table) > 10 else ''}")
            raise ValueError(
                "Mismatch between `schedules_BSS` and the equipment table: "
                + "; ".join(detail)
                + f". The schedule describes {len(scheduled_domestic)} domestic "
                f"batterie(s), the table installs {len(register)} (rows with "
                f"{BSS_SIZE_COLUMN} > 0)."
            )

        # --- BUS ID CONTRACT ---
        # Two domains, because the two kinds of battery are not interchangeable:
        # a domestic battery on an MV node would be a placement error, and the
        # central one has no LV bus to sit on. NOT exhaustive on either side --
        # a fleet is a sparse layer, most LV buses carry no battery.
        if register:
            check_bus_ids(
                (register[b]['bus_id'] for b in sorted(scheduled_domestic,
                                                       key=lambda i: int(i[1:]))),
                network, BUS_DOMAIN_LV, 'schedules_BSS (domestic batteries)',
            )
        if BSS_CENTRAL_ID in battery_ids:
            full_register = bss_register(equipment)
            central = full_register.get(BSS_CENTRAL_ID)
            if central is None:
                raise ValueError(
                    "`schedules_BSS` names 'b0' but the equipment table carries no "
                    "central battery row (see data_processing.equipment.place_central_bss)."
                )
            check_bus_ids(
                [central['bus_id']], network, BUS_DOMAIN_MV_NODE,
                f"equipment (central battery {BSS_CENTRAL_ID})",
            )
            # Specs are only load-bearing once b0 is actually injected, so
            # they are validated here rather than in the unconditional block
            # above. The domestic C-rate also derives from them, but
            # utils.bss_c_rate raises on its own if they cannot serve.
            if central['e_kwh'] <= 0:
                raise ValueError(
                    f"equipment: central battery e_kwh ({central['e_kwh']}) must be positive."
                )
            if central['p_max_kw'] <= 0:
                raise ValueError(
                    f"equipment: central battery p_max_kw ({central['p_max_kw']}) must be positive."
                )
        if not (0 < config['bss_eta'] <= 1):
            raise ValueError(
                f"config['bss_eta'] ({config['bss_eta']}) must be in (0, 1]."
            )

    return warnings


# ============================================================
#  2. LOAD ASSIGNMENT
#  Consumes the caller-supplied `network` and the `households` consumption
#  timeseries. `households` is an interface contract: who filled it and how does
#  not matter (see data_processing/profiles.py for the Bressoux generator).
# ============================================================
def _assign_loads(network, metadata, households, config):
    """Attach each LV charge bus to its household profile, plus the fixed
    deterministic profiles of the network's MV load facilities
    (network['mv_load_names']). Computes reactive power from cos_phi.

    `households` and the MV load profiles (mv_load_profiles.csv) both arrive in
    kW; the wide frame built here is converted to MW (_KW_TO_MW) right before
    being returned, since it is fed to net.load.p_mw as-is by _run_powerflow.
    """
    warnings = []

    net = network['net']
    buses = network['buses']
    mv_nodes = network['mv_nodes']
    mv_load_names = network['mv_load_names']

    declared = {}
    if 'bus_id' in metadata.columns:
        assigned = metadata.dropna(subset=['bus_id'])
        declared = dict(zip(assigned['bus_id'].astype(int), assigned['profile_id']))

    # Fallback machinery, built only when something is actually missing.
    rng = np.random.default_rng(config['seed'])
    profiles_by_street = None
    n_drawn = 0

    net.load = net.load.iloc[0:0]

    profiles_dict = {}
    bus_to_profile = {}

    for (street, num), b in buses.items():
        load_name = f"Load_{b}"
        pp.create_load(net, bus=b, p_mw=0, q_mvar=0, name=load_name)

        pick = declared.get(int(b))
        if pick is None:
            if profiles_by_street is None:
                profiles_by_street = (metadata.groupby('source_street')['profile_id']
                                      .apply(list).to_dict()
                                      if 'source_street' in metadata.columns else {})
            pool = profiles_by_street.get(street) or metadata['profile_id'].tolist()
            pick = rng.choice(pool)
            n_drawn += 1

        profiles_dict[load_name] = households[f"profile_{pick}"].values
        bus_to_profile[load_name] = pick

    if n_drawn:
        warnings.append(
            f"[LOAD/assign] {n_drawn} of the {len(buses)} LV buses have no assignment "
            "declared in `metadata` (column 'bus_id'): a profile was DRAWN at random "
            "for them. A draw puts several buses -- often neighbours -- on the same "
            "profile, whose peaks then add up on the same feeder. Generate metadata "
            "via data_processing.profiles.generate_household_metadata for an explicit "
            "1:1 assignment."
        )

    # --- Fixed profiles for the network's MV load buses (rest home,
    # horticulture centre, sports hall, school, ...), rather than zero load.
    mv_profiles_path = os.path.join(config['data_path'], "out", "processed", "mv_load_profiles.csv")
    if os.path.exists(mv_profiles_path):
        mv_load_profiles = pd.read_csv(mv_profiles_path, index_col=0)
        for name in mv_load_names:
            b = mv_nodes[name]
            load_name = f"Load_{b}"
            pp.create_load(net, bus=b, p_mw=0, q_mvar=0, name=load_name)
            profiles_dict[load_name] = mv_load_profiles[name].values
            bus_to_profile[load_name] = name
    else:
        warnings.append(
            f"MV load profiles file not found ({mv_profiles_path}) -- "
            f"{', '.join(mv_load_names)} left at zero load."
        )

    df_profiles_p = (pd.DataFrame(profiles_dict) * _KW_TO_MW).clip(lower=1e-4)

    tan_phi = np.tan(np.arccos(config['cos_phi']))
    df_profiles_q = df_profiles_p * tan_phi

    return df_profiles_p, df_profiles_q, bus_to_profile, warnings


# ============================================================
#  2bis. PV INJECTION (native capability)
# ============================================================
def _inject_pv_sgens(net, pv_profiles):
    """Create one sgen per 'pv_{bus_id}' column of `pv_profiles`, on `net`.

    pandapower convention: sgen p_mw POSITIVE = injection (NOT a negative load).
    p_mw=q_mvar=0 at creation (filled per timestep by ConstControl in
    _run_powerflow). q_mvar stays 0 for the whole run (inverters as pure active
    power). net.sgen is emptied first so sgens never accumulate across calls.
    """
    net.sgen = net.sgen.iloc[0:0]
    valid_bus = set(net.bus.index)
    sgen_bus = {}
    for col in pv_profiles.columns:
        bus_id = int(str(col).split('_', 1)[1])
        if bus_id not in valid_bus:
            raise ValueError(
                f"pv_profiles: column {col!r} targets a bus_id {bus_id} absent from the network."
            )
        idx = pp.create_sgen(net, bus=bus_id, p_mw=0.0, q_mvar=0.0, name=str(col))
        sgen_bus[idx] = bus_id
    return sgen_bus


def _verify_temporal_alignment(df_load_p, df_other_p, label='PV'):
    """Guardrail: loads and `df_other_p` (PV production, BSS schedule, ...) must
    share the SAME index (same length, same labels, same order). Step-by-step
    mapping in the power flow is only correct if they do. Raise a clear error
    rather than let a silent offset skew the net production/consumption balance.

    `label` names the other side in the error messages.

    KNOWN LIMITATION: every caller flattens both frames onto a RangeIndex before
    calling, because DFData/ConstControl index BY POSITION. That makes the label
    comparison below vacuous (two RangeIndex of equal length are always equal),
    so in practice only the LENGTH is verified and a same-length temporal offset
    passes unnoticed. Tolerable for PV and loads, both built from
    _WEEK_TIME_INDEX by this repo; NOT self-evident for an externally produced
    BSS schedule -- see the label-level pre-check in _read_bss_schedule.
    """
    if len(df_load_p) != len(df_other_p):
        raise ValueError(
            f"Different lengths: load={len(df_load_p)} steps, {label}={len(df_other_p)} steps."
        )
    if list(df_load_p.index) != list(df_other_p.index):
        raise ValueError(
            f"Different index between load and {label} profiles -- "
            "step-by-step alignment is not mechanically guaranteed."
        )
    return True


# ============================================================
#  2ter. BSS SCHEDULE (LONG FORMAT): fleet resolution and reading
#
#  FORMAT. `schedules_BSS` is LONG: an index of week-step labels repeated once
#  per battery, and exactly two columns -- 'battery_id' (b0..bn) and 'p_kw'.
#  A row reads "this battery, at this step, at this power". The identifier
#  column is checkable against the network, which is what lets each battery keep
#  its own storage element, bus and state of charge.
#
#  Units: the schedule arrives in kW, like households and pv_profiles, and is
#  converted to MW at _KW_TO_MW (module-level, see above) right before the
#  wide frame is handed to _run_powerflow.
# ============================================================

BSS_BATTERY_ID_COLUMN = 'battery_id'   # long-format identifier column
BSS_POWER_COLUMN = 'p_kw'              # long-format power column, kW
# BSS_CENTRAL_ID ('b0') lives in utils now -- utils.bss_c_rate and
# data_processing.bss need it too, both of which import from utils already.


def _bss_storage_name(battery_id):
    """Name of the pandapower storage element carrying `battery_id`.
    """
    return f"BSS_{battery_id}"


def _bss_battery_ids(schedules_BSS):
    """Ordered battery ids of a long-format schedule: b0 first, then b1, b2, ...
    """
    if BSS_BATTERY_ID_COLUMN not in schedules_BSS.columns:
        raise ValueError(
            f"`schedules_BSS` has no '{BSS_BATTERY_ID_COLUMN}' column: long format "
            f"expected (index = time steps, columns "
            f"['{BSS_BATTERY_ID_COLUMN}', '{BSS_POWER_COLUMN}']). Columns received: "
            f"{list(schedules_BSS.columns)}."
        )
    col = schedules_BSS[BSS_BATTERY_ID_COLUMN]
    if col.isna().any():
        raise ValueError(
            f"`schedules_BSS` has {int(col.isna().sum())} row(s) without "
            f"'{BSS_BATTERY_ID_COLUMN}' -- every row must name its battery."
        )
    ids = [str(i) for i in pd.unique(col)]
    bad = sorted(i for i in ids if not (i[:1] == 'b' and i[1:].isdigit()))
    if bad:
        raise ValueError(
            f"`schedules_BSS` has identifiers outside the 'b<integer>' format: "
            f"{bad[:10]}{' ...' if len(bad) > 10 else ''}. Expected "
            f"'{BSS_CENTRAL_ID}' for the central battery and b1..bn for the domestic ones."
        )
    return sorted(ids, key=lambda i: int(i[1:]))


def _bss_domestic_register(equipment):
    """{battery_id: {'bus_id', 'e_kwh'}} for the DOMESTIC batteries (b1..bn)
    only -- a thin filter over utils.bss_register, which now also covers the
    central battery 'b0' (see data_processing.equipment.place_central_bss).
    """
    return {b: {'bus_id': v['bus_id'], 'e_kwh': v['e_kwh']}
            for b, v in bss_register(equipment).items() if b != BSS_CENTRAL_ID}


def _bss_fleet_specs(config, equipment, battery_ids):
    """{battery_id: {'bus_id', 'e_kwh', 'p_max_kw'}} for every battery the
    schedule names, read from `equipment` -- b0 and b1..bn are now rows of the
    SAME table (see data_processing.equipment). b0's power is read directly
    (its own battery_p_max_kw); a domestic battery's is DERIVED via
    bss_c_rate(config, equipment).
    """
    register = bss_register(equipment)
    specs = {}
    c_rate = None  # lazily computed, only if a domestic battery is actually scheduled
    for battery_id in battery_ids:
        entry = register.get(battery_id)
        if entry is None:
            # _validate_inputs compares the two sets and reports both in full;
            # this guard only covers callers reaching the engine's internals
            # directly (digital_twin/plots/ does).
            raise ValueError(
                f"scheduled battery {battery_id!r} is absent from the equipment table."
            )
        if battery_id == BSS_CENTRAL_ID:
            specs[battery_id] = {'bus_id': entry['bus_id'], 'e_kwh': entry['e_kwh'],
                                 'p_max_kw': entry['p_max_kw']}
        else:
            if c_rate is None:
                c_rate = bss_c_rate(config, equipment)
            specs[battery_id] = {'bus_id': entry['bus_id'], 'e_kwh': entry['e_kwh'],
                                 'p_max_kw': entry['e_kwh'] * c_rate}
    return specs


def _read_bss_schedule(schedules_BSS, df_load_p, config, specs, load_time_index=None):
    """Pivot the long schedule into the wide frame the storage ConstControls read.
    """
    warnings = []

    if BSS_POWER_COLUMN not in schedules_BSS.columns:
        raise ValueError(
            f"`schedules_BSS` has no power column '{BSS_POWER_COLUMN}' (kW). "
            f"Columns received: {list(schedules_BSS.columns)}."
        )
    power = schedules_BSS[BSS_POWER_COLUMN]
    if not pd.api.types.is_numeric_dtype(power):
        # Not coerced to NaN: a schedule silently read as zero would look like a
        # fleet that simply never operates.
        raise ValueError(
            f"`schedules_BSS['{BSS_POWER_COLUMN}']` is not numeric (dtype "
            f"{power.dtype}) -- a power in kW is expected."
        )

    battery_ids = _bss_battery_ids(schedules_BSS)
    n_domestic = sum(1 for b in battery_ids if b != BSS_CENTRAL_ID)
    steps = list(pd.unique(schedules_BSS.index))

    # Surfaced unconditionally, so the caller can tell at a glance whether the
    # fleet received is the fleet expected.
    warnings.append(
        f"[BSS/read] {len(battery_ids)} batteries over {len(steps)} time steps: "
        f"{'central (b0) + ' if BSS_CENTRAL_ID in battery_ids else 'no central battery, '}"
        f"{n_domestic} domestic, kW at the interface. Check that this count matches "
        "the expected fleet."
    )

    n_nan = int(power.isna().sum())
    if n_nan:
        warnings.append(
            f"[BSS/read] {n_nan} missing power value(s) -- counted as 0 kW, which "
            "underestimates the fleet power at those steps."
        )

    # TEMPORAL ALIGNMENT: warn, never reject. Three cases:
    #   1. labelled index == load index      -> aligned, nothing to say.
    #   2. labelled index != load index      -> [BSS/align] below.
    #   3. bare RangeIndex                   -> accepted silently: it carries no
    #      time information, so there is nothing to compare against. Guaranteeing
    #      row k is the same instant on both sides is then the PROVIDER's job.
    # Case 3 combined with a shifted schedule is undetectable by construction --
    # see docs/interfacing.md ("Known limitations"). It matters because power is
    # injected SIGNED: an offset flips charge against discharge instead of
    # degrading gracefully.
    if load_time_index is not None and not isinstance(schedules_BSS.index, pd.RangeIndex):
        load_labels = list(load_time_index)
        if len(steps) == len(load_labels) and steps != load_labels:
            warnings.append(
                f"[BSS/align] the schedule's time index differs from the loads' "
                f"(schedule {steps[0]!r}..{steps[-1]!r}, loads {load_labels[0]!r}.."
                f"{load_labels[-1]!r}) -- same length, so the positional guardrail "
                "cannot detect it. Check that the schedule is not offset."
            )

    # --- long -> wide ---
    id_col = schedules_BSS[BSS_BATTERY_ID_COLUMN].astype(str).to_numpy()
    columns = {}
    for battery_id in battery_ids:
        block = schedules_BSS[id_col == battery_id]
        if list(block.index) != steps:
            raise ValueError(
                f"`schedules_BSS`: battery {battery_id!r} covers {len(block)} time steps "
                f"against {len(steps)} for the fleet, or in a different order. Every "
                "battery must cover the SAME time axis in the same order: the series are "
                "then injected by position."
            )
        columns[_bss_storage_name(battery_id)] = np.nan_to_num(
            block[BSS_POWER_COLUMN].to_numpy(dtype=float), nan=0.0)

    # --- [BSS/schedule]: instantaneous power feasibility, battery by battery ---
    # Simulated horizon only, like _track_bss_soc: steps the power flow never runs
    # must not raise feasibility warnings. Each battery is judged against ITS OWN
    # rating -- the central park's kW for b0, capacity * C-rate for a domestic one
    # -- which a single park-wide threshold cannot express.
    horizon = min(int(config['n_timesteps']), len(steps))
    enforce = bool(config['bss_enforce_limits'])
    over = []
    for battery_id in battery_ids:
        col = _bss_storage_name(battery_id)
        p_max_kw = float(specs[battery_id]['p_max_kw'])
        window = columns[col][:horizon]
        n_over = int((np.abs(window) > p_max_kw + 1e-9).sum())
        if n_over:
            k = int(np.argmax(np.abs(window)))
            over.append((battery_id, n_over, float(window[k]), p_max_kw))
        if enforce:
            # Clip only the simulated horizon; the tail is carried through
            # untouched so the frame keeps the FULL length the alignment
            # guardrail compared.
            columns[col][:horizon] = np.clip(window, -p_max_kw, p_max_kw)
    if over:
        # Aggregated, never one line per battery: a fleet of 21 would drown the
        # report. The worst case is picked on the RELATIVE overshoot, so a small
        # domestic battery at 3x its rating outranks the park at 1.05x.
        worst = max(over, key=lambda o: abs(o[2]) / o[3])
        warnings.append(
            f"[BSS/schedule] {len(over)} of {len(battery_ids)} batteries exceed their "
            f"rated power over the simulated horizon ({horizon} steps) -- worst case "
            f"{worst[0]}: {worst[2]:+.1f} kW for {worst[3]:.1f} kW rated, at "
            f"{worst[1]} steps"
            + (" -- clipped (bss_enforce_limits=True)." if enforce
               else " -- applied as-is (bss_enforce_limits=False).")
        )

    # kW -> MW (see _KW_TO_MW at the top of this module).
    df_bss_p = pd.DataFrame({col: values * _KW_TO_MW
                             for col, values in columns.items()})
    # Same discipline as the PV path: df_load_p lost its text index to .values in
    # _assign_loads, so both sides are compared on the positional index.
    _verify_temporal_alignment(df_load_p, df_bss_p, label='BSS schedule')
    return df_bss_p, warnings


# ============================================================
#  2quater. BSS INJECTION (battery fleet, native capability)
#  Same split as the PV layer above: the elements are created here, their
#  per-timestep power is applied by ConstControls in _run_powerflow.
# ============================================================
def _inject_bss_storage(net, specs):
    """Create ONE storage element per battery of `specs`, each on its own bus.
    """
    net.storage = net.storage.iloc[0:0]

    storage_bus = {}
    for battery_id, spec in specs.items():
        max_e_mwh = float(spec['e_kwh']) / 1000.0
        max_p_mw = float(spec['p_max_kw']) / 1000.0
        idx = pp.create_storage(
            net, bus=int(spec['bus_id']), p_mw=0.0,
            max_e_mwh=max_e_mwh, min_e_mwh=BSS_SOC_MIN_FRAC * max_e_mwh,
            max_p_mw=max_p_mw, min_p_mw=-max_p_mw, sn_mva=max_p_mw,
            soc_percent=100.0 * BSS_SOC_INIT, name=_bss_storage_name(battery_id),
        )
        storage_bus[idx] = int(spec['bus_id'])
    return storage_bus


def _track_one_soc(p_mw, e_max_mwh, config, soc_init=BSS_SOC_INIT):
    """Integrate ONE battery's schedule into its state of charge, step by step.
    """
    dt_h = 24.0 / STEPS_PER_DAY
    e_min_mwh = BSS_SOC_MIN_FRAC * e_max_mwh
    eta = float(config['bss_eta'])
    enforce = bool(config['bss_enforce_limits'])

    p_applied = np.asarray(p_mw, dtype=float).copy()
    soc = np.empty(len(p_applied), dtype=float)
    e_mwh = soc_init * e_max_mwh

    n_out, worst_step, worst_soc = 0, None, None
    for k, p in enumerate(p_applied):
        # Charge and discharge do not cost the same: efficiency multiplies what
        # reaches the cells, and divides what has to be drawn from them.
        delta = p * dt_h * eta if p > 0 else p * dt_h / eta
        e_new = e_mwh + delta
        if e_new > e_max_mwh + 1e-12 or e_new < e_min_mwh - 1e-12:
            n_out += 1
            frac = e_new / e_max_mwh
            # "worst" = furthest from mid-band, in either direction
            if worst_soc is None or abs(frac - 0.5) > abs(worst_soc - 0.5):
                worst_step, worst_soc = k, frac
            if enforce:
                e_clipped = min(max(e_new, e_min_mwh), e_max_mwh)
                delta = e_clipped - e_mwh
                p_applied[k] = (delta / (dt_h * eta)) if delta > 0 else (delta * eta / dt_h)
                e_new = e_clipped
        e_mwh = e_new
        soc[k] = e_mwh / e_max_mwh

    return soc, p_applied, n_out, worst_step, worst_soc


def _track_bss_soc(df_bss_p, specs, config, soc_init=BSS_SOC_INIT):
    """Track EACH battery's own state of charge over the fleet.
    """
    soc_columns, applied_columns, breaches = {}, {}, []

    for battery_id, spec in specs.items():
        col = _bss_storage_name(battery_id)
        soc, p_applied, n_out, worst_step, worst_soc = _track_one_soc(
            df_bss_p[col].to_numpy(dtype=float), float(spec['e_kwh']) / 1000.0,
            config, soc_init=soc_init)
        soc_columns[col] = soc
        applied_columns[col] = p_applied
        if n_out:
            breaches.append((battery_id, n_out, worst_step, worst_soc,
                             float(spec['e_kwh'])))

    warnings = []
    if breaches:
        # Worst = the battery whose SoC strays furthest from mid-band, the same
        # criterion _track_one_soc uses step by step.
        worst = max(breaches, key=lambda b: abs(b[3] - 0.5))
        n_steps = len(df_bss_p)
        warnings.append(
            f"[BSS/soc] {len(breaches)} of {len(specs)} batteries leave the "
            f"[{100 * BSS_SOC_MIN_FRAC:.0f}%, 100%] band -- worst case {worst[0]} "
            f"({worst[4]:.1f} kWh): {worst[1]} of {n_steps} steps out of band, down to "
            f"{100 * worst[3]:.1f}% at step {worst[2]}. Schedule inconsistent with the "
            "capacity of these batteries"
            + (" -- clipped (bss_enforce_limits=True)." if config['bss_enforce_limits']
               else " -- applied as-is (bss_enforce_limits=False).")
        )

    index = df_bss_p.index
    return (pd.DataFrame(soc_columns, index=index),
            pd.DataFrame(applied_columns, index=index), warnings)

# ============================================================
#  3. TIME-SERIES POWER FLOW
# ============================================================
def _run_powerflow(net, df_profiles_p, df_profiles_q, config, df_pv_p=None,
                   df_bss_p=None):
    
    net.controller = net.controller.iloc[0:0]  # controllers must not stack across calls

    data_source_p = DFData(df_profiles_p)
    data_source_q = DFData(df_profiles_q)

    for i, load in net.load.iterrows():
        load_column = load["name"]
        ConstControl(net, element='load', variable='p_mw', element_index=[i],
                     data_source=data_source_p, profile_name=load_column)
        ConstControl(net, element='load', variable='q_mvar', element_index=[i],
                     data_source=data_source_q, profile_name=load_column)

    with_pv = df_pv_p is not None
    if with_pv:
        data_source_pv = DFData(df_pv_p)
        for i, sgen in net.sgen.iterrows():
            ConstControl(net, element='sgen', variable='p_mw', element_index=[i],
                         data_source=data_source_pv, profile_name=sgen["name"])

    with_bss = df_bss_p is not None
    if with_bss:
        data_source_bss = DFData(df_bss_p)
        for i, storage in net.storage.iterrows():
            ConstControl(net, element='storage', variable='p_mw', element_index=[i],
                         data_source=data_source_bss, profile_name=storage["name"])

    n_timesteps = config['n_timesteps']
    # output_path=None keeps results in memory (ow.output) instead of round-
    # tripping through CSV on disk.
    ow = OutputWriter(net, time_steps=range(n_timesteps), output_path=None)
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_trafo', 'loading_percent')
    if with_pv:
        ow.log_variable('res_line', 'p_from_mw')     # sign: flow direction (reverse flows)
        ow.log_variable('res_trafo', 'p_hv_mw')      # sign: <0 = reverse HV->LV flow on this transformer
        ow.log_variable('res_ext_grid', 'p_mw')      # sign: <0 = network exports upstream
        ow.log_variable('res_sgen', 'p_mw')
    if with_bss:
        ow.log_variable('res_storage', 'p_mw')       # sign: >0 = charging

    # verbose=True makes pandapower print a tqdm progress bar over time_steps.
    run_timeseries(net, time_steps=range(n_timesteps), verbose=bool(config['verbose']))

    out = {
        'vm': ow.output['res_bus.vm_pu'],
        'line_load': ow.output['res_line.loading_percent'],
        'trafo_load': ow.output['res_trafo.loading_percent'],
    }
    if with_pv:
        out['line_p_from'] = ow.output['res_line.p_from_mw']
        out['trafo_p_hv'] = ow.output['res_trafo.p_hv_mw']
        out['ext_grid_p'] = ow.output['res_ext_grid.p_mw']
        out['sgen_p_out'] = ow.output['res_sgen.p_mw']
    if with_bss:
        out['storage_p'] = ow.output['res_storage.p_mw']
    return out


# ============================================================
#  4. RESULTS EVALUATION (verdicts)
# ============================================================
def _collect_violations(df, metric, threshold, mode):
    """Scan a (n_timesteps x n_element) DataFrame for out-of-bounds values.

    mode: 'below' -> df < threshold ; 'above' -> df > threshold.
    Returns a list of violation dicts, one per (timestep, element) breach.
    """
    mask = (df < threshold) if mode == "below" else (df > threshold)
    if not mask.values.any():
        return []
    detail = f"{'<' if mode == 'below' else '>'} {threshold}"
    violations = []
    for (timestep, element), value in df.where(mask).stack().items():
        violations.append({
            'element': element,
            'metric': metric,
            'value': float(value),
            'timestep': int(timestep) if isinstance(timestep, (int, np.integer)) else timestep,
            'detail': detail,
        })
    return violations


def _aggregate_profile_summary(df_profiles_p):
    """Aggregated metrics: sum of the active power of all loads at each step,
    reported in kW for readability."""
    agg_mw = df_profiles_p.sum(axis=1)
    peak_mw = float(agg_mw.max())
    mean_mw = float(agg_mw.mean())
    summary = {
        'peak_agg_kw': peak_mw * 1000.0,
        'mean_agg_kw': mean_mw * 1000.0,
        'load_factor_agg': (mean_mw / peak_mw) if peak_mw > 0 else None,
    }
    return summary, agg_mw


def _evaluate(vm, line_load, trafo_load, config):
    """Turn raw results into a network verdict + list of violations.
    """
    v_min, v_max = config['v_min'], config['v_max']
    load_max = config['loading_max']

    violations = []
    if vm is not None:
        violations += _collect_violations(vm, 'vm_pu', v_min, 'below')
        violations += _collect_violations(vm, 'vm_pu', v_max, 'above')
    if line_load is not None:
        violations += _collect_violations(line_load, 'line_loading_pct', load_max, 'above')
    if trafo_load is not None:
        violations += _collect_violations(trafo_load, 'trafo_loading_pct', load_max, 'above')

    n_voltage_violations = sum(1 for v in violations if v['metric'] == 'vm_pu')
    n_loading_violations = len(violations) - n_voltage_violations

    summary = {
        'network_valid': len(violations) == 0,
        'min_voltage_pu': float(vm.values.min()) if vm is not None else None,
        'max_voltage_pu': float(vm.values.max()) if vm is not None else None,
        'max_line_loading_pct': float(line_load.values.max()) if line_load is not None else None,
        'max_trafo_loading_pct': float(trafo_load.values.max()) if trafo_load is not None else None,
        'n_voltage_violations': n_voltage_violations,
        'n_loading_violations': n_loading_violations,
    }
    return summary, violations


# ============================================================
#  5. OUTPUT FORMATTING
# ============================================================
def _format_output(summary, violations, warnings):
    """Assemble the single output DataFrame (verdict + violations + warnings).
    """
    rows = []

    for metric, value in summary.items():
        rows.append({'category': 'summary', 'element': 'network',
                     'metric': metric, 'value': value,
                     'timestep': None, 'detail': None})

    for v in violations:
        rows.append({'category': 'violation', 'element': v.get('element'),
                     'metric': v.get('metric'), 'value': v.get('value'),
                     'timestep': v.get('timestep'), 'detail': v.get('detail')})

    for w in warnings:
        rows.append({'category': 'warning', 'element': None,
                     'metric': None, 'value': None,
                     'timestep': None, 'detail': w})

    return pd.DataFrame(rows, columns=[
        'category', 'element', 'metric', 'value', 'timestep', 'detail'])


def _format_report_value(value):
    """Round floats to 4 decimals for report readability; leave bools/ints/
    strings/None as-is (isinstance(True, int) is True in Python, so bool is
    checked first)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return round(value, 4)
    return value


def save_run_report(out, out_path):
    """Render the long-format `out` DataFrame (see run_digital_twin/
    _format_output) as a Markdown report and write it to
    `out_path`.
    """
    summary = out[out['category'] == 'summary']
    violations = out[out['category'] == 'violation']
    run_warnings = out[out['category'] == 'warning']

    network_valid = summary.set_index('metric')['value'].get('network_valid')

    lines = [
        "# Digital twin run report",
        "",
        f"**Generated:** {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"**Network valid:** {network_valid}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    lines += [f"| {row['metric']} | {_format_report_value(row['value'])} |" for _, row in summary.iterrows()]

    lines += ["", f"## Violations ({len(violations)})", ""]
    if len(violations) == 0:
        lines.append("No violations.")
    else:
        lines += ["| Element | Metric | Value | Timestep | Detail |", "|---|---|---|---|---|"]
        lines += [
            f"| {row['element']} | {row['metric']} | {row['value']:.4f} | "
            f"{row['timestep']} | {row['detail']} |"
            for _, row in violations.iterrows()
        ]

    lines += ["", f"## Warnings ({len(run_warnings)})", ""]
    if len(run_warnings) == 0:
        lines.append("No warnings.")
    else:
        lines += [f"- {row['detail']}" for _, row in run_warnings.iterrows()]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return out_path

def _print_scenario_info(cfg, metadata, network, pv_profiles, schedules_BSS):
    """config['verbose']: print what this run is about to simulate."""
    has_pv = pv_profiles is not None and len(pv_profiles) > 0
    has_bss = schedules_BSS is not None and len(schedules_BSS) > 0

    print("=" * 60)
    print("Digital twin run")
    print("=" * 60)
    print(f"  data_path       : {cfg['data_path']}")
    print(f"  month simulated : {cfg['mois_simule']}")
    print(f"  timesteps       : {cfg['n_timesteps']}")
    print(f"  households      : {len(metadata)} (LV buses: {len(network.get('buses', {}))})")
    print(f"  PV layer        : {'ON (' + str(len(pv_profiles.columns)) + ' buses)' if has_pv else 'OFF'}")
    if has_bss:
        print(f"  BSS layer       : ON ({len(_bss_battery_ids(schedules_BSS))} batteries)")
    else:
        print("  BSS layer       : OFF")
    print(f"  thresholds      : v in [{cfg['v_min']}, {cfg['v_max']}] pu, loading <= {cfg['loading_max']}%")
    print("-" * 60)


def _print_results_info(out, warnings):
    """config['verbose']: print the verdict once the run is done."""
    summary = out.loc[out['category'] == 'summary'].set_index('metric')['value']
    n_violations = int((out['category'] == 'violation').sum())

    print("-" * 60)
    print("Digital twin results")
    print(f"  network valid       : {summary.get('network_valid')}")
    min_v, max_v = summary.get('min_voltage_pu'), summary.get('max_voltage_pu')
    if min_v is not None and max_v is not None:
        print(f"  voltage range (pu)  : [{min_v:.4f}, {max_v:.4f}]")
    max_line = summary.get('max_line_loading_pct')
    if max_line is not None:
        print(f"  max line loading    : {max_line:.1f}%")
    max_trafo = summary.get('max_trafo_loading_pct')
    if max_trafo is not None:
        print(f"  max trafo loading   : {max_trafo:.1f}%")
    print(f"  violations          : {n_violations} "
         f"({summary.get('n_voltage_violations')} voltage, "
         f"{summary.get('n_loading_violations')} loading)")
    print(f"  warnings            : {len(warnings)}")
    print("=" * 60)


def run_digital_twin(config, metadata, network, households,
                     pv_profiles=None, schedules_BSS=None, equipment=None):
    """Run the digital twin on a given network and return a network verdict.

    Parameters
    ----------
    config : dict                # tunable simulation/validation parameters
    metadata : pd.DataFrame      # per-household attributes (profile_id, source_street, ...)
    network : dict               # caller-supplied network bundle
        {"net", "buses", "mv_nodes", "mv_load_names", ...} -- built by a
        site-specific inputs/network.py, see digital_twin.utils.load_site_network.
        `net` is mutated in place (loads/sgens cleared and repopulated) -- copy
        it yourself before calling if you plan to reuse `network` across calls.
        It is also the REFERENCE against which the bus ids of the optional
        layers are checked (see _validate_inputs): a caller supplying profiles
        for buses this network does not have, or for buses no equipment may sit
        on, is rejected here rather than deep inside pandapower.
    households : pd.DataFrame    # consumption profiles (n_timesteps x n_profiles), kW
    pv_profiles : pd.DataFrame, optional
        PV production. Columns 'pv_{bus_id}' (integer network bus_id), time index
        aligned on households ('Mon 00:00'..'Sun 23:45'), values in kW (SAME
        unit as households -- matches pv.build_pv_production's own output
        directly, no conversion needed upstream). When provided: injected as
        sgen, loads + PV power flow. When None: no PV (loads-only).
    schedules_BSS : pd.DataFrame, optional
        Battery fleet schedule, LONG format: index = the time steps (aligned on
        households) repeated once per battery, and two columns --
        'battery_id' and 'p_kw'.
        IDENTIFIERS: 'b0' is the CENTRAL battery, 'b1'..'bn' are DOMESTIC
        batteries -- both kinds placed and sized once, at data-generation time,
        as rows of the `equipment` table (see data_processing.equipment
        .place_central_bss / place_bss). A battery exists exactly when the
        schedule names it.
        Values in **kW**, like households and pv_profiles.
        SIGN: positive = CHARGING, negative = discharge -- the storage
        convention, OPPOSITE of the sgen one `pv_profiles` follows.
        When provided: one pandapower storage per battery, each driven per
        timestep with its own state of charge tracked and reported (bss_*
        metrics). When None: no BSS. Passing the schedule IS the activation, as
        for pv_profiles.
        Related config: bss_c_rate (optional override, else derived from `equipment`'s
        central row) / bss_eta / bss_enforce_limits / export_bss_soc (see
        digital_twin.utils._config_to_dict) -- NOT bss_bus/bss_park_e_kwh/
        bss_park_p_kw, which no longer exist: a battery's bus and capacity are a
        DATA fact (equipment.csv), never a runtime config scalar.
    equipment : pd.DataFrame, optional
        Bus-level equipment register (see data_processing.equipment): one row per
        equipable bus, carrying `battery_id`, `battery_size_kwh` and (on the
        central row only) `battery_p_max_kw`. Required as soon as
        `schedules_BSS` names ANY battery, central or domestic -- it is what
        turns b0..bn into bus ids and capacities, and what their identifiers
        are checked against (by SET: a mismatch in either direction is an
        error, not a warning). Not needed for the PV layer, whose placement
        already arrives baked into `pv_profiles`' column names.

    Returns
    -------
    out : pd.DataFrame
        Long-format result: summary (verdict) + violations + warnings.
    """
    # Loud guard rail: these three keys used to be how a caller sized/placed the
    # central battery; they are now a data fact (equipment.csv), generated once
    # by data_processing.equipment.place_central_bss. A caller still setting
    # them would otherwise be silently ignored -- a wrong-result trap far worse
    # than raising here. Checked on the RAW config, before _config_to_dict.
    if config and (bad := _REMOVED_BSS_CONFIG_KEYS & set(config)):
        raise ValueError(
            f"config carries {sorted(bad)}, which run_digital_twin no longer reads: "
            "the central battery's bus/capacity/power are now a row of `equipment` "
            "(see data_processing.equipment.place_central_bss), generated once rather "
            "than passed at run time. Remove these keys from your config."
        )

    cfg = _config_to_dict(config)

    if cfg['verbose']:
        _print_scenario_info(cfg, metadata, network, pv_profiles, schedules_BSS)

    warnings = _validate_inputs(cfg, metadata, network, households, pv_profiles,
                                schedules_BSS, equipment)
    warnings += network.get('warnings', [])

    df_p, df_q, bus_to_profile, w_assign = _assign_loads(
        network, metadata, households, cfg)
    warnings += w_assign

    net = network['net']

    # --- PV injection (activated by the presence of pv_profiles) ---
    df_pv_p = None
    if pv_profiles is not None and len(pv_profiles) > 0:
        # Positional alignment: _assign_loads builds df_p via .values, dropping
        # the text index 'Mon 00:00'.. for a RangeIndex. pv_profiles must sit on
        # the same RangeIndex before the alignment check and the power flow,
        # since DFData/ConstControl index by position. kW -> MW here (_KW_TO_MW),
        # the same conversion point as _assign_loads and _read_bss_schedule.
        df_pv_p = pv_profiles.reset_index(drop=True) * _KW_TO_MW
        _verify_temporal_alignment(df_p, df_pv_p)
        _inject_pv_sgens(net, df_pv_p)

    # --- BSS injection (activated by presence, like pv_profiles above) ---
    # Fleet resolved from the schedule's identifiers + the equipment table, one
    # storage per battery on its own bus, each SoC integrated separately.
    # households.index is passed for the offset pre-check: df_p has already lost
    # its own to .values in _assign_loads.
    df_bss_p, df_bss_soc, bss_specs = None, None, {}
    if schedules_BSS is not None and len(schedules_BSS) > 0:
        bss_specs = _bss_fleet_specs(cfg, equipment, _bss_battery_ids(schedules_BSS))
        df_bss_p, w_bss = _read_bss_schedule(
            schedules_BSS, df_p, cfg, bss_specs, load_time_index=households.index)
        warnings += w_bss

        # The SoC walk can clip, and its output -- not the raw schedule -- is
        # what drives the power flow, so bss_enforce_limits actually takes
        # effect instead of only raising a warning.
        df_bss_soc, df_bss_p, w_soc = _track_bss_soc(df_bss_p, bss_specs, cfg)
        warnings += w_soc

        _inject_bss_storage(net, bss_specs)

    pf = _run_powerflow(net, df_p, df_q, cfg, df_pv_p=df_pv_p, df_bss_p=df_bss_p)
    vm, line_load, trafo_load = pf['vm'], pf['line_load'], pf['trafo_load']

    summary, violations = _evaluate(vm, line_load, trafo_load, cfg)

    agg_summary, agg_mw = _aggregate_profile_summary(df_p)
    summary.update(agg_summary)

    out = _format_output(summary, violations, warnings)

    # Optional export of the full aggregate profile (672 steps), off by default
    # to keep the standard output light for MAS4TE consumers.
    if cfg.get('export_aggregate_profile'):
        profile_rows = pd.DataFrame({
            'category': 'profile_agg', 'element': None, 'metric': 'p_agg_mw',
            'value': agg_mw.values, 'timestep': range(len(agg_mw)), 'detail': None,
        })
        out = pd.concat([out, profile_rows], ignore_index=True)

    # Same discipline for the per-battery state of charge: the verdict carries
    # the fleet aggregate (bss_soc_min_pct / bss_soc_max_pct), the full walk is
    # opt-in because it is n_batteries x n_timesteps rows.
    if cfg.get('export_bss_soc') and df_bss_soc is not None:
        soc_rows = df_bss_soc.reset_index(drop=True).melt(
            var_name='element', value_name='value', ignore_index=False,
        ).reset_index(names='timestep')
        soc_rows['category'] = 'bss_soc'
        soc_rows['metric'] = 'soc_frac'
        soc_rows['detail'] = None
        out = pd.concat([out, soc_rows[out.columns]], ignore_index=True)

    if cfg['verbose']:
        _print_results_info(out, warnings)

    return out
