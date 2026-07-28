"""
Digital twin engine -- network-agnostic.

Exposes a single entry point `run_digital_twin` callable by the MAS4TE
framework. Given a network, household metadata and consumption profiles, it
assigns loads, runs a time-series power flow, and returns a verdict on whether
the network keeps voltages and line/transformer loadings within acceptable
limits.

Pipeline (see docs/pipeline.md):
    1. Data ingestion / input validation
    2. Load assignment (network reconstruction is now a CALLER responsibility,
       see `network` below -- this file no longer builds any topology itself)
    3. Time-series power flow
    4. Results evaluation (verdicts)
    5. Output formatting

Scope: LOADS + PV + BSS. Both optional layers are activated BY PRESENCE of their
argument -- no enable flag for either:
  * `pv_profiles`   -> sgen injection (_inject_pv_sgens).
  * `schedules_BSS` -> storage injection (_read_bss_schedule ->
    _inject_bss_storage -> _track_bss_soc): the columns are summed into one park
    power series driving ONE storage element on an MV node (config['bss_bus']).

Two OPPOSITE sign conventions live in this file:
  * sgen (PV):     positive p_mw = INJECTION.
  * storage (BSS): positive p_mw = CHARGING (consumption), negative = discharge.
The BSS schedule follows the storage one; _inject_bss_storage documents it for
the whole BSS path.

Units: schedules_BSS is in kW, households/pv_profiles in MW.

`network` is a caller-supplied input (see digital_twin.utils.load_site_network
and e.g. data/bressoux/inputs/network.py for how a site builds one), NOT
built internally: this is what makes the engine reusable for a different LV
network without touching this package. A caller that reuses the SAME
`network` object across repeated run_digital_twin calls is responsible for
copying it first (net.load/net.sgen are cleared and repopulated in place
every call) -- see digital_twin.data_processing.pv.run_pv_scenario for the
equivalent copy.deepcopy(network['net']) pattern.
"""

import datetime
import os

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import DFData, OutputWriter, run_timeseries

from .utils import STEPS_PER_DAY, _config_to_dict

# ============================================================
#  1. DATA INGESTION / INPUT VALIDATION
# ============================================================
def _validate_inputs(config, metadata, households, pv_profiles, schedules_BSS):
    """Check that injected DataFrames have the expected structure.

    Returns
    -------
    warnings : list[str]
        Non-fatal issues (e.g. optional inputs missing, unsupported columns).

    Raises
    ------
    ValueError
        On fatal problems (missing mandatory columns, empty households,
        inconsistent lengths).
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

    # --- config sanity ---
    # (required keys are always present at this point: _config_to_dict merges
    #  the caller's config with defaults before _validate_inputs is called;
    #  here we only check that the VALUES make physical sense.)
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
    # on households ('Mon 00:00'..'Sun 23:45'), values in MW (same unit as
    # households, see run_digital_twin). Here we validate the column shape and
    # the time coverage; bus existence and index alignment are checked at
    # injection time (_inject_pv_sgens / _verify_temporal_alignment).
    if pv_profiles is not None and len(pv_profiles) > 0:
        bad_cols = [c for c in pv_profiles.columns
                    if not (str(c).startswith('pv_') and str(c)[3:].isdigit())]
        if bad_cols:
            raise ValueError(
                "`pv_profiles` a des colonnes hors format 'pv_{bus_id}' (bus_id entier) : "
                f"{bad_cols[:10]}{' ...' if len(bad_cols) > 10 else ''}"
            )
        if config['n_timesteps'] > len(pv_profiles):
            raise ValueError(
                f"config['n_timesteps'] ({config['n_timesteps']}) depasse le nombre de pas "
                f"disponibles dans `pv_profiles` ({len(pv_profiles)})."
            )
        
    # --- schedules_BSS (optional) ---
    # One or more NUMERIC power columns, kW. Summed into one park aggregate by
    # _read_bss_schedule, so column names carry no meaning (unlike pv_profiles'
    # 'pv_{bus_id}'). Coverage only here; dtype and alignment live there.
    #
    # Malformed (non-numeric, or too short for n_timesteps) -> ValueError, as for
    # pv_profiles. Infeasible -> warning downstream, power as [BSS/schedule] and
    # energy as [BSS/soc]; both are clipped only when config['bss_enforce_limits']
    # is set, otherwise applied as-is.
    
    if schedules_BSS is not None and len(schedules_BSS) > 0:
        if schedules_BSS.shape[1] == 0:
            raise ValueError(
                "`schedules_BSS` has no column: at least one power column (kW) is required."
            )
        if config['n_timesteps'] > len(schedules_BSS):
            raise ValueError(
                f"config['n_timesteps'] ({config['n_timesteps']}) exceeds the number of "
                f"timesteps available in `schedules_BSS` ({len(schedules_BSS)})."
            )
        # Park specs are only load-bearing once a schedule is actually injected,
        # so they are validated here rather than in the unconditional block above.
        if config['bss_park_e_kwh'] <= 0:
            raise ValueError(
                f"config['bss_park_e_kwh'] ({config['bss_park_e_kwh']}) must be positive."
            )
        if config['bss_park_p_kw'] <= 0:
            raise ValueError(
                f"config['bss_park_p_kw'] ({config['bss_park_p_kw']}) must be positive."
            )
        if not (0 < config['bss_eta'] <= 1):
            raise ValueError(
                f"config['bss_eta'] ({config['bss_eta']}) must be in (0, 1]."
            )

    return warnings


# ============================================================
#  2. LOAD ASSIGNMENT
#  Consumes the `network` (caller-supplied, see module docstring) and the
#  `households` consumption timeseries (an interface contract: it does not
#  matter who filled it or how -- see digital_twin/data_processing/profiles.py
#  for the Bressoux upstream generator).
# ============================================================
def _assign_loads(network, metadata, households, config):
    """Assign a load profile to each LV charge bus, plus the fixed deterministic
    profiles of the network's MV load facilities (network['mv_load_names']).

    Uses the household->street mapping (light street-based association; global
    draw for orphan buses) and computes reactive power from cos_phi.

    Returns
    -------
    df_profiles_p : pd.DataFrame   # active power per load [MW], n_timesteps rows
    df_profiles_q : pd.DataFrame   # reactive power per load [MVAr]
    bus_to_profile : dict          # load name -> profile_id (metadata) or
                                    # facility name (MV load), traceability
    warnings : list[str]           # e.g. households not matched to a street
    """
    warnings = []
    rng = np.random.default_rng(config['seed'])

    net = network['net']
    buses = network['buses']
    mv_nodes = network['mv_nodes']
    mv_load_names = network['mv_load_names']

    # street -> [profile_id, ...] ; global fallback pool for unmatched buses
    profiles_by_street = metadata.groupby('source_street')['profile_id'].apply(list).to_dict()
    all_ids = metadata['profile_id'].tolist()

    net.load = net.load.iloc[0:0]

    profiles_dict = {}
    bus_to_profile = {}

    for (street, num), b in buses.items():
        load_name = f"Load_{b}"
        pp.create_load(net, bus=b, p_mw=0, q_mvar=0, name=load_name)

        if street in profiles_by_street:
            pick = rng.choice(profiles_by_street[street])
        else:
            # Global draw, expected for streets with no dedicated profile in
            # metadata (normal behaviour, not an anomaly to report).
            pick = rng.choice(all_ids)

        profiles_dict[load_name] = households[f"profile_{pick}"].values
        bus_to_profile[load_name] = pick

    # --- Fixed profiles for the network's MV load buses (rest home,
    # horticulture centre, sports hall, school, ...) -- attached here rather
    # than left at zero load.
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

    df_profiles_p = pd.DataFrame(profiles_dict).clip(lower=1e-4)

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

    Returns
    -------
    sgen_bus : dict {index in net.sgen -> bus_id}
    """
    net.sgen = net.sgen.iloc[0:0]
    valid_bus = set(net.bus.index)
    sgen_bus = {}
    for col in pv_profiles.columns:
        # 'pv_123' -> 123.
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

    `label` names the other side in the error messages ('PV' by default, so
    existing callers read unchanged).

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
            "step-by-step alignment is no longer mechanically guaranteed."
        )
    return True


# ============================================================
#  2ter. BSS SCHEDULE READING (reduction to the park aggregate)
#
#  Units: schedules_BSS arrives in kW -- the ONLY kW interface of
#  run_digital_twin -- and is converted to MW at _BSS_KW_TO_MW below and nowhere
#  else. pv_profiles is already MW at its interface (the /1000 happens upstream
#  in process_data), so the two are NOT symmetric.
# ============================================================
_BSS_KW_TO_MW = 1e-3

BSS_STORAGE_NAME = "BSS_central"   # storage element name == aggregate column name
BSS_SOC_MIN_FRAC = 0.1             # min_e_mwh = 10% of max_e_mwh (reserve)
BSS_SOC_INIT = 0.5                 # SoC at step 0; also the created soc_percent


def _read_bss_schedule(schedules_BSS, df_load_p, config, load_time_index=None):
    """Reduce a BSS schedule to the ONE-COLUMN frame the storage ConstControl reads.

    Any number of numeric columns is summed column-wise -- one per battery, agent
    or park sub-unit; the engine only needs the aggregate. A single pre-aggregated
    column is the degenerate case of that sum, so nothing branches on column count.

    Sign: positive = charging, negative = discharge (see _inject_bss_storage), so
    the power check below is on |p|, symmetric in both directions.

    The output column is named BSS_STORAGE_NAME so _run_powerflow's
    profile_name <-> element name matching works as it does for 'pv_{bus_id}'
    sgens -- that naming is why this returns a DataFrame rather than a Series.

    Warning prefixes, one meaning each:
      [BSS/read]     -- what was read (column count, missing values)
      [BSS/align]    -- labelled index differs from the loads'
      [BSS/schedule] -- instantaneous POWER over config['bss_park_p_kw']
    The last one is not redundant with [BSS/soc] (cumulative energy, raised in
    _track_bss_soc): both usually fire on the same file, for different causes.

    Parameters
    ----------
    schedules_BSS : pd.DataFrame  # >=1 numeric power column, kW at the interface
    df_load_p : pd.DataFrame      # load frame to align against (RangeIndex)
    config : dict                 # bss_park_p_kw, n_timesteps, bss_enforce_limits
    load_time_index : optional    # loads' labelled index (households.index),
                                   # for the offset pre-check
    Returns
    -------
    df_bss_p : pd.DataFrame  # single column BSS_STORAGE_NAME, MW, RangeIndex
    n_cols : int             # columns summed into the aggregate
    warnings : list[str]     # '[BSS/read]', '[BSS/align]', '[BSS/schedule]'
    """
    warnings = []

    non_numeric = [c for c in schedules_BSS.columns
                   if not pd.api.types.is_numeric_dtype(schedules_BSS[c])]
    if non_numeric:
        # Not coerced to NaN: a schedule silently read as zero would look like a
        # park that simply never operates.
        raise ValueError(
            f"`schedules_BSS` has non-numeric column(s): {non_numeric[:10]}"
            f"{' ...' if len(non_numeric) > 10 else ''}. Expected one numeric power "
            "column per battery/agent/park sub-unit, in kW."
        )

    # Surfaced unconditionally, so the caller can tell at a glance whether the
    # file received is the file expected (33 batteries vs 1 park profile).
    n_cols = int(schedules_BSS.shape[1])
    warnings.append(
        f"[BSS/read] {n_cols} schedule column(s) read over {len(schedules_BSS)} "
        f"timesteps, summed into a single park aggregate (kW at the interface). "
        "Check this column count matches the expected number of batteries/agents."
    )

    n_nan = int(schedules_BSS.isna().sum().sum())
    if n_nan:
        warnings.append(
            f"[BSS/read] {n_nan} missing value(s) in the schedule -- counted as 0 kW "
            "by the column-wise sum, which understates the park power at those steps."
        )

    # TEMPORAL ALIGNMENT: warn, never reject. Three cases:
    #   1. labelled index == load index      -> aligned, nothing to say.
    #   2. labelled index != load index      -> [BSS/align] below.
    #   3. bare RangeIndex                   -> accepted silently: it carries no
    #      time information, so there is nothing to compare against. Guaranteeing
    #      row k is the same instant on both sides is then the PROVIDER's job.
    # Case 3 combined with a shifted schedule is undetectable by construction --
    # the irreducible hole, listed in docs/interfacing.md ("Known limitations").
    # It matters because the aggregate is injected as SIGNED power: an offset
    # flips charge against discharge instead of degrading gracefully.
    #
    # Runs BEFORE the RangeIndex flattening below, which erases the evidence
    # (_verify_temporal_alignment then only sees lengths). Worth doing because a
    # BSS schedule is produced outside this repo, unlike households/pv_profiles.
    if load_time_index is not None and not isinstance(schedules_BSS.index, pd.RangeIndex):
        sched_labels, load_labels = list(schedules_BSS.index), list(load_time_index)
        if len(sched_labels) == len(load_labels) and sched_labels != load_labels:
            warnings.append(
                f"[BSS/align] the schedule's own time index differs from the load index "
                f"(schedule {sched_labels[0]!r}..{sched_labels[-1]!r}, loads "
                f"{load_labels[0]!r}..{load_labels[-1]!r}) -- same length, so the "
                "positional guardrail cannot catch it. Verify the schedule is not "
                "time-shifted before its values are injected."
            )

    # Same discipline as the PV path: df_load_p lost its text index to .values in
    # _assign_loads, so flatten onto the same RangeIndex before the guardrail.
    sched = schedules_BSS.reset_index(drop=True)
    _verify_temporal_alignment(df_load_p, sched, label='BSS schedule')

    total_kw = sched.sum(axis=1).to_numpy(dtype=float)

    # --- [BSS/schedule]: instantaneous power feasibility ---
    # Simulated horizon only, like _track_bss_soc: steps the power flow never runs
    # must not raise feasibility warnings. Compared in kW against the config value
    # itself, so the threshold printed is the threshold applied. |p| because the
    # rating is symmetric in charge and discharge.
    horizon = min(int(config['n_timesteps']), len(total_kw))
    p_max_kw = float(config['bss_park_p_kw'])
    enforce = bool(config['bss_enforce_limits'])
    window = total_kw[:horizon]
    n_over = int((np.abs(window) > p_max_kw + 1e-9).sum())
    if n_over:
        k = int(np.argmax(np.abs(window)))
        warnings.append(
            f"[BSS/schedule] scheduled power exceeds config['bss_park_p_kw'] "
            f"({p_max_kw:.1f} kW) at {n_over} of {horizon} step(s) -- worst at step "
            f"{k} ({window[k]:+.1f} kW)"
            + (" -- clipped (bss_enforce_limits=True)." if enforce
               else " -- applied as-is (bss_enforce_limits=False).")
        )
    if enforce:
        # Clip only the simulated horizon; the tail is carried through untouched
        # so the frame keeps the FULL length the alignment guardrail compared.
        total_kw[:horizon] = np.clip(window, -p_max_kw, p_max_kw)

    # THE kW -> MW conversion point (see the unit convention above).
    df_bss_p = pd.DataFrame({BSS_STORAGE_NAME: total_kw * _BSS_KW_TO_MW})
    return df_bss_p, n_cols, warnings


# ============================================================
#  2quater. BSS INJECTION (centralised battery park, native capability)
#  Same split as the PV layer above: the element is created here, its
#  per-timestep power is applied by a ConstControl in _run_powerflow.
# ============================================================
def _inject_bss_storage(net, config, mv_nodes):
    """Create the single centralised BSS storage element on `net`.

    SIGN CONVENTION for the whole BSS path -- everything else refers back here.
    pandapower storage: p_mw POSITIVE = CHARGE (a consumer seen from the grid),
    NEGATIVE = discharge. Verified on pandapower 3.4.0: a storage at p_mw=+0.1
    drives res_ext_grid.p_mw to +0.1, i.e. the network imports. This is the
    OPPOSITE of the sgen convention in _inject_pv_sgens just above, and the
    schedule follows the STORAGE one.

    p_mw=0 at creation (ConstControl fills it per timestep in _run_powerflow),
    q_mvar left at 0 for the whole run (pure active power, as for the PV sgens).
    net.storage is emptied first so storages never accumulate across calls.

    PLACEMENT on an MV node, resolved through the bundle's `mv_nodes` (9 buses on
    Bressoux). An LV secondary bus is out of scope: build_network keeps
    `cabine_lv_bus` local, so config has no stable handle to name one.
    config['bss_bus'] accepts an MV node NAME or an integer bus id.

    Returns
    -------
    storage_bus : dict {index in net.storage -> bus_id}
    """
    net.storage = net.storage.iloc[0:0]

    bss_bus = config['bss_bus']
    valid = {name: int(b) for name, b in mv_nodes.items()}
    if isinstance(bss_bus, str):
        if bss_bus not in valid:
            raise ValueError(
                f"config['bss_bus'] = {bss_bus!r} is not an MV node of this network. "
                f"Valid names: {sorted(valid)}."
            )
        bus_id = valid[bss_bus]
    else:
        try:
            bus_id = int(bss_bus)
        except (TypeError, ValueError):
            raise ValueError(
                f"config['bss_bus'] = {bss_bus!r} is neither an MV node name nor an "
                f"integer bus id. Valid names: {sorted(valid)}."
            )
        if bus_id not in set(valid.values()):
            raise ValueError(
                f"config['bss_bus'] = {bus_id} is not an MV bus of this network "
                f"(MV bus ids: {sorted(valid.values())}). Valid names: {sorted(valid)}."
            )

    max_e_mwh = float(config['bss_park_e_kwh']) / 1000.0
    max_p_mw = float(config['bss_park_p_kw']) / 1000.0
    idx = pp.create_storage(
        net, bus=bus_id, p_mw=0.0,
        max_e_mwh=max_e_mwh, min_e_mwh=BSS_SOC_MIN_FRAC * max_e_mwh,
        max_p_mw=max_p_mw, min_p_mw=-max_p_mw, sn_mva=max_p_mw,
        soc_percent=100.0 * BSS_SOC_INIT, name=BSS_STORAGE_NAME,
    )
    return {idx: bus_id}


def _track_bss_soc(p_mw, config, soc_init=BSS_SOC_INIT):
    """Integrate the aggregate schedule into a state of charge, step by step.

    pandapower does not update soc_percent during a time-series run -- ConstControl
    only writes p_mw -- so the SoC is tracked here by hand. It is the only way to
    tell whether a schedule is feasible for a park of config['bss_park_e_kwh'].

    Integrated from the SCHEDULED series rather than res_storage.p_mw, because
    ConstControl applies the scheduled value exactly. That equivalence holds only
    while the storage is NON-CURTAILABLE: add any controller that can override
    p_mw (voltage-driven curtailment, an OPF step) and this must switch to reading
    res_storage.p_mw back out of the OutputWriter.

    Step duration is derived from STEPS_PER_DAY, never hard-coded, so a change of
    time resolution propagates here.

    Sign as in _inject_bss_storage: p_mw > 0 charges, < 0 discharges.
    config['bss_eta'] is the ONE-WAY efficiency -- charging stores p*dt*eta,
    discharging draws p*dt/eta -- so a full cycle costs eta**2.

    With config['bss_enforce_limits'] the per-step power is clipped to keep the
    stored energy inside [min_e_mwh, max_e_mwh]; without it the schedule runs
    untouched and only the warning fires. Breaches are reported AGGREGATED (count
    + worst step), never one line per step.

    This is the [BSS/soc] level: cumulative ENERGY. Its instantaneous POWER
    counterpart, [BSS/schedule], lives in _read_bss_schedule.

    Returns
    -------
    soc : np.ndarray          # fraction 0..1, SoC at the END of each step
    p_applied : np.ndarray    # MW, clipped when bss_enforce_limits is True
    warnings : list[str]
    """
    dt_h = 24.0 / STEPS_PER_DAY
    e_max_mwh = float(config['bss_park_e_kwh']) / 1000.0
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
            # "worst" = furthest from the middle of the band, in either direction
            if worst_soc is None or abs(frac - 0.5) > abs(worst_soc - 0.5):
                worst_step, worst_soc = k, frac
            if enforce:
                e_clipped = min(max(e_new, e_min_mwh), e_max_mwh)
                delta = e_clipped - e_mwh
                p_applied[k] = (delta / (dt_h * eta)) if delta > 0 else (delta * eta / dt_h)
                e_new = e_clipped
        e_mwh = e_new
        soc[k] = e_mwh / e_max_mwh

    warnings = []
    if n_out:
        warnings.append(
            f"[BSS/soc] state of charge leaves the "
            f"[{100 * BSS_SOC_MIN_FRAC:.0f}%, 100%] band at {n_out} of "
            f"{len(p_applied)} step(s) -- schedule inconsistent with "
            f"config['bss_park_e_kwh'] ({float(config['bss_park_e_kwh']):.1f} kWh); "
            f"worst at step {worst_step} ({100 * worst_soc:.1f}%)"
            + (" -- clipped (bss_enforce_limits=True)." if enforce
               else " -- applied as-is (bss_enforce_limits=False).")
        )
    return soc, p_applied, warnings


# ============================================================
#  3. TIME-SERIES POWER FLOW
# ============================================================
def _run_powerflow(net, df_profiles_p, df_profiles_q, config, df_pv_p=None,
                   df_bss_p=None):
    """Attach controllers and run the time-series power flow.

    Loads are ALWAYS included. When `df_pv_p` is provided (columns 'pv_{bus_id}',
    MW; sgens already created by _inject_pv_sgens), the PV layer is added and the
    extra flow signals (direction/reversal) useful for PV impact analysis
    (overvoltage, reverse flows) are logged too. When `df_pv_p` is None, the
    power flow is loads-only.

    When `df_bss_p` is provided (single column BSS_STORAGE_NAME, MW; storage
    already created by _inject_bss_storage), the park is driven too. Positive =
    charge, so it ADDS to consumption where a PV sgen subtracts from it.

    Returns
    -------
    dict : 'vm', 'line_load', 'trafo_load' (always), with PV
    'line_p_from', 'trafo_p_hv', 'ext_grid_p', 'sgen_p_out', and with BSS
    'storage_p'.
    """
    net.controller = net.controller.iloc[0:0]  # avoid stacking controllers on repeated calls

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
    # output_path=None -> results collected in memory only (ow.output), no CSV
    # round-trip through disk: cleaner for a callable module.
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

    run_timeseries(net, time_steps=range(n_timesteps), verbose=False)

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

    Returns
    -------
    summary : dict           # network_valid, min_voltage, max_loading, counts
    violations : list[dict]  # one entry per (element, timestep) out of bounds
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

    Long format, one row per record, so a MAS4TE agent can filter by
    `category`:
        - category == 'summary'   -> the verdict (is it viable?)
        - category == 'violation' -> where/when limits were breached
        - category == 'warning'   -> issues during the run (unmatched households...)
    """
    rows = []

    # Verdict lines
    for metric, value in summary.items():
        rows.append({'category': 'summary', 'element': 'network',
                     'metric': metric, 'value': value,
                     'timestep': None, 'detail': None})

    # Violation lines
    for v in violations:
        rows.append({'category': 'violation', 'element': v.get('element'),
                     'metric': v.get('metric'), 'value': v.get('value'),
                     'timestep': v.get('timestep'), 'detail': v.get('detail')})

    # Warning lines
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
    _format_output) as a human-readable Markdown report and write it to
    `out_path`.

    Markdown rather than plain text: the summary/violations tables and the
    warnings list render directly, while the file still reads fine as plain
    text if opened without a renderer.

    Returns
    -------
    out_path : str
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


def run_digital_twin(config, metadata, network, households,
                     pv_profiles=None, schedules_BSS=None):
    """Run the digital twin on a given network and return a network verdict.

    Parameters
    ----------
    config : dict                # tunable simulation/validation parameters
    metadata : pd.DataFrame      # per-household attributes (profile_id, source_street, ...)
    network : dict                # caller-supplied network bundle, see
        {"net", "buses", "mv_nodes", "mv_load_names", ...} -- built by a
        site-specific inputs/network.py, see digital_twin.utils.load_site_network.
        `net` is mutated in place (loads/sgens cleared and repopulated) -- copy
        it yourself before calling if you plan to reuse `network` across calls.
    households : pd.DataFrame    # consumption profiles (n_timesteps x n_profiles), MW
    pv_profiles : pd.DataFrame, optional
        PV production. Columns 'pv_{bus_id}' (integer network bus_id), time index
        aligned on households ('Mon 00:00'..'Sun 23:45'), values in MW (SAME
        unit as households -- if the production comes from
        pv.build_pv_production, which outputs kW, convert /1000 upstream, see
        main.py). When provided: injected as sgen, loads + PV power flow. When
        None: no PV (loads-only).
    schedules_BSS : pd.DataFrame, optional
        Central battery park schedule. One or more NUMERIC power columns (one per
        battery / agent / park sub-unit -- names are free), summed column-wise
        into a single park aggregate; a pre-aggregated single column is the same
        case. Time index aligned on households. Values in **kW** -- the only kW
        interface here, households and pv_profiles are in MW.
        SIGN: positive = CHARGING, negative = discharge -- the storage
        convention, OPPOSITE of the sgen one `pv_profiles` follows.
        When provided: injected as ONE storage element on config['bss_bus'],
        driven per timestep, SoC tracked and reported (bss_* metrics). When None:
        no BSS. Passing the schedule IS the activation, as for pv_profiles.
        Related config: bss_bus / bss_park_e_kwh / bss_park_p_kw / bss_eta /
        bss_enforce_limits (see digital_twin.utils._config_to_dict).

    Returns
    -------
    out : pd.DataFrame
        Long-format result: summary (verdict) + violations + warnings.
    """
    # fill in any config keys the caller didn't set with their defaults
    cfg = _config_to_dict(config)

    warnings = _validate_inputs(cfg, metadata, households, pv_profiles, schedules_BSS)
    warnings += network.get('warnings', [])

    df_p, df_q, bus_to_profile, w_assign = _assign_loads(
        network, metadata, households, cfg)
    warnings += w_assign

    net = network['net']

    # --- PV injection (native capability, driven by the pv_profiles argument) ---
    # pv_profiles provided: create one sgen per 'pv_{bus_id}' column and run the
    # loads + PV power flow together. pv_profiles=None: no PV, loads-only.
    df_pv_p = None
    if pv_profiles is not None and len(pv_profiles) > 0:
        # Positional alignment: _assign_loads builds df_p via .values (text index
        # 'Mon 00:00'.. is lost -> RangeIndex 0..N-1). Put pv_profiles on the same
        # RangeIndex BEFORE the alignment check and the power flow (DFData/
        # ConstControl index by position).
        df_pv_p = pv_profiles.reset_index(drop=True)
        _verify_temporal_alignment(df_p, df_pv_p)   # index guardrail
        _inject_pv_sgens(net, df_pv_p)

    # --- BSS injection (activation by PRESENCE, like pv_profiles above) ---
    # Schedule reduced to one park column, storage created on config['bss_bus'],
    # SoC integrated, and the (possibly clipped) series drives it in _run_powerflow.
    # households.index is passed for the offset pre-check -- df_p has already lost
    # its own to .values in _assign_loads.
    df_bss_p, bss_summary = None, {}
    if schedules_BSS is not None and len(schedules_BSS) > 0:
        df_bss_p, n_bss_cols, w_bss = _read_bss_schedule(
            schedules_BSS, df_p, cfg, load_time_index=households.index)
        warnings += w_bss
        _inject_bss_storage(net, cfg, network['mv_nodes'])

        # Simulated horizon only: a statement about the steps that actually run.
        n_bss = min(int(cfg['n_timesteps']), len(df_bss_p))
        soc, p_applied, w_soc = _track_bss_soc(
            df_bss_p[BSS_STORAGE_NAME].to_numpy()[:n_bss], cfg)
        warnings += w_soc
        # Clipped values written back where the ConstControl reads them, so the
        # power flow applies what the SoC was integrated from (see _track_bss_soc).
        df_bss_p.iloc[:n_bss, df_bss_p.columns.get_loc(BSS_STORAGE_NAME)] = p_applied

        p_applied_kw = p_applied / _BSS_KW_TO_MW
        dt_h = 24.0 / STEPS_PER_DAY
        charge_kw = p_applied_kw[p_applied_kw > 0]
        discharge_kw = p_applied_kw[p_applied_kw < 0]
        bss_summary = {
            'bss_bus': cfg['bss_bus'],
            'bss_n_schedule_columns': n_bss_cols,
            'bss_timesteps': n_bss,
            'bss_park_e_kwh': cfg['bss_park_e_kwh'],
            'bss_park_p_kw': cfg['bss_park_p_kw'],
            # Signed peaks, now that the convention is fixed: positive = charge.
            'bss_peak_charge_kw': float(charge_kw.max()) if len(charge_kw) else 0.0,
            'bss_peak_discharge_kw': float(discharge_kw.min()) if len(discharge_kw) else 0.0,
            'bss_mean_kw': float(p_applied_kw.mean()),
            'bss_soc_min_pct': 100.0 * float(soc.min()),
            'bss_soc_max_pct': 100.0 * float(soc.max()),
            'bss_soc_final_pct': 100.0 * float(soc[-1]),
            'bss_energy_charged_kwh': float(charge_kw.sum()) * dt_h,
            'bss_energy_discharged_kwh': float(discharge_kw.sum()) * dt_h,
        }

    pf = _run_powerflow(net, df_p, df_q, cfg, df_pv_p=df_pv_p, df_bss_p=df_bss_p)
    vm, line_load, trafo_load = pf['vm'], pf['line_load'], pf['trafo_load']

    summary, violations = _evaluate(vm, line_load, trafo_load, cfg)

    agg_summary, agg_mw = _aggregate_profile_summary(df_p)
    summary.update(agg_summary)

    # In kW like the load aggregate, and only when the layer ran -- so a
    # loads(+PV) verdict keeps exactly the keys it had before BSS existed.
    summary.update(bss_summary)

    out = _format_output(summary, violations, warnings)

    # Optional export of the full aggregate profile (672 steps), off by default
    # to keep the standard output light for MAS4TE consumers.
    if cfg.get('export_aggregate_profile'):
        profile_rows = pd.DataFrame({
            'category': 'profile_agg', 'element': None, 'metric': 'p_agg_mw',
            'value': agg_mw.values, 'timestep': range(len(agg_mw)), 'detail': None,
        })
        out = pd.concat([out, profile_rows], ignore_index=True)

    return out
